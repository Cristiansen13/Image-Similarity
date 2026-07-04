"""Backbone-matched comparison against literature: train the SAME joint
global+patch recipe as finetune_cub_joint.py, but on ResNet50 (standard
ImageNet-supervised, comparable era/tier to BN-Inception -- what Proxy-Anchor/
MS/SoftTriple actually used) instead of DINOv2.

This is the piece that was missing: everything so far compared DINOv2+our-
architecture against BN-Inception+their-architecture (confounded by backbone).
This isolates architecture from backbone -- if our recipe beats Proxy-Anchor
et al. on a comparable-tier backbone, that's a real architecture win, not
just "DINOv2 is a strong backbone" again.

Patch representation: ResNet50 layer4 output (B,2048,7,7) treated as 49 patch
tokens (the conv analogue of ViT patches, same as resnet50_backbone_check.py
used earlier this session). K not separately tuned for this backbone (unlike
DINOv2's validated K=16) -- using a proportionally scaled-down K given far
fewer patches (49 vs 256), noted as a limitation, not exhaustively searched.

Usage: python finetune_cub_joint_resnet50.py --cub-dir /path/CUB_200_2011 --steps 1000
"""
import argparse
import os
import random
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import torchvision.transforms.functional as TF
from PIL import Image

IMAGE_SIZE = 224
AUG_SIZE = 256
PATCH_K = 4  # proportionally scaled down from DINOv2's K=16 (16/256 patches ~ 4/49)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class ResNet50JointModel(nn.Module):
    def __init__(self):
        super().__init__()
        base = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool,
                                   base.layer1, base.layer2, base.layer3)
        self.layer4 = base.layer4
        for p in self.stem.parameters():
            p.requires_grad = False
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Trainable (layer4 only): {trainable:,}/{total:,} ({100 * trainable / total:.1f}%)")

    def forward(self, pixel_values):
        feat = self.stem(pixel_values)
        feat = self.layer4(feat)  # (B, 2048, 7, 7)
        B, C, H, W = feat.shape
        patches = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
        patches = F.normalize(patches, dim=-1)
        global_emb = F.normalize(feat.mean(dim=(2, 3)), dim=-1)
        return global_emb, patches


def batched_topk_maxsim(patches, k):
    sim = torch.einsum("ipd,jqd->ijpq", patches, patches)
    max_i_to_j = sim.max(dim=3).values
    max_j_to_i = sim.max(dim=2).values
    k_eff = min(k, max_i_to_j.shape[-1])
    top_i = max_i_to_j.topk(k_eff, dim=-1).values.mean(dim=-1)
    top_j = max_j_to_i.topk(k_eff, dim=-1).values.mean(dim=-1)
    return 0.5 * (top_i + top_j)


def supcon_loss(scores, labels, temperature=0.1):
    device = scores.device
    labels = torch.tensor([int(l) for l in labels], device=device)
    same_class = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    same_class.fill_diagonal_(0)
    logits = scores / temperature
    logits = logits.masked_fill(torch.eye(len(labels), device=device, dtype=torch.bool), -1e9)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    denom = same_class.sum(1).clamp(min=1)
    mean_log_prob_pos = (same_class * log_prob).sum(1) / denom
    return -mean_log_prob_pos.mean()


def load_cub_index(cub_dir):
    by_class = defaultdict(list)
    image_paths = {}
    with open(os.path.join(cub_dir, "images.txt")) as f:
        for line in f:
            image_id, path = line.split()
            image_paths[image_id] = path
    with open(os.path.join(cub_dir, "image_class_labels.txt")) as f:
        for line in f:
            image_id, class_id = line.split()
            by_class[class_id].append(image_paths[image_id])
    return by_class


def load_image(path, images_root, rng, augment):
    img = Image.open(os.path.join(images_root, path)).convert("RGB")
    if augment:
        img = img.resize((AUG_SIZE, AUG_SIZE))
        off = AUG_SIZE - IMAGE_SIZE
        x, y = rng.randint(0, off), rng.randint(0, off)
        img = img.crop((x, y, x + IMAGE_SIZE, y + IMAGE_SIZE))
        if rng.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    return img


def to_pixel_values(images, device):
    batch = torch.stack([TF.to_tensor(img) for img in images])
    return (batch.to(device) - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)


def sample_batch(by_class, classes, images_root, P, K, rng, augment, device):
    chosen = rng.sample(classes, P)
    paths, labels = [], []
    for c in chosen:
        pool = by_class[c]
        imgs = rng.sample(pool, min(K, len(pool)))
        paths.extend(imgs)
        labels.extend([c] * len(imgs))
    images = [load_image(p, images_root, rng, augment) for p in paths]
    return to_pixel_values(images, device), labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cub-dir", required=True)
    ap.add_argument("--P", type=int, default=16)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--out-dir", default="checkpoints_cub_joint_resnet50")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    by_class = load_cub_index(args.cub_dir)
    train_classes = [str(i) for i in range(1, 101)]
    images_root = os.path.join(args.cub_dir, "images")

    model = ResNet50JointModel().to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    print(f"\nTraining ResNet50 JOINT: P={args.P} K={args.K} steps={args.steps} "
          f"lr={args.lr} patch_k={PATCH_K}")
    t0 = time.time()
    running_loss, running_g, running_p = 0.0, 0.0, 0.0
    for step in range(1, args.steps + 1):
        pixel_values, labels = sample_batch(by_class, train_classes, images_root,
                                             args.P, args.K, rng, True, device)
        opt.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            global_emb, patches = model(pixel_values)
            global_scores = global_emb.float() @ global_emb.float().T
            patch_scores = batched_topk_maxsim(patches.float(), PATCH_K)
        global_loss = supcon_loss(global_scores, labels)
        patch_loss = supcon_loss(patch_scores, labels)
        loss = global_loss + patch_loss
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        scaler.step(opt)
        scaler.update()
        scheduler.step()

        running_loss += loss.item(); running_g += global_loss.item(); running_p += patch_loss.item()
        if step % 20 == 0:
            elapsed = time.time() - t0
            print(f"  step {step:5d}/{args.steps}  loss={running_loss / 20:.4f}  "
                  f"global={running_g / 20:.4f}  patch={running_p / 20:.4f}  ({elapsed:.0f}s)")
            running_loss, running_g, running_p = 0.0, 0.0, 0.0

        if step % 500 == 0:
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_step{step}.pt"))

    torch.save(model.state_dict(), os.path.join(args.out_dir, "model_final.pt"))
    print(f"\nSaved final model to {args.out_dir}/model_final.pt")


if __name__ == "__main__":
    main()
