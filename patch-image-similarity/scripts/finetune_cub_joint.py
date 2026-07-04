"""Fine-tune DINOv2 with a JOINT global+patch loss -- everything in this
session's adaptive-scorer work so far used a ZERO-SHOT frozen backbone;
this tests whether actually training the backbone (with the validated
top-K MaxSim mechanism baked into the training objective, not just applied
post-hoc at eval time) improves both the global and patch representations
the adaptive system depends on.

Loss = SupCon(global CLS-cosine similarity matrix) + SupCon(top-K MaxSim
similarity matrix, K=16 -- CUB's own validated optimum from
check_topk_maxsim_cub.py) over the SAME P-classes-x-K-images batch. One
backbone, trained to be good at both scoring mechanisms simultaneously,
rather than the same frozen zero-shot backbone read out two different ways.

Keeps the proven augment + cosine-LR-schedule recipe from earlier sessions.

Usage: python finetune_cub_joint.py --cub-dir /path/CUB_200_2011 --steps 1000
"""
import argparse
import os
import random
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

MODEL_NAME = "facebook/dinov2-base"
IMAGE_SIZE = 224
AUG_SIZE = 256
PATCH_K = 16  # CUB's own validated top-K optimum


class JointModel(nn.Module):
    def __init__(self, unfreeze_last_n=4):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(MODEL_NAME)
        blocks = self.backbone.encoder.layer
        for p in self.backbone.embeddings.parameters():
            p.requires_grad = False
        for i, block in enumerate(blocks):
            grad = i >= len(blocks) - unfreeze_last_n
            for p in block.parameters():
                p.requires_grad = grad
        for p in self.backbone.layernorm.parameters():
            p.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Unfrozen last {unfreeze_last_n}/{len(blocks)} blocks: "
              f"{trainable:,}/{total:,} params ({100 * trainable / total:.1f}%)")

    def forward(self, pixel_values):
        out = self.backbone(pixel_values=pixel_values)
        cls = F.normalize(out.last_hidden_state[:, 0, :], dim=-1)
        patches = F.normalize(out.last_hidden_state[:, 1:, :], dim=-1)
        return cls, patches


def batched_topk_maxsim(patches, k):
    """patches: (B, N, D) -> (B, B) symmetric top-K MaxSim score matrix,
    fixed K for the whole batch (CUB is a fairly homogeneous-complexity
    dataset, so a single representative K is a reasonable simplification
    vs. full per-sample adaptive K, which the post-hoc adaptive system
    already handles at inference time across heterogeneous domains)."""
    sim = torch.einsum("ipd,jqd->ijpq", patches, patches)  # (B,B,N,N)
    max_i_to_j = sim.max(dim=3).values  # (B,B,N)
    max_j_to_i = sim.max(dim=2).values  # (B,B,N)
    k_eff = min(k, max_i_to_j.shape[-1])
    top_i_to_j = max_i_to_j.topk(k_eff, dim=-1).values.mean(dim=-1)  # (B,B)
    top_j_to_i = max_j_to_i.topk(k_eff, dim=-1).values.mean(dim=-1)  # (B,B)
    return 0.5 * (top_i_to_j + top_j_to_i)


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


def sample_batch(by_class, classes, images_root, processor, P, K, rng, augment):
    chosen = rng.sample(classes, P)
    paths, labels = [], []
    for c in chosen:
        pool = by_class[c]
        imgs = rng.sample(pool, min(K, len(pool)))
        paths.extend(imgs)
        labels.extend([c] * len(imgs))
    images = [load_image(p, images_root, rng, augment) for p in paths]
    pixel_values = processor(images=images, return_tensors="pt")["pixel_values"]
    return pixel_values, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cub-dir", required=True)
    ap.add_argument("--unfreeze-last-n", type=int, default=4)
    ap.add_argument("--P", type=int, default=16)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--global-weight", type=float, default=1.0)
    ap.add_argument("--patch-weight", type=float, default=1.0)
    ap.add_argument("--out-dir", default="checkpoints_cub_joint")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    by_class = load_cub_index(args.cub_dir)
    train_classes = [str(i) for i in range(1, 101)]
    images_root = os.path.join(args.cub_dir, "images")

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, use_fast=True)
    processor.size = {"height": IMAGE_SIZE, "width": IMAGE_SIZE}
    model = JointModel(unfreeze_last_n=args.unfreeze_last_n).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    print(f"\nTraining JOINT global+patch loss: P={args.P} K={args.K} steps={args.steps} "
          f"lr={args.lr} patch_k={PATCH_K} global_weight={args.global_weight} patch_weight={args.patch_weight}")
    t0 = time.time()
    running_loss, running_g, running_p = 0.0, 0.0, 0.0
    for step in range(1, args.steps + 1):
        pixel_values, labels = sample_batch(by_class, train_classes, images_root, processor,
                                             args.P, args.K, rng, augment=True)
        pixel_values = pixel_values.to(device)

        opt.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            cls, patches = model(pixel_values)
            global_scores = cls.float() @ cls.float().T
            patch_scores = batched_topk_maxsim(patches.float(), PATCH_K)
        global_loss = supcon_loss(global_scores, labels)
        patch_loss = supcon_loss(patch_scores, labels)
        loss = args.global_weight * global_loss + args.patch_weight * patch_loss
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        scaler.step(opt)
        scaler.update()
        scheduler.step()

        running_loss += loss.item()
        running_g += global_loss.item()
        running_p += patch_loss.item()
        if step % 20 == 0:
            elapsed = time.time() - t0
            print(f"  step {step:5d}/{args.steps}  loss={running_loss / 20:.4f}  "
                  f"global={running_g / 20:.4f}  patch={running_p / 20:.4f}  ({elapsed:.0f}s)")
            running_loss, running_g, running_p = 0.0, 0.0, 0.0

        if step % 500 == 0:
            torch.save(model.backbone.state_dict(), os.path.join(args.out_dir, f"backbone_step{step}.pt"))

    torch.save(model.backbone.state_dict(), os.path.join(args.out_dir, "backbone_final.pt"))
    print(f"\nSaved final backbone to {args.out_dir}/backbone_final.pt")


if __name__ == "__main__":
    main()
