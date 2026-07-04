"""Tuned ResNet50 recipe, attempting a genuinely fair comparison against
published SOTA (which all use Proxy-Anchor-family losses, not our earlier
ad-hoc SupCon substitute).

Changes from the untuned first attempt (finetune_cub_joint_resnet50.py,
47.87% R@1, 17-23pt behind published SOTA):
  1. Proxy-Anchor loss (global: one proxy vector/class; patch: M=8 prototype
     tokens/class, same design validated on DINOv2 earlier this session) --
     matches the actual method literature numbers come from, instead of a
     substitute loss that was never the fair comparison to begin with.
  2. Higher backbone LR (tunable via --lr, try higher than the 2e-5 that was
     copied verbatim from the DINOv2/ViT recipe -- CNNs conventionally fine-
     tune faster).
  3. More steps (default 3000, matching this project's "serious" recipes,
     not the untuned 1000).
  4. proxy_lr separate and higher (proxies need to move fast, matching what
     was validated on DINOv2: M=8, alpha=32, proxy_lr=5e-2).

Usage: python finetune_cub_joint_resnet50_v2.py --cub-dir /path/CUB_200_2011 \
    --steps 3000 --lr 1e-4 --proxy-lr 5e-2 --alpha 32 --M 8
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
PATCH_D = 2048
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class ResNet50Model(nn.Module):
    def __init__(self, unfreeze_layer3=False):
        super().__init__()
        base = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool, base.layer1, base.layer2)
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        for p in self.stem.parameters():
            p.requires_grad = False
        for p in self.layer3.parameters():
            p.requires_grad = unfreeze_layer3
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Trainable: {trainable:,}/{total:,} ({100 * trainable / total:.1f}%) "
              f"unfreeze_layer3={unfreeze_layer3}")

    def forward(self, pixel_values):
        feat = self.stem(pixel_values)
        feat = self.layer3(feat)
        feat = self.layer4(feat)
        B, C, H, W = feat.shape
        patches = F.normalize(feat.permute(0, 2, 3, 1).reshape(B, H * W, C), dim=-1)
        global_emb = F.normalize(feat.mean(dim=(2, 3)), dim=-1)
        return global_emb, patches


def image_to_proxy_maxsim(patches, proxies):
    sim = torch.einsum("bnd,cmd->bcnm", patches, proxies)
    img_to_proxy = sim.max(dim=3).values.mean(dim=2)
    proxy_to_img = sim.max(dim=2).values.mean(dim=2)
    return 0.5 * (img_to_proxy + proxy_to_img)


def proxy_anchor_loss(scores, label_idx, alpha, delta):
    B, C = scores.shape
    device = scores.device
    pos_mask = torch.zeros(B, C, device=device, dtype=torch.bool)
    pos_mask[torch.arange(B, device=device), label_idx] = True
    neg_mask = ~pos_mask
    with_pos = pos_mask.any(dim=0)
    num_pos = with_pos.sum().clamp(min=1)

    def masked_lse_with_zero(z, mask):
        z = z.masked_fill(~mask, float("-inf"))
        zeros = torch.zeros(1, z.shape[1], device=z.device, dtype=z.dtype)
        return torch.logsumexp(torch.cat([z, zeros], dim=0), dim=0)

    pos_term = masked_lse_with_zero(-alpha * (scores - delta), pos_mask)
    pos_loss = (pos_term * with_pos).sum() / num_pos
    neg_term = masked_lse_with_zero(alpha * (scores + delta), neg_mask)
    neg_loss = neg_term.sum() / C
    return pos_loss + neg_loss


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
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--proxy-lr", type=float, default=5e-2)
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--unfreeze-layer3", action="store_true")
    ap.add_argument("--out-dir", default="checkpoints_cub_joint_resnet50_v2")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    by_class = load_cub_index(args.cub_dir)
    train_classes = [str(i) for i in range(1, 101)]
    class_to_idx = {c: i for i, c in enumerate(train_classes)}
    num_classes = len(train_classes)
    images_root = os.path.join(args.cub_dir, "images")

    model = ResNet50Model(unfreeze_layer3=args.unfreeze_layer3).to(device)
    global_proxies = nn.Parameter(torch.randn(num_classes, PATCH_D, device=device))
    patch_proxies = nn.Parameter(torch.randn(num_classes, args.M, PATCH_D, device=device))

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": trainable_params, "lr": args.lr, "weight_decay": 1e-4},
        {"params": [global_proxies, patch_proxies], "lr": args.proxy_lr, "weight_decay": 0.0},
    ])
    scaler = torch.amp.GradScaler("cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    print(f"\nTraining ResNet50 PROXY-ANCHOR (tuned v2): P={args.P} K={args.K} steps={args.steps} "
          f"lr={args.lr} proxy_lr={args.proxy_lr} M={args.M} alpha={args.alpha}")
    t0 = time.time()
    running_loss, running_g, running_p = 0.0, 0.0, 0.0
    for step in range(1, args.steps + 1):
        pixel_values, labels = sample_batch(by_class, train_classes, images_root,
                                             args.P, args.K, rng, True, device)
        label_idx = torch.tensor([class_to_idx[c] for c in labels], device=device)

        opt.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            global_emb, patches = model(pixel_values)
            global_proxies_norm = F.normalize(global_proxies, dim=-1).to(global_emb.dtype)
            patch_proxies_norm = F.normalize(patch_proxies, dim=-1).to(patches.dtype)
            global_scores = global_emb.float() @ global_proxies_norm.float().T
            patch_scores = image_to_proxy_maxsim(patches.float(), patch_proxies_norm.float())
        global_loss = proxy_anchor_loss(global_scores, label_idx, args.alpha, args.delta)
        patch_loss = proxy_anchor_loss(patch_scores, label_idx, args.alpha, args.delta)
        loss = global_loss + patch_loss
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(trainable_params + [global_proxies, patch_proxies], 1.0)
        scaler.step(opt)
        scaler.update()
        scheduler.step()

        running_loss += loss.item(); running_g += global_loss.item(); running_p += patch_loss.item()
        if step % 50 == 0:
            elapsed = time.time() - t0
            print(f"  step {step:5d}/{args.steps}  loss={running_loss / 50:.4f}  "
                  f"global={running_g / 50:.4f}  patch={running_p / 50:.4f}  ({elapsed:.0f}s)")
            running_loss, running_g, running_p = 0.0, 0.0, 0.0

        if step % 1000 == 0:
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_step{step}.pt"))

    torch.save(model.state_dict(), os.path.join(args.out_dir, "model_final.pt"))
    print(f"\nSaved final model to {args.out_dir}/model_final.pt")


if __name__ == "__main__":
    main()
