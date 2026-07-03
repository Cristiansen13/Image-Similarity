"""CARS196 fine-tuning with the patch-level Proxy-Anchor loss, mirroring
finetune_cub_proxy.py. See that file's docstring for the proxy design
(learnable set of M prototype patch tokens per class, scored via the same
symmetric MaxSim, training-aid only -- eval stays pure image-image MaxSim).
Uses the winning knobs found on CUB (M=8, alpha=32, proxy_lr=5e-2) as defaults.
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
HIDDEN_D = 768


class FineTuneModel(nn.Module):
    def __init__(self, model_name=MODEL_NAME, unfreeze_last_n=4):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        blocks = self.backbone.encoder.layer
        for p in self.backbone.embeddings.parameters():
            p.requires_grad = False
        for i, block in enumerate(blocks):
            grad = i >= len(blocks) - unfreeze_last_n
            for p in block.parameters():
                p.requires_grad = grad
        for p in self.backbone.layernorm.parameters():
            p.requires_grad = True

    def forward(self, pixel_values):
        out = self.backbone(pixel_values=pixel_values)
        return F.normalize(out.last_hidden_state[:, 1:, :], dim=-1)


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


def load_cars_index(cars_dir):
    names_path = os.path.join(cars_dir, "names.csv")
    with open(names_path) as f:
        class_names = [line.strip() for line in f if line.strip()]
    name_to_id = {name: str(i + 1) for i, name in enumerate(class_names)}

    by_class = defaultdict(list)
    car_data_root = os.path.join(cars_dir, "car_data", "car_data")
    for split in ("train", "test"):
        split_dir = os.path.join(car_data_root, split)
        for class_name in os.listdir(split_dir):
            class_id = name_to_id[class_name]
            class_dir = os.path.join(split_dir, class_name)
            for fname in os.listdir(class_dir):
                by_class[class_id].append(os.path.join(class_dir, fname))
    return by_class


def load_image(path, rng, augment):
    img = Image.open(path).convert("RGB")
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


def sample_batch(by_class, classes, processor, P, K, rng, augment):
    chosen = rng.sample(classes, P)
    paths, labels = [], []
    for c in chosen:
        pool = by_class[c]
        imgs = rng.sample(pool, min(K, len(pool)))
        paths.extend(imgs)
        labels.extend([c] * len(imgs))
    images = [load_image(p, rng, augment) for p in paths]
    pixel_values = processor(images=images, return_tensors="pt")["pixel_values"]
    return pixel_values, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cars-dir", required=True)
    ap.add_argument("--unfreeze-last-n", type=int, default=4)
    ap.add_argument("--P", type=int, default=16)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--proxy-lr", type=float, default=5e-2)
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--lr-schedule", action="store_true")
    ap.add_argument("--out-dir", default="checkpoints_cars_proxy")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    by_class = load_cars_index(args.cars_dir)
    train_classes = [str(i) for i in range(1, 99)]
    num_classes = len(train_classes)
    print(f"{num_classes} train classes; proxy = {args.M} tokens/class")

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, use_fast=True)
    processor.size = {"height": IMAGE_SIZE, "width": IMAGE_SIZE}
    model = FineTuneModel(unfreeze_last_n=args.unfreeze_last_n).to(device)
    proxies = nn.Parameter(torch.randn(num_classes, args.M, HIDDEN_D, device=device))

    backbone_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr, "weight_decay": 1e-4},
        {"params": [proxies], "lr": args.proxy_lr, "weight_decay": 0.0},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps) if args.lr_schedule else None

    print(f"\nTraining: P={args.P} K={args.K} batch={args.P * args.K} steps={args.steps} "
          f"lr={args.lr} proxy_lr={args.proxy_lr} M={args.M} alpha={args.alpha} delta={args.delta} "
          f"augment={args.augment} lr_schedule={args.lr_schedule}")
    t0 = time.time()
    running_loss = 0.0
    class_to_idx = {c: i for i, c in enumerate(train_classes)}
    for step in range(1, args.steps + 1):
        pixel_values, labels = sample_batch(by_class, train_classes, processor, args.P, args.K, rng, args.augment)
        pixel_values = pixel_values.to(device)
        label_idx = torch.tensor([class_to_idx[c] for c in labels], device=device)

        opt.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            patches = model(pixel_values)
            proxies_norm = F.normalize(proxies, dim=-1).to(patches.dtype)
            scores = image_to_proxy_maxsim(patches, proxies_norm)
        loss = proxy_anchor_loss(scores.float(), label_idx, args.alpha, args.delta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(backbone_params + [proxies], 1.0)
        opt.step()
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()
        if step % 20 == 0:
            elapsed = time.time() - t0
            cur_lr = opt.param_groups[0]["lr"]
            print(f"  step {step:5d}/{args.steps}  loss={running_loss / 20:.4f}  lr={cur_lr:.2e}  "
                  f"({elapsed:.0f}s, {step / elapsed:.2f} steps/s)")
            running_loss = 0.0

    torch.save(model.backbone.state_dict(), os.path.join(args.out_dir, "backbone_final.pt"))
    print(f"\nSaved final backbone to {args.out_dir}/backbone_final.pt")


if __name__ == "__main__":
    main()
