"""CUB fine-tuning with a PATCH-LEVEL Proxy-Anchor loss -- an adaptation of the
Proxy-Anchor metric-learning loss (Kim et al. 2020, the method topping the
published SOP table at 79-80% R@1) to our patch-set MaxSim architecture.

Standard Proxy-Anchor: one learnable proxy VECTOR per class; score = cosine
between an image's single global embedding and each proxy. Every batch item is
compared against ALL class proxies, giving far denser gradient signal than
in-batch SupCon (which only sees the handful of classes in the current batch).

Our adaptation: our similarity is symmetric patch MaxSim between two SETS of
patch tokens, so a single proxy vector doesn't fit. Instead each class gets a
learnable SET of M "prototype patch tokens" (M, D), and an image is scored
against a proxy with the SAME symmetric MaxSim used image-to-image throughout
this project. Proxies are a TRAINING AID ONLY -- inference/eval is unchanged
(pure image-image MaxSim via eval_cub_test.py), so the resulting backbone
checkpoint is directly comparable to the SupCon-trained ones.

Keeps the two proven wins (augmentation + cosine LR schedule) so this isolates
the loss change: aug+sched+proxy-anchor  vs  aug+sched+supcon (CUB=77.92%).
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
HIDDEN_D = 768  # dinov2-base


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
    """patches (B,N,D) normalized, proxies (C,M,D) normalized -> (B,C) symmetric MaxSim."""
    sim = torch.einsum("bnd,cmd->bcnm", patches, proxies)  # (B,C,N,M)
    img_to_proxy = sim.max(dim=3).values.mean(dim=2)  # each img patch -> best proxy token
    proxy_to_img = sim.max(dim=2).values.mean(dim=2)  # each proxy token -> best img patch
    return 0.5 * (img_to_proxy + proxy_to_img)


def proxy_anchor_loss(scores, label_idx, alpha, delta):
    """scores (B,C) fp32, label_idx (B,) long. Numerically stable via logsumexp,
    which matters here because our MaxSim scores sit in a compressed positive band
    (~0.3-0.7) rather than the [-1,1] range Proxy-Anchor's alpha was tuned for, so a
    naive exp(alpha*(s+delta)) would overflow."""
    B, C = scores.shape
    device = scores.device
    pos_mask = torch.zeros(B, C, device=device, dtype=torch.bool)
    pos_mask[torch.arange(B, device=device), label_idx] = True
    neg_mask = ~pos_mask
    with_pos = pos_mask.any(dim=0)  # (C,) proxies with >=1 positive in batch
    num_pos = with_pos.sum().clamp(min=1)

    def masked_lse_with_zero(z, mask):  # over batch dim -> (C,)  == log(1 + sum_mask exp(z))
        z = z.masked_fill(~mask, float("-inf"))
        zeros = torch.zeros(1, z.shape[1], device=z.device, dtype=z.dtype)
        return torch.logsumexp(torch.cat([z, zeros], dim=0), dim=0)

    pos_term = masked_lse_with_zero(-alpha * (scores - delta), pos_mask)  # (C,)
    pos_loss = (pos_term * with_pos).sum() / num_pos
    neg_term = masked_lse_with_zero(alpha * (scores + delta), neg_mask)  # (C,)
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
        max_off = AUG_SIZE - IMAGE_SIZE
        x, y = rng.randint(0, max_off), rng.randint(0, max_off)
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
    ap.add_argument("--proxy-lr", type=float, default=1e-2)
    ap.add_argument("--M", type=int, default=8, help="prototype patch tokens per class proxy")
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--lr-schedule", action="store_true")
    ap.add_argument("--out-dir", default="checkpoints_cub_proxy")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    by_class = load_cub_index(args.cub_dir)
    train_classes = [str(i) for i in range(1, 101)]  # standard CUB metric-learning split
    class_to_idx = {c: i for i, c in enumerate(train_classes)}
    num_classes = len(train_classes)
    print(f"{num_classes} train classes; proxy = {args.M} prototype tokens/class")

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

    images_root = os.path.join(args.cub_dir, "images")
    print(f"\nTraining: P={args.P} K={args.K} batch={args.P * args.K} steps={args.steps} "
          f"lr={args.lr} proxy_lr={args.proxy_lr} M={args.M} alpha={args.alpha} delta={args.delta} "
          f"augment={args.augment} lr_schedule={args.lr_schedule}")
    t0 = time.time()
    running_loss = 0.0
    for step in range(1, args.steps + 1):
        pixel_values, labels = sample_batch(by_class, train_classes, images_root, processor,
                                             args.P, args.K, rng, args.augment)
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

    final_path = os.path.join(args.out_dir, "backbone_final.pt")
    torch.save(model.backbone.state_dict(), final_path)
    print(f"\nSaved final backbone to {final_path}")


if __name__ == "__main__":
    main()
