"""SOP fine-tuning with the patch-level Proxy-Anchor loss (see finetune_cub_proxy.py
for the design). SOP has ~11,318 train classes vs CUB's 100, so the proxy-score
tensor (B, C, N, M) = 128 x 11318 x 256 x 8 ~ 3B elements can't be materialized
at once. We compute scores in chunks over the proxy (class) dimension, each chunk
wrapped in gradient checkpointing so only one chunk's intermediate is resident in
both the forward and the backward pass -- this preserves EXACT Proxy-Anchor
semantics (every image compared against ALL class proxies) within a 24GB budget.

Keeps the proven augment + cosine-LR schedule. Trains only on official
Ebay_train.txt classes; eval (eval_full_test.py on Ebay_test.txt) is unchanged
pure image-image MaxSim, so the checkpoint is directly comparable to the
SupCon-trained ones. Baseline to beat: aug+sched+SupCon SOP R@1 = 0.8121.
"""
import argparse
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.checkpoint import checkpoint
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
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Unfrozen last {unfreeze_last_n}/{len(blocks)} blocks: "
              f"{trainable:,}/{total:,} params ({100 * trainable / total:.1f}%)")

    def forward(self, pixel_values):
        out = self.backbone(pixel_values=pixel_values)
        return F.normalize(out.last_hidden_state[:, 1:, :], dim=-1)


def scores_for_chunk(patches, proxy_chunk):
    """patches (B,N,D), proxy_chunk (Cc,M,D) both normalized+same dtype -> (B,Cc) MaxSim."""
    sim = torch.einsum("bnd,cmd->bcnm", patches, proxy_chunk)  # (B,Cc,N,M)
    i2p = sim.max(dim=3).values.mean(dim=2)
    p2i = sim.max(dim=2).values.mean(dim=2)
    return 0.5 * (i2p + p2i)


def chunked_scores(patches, proxies_norm, chunk):
    C = proxies_norm.shape[0]
    out = []
    for c0 in range(0, C, chunk):
        pc = proxies_norm[c0:c0 + chunk].to(patches.dtype)
        out.append(checkpoint(scores_for_chunk, patches, pc, use_reentrant=False))
    return torch.cat(out, dim=1)  # (B, C)


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


def load_ebay_index(ebay_info_path):
    by_class = {}
    with open(ebay_info_path) as f:
        next(f)
        for line in f:
            _, class_id, _, path = line.split()
            by_class.setdefault(class_id, []).append(path)
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
    ap.add_argument("--ebay-info", required=True, help="Ebay_train.txt ONLY (never Ebay_info.txt)")
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--unfreeze-last-n", type=int, default=4)
    ap.add_argument("--P", type=int, default=32)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--proxy-lr", type=float, default=1e-2)
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--proxy-chunk", type=int, default=1024)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--lr-schedule", action="store_true")
    ap.add_argument("--out-dir", default="checkpoints_sop_proxy")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading SOP train index (official split -- disjoint from Ebay_test.txt)...")
    by_class = load_ebay_index(args.ebay_info)
    train_classes = [c for c, imgs in by_class.items() if len(imgs) >= args.K]
    class_to_idx = {c: i for i, c in enumerate(train_classes)}
    num_classes = len(train_classes)
    print(f"{num_classes} train classes; proxy = {args.M} tokens/class, "
          f"scored in chunks of {args.proxy_chunk}")

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
    for step in range(1, args.steps + 1):
        pixel_values, labels = sample_batch(by_class, train_classes, args.images_root, processor,
                                             args.P, args.K, rng, args.augment)
        pixel_values = pixel_values.to(device)
        label_idx = torch.tensor([class_to_idx[c] for c in labels], device=device)

        opt.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            patches = model(pixel_values)
            proxies_norm = F.normalize(proxies, dim=-1)
            scores = chunked_scores(patches, proxies_norm, args.proxy_chunk)
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

        if step % 500 == 0:
            torch.save(model.backbone.state_dict(), os.path.join(args.out_dir, f"backbone_step{step}.pt"))
            print(f"  saved checkpoint at step {step}")

    torch.save(model.backbone.state_dict(), os.path.join(args.out_dir, "backbone_final.pt"))
    print(f"\nSaved final backbone to {args.out_dir}/backbone_final.pt")


if __name__ == "__main__":
    main()
