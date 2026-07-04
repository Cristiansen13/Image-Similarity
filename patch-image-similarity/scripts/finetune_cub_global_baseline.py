"""Backbone-controlled comparison baseline: standard GLOBAL-embedding
Proxy-Anchor (Kim et al. 2020) on our same DINOv2 backbone -- the literature-
standard recipe (single proxy VECTOR per class, cosine similarity between one
pooled embedding per image and each proxy, no patch-level anything).

Purpose: isolate how much of our result is "DINOv2 is just a strong backbone"
vs. "our patch-level MaxSim architecture adds value beyond a good backbone."
Published SOTA (Proxy-Anchor 79.1-80.3% on SOP, etc.) used ResNet50/
BN-Inception; if DINOv2 + this exact standard recipe already matches or beats
that, the gain is mostly backbone. If it falls notably short of our patch-
MaxSim+proxy result (82.66% CUB), the patch architecture is doing real work.

Same augment + cosine LR schedule as our real recipe, for a fair comparison --
only the architecture/loss differs (global pooled embedding + standard
Proxy-Anchor vs. patch-set + our MaxSim-based proxy adaptation).
"""
import argparse
import json
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


class GlobalEmbedModel(nn.Module):
    """Standard global-embedding model: CLS token -> normalized vector (the
    conventional way to get one embedding per image from a ViT, as opposed to
    our project's patch-set representation)."""
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
        cls = out.last_hidden_state[:, 0, :]  # CLS token = the standard global embedding
        return F.normalize(cls, dim=-1)


def global_proxy_anchor_loss(embeds, proxies, label_idx, alpha, delta):
    """Standard Proxy-Anchor loss: cosine sim between embeds (B,D) and proxies
    (C,D) -- one vector per class, not a set. Same logsumexp stabilization as
    our patch-level variant for consistency, though cosine sim in [-1,1] is
    the range Proxy-Anchor's alpha was originally tuned for."""
    scores = embeds @ proxies.T  # (B, C)
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


@torch.no_grad()
def evaluate_full(model, processor, cub_dir, device, batch_size=128):
    """Full R@1/2/4/8 + MAP@R on the test split (classes 101-200), scored by
    plain cosine similarity between global embeddings -- no MaxSim, no
    two-stage, just the standard global-embedding retrieval protocol."""
    by_class = load_cub_index(cub_dir)
    test_classes = [str(i) for i in range(101, 201)]
    images_root = os.path.join(cub_dir, "images")
    paths, classes = [], []
    for c in test_classes:
        for p in by_class[c]:
            paths.append(p)
            classes.append(c)
    N = len(paths)

    model.eval()
    all_embeds = torch.zeros((N, 768), device=device)
    for i in range(0, N, batch_size):
        batch_paths = paths[i:i + batch_size]
        imgs = [Image.open(os.path.join(images_root, p)).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
                for p in batch_paths]
        pixel_values = processor(images=imgs, return_tensors="pt")["pixel_values"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            embeds = model(pixel_values)
        all_embeds[i:i + len(batch_paths)] = embeds.float()
    model.train()

    classes_t = torch.tensor([hash(c) for c in classes], device=device)
    sims = all_embeds @ all_embeds.T  # (N, N)
    sims.fill_diagonal_(-1e9)

    class_counts = defaultdict(int)
    for c in classes:
        class_counts[c] += 1
    R_per_query = torch.tensor([class_counts[c] - 1 for c in classes], device=device).clamp(min=1, max=N - 1)

    K = min(100, N - 1)
    _, ranked_idx = sims.topk(K, dim=1)
    ranked_classes = classes_t[ranked_idx]
    query_classes = classes_t.unsqueeze(1)
    relevant = (ranked_classes == query_classes).float()

    recall_at = {}
    for r in (1, 2, 4, 8):
        recall_at[r] = relevant[:, :r].any(dim=1).float().mean().item()

    cum_relevant = relevant.cumsum(dim=1)
    ranks = torch.arange(1, K + 1, device=device, dtype=torch.float32).unsqueeze(0)
    precision_at_rank = cum_relevant / ranks
    R_batch = R_per_query.clamp(max=K)
    rank_mask = (torch.arange(1, K + 1, device=device).unsqueeze(0) <= R_batch.unsqueeze(1)).float()
    ap = (precision_at_rank * relevant * rank_mask).sum(dim=1) / R_batch.float()
    map_at_r = ap.mean().item()

    return {"recall_at_1": recall_at[1], "recall_at_2": recall_at[2],
            "recall_at_4": recall_at[4], "recall_at_8": recall_at[8], "map_at_r": map_at_r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cub-dir", required=True)
    ap.add_argument("--unfreeze-last-n", type=int, default=4)
    ap.add_argument("--P", type=int, default=16)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--proxy-lr", type=float, default=5e-2)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--lr-schedule", action="store_true")
    ap.add_argument("--out-dir", default="checkpoints_cub_global_baseline")
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

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, use_fast=True)
    processor.size = {"height": IMAGE_SIZE, "width": IMAGE_SIZE}
    model = GlobalEmbedModel(unfreeze_last_n=args.unfreeze_last_n).to(device)
    proxies = nn.Parameter(torch.randn(num_classes, 768, device=device))

    backbone_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr, "weight_decay": 1e-4},
        {"params": [proxies], "lr": args.proxy_lr, "weight_decay": 0.0},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps) if args.lr_schedule else None

    images_root = os.path.join(args.cub_dir, "images")
    print(f"\nTraining (GLOBAL baseline): P={args.P} K={args.K} steps={args.steps} lr={args.lr} "
          f"proxy_lr={args.proxy_lr} alpha={args.alpha} augment={args.augment} lr_schedule={args.lr_schedule}")
    t0 = time.time()
    running_loss = 0.0
    for step in range(1, args.steps + 1):
        pixel_values, labels = sample_batch(by_class, train_classes, images_root, processor,
                                             args.P, args.K, rng, args.augment)
        pixel_values = pixel_values.to(device)
        label_idx = torch.tensor([class_to_idx[c] for c in labels], device=device)

        opt.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            embeds = model(pixel_values)
            proxies_norm = F.normalize(proxies, dim=-1).to(embeds.dtype)
        loss = global_proxy_anchor_loss(embeds.float(), proxies_norm.float(), label_idx, args.alpha, args.delta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(backbone_params + [proxies], 1.0)
        opt.step()
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()
        if step % 20 == 0:
            elapsed = time.time() - t0
            print(f"  step {step:5d}/{args.steps}  loss={running_loss / 20:.4f}  ({elapsed:.0f}s)")
            running_loss = 0.0

    torch.save(model.backbone.state_dict(), os.path.join(args.out_dir, "backbone_final.pt"))
    print(f"Saved backbone to {args.out_dir}/backbone_final.pt")

    print("\nEvaluating (global cosine similarity, standard protocol)...")
    metrics = evaluate_full(model, processor, args.cub_dir, device)
    print(f"Results: {metrics}")

    with open(os.path.join(args.out_dir, "global_baseline_results.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved to {args.out_dir}/global_baseline_results.json")


if __name__ == "__main__":
    main()
