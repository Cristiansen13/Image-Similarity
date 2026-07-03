"""GPU fine-tuning of the DINOv2 backbone itself (not just a frozen-feature
probe) -- the lever the CPU-only exploration never pulled. Everything before
this trained a tiny linear/MLP head on FROZEN DINOv2 patch embeddings; this
unfreezes the last few transformer blocks and fine-tunes them end-to-end
with an in-batch hard-negative contrastive loss over patch-level MaxSim.

Batches are P classes x K images/class (standard metric-learning batching).
For each batch, the full BxB patch-level MaxSim similarity matrix is computed
on-GPU, and a supervised-contrastive (SupCon-style) loss is applied directly
to those MaxSim scores -- same scoring mechanism used throughout this
project, just now with in-batch mining (much stronger than the static
pre-mined single-hardest-negative approach used on CPU) and a backbone that
actually adapts to the data instead of staying frozen.

Usage:
  python finetune_backbone.py --ebay-info /path/Ebay_info.txt --images-root /path/images_root
"""
import argparse
import json
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

MODEL_NAME = "facebook/dinov2-base"
IMAGE_SIZE = 224  # divisible by patch_size=14 -> clean 16x16 grid, consistent with the rest of the project


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
        print(f"Unfrozen last {unfreeze_last_n}/{len(blocks)} blocks + final layernorm: "
              f"{trainable:,}/{total:,} params trainable ({100 * trainable / total:.1f}%)")

    def forward(self, pixel_values):
        out = self.backbone(pixel_values=pixel_values)
        patch_tokens = out.last_hidden_state[:, 1:, :]  # drop CLS
        return F.normalize(patch_tokens, dim=-1)  # (B, N, D)


def batched_symmetric_maxsim(patches):
    """patches: (B, N, D), L2-normalized. Returns (B, B) symmetric MaxSim score matrix."""
    sim = torch.einsum("ipd,jqd->ijpq", patches, patches)  # (B,B,N,N)
    a_to_b = sim.max(dim=3).values.mean(dim=2)  # (B,B): mean_p max_q
    scores = 0.5 * (a_to_b + a_to_b.transpose(0, 1))
    return scores


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


def load_ebay_index(ebay_info_path):
    by_class = {}
    with open(ebay_info_path) as f:
        next(f)  # header
        for line in f:
            _, class_id, _, path = line.split()
            by_class.setdefault(class_id, []).append(path)
    return by_class


def sample_batch(by_class, classes, images_root, processor, P, K, rng):
    chosen = rng.sample(classes, P)
    paths, labels = [], []
    for c in chosen:
        pool = by_class[c]
        imgs = rng.sample(pool, min(K, len(pool)))
        paths.extend(imgs)
        labels.extend([c] * len(imgs))
    images = [Image.open(os.path.join(images_root, p)).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
              for p in paths]
    pixel_values = processor(images=images, return_tensors="pt")["pixel_values"]
    return pixel_values, labels


@torch.no_grad()
def evaluate(model, processor, eval_triplets, device):
    model.eval()
    correct = 0
    for t in eval_triplets:
        imgs = [Image.open(p).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
                for p in (t["anchor"], t["positive"], t["negative"])]
        pixel_values = processor(images=imgs, return_tensors="pt")["pixel_values"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            patches = model(pixel_values)
        scores = batched_symmetric_maxsim(patches.float())
        pos_score, neg_score = scores[0, 1].item(), scores[0, 2].item()
        if pos_score > neg_score:
            correct += 1
    model.train()
    return correct / len(eval_triplets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ebay-info", required=True)
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--eval-triplets", default=None, help="JSON file of {anchor,positive,negative} triplets")
    ap.add_argument("--unfreeze-last-n", type=int, default=4)
    ap.add_argument("--P", type=int, default=16)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading SOP index...")
    by_class = load_ebay_index(args.ebay_info)
    eligible = [c for c, imgs in by_class.items() if len(imgs) >= args.K]
    rng.shuffle(eligible)
    split_point = int(len(eligible) * 0.9)
    train_classes, holdout_classes = eligible[:split_point], eligible[split_point:]
    print(f"{len(train_classes)} train classes, {len(holdout_classes)} held-out classes "
          f"(held out for future eval, not used here)")

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, use_fast=True)
    processor.size = {"height": IMAGE_SIZE, "width": IMAGE_SIZE}
    model = FineTuneModel(unfreeze_last_n=args.unfreeze_last_n).to(device)

    eval_triplets = None
    if args.eval_triplets:
        with open(args.eval_triplets) as f:
            eval_triplets = json.load(f)
        print(f"Loaded {len(eval_triplets)} eval triplets from {args.eval_triplets}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    print(f"\nTraining: P={args.P} K={args.K} batch={args.P * args.K}  steps={args.steps}  lr={args.lr}")
    t0 = time.time()
    running_loss = 0.0
    for step in range(1, args.steps + 1):
        pixel_values, labels = sample_batch(by_class, train_classes, args.images_root, processor,
                                             args.P, args.K, rng)
        pixel_values = pixel_values.to(device)

        opt.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            patches = model(pixel_values)
            scores = batched_symmetric_maxsim(patches)
            loss = supcon_loss(scores.float(), labels)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        scaler.step(opt)
        scaler.update()

        running_loss += loss.item()
        if step % 20 == 0:
            elapsed = time.time() - t0
            print(f"  step {step:5d}/{args.steps}  loss={running_loss / 20:.4f}  "
                  f"({elapsed:.0f}s, {step / elapsed:.2f} steps/s)")
            running_loss = 0.0

        if eval_triplets and step % args.eval_every == 0:
            acc = evaluate(model, processor, eval_triplets, device)
            print(f"  [eval] step {step}: triplet_accuracy={acc:.3f}")

        if step % 500 == 0:
            ckpt_path = os.path.join(args.out_dir, f"backbone_step{step}.pt")
            torch.save(model.backbone.state_dict(), ckpt_path)
            print(f"  saved checkpoint: {ckpt_path}")

    final_path = os.path.join(args.out_dir, "backbone_final.pt")
    torch.save(model.backbone.state_dict(), final_path)
    print(f"\nSaved final backbone to {final_path}")

    if eval_triplets:
        final_acc = evaluate(model, processor, eval_triplets, device)
        print(f"Final triplet accuracy: {final_acc:.3f}")


if __name__ == "__main__":
    main()
