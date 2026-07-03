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
        patch_tokens = out.last_hidden_state[:, 1:, :]
        return F.normalize(patch_tokens, dim=-1)

def batched_symmetric_maxsim(patches):
    sim = torch.einsum("ipd,jqd->ijpq", patches, patches)
    a_to_b = sim.max(dim=3).values.mean(dim=2)
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cub-dir", required=True)
    ap.add_argument("--unfreeze-last-n", type=int, default=4)
    ap.add_argument("--P", type=int, default=16)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--out-dir", default="checkpoints_cub")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading CUB index...")
    by_class = load_cub_index(args.cub_dir)
    
    # Standard metric-learning CUB split: classes 1-100 are train, 101-200 are test.
    train_classes = [str(i) for i in range(1, 101)]
    test_classes = [str(i) for i in range(101, 201)]
    print(f"{len(train_classes)} train classes, {len(test_classes)} test classes.")

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, use_fast=True)
    processor.size = {"height": IMAGE_SIZE, "width": IMAGE_SIZE}
    model = FineTuneModel(unfreeze_last_n=args.unfreeze_last_n).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    images_root = os.path.join(args.cub_dir, "images")
    print(f"\nTraining: P={args.P} K={args.K} batch={args.P * args.K}  steps={args.steps}  lr={args.lr}")
    t0 = time.time()
    running_loss = 0.0
    for step in range(1, args.steps + 1):
        pixel_values, labels = sample_batch(by_class, train_classes, images_root, processor,
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

        if step % 500 == 0:
            ckpt_path = os.path.join(args.out_dir, f"backbone_step{step}.pt")
            torch.save(model.backbone.state_dict(), ckpt_path)

    final_path = os.path.join(args.out_dir, "backbone_final.pt")
    torch.save(model.backbone.state_dict(), final_path)
    print(f"\nSaved final backbone to {final_path}")

if __name__ == "__main__":
    main()
