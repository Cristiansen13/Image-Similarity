"""CARS196 fine-tuning (SupCon variant), mirroring finetune_cub_backbone.py.

Standard CARS196 metric-learning protocol: classes 1-98 train, 99-196 test
(by class name, canonical order = names.csv line order, 1-indexed -- NOT the
dataset's own train/test folders, which are a classification split with every
class in both; we pool images from both folders per class then re-split by
class ourselves.
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


def batched_symmetric_maxsim(patches):
    sim = torch.einsum("ipd,jqd->ijpq", patches, patches)
    a_to_b = sim.max(dim=3).values.mean(dim=2)
    return 0.5 * (a_to_b + a_to_b.transpose(0, 1))


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


def load_cars_index(cars_dir):
    """Returns {class_id (1-indexed str): [absolute image paths]}, pooling both
    the dataset's own train/ and test/ folders per class (see module docstring)."""
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
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--lr-schedule", action="store_true")
    ap.add_argument("--out-dir", default="checkpoints_cars")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading CARS196 index...")
    by_class = load_cars_index(args.cars_dir)
    train_classes = [str(i) for i in range(1, 99)]
    test_classes = [str(i) for i in range(99, 197)]
    print(f"{len(train_classes)} train classes, {len(test_classes)} test classes.")

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, use_fast=True)
    processor.size = {"height": IMAGE_SIZE, "width": IMAGE_SIZE}
    model = FineTuneModel(unfreeze_last_n=args.unfreeze_last_n).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps) if args.lr_schedule else None

    print(f"\nTraining: P={args.P} K={args.K} batch={args.P * args.K} steps={args.steps} lr={args.lr} "
          f"augment={args.augment} lr_schedule={args.lr_schedule}")
    t0 = time.time()
    running_loss = 0.0
    for step in range(1, args.steps + 1):
        pixel_values, labels = sample_batch(by_class, train_classes, processor, args.P, args.K, rng, args.augment)
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

    torch.save(model.backbone.state_dict(), os.path.join(args.out_dir, "backbone_final.pt"))
    print(f"\nSaved final backbone to {args.out_dir}/backbone_final.pt")


if __name__ == "__main__":
    main()
