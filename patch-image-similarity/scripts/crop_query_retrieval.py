"""Partial-crop-query retrieval: the test global pooling can't win in
principle, not just one it currently loses by dataset luck.

Query = a single deterministic partial crop of each test image (default 50%
area, same crop for every method so the comparison is exactly fair). Gallery
= the FULL, uncropped test images (self excluded). Task: does the cropped
query retrieve another image of the same class as its top-1 match?

Global pooling collapses a crop into a single vector that lives in a
different region of embedding space than the full image's vector -- there is
no mechanism to recover "this crop corresponds to a sub-region of that
gallery image." Patch MaxSim never collapses spatial structure, so a crop's
patches can still find their best correspondence inside a full gallery
image's patch set. If patch-level matching has any real edge over global
pooling, this is where it should show up.

Uses two ALREADY-TRAINED checkpoints (no retraining needed for this test):
  --global-checkpoint: from finetune_cub_global_baseline.py (CLS-token model)
  --patch-checkpoint: from finetune_cub_proxy.py (patch-token model)
Each was trained on its own specialist objective -- this evaluates each
checkpoint doing what it was actually trained to do, just on a new query type.

Usage:
  python crop_query_retrieval.py --cub-dir /path/CUB_200_2011 \
      --global-checkpoint checkpoints_cub_global_baseline_s0/backbone_final.pt \
      --patch-checkpoint rigorous_suite/cub_seed0/backbone_final.pt \
      --crop-fraction 0.5
"""
import argparse
import json
import os
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

MODEL_NAME = "facebook/dinov2-base"
IMAGE_SIZE = 224


class DinoModel(nn.Module):
    """Loads either checkpoint's weights; forward returns BOTH the CLS token
    and the patch tokens so one model instance serves either representation."""
    def __init__(self, model_name=MODEL_NAME):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)

    def forward(self, pixel_values):
        out = self.backbone(pixel_values=pixel_values)
        cls = F.normalize(out.last_hidden_state[:, 0, :], dim=-1)
        patches = F.normalize(out.last_hidden_state[:, 1:, :], dim=-1)
        return cls, patches


def load_cub_test(cub_dir):
    by_class = defaultdict(list)
    image_paths = {}
    with open(f"{cub_dir}/images.txt") as f:
        for line in f:
            image_id, path = line.split()
            image_paths[image_id] = path
    with open(f"{cub_dir}/image_class_labels.txt") as f:
        for line in f:
            image_id, class_id = line.split()
            by_class[class_id].append(image_paths[image_id])
    test_classes = [str(i) for i in range(101, 201)]
    images = []
    for c in test_classes:
        for p in by_class[c]:
            images.append({"class_id": c, "path": p})
    return images


def make_crop(img, fraction, rng):
    """Deterministic (per-call rng) random square-ish crop covering `fraction`
    of the image area, then resized to IMAGE_SIZE like any other input."""
    w, h = img.size
    side_frac = fraction ** 0.5  # area fraction -> linear side fraction
    crop_w, crop_h = int(w * side_frac), int(h * side_frac)
    x0 = rng.randint(0, max(1, w - crop_w))
    y0 = rng.randint(0, max(1, h - crop_h))
    crop = img.crop((x0, y0, x0 + crop_w, y0 + crop_h))
    return crop.resize((IMAGE_SIZE, IMAGE_SIZE))


@torch.no_grad()
def encode_all(model, processor, images_root, images, device, batch_size, crop_fraction, seed):
    """Returns (full_cls, full_patches, crop_cls, crop_patches) for the whole set."""
    rng = random.Random(seed)  # same seed -> same crops regardless of which checkpoint runs first
    N = len(images)
    full_cls = torch.zeros((N, 768))
    full_patches = torch.zeros((N, 256, 768), dtype=torch.bfloat16)
    crop_cls = torch.zeros((N, 768))
    crop_patches = torch.zeros((N, 256, 768), dtype=torch.bfloat16)

    for i in range(0, N, batch_size):
        batch = images[i:i + batch_size]
        full_imgs = [Image.open(os.path.join(images_root, im["path"])).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
                     for im in batch]
        raw_imgs = [Image.open(os.path.join(images_root, im["path"])).convert("RGB") for im in batch]
        crop_imgs = [make_crop(img, crop_fraction, rng) for img in raw_imgs]

        pv_full = processor(images=full_imgs, return_tensors="pt")["pixel_values"].to(device)
        pv_crop = processor(images=crop_imgs, return_tensors="pt")["pixel_values"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            cls_f, patches_f = model(pv_full)
            cls_c, patches_c = model(pv_crop)
        n = len(batch)
        full_cls[i:i + n] = cls_f.float().cpu()
        full_patches[i:i + n] = patches_f.cpu()
        crop_cls[i:i + n] = cls_c.float().cpu()
        crop_patches[i:i + n] = patches_c.cpu()

    return full_cls, full_patches, crop_cls, crop_patches


def recall_at_1_global(query_embs, gallery_embs, classes, device):
    q = query_embs.to(device)
    g = gallery_embs.to(device)
    sims = q @ g.T
    sims.fill_diagonal_(-1e9)  # exclude self (query i's crop vs gallery i's full image)
    best_idx = sims.argmax(dim=1).cpu()
    hits = sum(1 for i in range(len(classes)) if classes[best_idx[i]] == classes[i])
    return hits / len(classes)


def recall_at_1_patch(query_patches, gallery_patches, classes, device, query_chunk=8, cand_chunk=1024):
    N = len(classes)
    best_scores = torch.full((N,), -1e9, device=device)
    best_idx = torch.zeros((N,), dtype=torch.long, device=device)
    for c0 in range(0, N, cand_chunk):
        c1 = min(N, c0 + cand_chunk)
        c_embs = gallery_patches[c0:c1].to(device).float()
        for q0 in range(0, N, query_chunk):
            q1 = min(N, q0 + query_chunk)
            q_embs = query_patches[q0:q1].to(device).float()
            sim = torch.einsum("qpd,crd->qcpr", q_embs, c_embs)
            a_to_b = sim.max(dim=3).values.mean(dim=2)
            b_to_a = sim.max(dim=2).values.mean(dim=2)
            scores = 0.5 * (a_to_b + b_to_a)
            for i in range(q1 - q0):
                gq = q0 + i
                if c0 <= gq < c1:
                    scores[i, gq - c0] = -1e9  # exclude self
            chunk_best, chunk_idx = scores.max(dim=1)
            mask = chunk_best > best_scores[q0:q1]
            best_scores[q0:q1] = torch.where(mask, chunk_best, best_scores[q0:q1])
            best_idx[q0:q1] = torch.where(mask, chunk_idx + c0, best_idx[q0:q1])
    best_idx = best_idx.cpu()
    hits = sum(1 for i in range(N) if classes[best_idx[i]] == classes[i])
    return hits / N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cub-dir", required=True)
    ap.add_argument("--global-checkpoint", required=True)
    ap.add_argument("--patch-checkpoint", required=True)
    ap.add_argument("--crop-fraction", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="crop_query_results.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    images = load_cub_test(args.cub_dir)
    classes = [im["class_id"] for im in images]
    images_root = f"{args.cub_dir}/images"
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, use_fast=True)
    processor.size = {"height": IMAGE_SIZE, "width": IMAGE_SIZE}

    print(f"Test set: {len(images)} images, crop_fraction={args.crop_fraction}")

    print("Encoding with GLOBAL checkpoint...")
    global_model = DinoModel().to(device)
    global_model.backbone.load_state_dict(torch.load(args.global_checkpoint, map_location=device))
    global_model.eval()
    g_full_cls, _, g_crop_cls, _ = encode_all(global_model, processor, images_root, images, device,
                                               args.batch_size, args.crop_fraction, args.seed)
    del global_model
    torch.cuda.empty_cache()

    print("Encoding with PATCH checkpoint...")
    patch_model = DinoModel().to(device)
    patch_model.backbone.load_state_dict(torch.load(args.patch_checkpoint, map_location=device))
    patch_model.eval()
    _, p_full_patches, _, p_crop_patches = encode_all(patch_model, processor, images_root, images, device,
                                                       args.batch_size, args.crop_fraction, args.seed)
    del patch_model
    torch.cuda.empty_cache()

    print("Computing crop-query Recall@1 (global checkpoint, cosine)...")
    global_r1 = recall_at_1_global(g_crop_cls, g_full_cls, classes, device)
    print(f"  Global R@1 (crop query -> full gallery): {global_r1:.4f}")

    print("Computing crop-query Recall@1 (patch checkpoint, MaxSim)...")
    patch_r1 = recall_at_1_patch(p_crop_patches, p_full_patches, classes, device)
    print(f"  Patch R@1 (crop query -> full gallery): {patch_r1:.4f}")

    result = {
        "crop_fraction": args.crop_fraction,
        "test_size": len(images),
        "global_checkpoint_r1": global_r1,
        "patch_checkpoint_r1": patch_r1,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
