"""Proper Recall@K on the fine-tuned backbone, using the TRUE held-out
classes (the 10% split that finetune_backbone.py set aside and never
trained on) -- reconstructed with the identical seed/filter/shuffle logic
so this is a genuinely unseen gallery, not classes the model has seen.

Compares the fine-tuned backbone directly against zero-shot (pretrained,
unmodified DINOv2) on the exact same gallery, so the two numbers are
honestly comparable -- same protocol used earlier for the frozen-probe
comparison (compute_recall_at_k.py), just with a real held-out class split
and a fine-tuned backbone instead of a frozen one + linear head.
"""
import argparse
import json
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

MODEL_NAME = "facebook/dinov2-base"
IMAGE_SIZE = 224


class FineTuneModel(nn.Module):
    def __init__(self, model_name=MODEL_NAME):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)

    def forward(self, pixel_values):
        out = self.backbone(pixel_values=pixel_values)
        return F.normalize(out.last_hidden_state[:, 1:, :], dim=-1)


def load_ebay_index(ebay_info_path):
    by_class = {}
    with open(ebay_info_path) as f:
        next(f)
        for line in f:
            _, class_id, _, path = line.split()
            by_class.setdefault(class_id, []).append(path)
    return by_class


def reconstruct_holdout_split(by_class, K=4, seed=0):
    """Must match finetune_backbone.py's main() exactly: same filter, same
    shuffle, same 90/10 split -- so this really is the untouched holdout."""
    rng = random.Random(seed)
    eligible = [c for c, imgs in by_class.items() if len(imgs) >= K]
    rng.shuffle(eligible)
    split_point = int(len(eligible) * 0.9)
    return eligible[:split_point], eligible[split_point:]


def symmetric_maxsim(a, b):
    sim = a @ b.T
    return 0.5 * (sim.max(dim=1).values.mean() + sim.max(dim=0).values.mean())


def build_gallery(by_class, holdout_classes, images_root, rng, n_classes, photos_per_class):
    chosen = rng.sample(holdout_classes, min(n_classes, len(holdout_classes)))
    gallery = {}  # path -> class_id
    for c in chosen:
        photos = by_class[c][:photos_per_class]
        for p in photos:
            gallery[os.path.join(images_root, p)] = c
    return gallery


@torch.no_grad()
def encode_gallery(model, processor, gallery, device, batch_size=32):
    paths = list(gallery.keys())
    embeddings = {}
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i + batch_size]
        images = [Image.open(p).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)) for p in batch_paths]
        pixel_values = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            patches = model(pixel_values)
        for p, patch in zip(batch_paths, patches.float().cpu()):
            embeddings[p] = patch
        if (i + batch_size) % 200 < batch_size:
            print(f"  encoded {min(i + batch_size, len(paths))}/{len(paths)}")
    return embeddings


@torch.no_grad()
def recall_at_k(gallery, embeddings, device, k_values=(1, 5)):
    paths = list(gallery.keys())
    embs = {p: embeddings[p].to(device) for p in paths}
    hits = {k: 0 for k in k_values}
    for qi, query in enumerate(paths):
        scores = []
        for cand in paths:
            if cand == query:
                continue
            scores.append((symmetric_maxsim(embs[query], embs[cand]).item(), cand))
        scores.sort(key=lambda x: -x[0])
        ranked_classes = [gallery[p] for _, p in scores]
        for k in k_values:
            if gallery[query] in ranked_classes[:k]:
                hits[k] += 1
        if (qi + 1) % 50 == 0:
            print(f"    {qi + 1}/{len(paths)} queries evaluated")
    return {k: hits[k] / len(paths) for k in k_values}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ebay-info", required=True)
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--finetuned-checkpoint", required=True)
    ap.add_argument("--n-gallery-classes", type=int, default=300)
    ap.add_argument("--photos-per-class", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(args.seed)

    print("Reconstructing the exact train/holdout split used during fine-tuning...")
    by_class = load_ebay_index(args.ebay_info)
    train_classes, holdout_classes = reconstruct_holdout_split(by_class)
    print(f"{len(train_classes)} classes were used in training, {len(holdout_classes)} were held out")

    gallery = build_gallery(by_class, holdout_classes, args.images_root, rng,
                             args.n_gallery_classes, args.photos_per_class)
    print(f"Gallery: {len(gallery)} images, {len(set(gallery.values()))} classes, "
          f"all from held-out (never-trained-on) classes")

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, use_fast=True)
    processor.size = {"height": IMAGE_SIZE, "width": IMAGE_SIZE}

    print("\n=== Zero-shot (pretrained, unmodified DINOv2) ===")
    zs_model = FineTuneModel().to(device).eval()
    zs_embeddings = encode_gallery(zs_model, processor, gallery, device)
    zs_recall = recall_at_k(gallery, zs_embeddings, device)
    print(f"zero-shot: R@1={zs_recall[1]:.3f}  R@5={zs_recall[5]:.3f}")
    del zs_model
    torch.cuda.empty_cache()

    print("\n=== Fine-tuned backbone ===")
    ft_model = FineTuneModel().to(device)
    state_dict = torch.load(args.finetuned_checkpoint, map_location=device)
    ft_model.backbone.load_state_dict(state_dict)
    ft_model.eval()
    ft_embeddings = encode_gallery(ft_model, processor, gallery, device)
    ft_recall = recall_at_k(gallery, ft_embeddings, device)
    print(f"fine-tuned: R@1={ft_recall[1]:.3f}  R@5={ft_recall[5]:.3f}")

    print(f"\n=== SUMMARY (gallery: {len(gallery)} images / {len(set(gallery.values()))} "
          f"held-out classes, never touched during training) ===")
    print(f"zero-shot:  R@1={zs_recall[1]:.3f}  R@5={zs_recall[5]:.3f}")
    print(f"fine-tuned: R@1={ft_recall[1]:.3f}  R@5={ft_recall[5]:.3f}")
    print("For reference, published SOP full-test-set (60,502 images) Recall@1 with fully "
          "fine-tuned backbones: Clustering 67.0%, Proxy-NCA 73.7%, Margin 72.7%, MS 78.2%, "
          "SoftTriple 78.3%, Proxy-Anchor 79.1-80.3%. Our gallery is much smaller/easier.")

    out_path = os.path.join(os.path.dirname(args.finetuned_checkpoint), "recall_comparison.json")
    with open(out_path, "w") as f:
        json.dump({"gallery_size": len(gallery), "n_classes": len(set(gallery.values())),
                   "zero_shot": zs_recall, "fine_tuned": ft_recall}, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
