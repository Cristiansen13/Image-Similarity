"""Attention-weighted MaxSim: instead of (or alongside) top-K patch
selection, weight each patch's contribution by DINOv2's own CLS-token
attention to it -- a well-documented emergent unsupervised foreground
segmentation signal (the original DINO paper's headline visualization).
No extra model or training needed, just output_attentions=True on the
already-frozen backbone.

This targets the same failure mode top-K targets (background patches
diluting genuine object correspondence) via a different, more principled
mechanism: instead of implicitly assuming "the strongest matches are the
real object" (top-K), it explicitly asks the model "which patches do you
consider foreground" and downweights the rest directly.

Runs the same CUB (simple) + COCO confusion (complex) robust multi-seed
benchmark as adaptive_scorer_v2.py, on GPU, comparing:
  - plain MaxSim (mean over all patches)
  - top-K MaxSim (complexity-adaptive K, from adaptive_scorer_v2.py)
  - attention-weighted MaxSim (weight by CLS attention, no K cutoff)
  - attention-weighted top-K MaxSim (both mechanisms combined)

Usage: python attention_weighted_maxsim.py --cub-dir /path/CUB_200_2011 \
    --coco-triplets /path/coco_confusion_triplets.json --n-seeds 5
"""
import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

MODEL_NAME = "facebook/dinov2-base"
IMAGE_SIZE = 224

DIVERSITY_SIMPLE, DIVERSITY_COMPLEX = 0.296, 0.220
K_SIMPLE, K_COMPLEX = 16, 4
MIDPOINT = 0.258
STEEPNESS = 60.0


class AttentionEncoder:
    """Like PatchEncoder, but also returns per-patch CLS-attention weights
    from the last transformer block (averaged over heads, normalized to
    sum to 1 over patches -- the standard DINO foreground-saliency signal)."""

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(MODEL_NAME, use_fast=True)
        self.processor.size = {"height": IMAGE_SIZE, "width": IMAGE_SIZE}
        self.model = AutoModel.from_pretrained(MODEL_NAME, attn_implementation="eager")
        self.model.to(device).eval()
        self.num_register_tokens = getattr(self.model.config, "num_register_tokens", 0)

    @torch.no_grad()
    def encode(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        out = self.model(**inputs, output_attentions=True)
        last_hidden = out.last_hidden_state[0]  # (1+reg+N, D)
        cls_token = last_hidden[0]
        patch_tokens = last_hidden[1 + self.num_register_tokens:]

        # last layer's attention, CLS (query) -> patches (key), averaged over heads
        last_attn = out.attentions[-1][0]  # (heads, seq, seq)
        cls_to_patches = last_attn[:, 0, 1 + self.num_register_tokens:]  # (heads, N)
        attn_weight = cls_to_patches.mean(dim=0)  # (N,) average over heads
        attn_weight = attn_weight / (attn_weight.sum() + 1e-8)  # normalize to sum=1

        return {
            "patch_embeddings": F.normalize(patch_tokens, dim=-1),
            "global_embedding": F.normalize(cls_token, dim=-1),
            "attn_weight": attn_weight,
        }


def plain_maxsim(a, b):
    sim = a @ b.T
    return 0.5 * (sim.max(dim=1).values.mean().item() + sim.max(dim=0).values.mean().item())


def topk_maxsim(a, b, k):
    sim = a @ b.T
    max_a, max_b = sim.max(dim=1).values, sim.max(dim=0).values
    ka, kb = min(k, max_a.shape[0]), min(k, max_b.shape[0])
    return 0.5 * (max_a.topk(ka).values.mean().item() + max_b.topk(kb).values.mean().item())


def attn_weighted_maxsim(a, b, wa, wb):
    """wa, wb: (N,) normalized attention weights for a's and b's own patches.
    Weight each patch's best-match contribution by ITS OWN attention weight
    (how much the model itself considers that patch foreground) before
    averaging, instead of a uniform mean."""
    sim = a @ b.T
    max_a, max_b = sim.max(dim=1).values, sim.max(dim=0).values
    score_a_to_b = (max_a * wa).sum().item()  # wa already sums to 1 -> weighted mean
    score_b_to_a = (max_b * wb).sum().item()
    return 0.5 * (score_a_to_b + score_b_to_a)


def attn_weighted_topk_maxsim(a, b, wa, wb, k):
    """Combine both mechanisms: restrict to top-K by RAW match strength, then
    weight those K contributions by attention (renormalized over the K)."""
    sim = a @ b.T
    max_a, max_b = sim.max(dim=1).values, sim.max(dim=0).values
    ka, kb = min(k, max_a.shape[0]), min(k, max_b.shape[0])
    top_a_vals, top_a_idx = max_a.topk(ka)
    top_b_vals, top_b_idx = max_b.topk(kb)
    wa_top = wa[top_a_idx]; wa_top = wa_top / (wa_top.sum() + 1e-8)
    wb_top = wb[top_b_idx]; wb_top = wb_top / (wb_top.sum() + 1e-8)
    score_a_to_b = (top_a_vals * wa_top).sum().item()
    score_b_to_a = (top_b_vals * wb_top).sum().item()
    return 0.5 * (score_a_to_b + score_b_to_a)


def mean_patch_similarity(patch_embeddings):
    sim = patch_embeddings @ patch_embeddings.T
    n = sim.shape[0]
    return ((sim.sum() - torch.diagonal(sim).sum()) / (n * (n - 1))).item()


def adaptive_k(diversity):
    t = (diversity - DIVERSITY_COMPLEX) / (DIVERSITY_SIMPLE - DIVERSITY_COMPLEX)
    t = max(0.0, min(1.0, t))
    return max(1, round(K_COMPLEX + t * (K_SIMPLE - K_COMPLEX)))


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


def build_cub_triplets(cub_dir, n, seed):
    rng = random.Random(seed)
    by_class = load_cub_index(cub_dir)
    test_classes = [str(i) for i in range(101, 201)]
    images_root = os.path.join(cub_dir, "images")
    triplets = []
    for _ in range(n):
        pos_class = rng.choice(test_classes)
        neg_class = rng.choice([c for c in test_classes if c != pos_class])
        if len(by_class[pos_class]) < 2:
            continue
        anchor, positive = rng.sample(by_class[pos_class], 2)
        negative = rng.choice(by_class[neg_class])
        triplets.append({"anchor": os.path.join(images_root, anchor),
                          "positive": os.path.join(images_root, positive),
                          "negative": os.path.join(images_root, negative)})
    return triplets


def mean_ci95(values):
    arr = np.array(values)
    n = len(arr)
    mean = arr.mean()
    if n <= 1:
        return mean, 0.0
    return mean, 1.96 * arr.std(ddof=1) / (n ** 0.5)


def standardized_elevation(confusable, random_):
    c, r = np.array(confusable), np.array(random_)
    pooled_std = np.sqrt((c.var() + r.var()) / 2) + 1e-8
    return (c.mean() - r.mean()) / pooled_std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cub-dir", required=True)
    ap.add_argument("--coco-triplets", required=True)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--n-cub-triplets", type=int, default=200)
    args = ap.parse_args()

    encoder = AttentionEncoder()
    cache = {}

    def get(p):
        if p not in cache:
            cache[p] = encoder.encode(p)
        return cache[p]

    with open(args.coco_triplets) as f:
        coco_triplets = json.load(f)

    scorers = ["plain", "topk", "attn", "attn_topk"]

    def score_pair(a, b, name):
        ea, eb = get(a), get(b)
        if name == "plain":
            return plain_maxsim(ea["patch_embeddings"], eb["patch_embeddings"])
        if name == "topk":
            k = adaptive_k(mean_patch_similarity(ea["patch_embeddings"]))
            return topk_maxsim(ea["patch_embeddings"], eb["patch_embeddings"], k)
        if name == "attn":
            return attn_weighted_maxsim(ea["patch_embeddings"], eb["patch_embeddings"],
                                         ea["attn_weight"], eb["attn_weight"])
        if name == "attn_topk":
            k = adaptive_k(mean_patch_similarity(ea["patch_embeddings"]))
            return attn_weighted_topk_maxsim(ea["patch_embeddings"], eb["patch_embeddings"],
                                              ea["attn_weight"], eb["attn_weight"], k)

    cub_results = {s: [] for s in scorers}
    coco_results = {s: {"effect_size": [], "pos_vs_conf": []} for s in scorers}

    for seed in range(args.n_seeds):
        print(f"\n--- Seed {seed} ---")
        cub_triplets = build_cub_triplets(args.cub_dir, args.n_cub_triplets, seed)
        for i, t in enumerate(cub_triplets):
            if (i + 1) % 50 == 0:
                print(f"  CUB {i + 1}/{len(cub_triplets)}")
            get(t["anchor"]); get(t["positive"]); get(t["negative"])
        for s in scorers:
            pos = [score_pair(t["anchor"], t["positive"], s) for t in cub_triplets]
            neg = [score_pair(t["anchor"], t["negative"], s) for t in cub_triplets]
            acc = float(np.mean(np.array(pos) > np.array(neg)))
            cub_results[s].append(acc)
        print(f"  CUB acc this seed: {[f'{s}={cub_results[s][-1]:.3f}' for s in scorers]}")

        anchors = [t["anchor"] for t in coco_triplets]
        positives = [t["positive"] for t in coco_triplets]
        confusable_negs = [t["negative"] for t in coco_triplets]
        rng = random.Random(seed)
        shuffled = confusable_negs[:]
        rng.shuffle(shuffled)
        for i in range(len(shuffled)):
            if shuffled[i] == confusable_negs[i]:
                shuffled[i], shuffled[(i + 1) % len(shuffled)] = shuffled[(i + 1) % len(shuffled)], shuffled[i]
        random_negs = shuffled
        for i, (a, p, cn, rn) in enumerate(zip(anchors, positives, confusable_negs, random_negs)):
            if (i + 1) % 50 == 0:
                print(f"  COCO {i + 1}/{len(anchors)}")
            get(a); get(p); get(cn); get(rn)
        for s in scorers:
            pos = [score_pair(a, p, s) for a, p in zip(anchors, positives)]
            conf = [score_pair(a, cn, s) for a, cn in zip(anchors, confusable_negs)]
            rand = [score_pair(a, rn, s) for a, rn in zip(anchors, random_negs)]
            eff = standardized_elevation(conf, rand)
            pos_vs_conf = float(np.mean(np.array(pos) > np.array(conf)))
            coco_results[s]["effect_size"].append(eff)
            coco_results[s]["pos_vs_conf"].append(pos_vs_conf)
        effect_size_strs = [f"{s}={coco_results[s]['effect_size'][-1]:.3f}" for s in scorers]
        print(f"  COCO effect_size this seed: {effect_size_strs}")

    print("\n" + "=" * 70)
    print(f"FINAL RESULTS (n={args.n_seeds} seeds, mean +/- 95% CI)")
    print("=" * 70)
    print("\nCUB (simple images) triplet accuracy:")
    for s in scorers:
        mean, ci = mean_ci95(cub_results[s])
        print(f"  {s:12s}: {mean:.4f} +/- {ci:.4f}")

    print("\nCOCO confusion (complex scenes) standardized effect size (lower=better):")
    for s in scorers:
        mean, ci = mean_ci95(coco_results[s]["effect_size"])
        mean_pvc, ci_pvc = mean_ci95(coco_results[s]["pos_vs_conf"])
        print(f"  {s:12s}: {mean:.4f} +/- {ci:.4f}   pos_vs_conf_acc={mean_pvc:.4f} +/- {ci_pvc:.4f}")

    out = {"cub": cub_results, "coco": coco_results}
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "data", "attention_weighted_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
