"""Evaluate a jointly fine-tuned CUB backbone (finetune_cub_joint.py) on the
standard held-out test classes (101-200): global-only, patch-only (top-K=16),
and adaptive (using the trained learned gate, not the hand-tuned sigmoid) --
compared against the zero-shot baseline numbers already established this
session, to see if actually training the backbone (vs. applying these
scoring mechanisms post-hoc to a frozen one) helps further.

Usage: python eval_cub_joint.py --cub-dir /path/CUB_200_2011 \
    --checkpoint checkpoints_cub_joint/backbone_final.pt --gate-path data/learned_gate.pt
"""
import argparse
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

MODEL_NAME = "facebook/dinov2-base"
IMAGE_SIZE = 224
PATCH_K = 16
DIVERSITY_SIMPLE, DIVERSITY_COMPLEX = 0.296, 0.220
K_SIMPLE, K_COMPLEX = 16, 4


class Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


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


def build_cub_triplets(cub_dir, classes, n, seed):
    rng = random.Random(seed)
    by_class = load_cub_index(cub_dir)
    images_root = os.path.join(cub_dir, "images")
    triplets = []
    for _ in range(n):
        pos_class = rng.choice(classes)
        neg_class = rng.choice([c for c in classes if c != pos_class])
        if len(by_class[pos_class]) < 2:
            continue
        anchor, positive = rng.sample(by_class[pos_class], 2)
        negative = rng.choice(by_class[neg_class])
        triplets.append({"anchor": os.path.join(images_root, anchor),
                          "positive": os.path.join(images_root, positive),
                          "negative": os.path.join(images_root, negative)})
    return triplets


def mean_patch_similarity(patches):
    sim = patches @ patches.T
    n = sim.shape[0]
    return ((sim.sum() - torch.diagonal(sim).sum()) / (n * (n - 1))).item()


def adaptive_k(diversity):
    t = (diversity - DIVERSITY_COMPLEX) / (DIVERSITY_SIMPLE - DIVERSITY_COMPLEX)
    t = max(0.0, min(1.0, t))
    return max(1, round(K_COMPLEX + t * (K_SIMPLE - K_COMPLEX)))


def topk_maxsim(a, b, k):
    sim = a @ b.T
    max_a, max_b = sim.max(dim=1).values, sim.max(dim=0).values
    ka, kb = min(k, max_a.shape[0]), min(k, max_b.shape[0])
    return 0.5 * (max_a.topk(ka).values.mean().item() + max_b.topk(kb).values.mean().item())


def mean_ci95(values):
    arr = np.array(values)
    n = len(arr)
    mean = arr.mean()
    if n <= 1:
        return mean, 0.0
    return mean, 1.96 * arr.std(ddof=1) / (n ** 0.5)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cub-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--gate-path", required=True)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--n-triplets", type=int, default=200)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, use_fast=True)
    processor.size = {"height": IMAGE_SIZE, "width": IMAGE_SIZE}
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    gate_data = torch.load(args.gate_path, map_location=device)
    gate = Gate().to(device)
    gate.load_state_dict(gate_data["state_dict"])
    gate.eval()
    gate_mean = torch.tensor(gate_data["mean"], device=device, dtype=torch.float32)
    gate_std = torch.tensor(gate_data["std"], device=device, dtype=torch.float32)

    cache = {}

    def encode(path):
        if path not in cache:
            image = Image.open(path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            out = model(**inputs)
            last_hidden = out.last_hidden_state[0]
            cache[path] = {
                "global": F.normalize(last_hidden[0], dim=-1),
                "patches": F.normalize(last_hidden[1:], dim=-1),
            }
        return cache[path]

    def pair_scores(a, b):
        ea, eb = encode(a), encode(b)
        g = (ea["global"] @ eb["global"]).item()
        div_a = mean_patch_similarity(ea["patches"])
        k = adaptive_k(div_a)
        v = topk_maxsim(ea["patches"], eb["patches"], k)
        return g, v, div_a

    def gate_score(g, v, div):
        x = torch.tensor([g, v, div], device=device, dtype=torch.float32)
        xn = (x - gate_mean) / gate_std
        return torch.sigmoid(gate(xn.unsqueeze(0))).item()

    test_classes = [str(i) for i in range(101, 201)]
    results = {"global": [], "patch": [], "adaptive": []}
    for seed in range(args.n_seeds):
        triplets = build_cub_triplets(args.cub_dir, test_classes, args.n_triplets, seed)
        g_pos, g_neg, v_pos, v_neg, a_pos, a_neg = [], [], [], [], [], []
        for i, t in enumerate(triplets):
            g_p, v_p, d_p = pair_scores(t["anchor"], t["positive"])
            g_n, v_n, d_n = pair_scores(t["anchor"], t["negative"])
            g_pos.append(g_p); g_neg.append(g_n)
            v_pos.append(v_p); v_neg.append(v_n)
            a_pos.append(gate_score(g_p, v_p, d_p)); a_neg.append(gate_score(g_n, v_n, d_n))
            if (i + 1) % 50 == 0:
                print(f"  seed {seed}: {i + 1}/{len(triplets)}")
        results["global"].append(float(np.mean(np.array(g_pos) > np.array(g_neg))))
        results["patch"].append(float(np.mean(np.array(v_pos) > np.array(v_neg))))
        results["adaptive"].append(float(np.mean(np.array(a_pos) > np.array(a_neg))))
        print(f"  seed {seed} done: global={results['global'][-1]:.4f} "
              f"patch={results['patch'][-1]:.4f} adaptive={results['adaptive'][-1]:.4f}")

    print("\n" + "=" * 60)
    print(f"JOINTLY FINE-TUNED backbone, held-out CUB classes, n={args.n_seeds} seeds")
    print("=" * 60)
    for key in ("global", "patch", "adaptive"):
        mean, ci = mean_ci95(results[key])
        print(f"  {key:10s}: {mean:.4f} +/- {ci:.4f}")


if __name__ == "__main__":
    main()
