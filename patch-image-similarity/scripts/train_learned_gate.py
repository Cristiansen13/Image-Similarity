"""Replace the hand-tuned sigmoid blend weight with a small learned gate --
the biggest reviewer-vulnerability in the adaptive-scorer work so far ("you
fit two numbers to two curated datasets, does this actually generalize?").

Gate: tiny MLP taking [global_score, patch_score, complexity_signal] ->
match probability, trained via BCE on labeled match/non-match pairs.
Normalization constants (mean/std per feature) are computed ONCE from the
training set only, then applied fixed at eval time -- not per-batch
z-scoring, which would leak test-set statistics and isn't realistic for a
deployed model that scores one pair at a time.

Splits designed to actually test generalization, not just fit two numbers
harder:
  - CUB: train on classes 1-100 (never used for the hand-tuned heuristic's
    design), evaluate on the same held-out classes 101-200 used throughout
    this project.
  - COCO confusion: split by STRUCTURAL TEMPLATE (not just re-shuffled
    negatives of the same 115 pairs) -- train on a subset of templates,
    evaluate on templates never seen during training, so the eval set tests
    genuinely different structural-confusion patterns.
  - CARS196: entirely held out, ZERO training data drawn from it -- the real
    test of "did this learn something general or just curve-fit."

Usage: python train_learned_gate.py --cub-dir /path/CUB_200_2011 \
    --coco-triplets /path/coco_confusion_triplets.json --cars-dir /path/cars_raw
"""
import argparse
import json
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
PATCH_TOPK_SIMPLE, PATCH_TOPK_COMPLEX = 16, 4
DIVERSITY_SIMPLE, DIVERSITY_COMPLEX = 0.296, 0.220

TRAIN_TEMPLATES_IDX = list(range(0, 10))  # first 10 of 16 templates for training
TEST_TEMPLATES_IDX = list(range(10, 16))  # last 6, never seen during gate training


class Encoder:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(MODEL_NAME, use_fast=True)
        self.processor.size = {"height": IMAGE_SIZE, "width": IMAGE_SIZE}
        self.model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    @torch.no_grad()
    def encode(self, path):
        image = Image.open(path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        last_hidden = out.last_hidden_state[0]
        cls = F.normalize(last_hidden[0], dim=-1)
        patches = F.normalize(last_hidden[1:], dim=-1)
        return {"global": cls, "patches": patches}


def mean_patch_similarity(patches):
    sim = patches @ patches.T
    n = sim.shape[0]
    return ((sim.sum() - torch.diagonal(sim).sum()) / (n * (n - 1))).item()


def adaptive_k(diversity):
    t = (diversity - DIVERSITY_COMPLEX) / (DIVERSITY_SIMPLE - DIVERSITY_COMPLEX)
    t = max(0.0, min(1.0, t))
    return max(1, round(PATCH_TOPK_COMPLEX + t * (PATCH_TOPK_SIMPLE - PATCH_TOPK_COMPLEX)))


def topk_maxsim(a, b, k):
    sim = a @ b.T
    max_a, max_b = sim.max(dim=1).values, sim.max(dim=0).values
    ka, kb = min(k, max_a.shape[0]), min(k, max_b.shape[0])
    return 0.5 * (max_a.topk(ka).values.mean().item() + max_b.topk(kb).values.mean().item())


def pair_features(cache, a, b):
    ea, eb = cache[a], cache[b]
    g = (ea["global"] @ eb["global"]).item()
    div_a = mean_patch_similarity(ea["patches"])
    k = adaptive_k(div_a)
    v = topk_maxsim(ea["patches"], eb["patches"], k)
    return [g, v, div_a]


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


def build_cub_pairs(cub_dir, classes, n, seed):
    rng = random.Random(seed)
    by_class = load_cub_index(cub_dir)
    images_root = os.path.join(cub_dir, "images")
    pairs = []  # (path_a, path_b, label)
    for _ in range(n):
        pos_class = rng.choice(classes)
        neg_class = rng.choice([c for c in classes if c != pos_class])
        if len(by_class[pos_class]) < 2:
            continue
        anchor, positive = rng.sample(by_class[pos_class], 2)
        negative = rng.choice(by_class[neg_class])
        a = os.path.join(images_root, anchor)
        pairs.append((a, os.path.join(images_root, positive), 1))
        pairs.append((a, os.path.join(images_root, negative), 0))
    return pairs


def build_coco_pairs(triplets, template_idx, seed):
    rng = random.Random(seed)
    unique_templates = sorted(set(t["structure"] for t in triplets))
    selected = {unique_templates[i] for i in template_idx if i < len(unique_templates)}
    subset = [t for t in triplets if t["structure"] in selected]
    pairs = []
    negs = [t["negative"] for t in subset]
    shuffled = negs[:]
    rng.shuffle(shuffled)
    for i in range(len(shuffled)):
        if shuffled[i] == negs[i]:
            shuffled[i], shuffled[(i + 1) % len(shuffled)] = shuffled[(i + 1) % len(shuffled)], shuffled[i]
    for t, rand_neg in zip(subset, shuffled):
        pairs.append((t["anchor"], t["positive"], 1))
        pairs.append((t["anchor"], t["negative"], 0))
        pairs.append((t["anchor"], rand_neg, 0))
    return pairs


def load_cars_index(cars_dir):
    names_path = os.path.join(cars_dir, "names.csv")
    with open(names_path) as f:
        class_names = [line.strip() for line in f if line.strip()]
    name_to_id = {}
    for i, name in enumerate(class_names):
        name_to_id[name] = str(i + 1)
        name_to_id[name.replace("/", "-")] = str(i + 1)
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


def build_cars_pairs(cars_dir, n, seed):
    rng = random.Random(seed)
    by_class = load_cars_index(cars_dir)
    classes = [c for c, imgs in by_class.items() if len(imgs) >= 2]
    pairs = []
    for _ in range(n):
        pos_class = rng.choice(classes)
        neg_class = rng.choice([c for c in classes if c != pos_class])
        anchor, positive = rng.sample(by_class[pos_class], 2)
        negative = rng.choice(by_class[neg_class])
        pairs.append((anchor, positive, 1))
        pairs.append((anchor, negative, 0))
    return pairs


def featurize(encoder, cache, pairs):
    X, y = [], []
    for i, (a, b, label) in enumerate(pairs):
        if a not in cache:
            cache[a] = encoder.encode(a)
        if b not in cache:
            cache[b] = encoder.encode(b)
        X.append(pair_features(cache, a, b))
        y.append(label)
        if (i + 1) % 100 == 0:
            print(f"    featurized {i + 1}/{len(pairs)}")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def accuracy_at_threshold(scores, labels, threshold=0.5):
    preds = (scores > threshold).astype(np.float32)
    return float((preds == labels).mean())


def pairwise_auc(model_or_none, X, y, mean, std, device, hand_tuned_weight_fn=None):
    """Fraction of all (positive, negative) pairs where positive outranks
    negative (+0.5 credit for ties) -- the same "does the true match outrank
    the non-match" concept used throughout this project (triplet accuracy),
    generalized to work regardless of the positive:negative ratio (COCO pairs
    are 1 positive : 2 negatives -- confusable + random -- not 1:1 like CUB/
    CARS, so a naive positional pos[i]-vs-neg[i] comparison would misalign)."""
    scores = score_batch(model_or_none, X, mean, std, device, hand_tuned_weight_fn)
    pos_scores = scores[y == 1]
    neg_scores = scores[y == 0]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return float("nan")
    diff = pos_scores[:, None] - neg_scores[None, :]
    return float(np.mean((diff > 0).astype(np.float32) + 0.5 * (diff == 0).astype(np.float32)))


def score_batch(model, X, mean, std, device, hand_tuned_weight_fn=None):
    if model is not None:
        Xn = (X - mean) / std
        with torch.no_grad():
            logits = model(torch.tensor(Xn, dtype=torch.float32, device=device))
        return torch.sigmoid(logits).cpu().numpy()
    else:
        # hand-tuned baseline: blend global(col0)/patch(col1) by weight(diversity=col2)
        g, v, div = X[:, 0], X[:, 1], X[:, 2]
        w = hand_tuned_weight_fn(div)
        gm, gs = mean[0], std[0]
        vm, vs = mean[1], std[1]
        gz = (g - gm) / gs
        vz = (v - vm) / vs
        return w * vz + (1 - w) * gz


def hand_tuned_weight(diversity, midpoint=0.258, steepness=60.0):
    return 1.0 / (1.0 + np.exp(steepness * (diversity - midpoint)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cub-dir", required=True)
    ap.add_argument("--coco-triplets", required=True)
    ap.add_argument("--cars-dir", required=True)
    ap.add_argument("--n-cub-train", type=int, default=300)
    ap.add_argument("--n-cub-test", type=int, default=200)
    ap.add_argument("--n-cars-test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=500)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = Encoder(device)
    cache = {}

    print("Building training pairs: CUB classes 1-100 + COCO train templates...")
    cub_train_pairs = build_cub_pairs(args.cub_dir, [str(i) for i in range(1, 101)],
                                       args.n_cub_train, args.seed)
    with open(args.coco_triplets) as f:
        coco_triplets = json.load(f)
    coco_train_pairs = build_coco_pairs(coco_triplets, TRAIN_TEMPLATES_IDX, args.seed)
    train_pairs = cub_train_pairs + coco_train_pairs
    print(f"  {len(cub_train_pairs)} CUB-train pairs + {len(coco_train_pairs)} COCO-train pairs")

    print("Featurizing training pairs...")
    X_train, y_train = featurize(encoder, cache, train_pairs)
    mean, std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
    print(f"  feature mean={mean}, std={std}")

    print("Training gate MLP...")
    gate = Gate().to(device)
    opt = torch.optim.Adam(gate.parameters(), lr=1e-2, weight_decay=1e-4)
    Xn = (X_train - mean) / std
    Xt = torch.tensor(Xn, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.float32, device=device)
    for step in range(args.steps):
        opt.zero_grad()
        logits = gate(Xt)
        loss = F.binary_cross_entropy_with_logits(logits, yt)
        loss.backward()
        opt.step()
        if (step + 1) % 100 == 0:
            print(f"  step {step + 1}/{args.steps}  loss={loss.item():.4f}")
    gate.eval()

    print("\nBuilding held-out eval sets: CUB classes 101-200, COCO test templates, CARS (fully unseen)...")
    cub_test_pairs = build_cub_pairs(args.cub_dir, [str(i) for i in range(101, 201)],
                                      args.n_cub_test, args.seed + 1000)
    coco_test_pairs = build_coco_pairs(coco_triplets, TEST_TEMPLATES_IDX, args.seed + 1000)
    cars_test_pairs = build_cars_pairs(args.cars_dir, args.n_cars_test, args.seed + 1000)

    results = {}
    for name, pairs in [("CUB (held-out classes)", cub_test_pairs),
                         ("COCO (held-out templates)", coco_test_pairs),
                         ("CARS196 (fully unseen domain)", cars_test_pairs)]:
        print(f"\nFeaturizing {name} ({len(pairs)} pairs)...")
        X, y = featurize(encoder, cache, pairs)
        gate_acc = pairwise_auc(gate, X, y, mean, std, device)
        hand_acc = pairwise_auc(None, X, y, mean, std, device, hand_tuned_weight)
        results[name] = {"learned_gate": gate_acc, "hand_tuned": hand_acc, "n_pairs": len(pairs)}
        print(f"  learned gate AUC: {gate_acc:.4f}   hand-tuned AUC: {hand_acc:.4f}")

    print("\n" + "=" * 70)
    print("FINAL: learned gate vs hand-tuned sigmoid, on held-out data")
    print("=" * 70)
    for name, r in results.items():
        print(f"  {name:32s}: learned={r['learned_gate']:.4f}  hand-tuned={r['hand_tuned']:.4f}  (n={r['n_pairs']})")

    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "data", "learned_gate_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "data", "learned_gate.pt")
    torch.save({"state_dict": gate.state_dict(), "mean": mean.tolist(), "std": std.tolist()}, model_path)
    print(f"Saved gate model to {model_path}")


if __name__ == "__main__":
    main()
