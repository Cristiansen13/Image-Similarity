"""On the hard-mined SOP triplets (make_sop_hard_subset.py), compare:

1. Zero-shot baselines (global, Phase0 vision MaxSim, Phase1 caption MaxSim
   at grid 3/4) -- same as before, but now on negatives deliberately mined to
   be confusable, so Phase 0 should no longer be at ceiling.
2. A small trained combiner (logistic regression over the four zero-shot
   scores) evaluated with leave-one-triplet-out cross-validation, to test
   whether *learning* to combine vision + caption signal recovers accuracy
   that vision alone misses -- the actual test of whether training can
   rescue the caption-based approach, as opposed to zero-shot averaging.
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from encoders import PatchEncoder
from nim_caption_regions import NimRegionCaptioner
from text_embed import TextEmbedder
from maxsim import symmetric_maxsim, global_cosine
from eval import triplet_accuracy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRIPLETS_PATH = os.path.join(ROOT, "data", "sop_hard_triplets.json")
CAPTION_CACHE = os.path.join(ROOT, "data", "sop_hard_subset", "region_captions_cache_nim.json")
GRID_SIZES = (3, 4)
FEATURE_NAMES = ["global", "phase0_maxsim", "phase1_grid3", "phase1_grid4"]


def compute_scores(triplets):
    unique_images = sorted({t[k] for t in triplets for k in ("anchor", "positive", "negative")})
    print(f"{len(triplets)} triplets, {len(unique_images)} unique images")

    patch_encoder = PatchEncoder()
    captioner = NimRegionCaptioner(cache_path=CAPTION_CACHE)
    text_embedder = TextEmbedder()

    patch_cache = {}
    print("Encoding patch + global embeddings...")
    for path in unique_images:
        patch_cache[path] = patch_encoder.encode(path)

    region_emb = {g: {} for g in GRID_SIZES}
    for grid_size in GRID_SIZES:
        print(f"Captioning regions at grid_size={grid_size}...")
        for i, path in enumerate(unique_images):
            regions = captioner.caption_regions(path, grid_size=grid_size)
            region_emb[grid_size][path] = text_embedder.embed([r.description for r in regions])
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(unique_images)}")

    def feature_vector(a, b):
        g = global_cosine(patch_cache[a].global_embedding, patch_cache[b].global_embedding)
        p0 = symmetric_maxsim(patch_cache[a].patch_embeddings, patch_cache[b].patch_embeddings).similarity
        p1_3 = symmetric_maxsim(region_emb[3][a], region_emb[3][b]).similarity
        p1_4 = symmetric_maxsim(region_emb[4][a], region_emb[4][b]).similarity
        return [g, p0, p1_3, p1_4]

    return [
        {"anchor": t["anchor"], "positive": t["positive"], "negative": t["negative"],
         "pos_features": feature_vector(t["anchor"], t["positive"]),
         "neg_features": feature_vector(t["anchor"], t["negative"])}
        for t in triplets
    ]


def zero_shot_accuracy(scored, feature_idx):
    correct = sum(1 for s in scored if s["pos_features"][feature_idx] > s["neg_features"][feature_idx])
    return correct / len(scored)


def train_logreg(X, y, epochs=500, lr=0.1):
    """X: (n, d) standardized features, y: (n,) in {0,1}. Returns weight, bias."""
    n, d = X.shape
    w = torch.zeros(d, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        logits = X @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        opt.step()
    return w.detach(), b.detach()


def loo_combiner_accuracy(scored):
    """Leave-one-triplet-out cross-validated accuracy of a trained logistic
    regression combining the four zero-shot scores."""
    n = len(scored)
    all_pos = torch.tensor([s["pos_features"] for s in scored], dtype=torch.float32)
    all_neg = torch.tensor([s["neg_features"] for s in scored], dtype=torch.float32)

    correct = 0
    weights_per_fold = []
    for held_out in range(n):
        train_idx = [i for i in range(n) if i != held_out]
        X_train = torch.cat([all_pos[train_idx], all_neg[train_idx]], dim=0)
        y_train = torch.cat([torch.ones(len(train_idx)), torch.zeros(len(train_idx))])

        mean, std = X_train.mean(dim=0), X_train.std(dim=0).clamp_min(1e-6)
        X_train_norm = (X_train - mean) / std

        w, b = train_logreg(X_train_norm, y_train)
        weights_per_fold.append(w)

        pos_feat = (all_pos[held_out] - mean) / std
        neg_feat = (all_neg[held_out] - mean) / std
        pos_score = (pos_feat @ w + b).item()
        neg_score = (neg_feat @ w + b).item()
        if pos_score > neg_score:
            correct += 1

    avg_weights = torch.stack(weights_per_fold).mean(dim=0)
    return correct / n, avg_weights


def main():
    with open(TRIPLETS_PATH) as f:
        triplets = json.load(f)

    scored = compute_scores(triplets)

    print()
    print(f"{'scorer':20s}{'accuracy':>12s}")
    print("-" * 32)
    for i, name in enumerate(FEATURE_NAMES):
        acc = zero_shot_accuracy(scored, i)
        print(f"{name:20s}{acc:12.3f}")

    combo_acc = sum(
        1 for s in scored
        if (s["pos_features"][2] + s["pos_features"][3]) > (s["neg_features"][2] + s["neg_features"][3])
    ) / len(scored)
    print(f"{'phase1_combo_3+4':20s}{combo_acc:12.3f}")

    loo_acc, avg_weights = loo_combiner_accuracy(scored)
    print(f"{'trained_combiner':20s}{loo_acc:12.3f}   (leave-one-out cross-validated)")

    print("\nAverage learned feature weights (standardized features, higher = more trusted):")
    for name, w in zip(FEATURE_NAMES, avg_weights.tolist()):
        print(f"  {name:20s}{w:8.3f}")

    out_path = os.path.join(ROOT, "data", "sop_hard_eval_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "zero_shot_accuracy": {name: zero_shot_accuracy(scored, i) for i, name in enumerate(FEATURE_NAMES)},
            "phase1_combo_3+4_accuracy": combo_acc,
            "trained_combiner_loo_accuracy": loo_acc,
            "avg_learned_weights": dict(zip(FEATURE_NAMES, avg_weights.tolist())),
            "scored_triplets": scored,
        }, f, indent=2)
    print(f"\nWrote results to {out_path}")


if __name__ == "__main__":
    main()
