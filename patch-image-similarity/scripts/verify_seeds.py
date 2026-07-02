"""Verification run: is the +10pt projection-head result stable, or was it
one lucky training run?

Fixes the train/val/test split (same data as train_projection_head.py) and
retrains the projection head with 5 different random seeds (weight init +
training stochasticity), reporting mean +/- std test accuracy. If the gain
is real, it should show up consistently across seeds, not just once.
"""
import json
import os
import random
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_projection_head import (
    ProjectionHead, train, score_triplets, meanmax_scorer, best_threshold_loo, threshold_hit_rate
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_DIR = os.path.join(ROOT, "data", "sop_split")
DATA_SPLIT_SEED = 0  # fixed: which triplets go to internal-val vs train (isolates training variance)
TRAINING_SEEDS = [0, 1, 2, 3, 4]
VAL_FRACTION = 0.2


def main():
    with open(os.path.join(SPLIT_DIR, "train_triplets.json")) as f:
        all_train_triplets = json.load(f)
    with open(os.path.join(SPLIT_DIR, "test_triplets.json")) as f:
        test_triplets = json.load(f)
    raw_embeddings = torch.load(os.path.join(SPLIT_DIR, "patch_embeddings.pt"))

    rng = random.Random(DATA_SPLIT_SEED)
    rng.shuffle(all_train_triplets)
    n_val = max(1, int(len(all_train_triplets) * VAL_FRACTION))
    val_triplets, train_triplets = all_train_triplets[:n_val], all_train_triplets[n_val:]
    print(f"Fixed split: {len(train_triplets)} train / {len(val_triplets)} val / {len(test_triplets)} test")

    zs_meanmax_test = score_triplets(test_triplets, raw_embeddings, meanmax_scorer)
    print(f"zero-shot meanmax test accuracy (reference): {zs_meanmax_test:.3f}\n")

    meanmax_results, threshold_results = [], []
    for seed in TRAINING_SEEDS:
        torch.manual_seed(seed)
        head = ProjectionHead()
        head, best_val_acc = train(head, train_triplets, val_triplets, raw_embeddings)

        with torch.no_grad():
            all_paths = sorted({t[k] for triplets in (all_train_triplets, test_triplets)
                                 for t in triplets for k in ("anchor", "positive", "negative")})
            projected = {p: head(raw_embeddings[p]) for p in all_paths}

        tr_meanmax_test = score_triplets(test_triplets, projected, meanmax_scorer)
        tr_thresh, _ = best_threshold_loo(all_train_triplets, projected)
        tr_thresh_test = score_triplets(test_triplets, projected,
                                         lambda a, b: threshold_hit_rate(a, b, tr_thresh))

        meanmax_results.append(tr_meanmax_test)
        threshold_results.append(tr_thresh_test)
        print(f"seed={seed}  val_acc={best_val_acc:.3f}  test_meanmax={tr_meanmax_test:.3f}  "
              f"test_threshold={tr_thresh_test:.3f}")

    print(f"\n=== Across {len(TRAINING_SEEDS)} training seeds (fixed data split) ===")
    print(f"zero-shot meanmax test accuracy:      {zs_meanmax_test:.3f}")
    print(f"trained meanmax test accuracy:        mean={statistics.mean(meanmax_results):.3f}  "
          f"std={statistics.stdev(meanmax_results):.3f}  min={min(meanmax_results):.3f}  max={max(meanmax_results):.3f}")
    print(f"trained threshold test accuracy:      mean={statistics.mean(threshold_results):.3f}  "
          f"std={statistics.stdev(threshold_results):.3f}  min={min(threshold_results):.3f}  max={max(threshold_results):.3f}")

    out_path = os.path.join(SPLIT_DIR, "seed_verification_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "zero_shot_meanmax_test": zs_meanmax_test,
            "training_seeds": TRAINING_SEEDS,
            "meanmax_results": meanmax_results,
            "threshold_results": threshold_results,
            "meanmax_mean": statistics.mean(meanmax_results),
            "meanmax_std": statistics.stdev(meanmax_results),
        }, f, indent=2)
    print(f"\nWrote results to {out_path}")


if __name__ == "__main__":
    main()
