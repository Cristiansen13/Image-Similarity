"""Honest, cross-validated version of analyze_vision_threshold.py.

Picking the best threshold by sweeping and evaluating on the same 40
triplets is optimistic (the threshold is fit to the test set). This selects
the threshold via leave-one-out: for each held-out triplet, pick whichever
threshold scored best on the *other* 39, then evaluate on the held-out one.
That's the honest number to compare against raw mean-of-max.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from encoders import PatchEncoder
from maxsim import symmetric_maxsim, threshold_hit_rate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRIPLETS_PATH = os.path.join(ROOT, "data", "sop_hard_triplets.json")
THRESHOLDS = np.arange(0.30, 0.86, 0.05)


def main():
    with open(TRIPLETS_PATH) as f:
        triplets = json.load(f)
    unique_images = sorted({t[k] for t in triplets for k in ("anchor", "positive", "negative")})

    encoder = PatchEncoder()
    print("Encoding patch embeddings...")
    patch_cache = {path: encoder.encode(path) for path in unique_images}

    n = len(triplets)
    meanmax_correct = np.zeros(n, dtype=bool)
    # per_threshold_correct[t_idx, i] = is triplet i correctly ranked at that threshold
    per_threshold_correct = np.zeros((len(THRESHOLDS), n), dtype=bool)

    for i, t in enumerate(triplets):
        a, p, neg = t["anchor"], t["positive"], t["negative"]
        pa, pp, pn = patch_cache[a].patch_embeddings, patch_cache[p].patch_embeddings, patch_cache[neg].patch_embeddings

        meanmax_correct[i] = symmetric_maxsim(pa, pp).similarity > symmetric_maxsim(pa, pn).similarity

        for ti, thresh in enumerate(THRESHOLDS):
            pos_score = threshold_hit_rate(pa, pp, thresh)
            neg_score = threshold_hit_rate(pa, pn, thresh)
            per_threshold_correct[ti, i] = pos_score > neg_score

    print(f"\nmeanmax accuracy: {meanmax_correct.mean():.3f}")
    print("\nPer-threshold accuracy (fit on full set, for reference -- optimistic):")
    for ti, thresh in enumerate(THRESHOLDS):
        print(f"  t={thresh:.2f}: {per_threshold_correct[ti].mean():.3f}")

    # leave-one-out: for each held-out triplet, pick the threshold with the
    # best accuracy on the *other* 39, then use its correctness on the held-out one
    loo_correct = np.zeros(n, dtype=bool)
    chosen_thresholds = []
    for i in range(n):
        train_mask = np.ones(n, dtype=bool)
        train_mask[i] = False
        train_acc_per_threshold = per_threshold_correct[:, train_mask].mean(axis=1)
        best_ti = int(np.argmax(train_acc_per_threshold))
        chosen_thresholds.append(THRESHOLDS[best_ti])
        loo_correct[i] = per_threshold_correct[best_ti, i]

    print(f"\nThreshold hit-rate, LEAVE-ONE-OUT cross-validated accuracy: {loo_correct.mean():.3f}")
    print(f"Chosen thresholds across folds: min={min(chosen_thresholds):.2f} "
          f"max={max(chosen_thresholds):.2f} most_common={max(set(chosen_thresholds), key=chosen_thresholds.count):.2f}")
    print(f"\nmeanmax (uncalibrated, no threshold to tune): {meanmax_correct.mean():.3f}")
    print(f"threshold hit-rate (honest, cross-validated):  {loo_correct.mean():.3f}")


if __name__ == "__main__":
    main()
