"""Complement to triplet accuracy: is the raw similarity score itself a
meaningful, well-calibrated 0-1 "how similar are these" number, not just
correctly *ranked* within each triplet?

Triplet accuracy only checks score(anchor,pos) > score(anchor,neg) -- a
method could pass that 100% of the time while its raw scores are poorly
separated or clustered in a narrow, unintuitive range (e.g. everything
between 0.85-0.95). This script reports, per scorer, on the hard-mined SOP
triplets:

- positive-pair vs negative-pair score distributions (mean/std/min/max)
- AUC (P(random positive-pair score > random negative-pair score) over all
  40x40 cross pairs, not just matched triplets) -- a threshold-free measure
  of how well the raw score separates matches from non-matches
- a simple data-driven calibration mapping raw scores onto an intuitive 0-1
  scale, anchored at this dataset's own negative-pair mean (~0) and
  positive-pair mean (~1), since raw cosine similarity is not naturally
  anchored that way (unrelated images often score 0.1-0.3, not ~0)
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_eval_hard_combiner import FEATURE_NAMES, train_logreg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = os.path.join(ROOT, "data", "sop_hard_eval_results.json")


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum()
    ties = (diff == 0).sum()
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def loo_combiner_scores(scored):
    """Like run_eval_hard_combiner.loo_combiner_accuracy, but returns the
    actual held-out predicted scores instead of just correct/incorrect."""
    n = len(scored)
    all_pos = torch.tensor([s["pos_features"] for s in scored], dtype=torch.float32)
    all_neg = torch.tensor([s["neg_features"] for s in scored], dtype=torch.float32)

    pos_scores, neg_scores = [], []
    for held_out in range(n):
        train_idx = [i for i in range(n) if i != held_out]
        X_train = torch.cat([all_pos[train_idx], all_neg[train_idx]], dim=0)
        y_train = torch.cat([torch.ones(len(train_idx)), torch.zeros(len(train_idx))])
        mean, std = X_train.mean(dim=0), X_train.std(dim=0).clamp_min(1e-6)
        w, b = train_logreg((X_train - mean) / std, y_train)

        pos_feat = (all_pos[held_out] - mean) / std
        neg_feat = (all_neg[held_out] - mean) / std
        pos_scores.append(torch.sigmoid(pos_feat @ w + b).item())
        neg_scores.append(torch.sigmoid(neg_feat @ w + b).item())
    return pos_scores, neg_scores


def report(name, pos, neg):
    pos_arr, neg_arr = np.array(pos), np.array(neg)
    the_auc = auc(pos, neg)
    neg_mean, pos_mean = neg_arr.mean(), pos_arr.mean()
    span = pos_mean - neg_mean
    calibrated_pos = np.clip((pos_arr - neg_mean) / span, 0, 1) if span > 1e-6 else pos_arr
    calibrated_neg = np.clip((neg_arr - neg_mean) / span, 0, 1) if span > 1e-6 else neg_arr

    print(f"\n{name}")
    print(f"  positive pairs: mean={pos_mean:.3f} std={pos_arr.std():.3f} "
          f"min={pos_arr.min():.3f} max={pos_arr.max():.3f}")
    print(f"  negative pairs: mean={neg_mean:.3f} std={neg_arr.std():.3f} "
          f"min={neg_arr.min():.3f} max={neg_arr.max():.3f}")
    print(f"  AUC (raw score separates match/non-match across all 40x40 pairs): {the_auc:.3f}")
    print(f"  calibrated to this dataset's own 0-1 scale: "
          f"positive mean={calibrated_pos.mean():.3f}, negative mean={calibrated_neg.mean():.3f}")


def main():
    with open(RESULTS_PATH) as f:
        results = json.load(f)
    scored = results["scored_triplets"]

    for i, name in enumerate(FEATURE_NAMES):
        pos = [s["pos_features"][i] for s in scored]
        neg = [s["neg_features"][i] for s in scored]
        report(name, pos, neg)

    combo_pos = [(s["pos_features"][2] + s["pos_features"][3]) / 2 for s in scored]
    combo_neg = [(s["neg_features"][2] + s["neg_features"][3]) / 2 for s in scored]
    report("phase1_combo_3+4", combo_pos, combo_neg)

    print("\nTraining LOO combiner to get calibrated held-out probability scores...")
    combiner_pos, combiner_neg = loo_combiner_scores(scored)
    report("trained_combiner (sigmoid output, already ~0-1)", combiner_pos, combiner_neg)


if __name__ == "__main__":
    main()
