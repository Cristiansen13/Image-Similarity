"""Test whether threshold/hit-rate rescoring -- which lost badly on caption
embeddings -- behaves differently on raw DINOv2 vision patches, where the
signal is far richer (768-dim, not compressed through a lossy captioning
bottleneck first).

Directly tests the calibration complaint: raw MaxSim (mean-of-max cosine)
over vision patches is known to be permissive and compressed into a narrow
band (embedding anisotropy -- even unrelated images share low-level visual
structure). A hit-rate score ("N of 256 patches clearly correspond") is
bounded 0-1 by construction and may separate matches from non-matches more
cleanly. Fully local (DINOv2 only) -- no NIM API calls, so this runs
independently of any captioning job.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from encoders import PatchEncoder
from maxsim import symmetric_maxsim, threshold_hit_rate
from eval import triplet_accuracy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRIPLETS_PATH = os.path.join(ROOT, "data", "sop_hard_triplets.json")
THRESHOLDS = (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7)


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum()
    ties = (diff == 0).sum()
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def main():
    with open(TRIPLETS_PATH) as f:
        triplets = json.load(f)
    unique_images = sorted({t[k] for t in triplets for k in ("anchor", "positive", "negative")})
    print(f"{len(triplets)} triplets, {len(unique_images)} unique images")

    encoder = PatchEncoder()
    print("Encoding patch embeddings (local, no API calls)...")
    patch_cache = {path: encoder.encode(path) for path in unique_images}

    def meanmax_score(a, b):
        return symmetric_maxsim(patch_cache[a].patch_embeddings, patch_cache[b].patch_embeddings).similarity

    def make_threshold_score(t):
        def score(a, b):
            return threshold_hit_rate(patch_cache[a].patch_embeddings, patch_cache[b].patch_embeddings, t)
        return score

    scorers = {"meanmax": meanmax_score}
    for t in THRESHOLDS:
        scorers[f"threshold_t{t}"] = make_threshold_score(t)

    print()
    results = {}
    pos_neg = {}
    for name, fn in scorers.items():
        result = triplet_accuracy(triplets, fn)
        results[name] = result.accuracy_overall
        pos_scores = [fn(t["anchor"], t["positive"]) for t in triplets]
        neg_scores = [fn(t["anchor"], t["negative"]) for t in triplets]
        pos_neg[name] = (pos_scores, neg_scores)

    print(f"{'scorer':16s}{'accuracy':>10s}{'AUC':>10s}{'pos_mean':>10s}{'neg_mean':>10s}"
          f"{'pos_std':>10s}{'neg_std':>10s}")
    print("-" * 76)
    for name in scorers:
        pos, neg = pos_neg[name]
        pos_a, neg_a = np.array(pos), np.array(neg)
        the_auc = auc(pos, neg)
        print(f"{name:16s}{results[name]:10.3f}{the_auc:10.3f}{pos_a.mean():10.3f}"
              f"{neg_a.mean():10.3f}{pos_a.std():10.3f}{neg_a.std():10.3f}")

    out_path = os.path.join(ROOT, "data", "vision_threshold_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "accuracy": results,
            "pos_neg_scores": {k: {"pos": v[0], "neg": v[1]} for k, v in pos_neg.items()},
        }, f, indent=2)
    print(f"\nWrote results to {out_path}")


if __name__ == "__main__":
    main()
