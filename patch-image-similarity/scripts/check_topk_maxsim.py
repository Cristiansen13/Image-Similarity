"""Isolated check: does top-K MaxSim (average only the K strongest per-patch
correspondences, instead of all 256) improve calibration on the COCO
structural-confusion set, before folding it into the adaptive scorer?

Compares plain symmetric_maxsim against topk_symmetric_maxsim at a few K
values, using both raw elevation and standardized effect size (the two
metrics that disagreed in the earlier adaptive-scorer test) so any
improvement is validated under both readings, not just whichever looks better.
"""
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from encoders import PatchEncoder
from maxsim import symmetric_maxsim, topk_symmetric_maxsim

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COCO_TRIPLETS_PATH = os.path.join(ROOT, "data", "coco_confusion_triplets.json")
SEED = 0


def standardized_elevation(confusable, random_):
    c, r = np.array(confusable), np.array(random_)
    pooled_std = np.sqrt((c.var() + r.var()) / 2) + 1e-8
    return (c.mean() - r.mean()) / pooled_std


def main():
    with open(COCO_TRIPLETS_PATH) as f:
        triplets = json.load(f)
    anchors = [t["anchor"] for t in triplets]
    positives = [t["positive"] for t in triplets]
    confusable_negs = [t["negative"] for t in triplets]
    rng = random.Random(SEED)
    shuffled = confusable_negs[:]
    rng.shuffle(shuffled)
    for i in range(len(shuffled)):
        if shuffled[i] == confusable_negs[i]:
            shuffled[i], shuffled[(i + 1) % len(shuffled)] = shuffled[(i + 1) % len(shuffled)], shuffled[i]
    random_negs = shuffled

    all_paths = sorted(set(anchors) | set(positives) | set(confusable_negs))
    encoder = PatchEncoder()
    print(f"Encoding {len(all_paths)} images...")
    cache = {}
    for i, p in enumerate(all_paths):
        cache[p] = encoder.encode(p).patch_embeddings
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(all_paths)}")

    ks = [1, 2, 4, 8, 16, 32, 64, 128, 256]  # 256 = equivalent to plain mean-of-max
    print(f"\n{'K':>6s}{'pos_mean':>11s}{'conf_mean':>11s}{'rand_mean':>11s}"
          f"{'pos_vs_conf_acc':>17s}{'raw_elevation':>15s}{'std_effect_size':>18s}")
    print("-" * 100)
    for k in ks:
        pos_scores = [topk_symmetric_maxsim(cache[a], cache[p], k=k).similarity
                      for a, p in zip(anchors, positives)]
        conf_scores = [topk_symmetric_maxsim(cache[a], cache[n], k=k).similarity
                       for a, n in zip(anchors, confusable_negs)]
        rand_scores = [topk_symmetric_maxsim(cache[a], cache[n], k=k).similarity
                        for a, n in zip(anchors, random_negs)]
        raw_elev = np.mean(conf_scores) - np.mean(rand_scores)
        std_elev = standardized_elevation(conf_scores, rand_scores)
        pos_vs_conf_acc = np.mean(np.array(pos_scores) > np.array(conf_scores))
        label = f"{k}" if k < 256 else "256(=plain)"
        print(f"{label:>6s}{np.mean(pos_scores):11.4f}{np.mean(conf_scores):11.4f}{np.mean(rand_scores):11.4f}"
              f"{pos_vs_conf_acc:17.4f}{raw_elev:15.4f}{std_elev:18.4f}")


if __name__ == "__main__":
    main()
