"""Same top-K MaxSim sweep as check_topk_maxsim.py, but on CUB triplets --
finding CUB's own optimal K before designing a complexity-adaptive K function
that interpolates between CUB's optimum (simple images) and COCO's K=4
(complex scenes), rather than assuming plain full-256 MaxSim is CUB-optimal
just because it beat K=4 there.
"""
import os
import random
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from encoders import PatchEncoder
from maxsim import topk_symmetric_maxsim

SEED = 0
N_TRIPLETS = 200


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
        triplets.append({
            "anchor": os.path.join(images_root, anchor),
            "positive": os.path.join(images_root, positive),
            "negative": os.path.join(images_root, negative),
        })
    return triplets


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cub-dir", required=True)
    args = ap.parse_args()

    triplets = build_cub_triplets(args.cub_dir, N_TRIPLETS, SEED)
    all_paths = sorted(set(p for t in triplets for p in (t["anchor"], t["positive"], t["negative"])))
    encoder = PatchEncoder()
    print(f"Encoding {len(all_paths)} CUB images...")
    cache = {}
    for i, p in enumerate(all_paths):
        cache[p] = encoder.encode(p).patch_embeddings
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(all_paths)}")

    ks = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    print(f"\n{'K':>12s}{'triplet_acc':>14s}{'pos_mean':>11s}{'neg_mean':>11s}{'gap':>10s}")
    print("-" * 60)
    for k in ks:
        pos_scores = [topk_symmetric_maxsim(cache[t["anchor"]], cache[t["positive"]], k=k).similarity
                      for t in triplets]
        neg_scores = [topk_symmetric_maxsim(cache[t["anchor"]], cache[t["negative"]], k=k).similarity
                      for t in triplets]
        acc = np.mean(np.array(pos_scores) > np.array(neg_scores))
        label = f"{k}" if k < 256 else "256(=plain)"
        print(f"{label:>12s}{acc:14.4f}{np.mean(pos_scores):11.4f}{np.mean(neg_scores):11.4f}"
              f"{np.mean(pos_scores) - np.mean(neg_scores):10.4f}")


if __name__ == "__main__":
    main()
