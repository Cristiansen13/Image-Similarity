"""Generalization test: same methodology as the SOP pipeline (mine hard
negatives via DINOv2 global-embedding similarity, cache patch embeddings),
applied to CUB-200-2011 (birds) instead of Stanford Online Products
(products) -- does the technique (frozen DINOv2 + trained linear
projection via triplet loss) generalize to a completely different visual
domain?

Uses the standard metric-learning class split for CUB (not the official
per-image classification split): classes 1-100 for training, 101-200 for
testing -- this is what CUB-based retrieval papers actually use, and
matches our SOP methodology of disjoint *classes* between train/test.
"""
import json
import os
import random
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from encoders import PatchEncoder
from maxsim import global_cosine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUB_DIR = os.path.join(ROOT, "data", "cub_raw", "CUB_200_2011")
IMAGES_DIR = os.path.join(CUB_DIR, "images")
SPLIT_DIR = os.path.join(ROOT, "data", "cub_split")
MIN_PHOTOS_PER_CLASS = 3
SEED = 42


def load_cub_index():
    by_class = defaultdict(list)
    with open(os.path.join(CUB_DIR, "images.txt")) as f:
        image_paths = {}
        for line in f:
            image_id, path = line.split()
            image_paths[image_id] = path
    with open(os.path.join(CUB_DIR, "image_class_labels.txt")) as f:
        for line in f:
            image_id, class_id = line.split()
            by_class[class_id].append(image_paths[image_id])
    return by_class


def mine_triplets(anchor_classes, candidate_classes, by_class, rng, encoder, full_cache,
                  triplets_per_class=1):
    """Mine up to `triplets_per_class` triplets per anchor class, using
    disjoint (anchor, positive) photo pairs within each class and the
    hardest (most-confusable by DINOv2 global cosine) same-domain negative
    per anchor photo."""
    anchor_pairs = {}
    for c in anchor_classes:
        photos = by_class[c][:]
        rng.shuffle(photos)
        n_pairs = min(triplets_per_class, len(photos) // 2)
        anchor_pairs[c] = [(photos[2 * i], photos[2 * i + 1]) for i in range(n_pairs)]

    pool_photo = {c: by_class[c][-1] for c in candidate_classes}

    needed = set()
    for pairs in anchor_pairs.values():
        for a, p in pairs:
            needed.update([a, p])
    needed.update(pool_photo.values())

    embeddings = {}
    for i, rel_path in enumerate(needed):
        local_path = os.path.join(IMAGES_DIR, rel_path)
        if local_path not in full_cache:
            full_cache[local_path] = encoder.encode(local_path)
        embeddings[rel_path] = full_cache[local_path].global_embedding
        if (i + 1) % 100 == 0:
            print(f"    encoded {i + 1}/{len(needed)} in this mining pass ({len(full_cache)} total cached)")

    triplets = []
    for c in anchor_classes:
        candidates = [cand for cand in candidate_classes if cand != c]
        for anchor_path, positive_path in anchor_pairs[c]:
            best_class, best_sim = None, -2.0
            for cand in candidates:
                sim = global_cosine(embeddings[anchor_path], embeddings[pool_photo[cand]])
                if sim > best_sim:
                    best_sim, best_class = sim, cand
            triplets.append({
                "anchor": os.path.join(IMAGES_DIR, anchor_path),
                "positive": os.path.join(IMAGES_DIR, positive_path),
                "negative": os.path.join(IMAGES_DIR, pool_photo[best_class]),
                "class_id": c,
            })
    return triplets


def main():
    rng = random.Random(SEED)
    os.makedirs(SPLIT_DIR, exist_ok=True)

    by_class = load_cub_index()
    train_classes = [str(i) for i in range(1, 101)]
    test_classes = [str(i) for i in range(101, 201)]
    train_classes = [c for c in train_classes if len(by_class[c]) >= MIN_PHOTOS_PER_CLASS]
    test_classes = [c for c in test_classes if len(by_class[c]) >= MIN_PHOTOS_PER_CLASS]
    print(f"{len(train_classes)} train classes, {len(test_classes)} test classes "
          f"(standard CUB metric-learning split: 1-100 train, 101-200 test)")

    encoder = PatchEncoder()
    full_cache = {}
    triplets_per_train_class = int(os.environ.get("CUB_TRIPLETS_PER_TRAIN_CLASS", "8"))
    triplets_per_test_class = int(os.environ.get("CUB_TRIPLETS_PER_TEST_CLASS", "3"))
    print(f"Mining train triplets ({triplets_per_train_class}/class)...")
    train_triplets = mine_triplets(train_classes, train_classes, by_class, rng, encoder, full_cache,
                                    triplets_per_class=triplets_per_train_class)
    print(f"  ({len(full_cache)} images encoded so far)")
    print(f"Mining test triplets ({triplets_per_test_class}/class)...")
    test_triplets = mine_triplets(test_classes, test_classes, by_class, rng, encoder, full_cache,
                                   triplets_per_class=triplets_per_test_class)
    print(f"  ({len(full_cache)} images encoded so far)")

    assert set(t["class_id"] for t in train_triplets).isdisjoint(t["class_id"] for t in test_triplets)

    with open(os.path.join(SPLIT_DIR, "train_triplets.json"), "w") as f:
        json.dump(train_triplets, f, indent=2)
    with open(os.path.join(SPLIT_DIR, "test_triplets.json"), "w") as f:
        json.dump(test_triplets, f, indent=2)
    print(f"Wrote {len(train_triplets)} train / {len(test_triplets)} test triplets")

    all_paths = sorted({t[k] for triplets in (train_triplets, test_triplets)
                         for t in triplets for k in ("anchor", "positive", "negative")})
    print(f"Collecting patch embeddings for {len(all_paths)} images "
          f"({sum(1 for p in all_paths if p in full_cache)} already cached from mining)...")
    patch_embeddings = {}
    for i, path in enumerate(all_paths):
        if path not in full_cache:
            full_cache[path] = encoder.encode(path)
        patch_embeddings[path] = full_cache[path].patch_embeddings
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(all_paths)}")

    torch.save(patch_embeddings, os.path.join(SPLIT_DIR, "patch_embeddings.pt"))
    print(f"Saved patch embeddings to {os.path.join(SPLIT_DIR, 'patch_embeddings.pt')}")


if __name__ == "__main__":
    main()
