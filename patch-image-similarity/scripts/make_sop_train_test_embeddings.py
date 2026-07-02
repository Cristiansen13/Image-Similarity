"""Build a genuinely held-out train/test split from Stanford Online Products
for training a projection head: train and test draw from DISJOINT product
classes (never overlapping), with hard negatives mined separately within
each split so there's no leakage from test into training or threshold
selection.

Also precomputes and caches raw (frozen) DINOv2 patch embeddings for every
image to disk, since that's the expensive step (~1-2s/image on CPU) and the
training script needs to iterate quickly.
"""
import json
import os
import random
import sys
import zipfile
from collections import defaultdict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from encoders import PatchEncoder
from maxsim import global_cosine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZIP_PATH = os.path.join(ROOT, "data", "sop_raw", "Stanford_Online_Products.zip")
SPLIT_DIR = os.path.join(ROOT, "data", os.environ.get("SOP_SPLIT_NAME", "sop_split"))
IMAGES_DIR = os.path.join(SPLIT_DIR, "images")
N_TRAIN_CLASSES = 1000
N_TEST_CLASSES = 300
POOL_SIZE_PER_CATEGORY = 30
MIN_PHOTOS_PER_CLASS = 3
SEED = int(os.environ.get("SOP_SPLIT_SEED", "123"))


def load_sop_index():
    with zipfile.ZipFile(ZIP_PATH) as zf:
        with zf.open("Stanford_Online_Products/Ebay_info.txt") as f:
            lines = f.read().decode("utf-8").splitlines()
        rows = [line.split() for line in lines[1:]]

    by_class, by_super_class, class_to_super = defaultdict(list), defaultdict(set), {}
    for _, class_id, super_class_id, path in rows:
        by_class[class_id].append(path)
        by_super_class[super_class_id].add(class_id)
        class_to_super[class_id] = super_class_id
    return by_class, by_super_class, class_to_super


def split_classes(by_class, by_super_class, rng):
    """Disjoint per-super-category train/test pools, so hard-negative mining
    for test never touches a class used anywhere in train."""
    eligible = {c for c, paths in by_class.items() if len(paths) >= MIN_PHOTOS_PER_CLASS}
    train_pool_by_super, test_pool_by_super = {}, {}
    for sup, classes in by_super_class.items():
        elig = [c for c in classes if c in eligible]
        rng.shuffle(elig)
        split_point = int(len(elig) * 0.75)
        train_pool_by_super[sup] = elig[:split_point]
        test_pool_by_super[sup] = elig[split_point:]

    all_train_candidates = [c for pool in train_pool_by_super.values() for c in pool]
    all_test_candidates = [c for pool in test_pool_by_super.values() for c in pool]
    train_anchors = rng.sample(all_train_candidates, min(N_TRAIN_CLASSES, len(all_train_candidates)))
    test_anchors = rng.sample(all_test_candidates, min(N_TEST_CLASSES, len(all_test_candidates)))
    assert set(train_anchors).isdisjoint(test_anchors)
    return train_anchors, test_anchors, train_pool_by_super, test_pool_by_super


def mine_triplets(anchor_classes, pool_by_super, by_class, class_to_super, rng, encoder, path_map, zf, full_cache,
                  triplets_per_class=1):
    """Extract images + encode them once each into full_cache (patch + global
    embeddings); the final embeddings-collection pass in main() reuses these
    instead of re-encoding. Mines up to `triplets_per_class` triplets per
    class from disjoint (anchor, positive) photo pairs."""
    needed_supers = {class_to_super[c] for c in anchor_classes}
    pool = {sup: rng.sample(pool_by_super[sup], min(POOL_SIZE_PER_CATEGORY, len(pool_by_super[sup])))
            for sup in needed_supers}

    anchor_pairs = {}
    for c in anchor_classes:
        photos = by_class[c][:]
        rng.shuffle(photos)
        n_pairs = min(triplets_per_class, len(photos) // 2)
        anchor_pairs[c] = [(photos[2 * i], photos[2 * i + 1]) for i in range(n_pairs)]

    pool_photo = {c: by_class[c][0] for sup in pool for c in pool[sup]}

    needed = set()
    for pairs in anchor_pairs.values():
        for a, p in pairs:
            needed.update([a, p])
    needed.update(pool_photo.values())
    extract_images(needed, zf, path_map)

    embeddings = {}
    for i, path in enumerate(needed):
        local_path = path_map[path]
        if local_path not in full_cache:
            full_cache[local_path] = encoder.encode(local_path)
        embeddings[path] = full_cache[local_path].global_embedding
        if (i + 1) % 200 == 0:
            print(f"    encoded {i + 1}/{len(needed)} in this mining pass "
                  f"({len(full_cache)} total cached)")

    triplets = []
    for c in anchor_classes:
        sup = class_to_super[c]
        candidates = [cand for cand in pool[sup] if cand != c]
        if not candidates:
            continue
        for anchor_path, positive_path in anchor_pairs[c]:
            best_class, best_sim = None, -2.0
            for cand in candidates:
                sim = global_cosine(embeddings[anchor_path], embeddings[pool_photo[cand]])
                if sim > best_sim:
                    best_sim, best_class = sim, cand
            triplets.append({
                "anchor": path_map[anchor_path], "positive": path_map[positive_path],
                "negative": path_map[pool_photo[best_class]], "class_id": c,
            })
    return triplets


def extract_images(paths, zf, path_map):
    os.makedirs(IMAGES_DIR, exist_ok=True)
    names = zf.namelist()
    for path in paths:
        if path in path_map:
            continue
        out_path = os.path.join(IMAGES_DIR, os.path.basename(path))
        if not os.path.exists(out_path):
            member_name = path if path in names else f"Stanford_Online_Products/{path}"
            with zf.open(member_name) as src, open(out_path, "wb") as dst:
                dst.write(src.read())
        path_map[path] = out_path


def main():
    rng = random.Random(SEED)
    os.makedirs(SPLIT_DIR, exist_ok=True)

    print("Loading SOP index...")
    by_class, by_super_class, class_to_super = load_sop_index()
    train_anchors, test_anchors, train_pool, test_pool = split_classes(by_class, by_super_class, rng)
    print(f"{len(train_anchors)} train classes, {len(test_anchors)} test classes (disjoint)")

    encoder = PatchEncoder()
    path_map = {}
    full_cache = {}  # local_path -> EncodedImage, populated during mining, reused below (avoids re-encoding)
    tpc_train = int(os.environ.get("SOP_TRIPLETS_PER_TRAIN_CLASS", "1"))
    tpc_test = int(os.environ.get("SOP_TRIPLETS_PER_TEST_CLASS", "1"))
    with zipfile.ZipFile(ZIP_PATH) as zf:
        print(f"Mining train triplets ({tpc_train}/class)...")
        train_triplets = mine_triplets(train_anchors, train_pool, by_class, class_to_super, rng, encoder, path_map, zf, full_cache,
                                        triplets_per_class=tpc_train)
        print(f"  ({len(full_cache)} images encoded so far)")
        print(f"Mining test triplets ({tpc_test}/class)...")
        test_triplets = mine_triplets(test_anchors, test_pool, by_class, class_to_super, rng, encoder, path_map, zf, full_cache,
                                       triplets_per_class=tpc_test)
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
