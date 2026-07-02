"""Build a small real-photo triplet set from Stanford Online Products.

SOP groups multiple real e-commerce photos (different angles/backgrounds/
lighting) under the same product `class_id`, with a `super_class_id` category
(bicycle, chair, mug, ...). That gives us real (not synthetic) same-instance
positive pairs -- exactly our "same object, different background" research
question, at scale, unlike the skimage toy set.

Only extracts a small subset of images from the local zip (no need to unpack
all 120k files) and picks *hard* negatives (same category, different
product) alongside easy ones, since same-category negatives are the more
diagnostic test for whether patch/caption matching actually distinguishes
instances rather than just categories.
"""
import json
import os
import random
import zipfile
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZIP_PATH = os.path.join(ROOT, "data", "sop_raw", "Stanford_Online_Products.zip")
SUBSET_DIR = os.path.join(ROOT, "data", "sop_subset")
N_CLASSES = 40
MIN_PHOTOS_PER_CLASS = 3
SEED = 42


def find_info_file(zf: zipfile.ZipFile) -> str:
    names = zf.namelist()
    for candidate in ("Ebay_info.txt", "Stanford_Online_Products/Ebay_info.txt"):
        if candidate in names:
            return candidate
    matches = [n for n in names if n.endswith("Ebay_info.txt")]
    if matches:
        return matches[0]
    raise RuntimeError(f"Could not find Ebay_info.txt in zip. First 20 entries: {names[:20]}")


def main():
    rng = random.Random(SEED)
    os.makedirs(SUBSET_DIR, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH) as zf:
        info_name = find_info_file(zf)
        print(f"Reading {info_name}")
        with zf.open(info_name) as f:
            lines = f.read().decode("utf-8").splitlines()

        header = lines[0].split()
        print(f"Columns: {header}")
        rows = [line.split() for line in lines[1:]]
        # expected columns: image_id class_id super_class_id path
        by_class = defaultdict(list)
        by_super_class = defaultdict(set)
        for row in rows:
            _, class_id, super_class_id, path = row
            by_class[class_id].append(path)
            by_super_class[super_class_id].add(class_id)

        eligible_classes = [c for c, paths in by_class.items() if len(paths) >= MIN_PHOTOS_PER_CLASS]
        print(f"{len(by_class)} total classes, {len(eligible_classes)} with >= {MIN_PHOTOS_PER_CLASS} photos")

        chosen_classes = rng.sample(eligible_classes, min(N_CLASSES, len(eligible_classes)))

        class_to_super = {}
        for row in rows:
            _, class_id, super_class_id, path = row
            class_to_super[class_id] = super_class_id

        triplets_meta = []
        needed_paths = set()
        for class_id in chosen_classes:
            photos = by_class[class_id]
            anchor_path, positive_path = rng.sample(photos, 2)

            super_class_id = class_to_super[class_id]
            same_category_others = [c for c in by_super_class[super_class_id]
                                     if c != class_id and c in by_class]
            hard_negative_class = rng.choice(same_category_others) if same_category_others else None
            other_class = rng.choice([c for c in chosen_classes if c != class_id])

            negative_path = rng.choice(by_class[hard_negative_class]) if hard_negative_class \
                else rng.choice(by_class[other_class])
            negative_type = "sop_hard" if hard_negative_class else "sop_easy"

            needed_paths.update([anchor_path, positive_path, negative_path])
            triplets_meta.append({
                "class_id": class_id, "super_class_id": super_class_id,
                "anchor": anchor_path, "positive": positive_path, "negative": negative_path,
                "negative_type": negative_type,
            })

        print(f"Extracting {len(needed_paths)} images to {SUBSET_DIR} ...")
        path_map = {}
        for path in needed_paths:
            member_name = path if path in zf.namelist() else f"Stanford_Online_Products/{path}"
            out_path = os.path.join(SUBSET_DIR, os.path.basename(path))
            with zf.open(member_name) as src, open(out_path, "wb") as dst:
                dst.write(src.read())
            path_map[path] = out_path

    triplets = []
    for t in triplets_meta:
        triplets.append({
            "anchor": path_map[t["anchor"]],
            "positive": path_map[t["positive"]],
            "negative": path_map[t["negative"]],
            "type": t["negative_type"],
            "note": f"SOP class {t['class_id']} (super_class {t['super_class_id']})",
        })

    manifest_path = os.path.join(ROOT, "data", "sop_triplets.json")
    with open(manifest_path, "w") as f:
        json.dump(triplets, f, indent=2)

    n_hard = sum(1 for t in triplets if t["type"] == "sop_hard")
    print(f"Wrote {len(triplets)} triplets ({n_hard} hard/same-category negatives) to {manifest_path}")


if __name__ == "__main__":
    main()
