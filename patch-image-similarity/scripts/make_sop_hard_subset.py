"""Build a *hard* real-photo triplet set from Stanford Online Products.

The earlier sop_triplets.json used a random same-category class as the
negative, and Phase 0 (vision patches) hit 100% accuracy on it -- meaning
there was no headroom left to show whether a learned combiner (Phase-2-style)
could add value on top of vision alone. This script mines negatives that are
deliberately confusable: for each anchor class, search a pool of other
same-super-category classes and pick the one whose representative photo has
the HIGHEST DINOv2 global-embedding cosine similarity to the anchor
(excluding the true class). That guarantees genuinely hard cases where
vision-only matching has a real chance to fail, which is the only way to
meaningfully test whether adding caption signal (via a trained combiner)
recovers accuracy vision alone misses.

All embedding/mining here is local DINOv2 inference (free, no API calls);
NIM captioning only happens later, on the small set of images actually
selected into the final triplets.
"""
import json
import os
import random
import sys
import zipfile
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from encoders import PatchEncoder
from maxsim import global_cosine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZIP_PATH = os.path.join(ROOT, "data", "sop_raw", "Stanford_Online_Products.zip")
SUBSET_DIR = os.path.join(ROOT, "data", "sop_hard_subset")
N_ANCHOR_CLASSES = 40
POOL_SIZE_PER_CATEGORY = 40
MIN_PHOTOS_PER_CLASS = 3
SEED = 7


def main():
    rng = random.Random(SEED)
    os.makedirs(SUBSET_DIR, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH) as zf:
        with zf.open("Stanford_Online_Products/Ebay_info.txt") as f:
            lines = f.read().decode("utf-8").splitlines()
        rows = [line.split() for line in lines[1:]]

        by_class = defaultdict(list)
        by_super_class = defaultdict(set)
        class_to_super = {}
        for _, class_id, super_class_id, path in rows:
            by_class[class_id].append(path)
            by_super_class[super_class_id].add(class_id)
            class_to_super[class_id] = super_class_id

        eligible = [c for c, paths in by_class.items() if len(paths) >= MIN_PHOTOS_PER_CLASS]
        anchor_classes = rng.sample(eligible, min(N_ANCHOR_CLASSES, len(eligible)))

        # candidate pool per super-category actually used by our anchors
        needed_supers = {class_to_super[c] for c in anchor_classes}
        pool_by_super = {}
        for sup in needed_supers:
            candidates = list(by_super_class[sup])
            pool_by_super[sup] = rng.sample(candidates, min(POOL_SIZE_PER_CATEGORY, len(candidates)))

        # figure out every image path we need to extract: anchor pairs + one representative photo per pool candidate
        needed_paths = set()
        anchor_photos = {}
        for c in anchor_classes:
            a, p = rng.sample(by_class[c], 2)
            anchor_photos[c] = (a, p)
            needed_paths.update([a, p])

        pool_photo = {}  # class_id -> representative photo path
        for sup, candidates in pool_by_super.items():
            for c in candidates:
                photo = by_class[c][0]
                pool_photo[c] = photo
                needed_paths.add(photo)

        print(f"Extracting {len(needed_paths)} images to {SUBSET_DIR} ...")
        path_map = {}
        names = zf.namelist()
        for path in needed_paths:
            member_name = path if path in names else f"Stanford_Online_Products/{path}"
            out_path = os.path.join(SUBSET_DIR, os.path.basename(path))
            if not os.path.exists(out_path):
                with zf.open(member_name) as src, open(out_path, "wb") as dst:
                    dst.write(src.read())
            path_map[path] = out_path

    print("Encoding all extracted images with DINOv2 (local, no API calls)...")
    encoder = PatchEncoder()
    embeddings = {}
    for path, local_path in path_map.items():
        embeddings[path] = encoder.encode(local_path).global_embedding

    triplets = []
    for c in anchor_classes:
        anchor_path, positive_path = anchor_photos[c]
        sup = class_to_super[c]
        pool = pool_by_super[sup]

        best_neg_class, best_sim = None, -2.0
        for cand_class in pool:
            if cand_class == c:
                continue
            sim = global_cosine(embeddings[anchor_path], embeddings[pool_photo[cand_class]])
            if sim > best_sim:
                best_sim = sim
                best_neg_class = cand_class

        positive_sim = global_cosine(embeddings[anchor_path], embeddings[positive_path])
        negative_path = pool_photo[best_neg_class]

        triplets.append({
            "anchor": path_map[anchor_path],
            "positive": path_map[positive_path],
            "negative": path_map[negative_path],
            "type": "sop_hard_mined",
            "note": (f"SOP class {c} (super_class {sup}); mined negative class "
                     f"{best_neg_class}; global_sim(anchor,positive)={positive_sim:.3f} "
                     f"global_sim(anchor,negative)={best_sim:.3f}"),
        })

    manifest_path = os.path.join(ROOT, "data", "sop_hard_triplets.json")
    with open(manifest_path, "w") as f:
        json.dump(triplets, f, indent=2)

    n_where_neg_closer = sum(1 for t in triplets
                              if float(t["note"].split("global_sim(anchor,negative)=")[1]) >
                              float(t["note"].split("global_sim(anchor,positive)=")[1].split(" ")[0]))
    print(f"Wrote {len(triplets)} mined-hard triplets to {manifest_path}")
    print(f"{n_where_neg_closer}/{len(triplets)} triplets already fool the raw global embedding "
          f"(negative closer than positive) -- these should be genuinely hard for Phase 0 too.")


if __name__ == "__main__":
    main()
