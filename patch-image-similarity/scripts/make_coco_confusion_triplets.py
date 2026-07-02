"""Build a quantitative eval set for the structural-vs-semantic confusion
problem: triplets where
  anchor   = a real COCO image
  positive = an augmented (crop/jitter/jpeg) version of the same image
             (unambiguously "the same picture" to a human)
  negative = a DIFFERENT image whose caption shares the anchor's structural
             pose/composition phrase but differs in subject/context --
             exactly the "0.9 similarity for an old man on a chair vs
             teenagers on a chair" failure mode.

A good scorer must rank the augmented positive above the structurally-
confusable negative. Global embeddings that latch onto pose/composition
are expected to fail some of these; the hybrid caption gate should help.
"""
import io
import json
import os
import re
import urllib.request
from collections import defaultdict

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAPTIONS_PATH = os.path.join(ROOT, "data", "coco_raw", "captions_val2017.json")
IMG_DIR = os.path.join(ROOT, "data", "coco_confusion_images")
SEED = 0

TEMPLATES = [
    (r"(standing|sitting) on (a|an|the) (bench|chair|table)", [
        ("adult", r"\b(man|woman)\b"),
        ("child_or_group", r"\b(child|children|kids?|boys?|girls?|teenagers?|group of|crowd)\b"),
    ]),
    (r"riding (a|an|the) (bike|bicycle|motorcycle|horse|skateboard)", [
        ("adult", r"\b(man|woman)\b"),
        ("child_or_group", r"\b(child|children|kids?|boys?|girls?|teenagers?|couple|two people)\b"),
    ]),
    (r"holding (a|an|the) \w+", [
        ("adult", r"\b(man|woman)\b"),
        ("child_or_group", r"\b(child|children|kids?|boys?|girls?|toddlers?|baby)\b"),
    ]),
    (r"(walking|running) (on|down|along) (a|an|the) (street|beach|road|sidewalk)", [
        ("adult", r"\b(man|woman)\b"),
        ("child_or_group", r"\b(child|children|kids?|boys?|girls?|group|crowd|people)\b"),
    ]),
    (r"(eating|cutting) (a|an|the) \w+", [
        ("adult", r"\b(man|woman)\b"),
        ("child_or_group", r"\b(child|children|kids?|boys?|girls?|toddlers?|baby)\b"),
    ]),
    (r"(playing|swinging) (a|an|the)? ?(tennis|baseball|frisbee|racket|bat)", [
        ("adult", r"\b(man|woman)\b"),
        ("child_or_group", r"\b(child|children|kids?|boys?|girls?|teenagers?)\b"),
    ]),
]

MAX_PAIRS_PER_TEMPLATE = 6


def augment(img, rng):
    """Crop + brightness jitter + jpeg recompress -- clearly the same image
    to a human, but not pixel-identical."""
    w, h = img.size
    frac = rng.uniform(0.75, 0.9)
    cw, ch = int(w * frac), int(h * frac)
    x0 = rng.integers(0, w - cw + 1)
    y0 = rng.integers(0, h - ch + 1)
    out = img.crop((x0, y0, x0 + cw, y0 + ch))
    arr = np.array(out).astype(np.int16)
    arr = np.clip(arr + rng.uniform(-25, 25), 0, 255).astype(np.uint8)
    out = Image.fromarray(arr)
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=int(rng.integers(25, 60)))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def download(url, dest):
    if os.path.exists(dest):
        return True
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        print(f"  [warn] failed to download {url}: {e}")
        return False


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(IMG_DIR, exist_ok=True)

    with open(CAPTIONS_PATH) as f:
        data = json.load(f)
    captions_by_image = defaultdict(list)
    for ann in data["annotations"]:
        captions_by_image[ann["image_id"]].append(ann["caption"].strip())
    image_url = {img["id"]: img["coco_url"] for img in data["images"]}

    triplets = []
    used_ids = set()
    for structure_pattern, subject_groups in TEMPLATES:
        matches_by_group = defaultdict(list)
        for image_id, captions in captions_by_image.items():
            matching = next((c for c in captions if re.search(structure_pattern, c.lower())), None)
            if matching is None:
                continue
            for label, subject_pattern in subject_groups:
                if re.search(subject_pattern, matching.lower()):
                    matches_by_group[label].append((image_id, matching))
                    break

        groups = list(matches_by_group.keys())
        if len(groups) < 2:
            continue
        group_a, group_b = matches_by_group[groups[0]], matches_by_group[groups[1]]
        n = min(len(group_a), len(group_b), MAX_PAIRS_PER_TEMPLATE)
        for i in range(n):
            a_id, a_cap = group_a[i]
            b_id, b_cap = group_b[i]
            if a_id in used_ids or b_id in used_ids or a_id == b_id:
                continue

            a_path = os.path.join(IMG_DIR, f"{a_id}.jpg")
            b_path = os.path.join(IMG_DIR, f"{b_id}.jpg")
            if not (download(image_url[a_id], a_path) and download(image_url[b_id], b_path)):
                continue

            aug_path = os.path.join(IMG_DIR, f"{a_id}_aug.jpg")
            if not os.path.exists(aug_path):
                augment(Image.open(a_path).convert("RGB"), rng).save(aug_path)

            used_ids.update([a_id, b_id])
            triplets.append({
                "anchor": a_path, "positive": aug_path, "negative": b_path,
                "structure": structure_pattern,
                "anchor_caption": a_cap, "negative_caption": b_cap,
                "type": "coco_confusion",
            })

    out_path = os.path.join(ROOT, "data", "coco_confusion_triplets.json")
    with open(out_path, "w") as f:
        json.dump(triplets, f, indent=2)
    print(f"Built {len(triplets)} confusion triplets -> {out_path}")
    for t in triplets:
        print(f"  [{t['structure'][:40]:40s}] {t['anchor_caption'][:60]:60s} VS {t['negative_caption'][:60]}")


if __name__ == "__main__":
    main()
