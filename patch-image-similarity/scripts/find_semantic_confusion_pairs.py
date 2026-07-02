"""Search real COCO val2017 captions for genuine structural-similarity /
semantic-difference pairs: images whose captions describe the same pose or
composition (e.g. "standing on a chair") but a different subject (age,
number of people, or context) -- exactly the kind of case where raw visual
similarity can be deceptively high while a human would say they're not that
similar.
"""
import json
import os
import re
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAPTIONS_PATH = os.path.join(ROOT, "data", "coco_raw", "captions_val2017.json")

# (structural pattern regex, list of (label, subject-word regex) to contrast)
TEMPLATES = [
    (r"standing on (a|an|the) \w+", [
        ("solo_adult", r"\b(man|woman)\b"),
        ("child_or_group", r"\b(child|children|kids?|boys?|girls?|teenagers?|group|people)\b"),
    ]),
    (r"sitting on (a|an|the) \w+", [
        ("solo_adult", r"\b(man|woman)\b"),
        ("child_or_group", r"\b(child|children|kids?|boys?|girls?|teenagers?|group|people)\b"),
    ]),
    (r"riding (a|an|the) \w+", [
        ("solo_adult", r"\b(man|woman)\b"),
        ("child_or_group", r"\b(child|children|kids?|boys?|girls?|teenagers?|group|people)\b"),
    ]),
    (r"holding (a|an|the) \w+", [
        ("solo_adult", r"\b(man|woman)\b"),
        ("child_or_group", r"\b(child|children|kids?|boys?|girls?|teenagers?|group|people)\b"),
    ]),
]


def main():
    with open(CAPTIONS_PATH) as f:
        data = json.load(f)

    captions_by_image = defaultdict(list)
    for ann in data["annotations"]:
        captions_by_image[ann["image_id"]].append(ann["caption"].strip())
    image_url = {img["id"]: img["coco_url"] for img in data["images"]}
    image_filename = {img["id"]: img["file_name"] for img in data["images"]}

    found_pairs = []
    for structure_pattern, subject_groups in TEMPLATES:
        matches_by_group = defaultdict(list)
        for image_id, captions in captions_by_image.items():
            matching_caption = next((c for c in captions if re.search(structure_pattern, c.lower())), None)
            if matching_caption is None:
                continue
            for label, subject_pattern in subject_groups:
                if re.search(subject_pattern, matching_caption.lower()):
                    matches_by_group[label].append((image_id, matching_caption))
                    break

        groups = list(matches_by_group.keys())
        if len(groups) >= 2 and matches_by_group[groups[0]] and matches_by_group[groups[1]]:
            # take a few candidates from each group, not just the first
            for a_id, a_text in matches_by_group[groups[0]][:3]:
                for b_id, b_text in matches_by_group[groups[1]][:3]:
                    found_pairs.append({
                        "structure": structure_pattern,
                        "image_a_id": a_id, "image_a_url": image_url[a_id], "image_a_file": image_filename[a_id],
                        "image_a_caption": a_text,
                        "image_b_id": b_id, "image_b_url": image_url[b_id], "image_b_file": image_filename[b_id],
                        "image_b_caption": b_text,
                    })

    out_path = os.path.join(ROOT, "data", "coco_confusion_pairs.json")
    with open(out_path, "w") as f:
        json.dump(found_pairs, f, indent=2)

    print(f"Found {len(found_pairs)} candidate pairs")
    for p in found_pairs:
        print(f"\n[{p['structure']}]")
        print(f"  A ({p['image_a_id']}): {p['image_a_caption']}")
        print(f"  B ({p['image_b_id']}): {p['image_b_caption']}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
