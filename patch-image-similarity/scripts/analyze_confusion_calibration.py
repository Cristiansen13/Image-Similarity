"""Corrected test of the structural-vs-semantic confusion phenomenon.

Triplet accuracy (positive=near-duplicate crop, negative=confusable-but-
different photo) turned out to be the wrong metric -- a near-duplicate
crop trivially outranks any different photo regardless of pose, so every
scorer hit 1.000 and the test didn't actually stress anything.

The real question is about ABSOLUTE score calibration: does a
structurally-confusable negative (shares the anchor's pose/composition
phrase) score suspiciously HIGH compared to a genuinely random,
unconfusable negative? That's the actual "0.9 for an old man vs teenagers
on a chair" phenomenon. This script computes, per scorer:
  - anchor-vs-confusable-negative score (the 33 mined pairs)
  - anchor-vs-random-other-negative score (shuffled pairing across the
    whole triplet set, i.e. genuinely unrelated most of the time)
and reports the gap. A well-calibrated method should score confusable
negatives close to random negatives (low), not elevated toward positives.
A method exhibiting the "pose trap" will show confusable negatives
sitting well above random negatives.
"""
import json
import math
import os
import random
import sys

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from encoders import PatchEncoder
from maxsim import symmetric_maxsim, global_cosine
from nim_caption_regions import NimRegionCaptioner
from text_embed import TextEmbedder

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COCO_TRIPLETS = os.path.join(ROOT, "data", "coco_confusion_triplets.json")
COCO_CAPTION_CACHE = os.path.join(ROOT, "data", "coco_confusion_captions_cache.json")
GRID = 3
SEED = 0


def main():
    rng = random.Random(SEED)
    with open(COCO_TRIPLETS) as f:
        triplets = json.load(f)
    anchors = [t["anchor"] for t in triplets]
    confusable_negs = [t["negative"] for t in triplets]

    # random pairing: shuffle negatives against different anchors so most
    # pairs are genuinely unrelated (not sharing the structural phrase)
    shuffled = confusable_negs[:]
    rng.shuffle(shuffled)
    # avoid any accidental self-pairing with the anchor's own real confusable match
    for i in range(len(shuffled)):
        if shuffled[i] == confusable_negs[i]:
            shuffled[i], shuffled[(i + 1) % len(shuffled)] = shuffled[(i + 1) % len(shuffled)], shuffled[i]
    random_negs = shuffled

    all_paths = sorted(set(anchors) | set(confusable_negs))
    encoder = PatchEncoder()
    text_embedder = TextEmbedder()
    captioner = NimRegionCaptioner(cache_path=COCO_CAPTION_CACHE)  # reuses cache from eval_hybrid.py run
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()

    patch, dino_global, clip_emb, cap_emb = {}, {}, {}, {}
    print(f"Encoding {len(all_paths)} images...")
    for i, p in enumerate(all_paths):
        encoded = encoder.encode(p)
        patch[p] = encoded.patch_embeddings
        dino_global[p] = encoded.global_embedding
        image = Image.open(p).convert("RGB")
        with torch.no_grad():
            feat = clip_model.get_image_features(**clip_proc(images=image, return_tensors="pt"))
        clip_emb[p] = torch.nn.functional.normalize(feat[0], dim=0)
        regions = captioner.caption_regions(p, grid_size=GRID)
        cap_emb[p] = text_embedder.embed([r.description for r in regions])
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(all_paths)}")

    def V(a, b):
        return symmetric_maxsim(patch[a], patch[b]).similarity

    def C(a, b):
        return symmetric_maxsim(cap_emb[a], cap_emb[b]).similarity

    scorers = {
        "clip_global": lambda a, b: global_cosine(clip_emb[a], clip_emb[b]),
        "dinov2_global": lambda a, b: global_cosine(dino_global[a], dino_global[b]),
        "vision_maxsim": V,
        "caption_maxsim": C,
        "hybrid_geometric": lambda a, b: math.sqrt(max(V(a, b), 0) * max(C(a, b), 0)),
        "hybrid_gated": lambda a, b: V(a, b) / (1 + math.exp(-8 * (C(a, b) - 0.35))),
    }

    print(f"\n{'scorer':20s}{'confusable_mean':>18s}{'random_mean':>15s}{'elevation':>12s}")
    print("-" * 65)
    results = {}
    for name, fn in scorers.items():
        confusable_scores = [fn(a, n) for a, n in zip(anchors, confusable_negs)]
        random_scores = [fn(a, n) for a, n in zip(anchors, random_negs)]
        conf_mean, rand_mean = np.mean(confusable_scores), np.mean(random_scores)
        elevation = conf_mean - rand_mean
        print(f"{name:20s}{conf_mean:18.3f}{rand_mean:15.3f}{elevation:12.3f}")
        results[name] = {"confusable_mean": float(conf_mean), "random_mean": float(rand_mean),
                          "elevation": float(elevation),
                          "confusable_scores": confusable_scores, "random_scores": random_scores}

    print("\n'elevation' = how much higher confusable (pose-sharing but different) negatives score")
    print("than genuinely random negatives. Lower/negative is better calibrated;")
    print("large positive elevation = the method is fooled by shared pose/composition.")

    out_path = os.path.join(ROOT, "data", "confusion_calibration_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
