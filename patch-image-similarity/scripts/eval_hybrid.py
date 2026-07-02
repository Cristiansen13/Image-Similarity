"""Quantitative test of the hybrid vision+caption disambiguation idea.

Scorers compared, on two eval sets at once:
  - CLIP global cosine (baseline)
  - DINOv2 patch MaxSim, zero-shot (our vision base)
  - caption-only MaxSim (NIM grid-3 region captions -> MiniLM embeddings)
  - hybrid variants combining vision V and caption C per pair:
      geometric  sqrt(V*C)      -- penalizes disagreement symmetrically
      arithmetic 0.5(V+C)
      min        min(V, C)      -- hardest gate
      gated      V * sigmoid(8*(C-0.35)) -- vision score, suppressed when
                 captions clearly disagree (fixed gate, no tuning)

Eval sets:
  1. COCO confusion triplets (33): anchor vs augmented-self vs structurally-
     similar-semantically-different image. The hybrid must HELP here.
  2. SOP hard triplets (40, captions already cached): fine-grained product
     discrimination. The hybrid must NOT HURT here (captions were already
     shown to be weak at this).

All caption work uses grid_size=3 (9 regions/image).
"""
import json
import math
import os
import sys

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from encoders import PatchEncoder
from maxsim import symmetric_maxsim, global_cosine
from nim_caption_regions import NimRegionCaptioner
from text_embed import TextEmbedder
from eval import triplet_accuracy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COCO_TRIPLETS = os.path.join(ROOT, "data", "coco_confusion_triplets.json")
SOP_TRIPLETS = os.path.join(ROOT, "data", "sop_hard_triplets.json")
COCO_CAPTION_CACHE = os.path.join(ROOT, "data", "coco_confusion_captions_cache.json")
SOP_CAPTION_CACHE = os.path.join(ROOT, "data", "sop_hard_subset", "region_captions_cache_nim.json")
GRID = 3


def build_scorers(paths, captioner, cache_label):
    encoder = PatchEncoder()
    text_embedder = TextEmbedder()
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()

    patch, clip_emb, cap_emb = {}, {}, {}
    print(f"Encoding {len(paths)} images (DINOv2 + CLIP + captions) for {cache_label}...")
    for i, p in enumerate(paths):
        patch[p] = encoder.encode(p).patch_embeddings
        image = Image.open(p).convert("RGB")
        with torch.no_grad():
            feat = clip_model.get_image_features(**clip_proc(images=image, return_tensors="pt"))
        clip_emb[p] = torch.nn.functional.normalize(feat[0], dim=0)
        regions = captioner.caption_regions(p, grid_size=GRID)
        cap_emb[p] = text_embedder.embed([r.description for r in regions])
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(paths)}")

    def V(a, b):
        return symmetric_maxsim(patch[a], patch[b]).similarity

    def C(a, b):
        return symmetric_maxsim(cap_emb[a], cap_emb[b]).similarity

    scorers = {
        "clip_global": lambda a, b: global_cosine(clip_emb[a], clip_emb[b]),
        "vision_maxsim": V,
        "caption_maxsim": C,
        "hybrid_geometric": lambda a, b: math.sqrt(max(V(a, b), 0) * max(C(a, b), 0)),
        "hybrid_arithmetic": lambda a, b: 0.5 * (V(a, b) + C(a, b)),
        "hybrid_min": lambda a, b: min(V(a, b), C(a, b)),
        "hybrid_gated": lambda a, b: V(a, b) / (1 + math.exp(-8 * (C(a, b) - 0.35))),
    }
    return scorers


def run_eval(name, triplets_path, caption_cache):
    with open(triplets_path) as f:
        triplets = json.load(f)
    paths = sorted({t[k] for t in triplets for k in ("anchor", "positive", "negative")})
    captioner = NimRegionCaptioner(cache_path=caption_cache)
    scorers = build_scorers(paths, captioner, name)

    print(f"\n=== {name} ({len(triplets)} triplets) ===")
    results = {}
    for scorer_name, fn in scorers.items():
        acc = triplet_accuracy(triplets, fn).accuracy_overall
        results[scorer_name] = acc
        print(f"  {scorer_name:20s} {acc:.3f}")
    return results


def main():
    all_results = {
        "coco_confusion": run_eval("COCO confusion (hybrid must HELP)", COCO_TRIPLETS, COCO_CAPTION_CACHE),
        "sop_hard": run_eval("SOP hard (hybrid must NOT HURT)", SOP_TRIPLETS, SOP_CAPTION_CACHE),
    }
    out_path = os.path.join(ROOT, "data", "hybrid_eval_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
