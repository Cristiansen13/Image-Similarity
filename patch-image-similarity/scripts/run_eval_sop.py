"""Re-run the same scorer comparison (global / Phase0 vision MaxSim / Phase1
caption MaxSim, single-grid and the 3+4 ensemble found to work best) on real
Stanford Online Products photos instead of the skimage toy set, to see
whether the earlier findings replicate at real-world scale and noise.

All 40 SOP triplets use hard (same-category, different-product) negatives,
which is a harder test than most of the skimage triplets.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from encoders import PatchEncoder
from nim_caption_regions import NimRegionCaptioner
from text_embed import TextEmbedder
from maxsim import symmetric_maxsim, global_cosine
from eval import triplet_accuracy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRIPLETS_PATH = os.path.join(ROOT, "data", "sop_triplets.json")
CAPTION_CACHE = os.path.join(ROOT, "data", "sop_subset", "region_captions_cache_nim.json")
GRID_SIZES = (3, 4)


def main():
    with open(TRIPLETS_PATH) as f:
        triplets = json.load(f)
    unique_images = sorted({t[k] for t in triplets for k in ("anchor", "positive", "negative")})
    print(f"{len(triplets)} triplets, {len(unique_images)} unique images")

    patch_encoder = PatchEncoder()
    captioner = NimRegionCaptioner(cache_path=CAPTION_CACHE)
    text_embedder = TextEmbedder()

    patch_cache = {}
    print("Encoding patch + global embeddings...")
    for path in unique_images:
        patch_cache[path] = patch_encoder.encode(path)

    region_emb = {g: {} for g in GRID_SIZES}
    for grid_size in GRID_SIZES:
        print(f"Captioning regions at grid_size={grid_size}...")
        for i, path in enumerate(unique_images):
            regions = captioner.caption_regions(path, grid_size=grid_size)
            region_emb[grid_size][path] = text_embedder.embed([r.description for r in regions])
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(unique_images)}")

    def global_score(a, b):
        return global_cosine(patch_cache[a].global_embedding, patch_cache[b].global_embedding)

    def phase0_score(a, b):
        return symmetric_maxsim(patch_cache[a].patch_embeddings, patch_cache[b].patch_embeddings).similarity

    def grid_score(g, a, b):
        return symmetric_maxsim(region_emb[g][a], region_emb[g][b]).similarity

    def combo_score(a, b):
        return sum(grid_score(g, a, b) for g in GRID_SIZES) / len(GRID_SIZES)

    scorers = {
        "global": global_score,
        "phase0_maxsim": phase0_score,
        "phase1_grid3": lambda a, b: grid_score(3, a, b),
        "phase1_grid4": lambda a, b: grid_score(4, a, b),
        "phase1_combo_3+4": combo_score,
    }

    print()
    results = {name: triplet_accuracy(triplets, fn) for name, fn in scorers.items()}

    print(f"{'scorer':20s}{'accuracy':>12s}")
    print("-" * 32)
    for name, result in results.items():
        print(f"{name:20s}{result.accuracy_overall:12.3f}")

    out_path = os.path.join(ROOT, "data", "sop_eval_results.json")
    with open(out_path, "w") as f:
        json.dump({
            name: {
                "accuracy_overall": r.accuracy_overall,
                "failures": r.failures,
            } for name, r in results.items()
        }, f, indent=2)
    print(f"\nWrote results to {out_path}")


if __name__ == "__main__":
    main()
