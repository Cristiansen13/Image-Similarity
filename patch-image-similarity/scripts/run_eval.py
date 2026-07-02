"""Run triplet accuracy (spec section 4) for: global embedding baseline,
Phase 0 MaxSim (DINOv2 patches), and Phase 1 MaxSim (NIM region captions) at
two grid resolutions (4x4 and 6x6), to see whether a finer region grid closes
the accuracy gap seen in the 11-pair sanity check.
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
TRIPLETS_PATH = os.path.join(ROOT, "data", "triplets.json")
CAPTION_CACHE = os.path.join(ROOT, "data", "test_images", "region_captions_cache_nim.json")
GRID_SIZES = (4, 6)


def main():
    with open(TRIPLETS_PATH) as f:
        triplets = json.load(f)

    unique_images = sorted({t[k] for t in triplets for k in ("anchor", "positive", "negative")})
    print(f"{len(triplets)} triplets, {len(unique_images)} unique images")

    patch_encoder = PatchEncoder()
    captioner = NimRegionCaptioner(cache_path=CAPTION_CACHE)
    text_embedder = TextEmbedder()

    patch_cache = {}
    region_emb_cache = {g: {} for g in GRID_SIZES}

    print("Encoding patch + global embeddings...")
    for path in unique_images:
        patch_cache[path] = patch_encoder.encode(path)

    for grid_size in GRID_SIZES:
        print(f"Captioning regions at grid_size={grid_size}...")
        for i, path in enumerate(unique_images):
            regions = captioner.caption_regions(path, grid_size=grid_size)
            descriptions = [r.description for r in regions]
            region_emb_cache[grid_size][path] = text_embedder.embed(descriptions)
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(unique_images)}")

    def global_score(a, b):
        return global_cosine(patch_cache[a].global_embedding, patch_cache[b].global_embedding)

    def phase0_score(a, b):
        return symmetric_maxsim(patch_cache[a].patch_embeddings, patch_cache[b].patch_embeddings).similarity

    def make_phase1_score(grid_size):
        def score(a, b):
            return symmetric_maxsim(region_emb_cache[grid_size][a], region_emb_cache[grid_size][b]).similarity
        return score

    scorers = {
        "global": global_score,
        "phase0_maxsim": phase0_score,
        **{f"phase1_maxsim_g{g}": make_phase1_score(g) for g in GRID_SIZES},
    }

    print()
    results = {name: triplet_accuracy(triplets, fn) for name, fn in scorers.items()}

    types = sorted({t["type"] for t in triplets})
    header = f"{'scorer':20s}" + "".join(f"{t:>12s}" for t in types) + f"{'overall':>12s}"
    print(header)
    print("-" * len(header))
    for name, result in results.items():
        row = f"{name:20s}"
        for t in types:
            row += f"{result.accuracy_by_type.get(t, float('nan')):12.3f}"
        row += f"{result.accuracy_overall:12.3f}"
        print(row)

    n_by_type = results["global"].n_by_type
    print(f"\nn per type: {n_by_type}")

    out_path = os.path.join(ROOT, "data", "eval_results.json")
    with open(out_path, "w") as f:
        json.dump({
            name: {
                "accuracy_by_type": r.accuracy_by_type,
                "accuracy_overall": r.accuracy_overall,
                "n_by_type": r.n_by_type,
                "failures": r.failures,
            } for name, r in results.items()
        }, f, indent=2)
    print(f"\nWrote results to {out_path}")


if __name__ == "__main__":
    main()
