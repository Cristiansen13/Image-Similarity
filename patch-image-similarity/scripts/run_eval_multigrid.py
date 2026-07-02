"""Multi-grid ensemble: instead of a single caption grid resolution, compute
mean-of-max MaxSim at several grid sizes (3x3, 4x4, 6x6) and combine (mean)
into one Phase-1 score, on the theory that averaging across resolutions
smooths out any one resolution's generic-caption noise.

Grid 4 and 6 captions are already cached (no new calls); grid 3 is new.
"""
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from nim_caption_regions import NimRegionCaptioner
from text_embed import TextEmbedder
from maxsim import symmetric_maxsim
from eval import triplet_accuracy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRIPLETS_PATH = os.path.join(ROOT, "data", "triplets.json")
CAPTION_CACHE = os.path.join(ROOT, "data", "test_images", "region_captions_cache_nim.json")
GRID_SIZES = (3, 4, 6)


def main():
    with open(TRIPLETS_PATH) as f:
        triplets = json.load(f)
    unique_images = sorted({t[k] for t in triplets for k in ("anchor", "positive", "negative")})

    captioner = NimRegionCaptioner(cache_path=CAPTION_CACHE)
    text_embedder = TextEmbedder()

    region_emb = {g: {} for g in GRID_SIZES}
    for grid_size in GRID_SIZES:
        print(f"Captioning at grid_size={grid_size} (cache hit unless new)...")
        for i, path in enumerate(unique_images):
            regions = captioner.caption_regions(path, grid_size=grid_size)
            region_emb[grid_size][path] = text_embedder.embed([r.description for r in regions])
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(unique_images)}")

    per_grid_score = {
        g: {} for g in GRID_SIZES
    }
    # cache per-grid pairwise scores lazily via closures below

    def grid_score(g, a, b):
        return symmetric_maxsim(region_emb[g][a], region_emb[g][b]).similarity

    def make_single_scorer(g):
        return lambda a, b: grid_score(g, a, b)

    def make_combo_scorer(grids):
        def score(a, b):
            return sum(grid_score(g, a, b) for g in grids) / len(grids)
        return score

    scorers = {f"grid{g}": make_single_scorer(g) for g in GRID_SIZES}
    for r in (2, 3):
        for combo in itertools.combinations(GRID_SIZES, r):
            name = "combo_" + "+".join(str(g) for g in combo)
            scorers[name] = make_combo_scorer(combo)

    print()
    results = {name: triplet_accuracy(triplets, fn) for name, fn in scorers.items()}

    types = sorted({t["type"] for t in triplets})
    header = f"{'scorer':16s}" + "".join(f"{t:>12s}" for t in types) + f"{'overall':>12s}"
    print(header)
    print("-" * len(header))
    for name, result in results.items():
        row = f"{name:16s}"
        for t in types:
            row += f"{result.accuracy_by_type.get(t, float('nan')):12.3f}"
        row += f"{result.accuracy_overall:12.3f}"
        print(row)


if __name__ == "__main__":
    main()
