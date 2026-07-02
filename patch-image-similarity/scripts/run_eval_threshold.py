"""Side-check experiment: does threshold/hit-rate aggregation (fraction of
patches whose best match clears a cosine threshold) fix the grid=6 accuracy
collapse that plain mean-of-max MaxSim showed?

Reuses already-cached region captions (region_captions_cache_nim.json) -- no
new NIM API calls, this is purely a local rescoring experiment.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from nim_caption_regions import NimRegionCaptioner
from text_embed import TextEmbedder
from maxsim import symmetric_maxsim, threshold_hit_rate
from eval import triplet_accuracy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRIPLETS_PATH = os.path.join(ROOT, "data", "triplets.json")
CAPTION_CACHE = os.path.join(ROOT, "data", "test_images", "region_captions_cache_nim.json")
GRID_SIZES = (4, 6)
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def main():
    with open(TRIPLETS_PATH) as f:
        triplets = json.load(f)
    unique_images = sorted({t[k] for t in triplets for k in ("anchor", "positive", "negative")})

    captioner = NimRegionCaptioner(cache_path=CAPTION_CACHE)  # cache hit for all -- no API calls
    text_embedder = TextEmbedder()

    region_emb = {g: {} for g in GRID_SIZES}
    for grid_size in GRID_SIZES:
        for path in unique_images:
            regions = captioner.caption_regions(path, grid_size=grid_size)
            region_emb[grid_size][path] = text_embedder.embed([r.description for r in regions])

    def make_meanmax_score(grid_size):
        def score(a, b):
            return symmetric_maxsim(region_emb[grid_size][a], region_emb[grid_size][b]).similarity
        return score

    def make_threshold_score(grid_size, threshold):
        def score(a, b):
            return threshold_hit_rate(region_emb[grid_size][a], region_emb[grid_size][b], threshold)
        return score

    scorers = {}
    for g in GRID_SIZES:
        scorers[f"meanmax_g{g}"] = make_meanmax_score(g)
        for t in THRESHOLDS:
            scorers[f"threshold_g{g}_t{t}"] = make_threshold_score(g, t)

    results = {name: triplet_accuracy(triplets, fn) for name, fn in scorers.items()}

    types = sorted({t["type"] for t in triplets})
    header = f"{'scorer':22s}" + "".join(f"{t:>12s}" for t in types) + f"{'overall':>12s}"
    print(header)
    print("-" * len(header))
    for name, result in results.items():
        row = f"{name:22s}"
        for t in types:
            row += f"{result.accuracy_by_type.get(t, float('nan')):12.3f}"
        row += f"{result.accuracy_overall:12.3f}"
        print(row)


if __name__ == "__main__":
    main()
