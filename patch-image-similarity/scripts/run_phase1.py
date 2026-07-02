"""Phase 1: LLM-generated region descriptions (via NVIDIA NIM, see
src/nim_caption_regions.py -- grid overlay + single structured-JSON call per
image, the spec's actual recipe) + symmetric MaxSim over their text
embeddings, compared against Phase 0's vision-patch MaxSim and the
global-embedding baseline.

Prints all three numbers side by side per pair, plus a readable match trace
(which region in A matched which region in B, with captions) for a few pairs.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from encoders import PatchEncoder
from nim_caption_regions import NimRegionCaptioner
from text_embed import TextEmbedder
from maxsim import symmetric_maxsim, global_cosine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIRS_MANIFEST = os.path.join(ROOT, "data", "test_images", "pairs.json")
CAPTION_CACHE = os.path.join(ROOT, "data", "test_images", "region_captions_cache_nim.json")
GRID_SIZE = 4

EXPECTED_ORDER = {"high": 0, "medium-high": 1, "medium": 2, "low": 3}

TRACE_PAIRS = {"same_scene_shifted_motorcycle", "same_object_diff_background_rocket_grass"}


def main():
    with open(PAIRS_MANIFEST) as f:
        pairs = json.load(f)

    patch_encoder = PatchEncoder()
    captioner = NimRegionCaptioner(cache_path=CAPTION_CACHE)
    text_embedder = TextEmbedder()

    patch_cache = {}
    region_cache = {}

    def get_patch(path):
        if path not in patch_cache:
            patch_cache[path] = patch_encoder.encode(path)
        return patch_cache[path]

    def get_regions(path):
        if path not in region_cache:
            print(f"  captioning regions for {os.path.basename(path)} ...")
            regions = captioner.caption_regions(path, grid_size=GRID_SIZE)
            descriptions = [r.description for r in regions]
            embeddings = text_embedder.embed(descriptions)
            region_cache[path] = (regions, embeddings)
        return region_cache[path]

    print(f"Processing {len(pairs)} pairs...")
    results = []
    traces = []
    for pair in pairs:
        enc_a = get_patch(pair["a"])
        enc_b = get_patch(pair["b"])
        phase0_maxsim = symmetric_maxsim(enc_a.patch_embeddings, enc_b.patch_embeddings).similarity
        global_sim = global_cosine(enc_a.global_embedding, enc_b.global_embedding)

        regions_a, emb_a = get_regions(pair["a"])
        regions_b, emb_b = get_regions(pair["b"])
        phase1_result = symmetric_maxsim(emb_a, emb_b)

        results.append({
            "name": pair["name"],
            "category": pair["category"],
            "expected": pair["expected"],
            "global": global_sim,
            "phase0_maxsim": phase0_maxsim,
            "phase1_maxsim": phase1_result.similarity,
        })

        if pair["name"] in TRACE_PAIRS:
            trace_lines = [f"\n--- match trace: {pair['name']} ---"]
            for i, region in enumerate(regions_a):
                j = phase1_result.best_match_a_to_b[i].item()
                sim = phase1_result.similarity_matrix[i, j].item()
                trace_lines.append(
                    f"  A[{region.row},{region.col}] '{region.description}' "
                    f"<-> B[{regions_b[j].row},{regions_b[j].col}] '{regions_b[j].description}' "
                    f"(sim={sim:.3f})"
                )
            traces.append("\n".join(trace_lines))

    results.sort(key=lambda r: EXPECTED_ORDER.get(r["expected"], 99))

    print()
    header = f"{'pair':45s} {'expected':12s} {'Global':>8s} {'Phase0':>8s} {'Phase1':>8s}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['name']:45s} {r['expected']:12s} {r['global']:8.4f} "
              f"{r['phase0_maxsim']:8.4f} {r['phase1_maxsim']:8.4f}")

    for trace in traces:
        print(trace)

    out_path = os.path.join(ROOT, "data", "test_images", "phase1_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote results to {out_path}")


if __name__ == "__main__":
    main()
