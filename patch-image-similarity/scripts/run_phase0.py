"""Phase 0: zero-shot DINOv2 patches + symmetric MaxSim vs global-embedding baseline.

Loads the pairs manifest built by make_test_pairs.py, encodes every unique
image once, computes MaxSim and global cosine similarity for every pair, and
prints both numbers side by side sorted by expected similarity so the ranking
can be eyeballed.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from encoders import PatchEncoder
from maxsim import symmetric_maxsim, global_cosine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIRS_MANIFEST = os.path.join(ROOT, "data", "test_images", "pairs.json")

EXPECTED_ORDER = {"high": 0, "medium-high": 1, "medium": 2, "low": 3}


def main():
    with open(PAIRS_MANIFEST) as f:
        pairs = json.load(f)

    encoder = PatchEncoder()
    cache = {}

    def get(path):
        if path not in cache:
            print(f"  encoding {os.path.basename(path)} ...")
            cache[path] = encoder.encode(path)
        return cache[path]

    print(f"Encoding images for {len(pairs)} pairs...")
    results = []
    for pair in pairs:
        enc_a = get(pair["a"])
        enc_b = get(pair["b"])

        maxsim_result = symmetric_maxsim(enc_a.patch_embeddings, enc_b.patch_embeddings)
        global_sim = global_cosine(enc_a.global_embedding, enc_b.global_embedding)

        results.append({
            "name": pair["name"],
            "category": pair["category"],
            "expected": pair["expected"],
            "maxsim": maxsim_result.similarity,
            "global": global_sim,
        })

    results.sort(key=lambda r: EXPECTED_ORDER.get(r["expected"], 99))

    print()
    header = f"{'pair':45s} {'expected':12s} {'MaxSim':>8s} {'Global':>8s}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['name']:45s} {r['expected']:12s} {r['maxsim']:8.4f} {r['global']:8.4f}")

    print()
    print("Sanity check: within each expected-similarity bucket, do MaxSim and Global")
    print("roughly agree with the expected ranking (high > medium > low)?")

    out_path = os.path.join(ROOT, "data", "test_images", "phase0_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote results to {out_path}")


if __name__ == "__main__":
    main()
