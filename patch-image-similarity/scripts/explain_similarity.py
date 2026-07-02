"""The complete claim: a similarity SCORE from the validated trained vision
projection head (0.890 test accuracy, see train_projection_head.py), paired
with a human-readable EXPLANATION grounded in that same model's own learned
patch correspondences -- not a separate, weaker zero-shot caption-based
score. The language model never touches the number; it only describes
regions the trained vision model already decided correspond.

Pipeline per image pair:
  1. Encode both images with frozen DINOv2 (16x16 = 256 patch grid).
  2. Project patches through the trained head, compute symmetric MaxSim ->
     this *is* the reported similarity score.
  3. Group the 256 fine patches into a 4x4 coarse grid (matching the caption
     grid) and aggregate each coarse cell's average best-match similarity,
     plus which coarse cell in the other image it mostly matched into.
  4. Rank coarse cells by match strength, caption both images once each
     (grid overlay + single structured-JSON call, the real spec recipe),
     and print the strongest-matching region pairs with their captions.
"""
import json
import os
import sys
from collections import Counter

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from encoders import PatchEncoder
from maxsim import symmetric_maxsim
from nim_caption_regions import NimRegionCaptioner

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_DIR = os.path.join(ROOT, "data", os.environ.get("SOP_SPLIT_NAME", "sop_split"))
HEAD_PATH = os.path.join(SPLIT_DIR, "projection_head.pt")
CAPTION_CACHE = os.path.join(ROOT, "data", "explain_captions_cache.json")
VISION_GRID = 16   # DINOv2 patch grid (224 / patch_size 14)
CAPTION_GRID = 4    # coarse region grid for captions
CELLS_PER_SIDE = VISION_GRID // CAPTION_GRID  # 4x4 fine patches per coarse cell
TOP_K_REGIONS = 6


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        return torch.nn.functional.normalize(self.proj(x), dim=-1)


def load_head():
    ckpt = torch.load(HEAD_PATH)
    head = ProjectionHead(ckpt["in_dim"], ckpt["out_dim"])
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    return head


def coarse_cell(patch_index):
    row, col = divmod(patch_index, VISION_GRID)
    return row // CELLS_PER_SIDE, col // CELLS_PER_SIDE


def explain(image_a, image_b, encoder, head, captioner):
    with torch.no_grad():
        enc_a, enc_b = encoder.encode(image_a), encoder.encode(image_b)
        pa, pb = head(enc_a.patch_embeddings), head(enc_b.patch_embeddings)
        result = symmetric_maxsim(pa, pb)

    # aggregate fine-patch matches up to the coarse caption grid
    coarse_matches = {}  # (r,c) in A -> list of (similarity, target coarse cell in B)
    for i in range(VISION_GRID * VISION_GRID):
        cell_a = coarse_cell(i)
        j = result.best_match_a_to_b[i].item()
        sim = result.similarity_matrix[i, j].item()
        cell_b = coarse_cell(j)
        coarse_matches.setdefault(cell_a, []).append((sim, cell_b))

    region_summary = []
    for cell_a, matches in coarse_matches.items():
        avg_sim = sum(s for s, _ in matches) / len(matches)
        target_cell_b = Counter(c for _, c in matches).most_common(1)[0][0]
        region_summary.append((avg_sim, cell_a, target_cell_b))
    region_summary.sort(reverse=True)

    regions_a = {(r.row, r.col): r.description for r in captioner.caption_regions(image_a, grid_size=CAPTION_GRID)}
    regions_b = {(r.row, r.col): r.description for r in captioner.caption_regions(image_b, grid_size=CAPTION_GRID)}

    print(f"\n=== {os.path.basename(image_a)}  <->  {os.path.basename(image_b)} ===")
    print(f"Similarity score (trained vision model): {result.similarity:.3f}")
    print(f"Strongest matching regions (grounded in the model's own patch correspondences):")
    for avg_sim, cell_a, cell_b in region_summary[:TOP_K_REGIONS]:
        desc_a = regions_a.get(cell_a, "?")
        desc_b = regions_b.get(cell_b, "?")
        print(f"  A{cell_a} '{desc_a}'  <->  B{cell_b} '{desc_b}'   (patch match strength={avg_sim:.3f})")

    return result.similarity, region_summary


def main():
    encoder = PatchEncoder()
    head = load_head()
    captioner = NimRegionCaptioner(cache_path=CAPTION_CACHE)

    with open(os.path.join(SPLIT_DIR, "test_triplets.json")) as f:
        test_triplets = json.load(f)

    # demo on a few held-out test triplets: one positive pair, one hard-negative pair
    for t in test_triplets[:3]:
        explain(t["anchor"], t["positive"], encoder, head, captioner)
        explain(t["anchor"], t["negative"], encoder, head, captioner)


if __name__ == "__main__":
    main()
