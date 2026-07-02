"""Proper Recall@1 (retrieval, not triplet comparison) on the SOP test
split, for a genuinely comparable number against published literature.

Literature (Proxy-Anchor paper, Table 3, SOP full test set: 60,502 images,
11,316 classes, ~5.3 photos/class, fully fine-tuned backbones):
  Clustering 67.0, Proxy-NCA 73.7, Margin 72.7, MS 78.2, SoftTriple 78.3,
  Proxy-Anchor 79.1-80.3 (all Recall@1, %).

IMPORTANT CAVEAT this script's numbers are NOT apples-to-apples with those:
our gallery here is our own held-out test split (300 classes, exactly 2
photos/class = 600 images) -- three orders of magnitude smaller than SOP's
full 60,502-image test set, with far fewer distractor classes. A smaller
gallery is a strictly easier retrieval problem. This number contextualizes
our method against the literature's scale of "what good looks like," not a
head-to-head claim of beating fully fine-tuned, full-gallery SOTA methods.
"""
import json
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from maxsim import symmetric_maxsim, global_cosine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        return torch.nn.functional.normalize(self.proj(x), dim=-1)


def build_gallery(test_triplets):
    """Clean, fully-labeled gallery: anchor+positive photos only (exactly
    2 known-class photos per class) -- ignores the mined 'negative' images
    since we don't have reliable secondary-instance labels for them."""
    gallery = {}  # path -> class_id
    for t in test_triplets:
        gallery[t["anchor"]] = t["class_id"]
        gallery[t["positive"]] = t["class_id"]
    return gallery


def recall_at_k(gallery, score_fn, k_values=(1, 5)):
    paths = list(gallery.keys())
    hits = {k: 0 for k in k_values}
    for query in paths:
        scored = sorted(
            ((score_fn(query, cand), cand) for cand in paths if cand != query),
            key=lambda x: -x[0],
        )
        ranked_classes = [gallery[p] for _, p in scored]
        for k in k_values:
            if gallery[query] in ranked_classes[:k]:
                hits[k] += 1
    return {k: hits[k] / len(paths) for k in k_values}


def main(split_dir_name="sop_split2"):
    split_dir = os.path.join(ROOT, "data", split_dir_name)
    with open(os.path.join(split_dir, "test_triplets.json")) as f:
        test_triplets = json.load(f)
    raw_embeddings = torch.load(os.path.join(split_dir, "patch_embeddings.pt"))
    gallery = build_gallery(test_triplets)
    print(f"Gallery: {len(gallery)} images, {len(set(gallery.values()))} classes "
          f"(2 photos/class) -- SOP's full test set is 60,502 images / 11,316 classes")

    def zs_meanmax(a, b):
        return symmetric_maxsim(raw_embeddings[a], raw_embeddings[b]).similarity

    def zs_global(a, b):
        # mean-pool patches as a cheap global proxy (consistent with maxsim.py's normalized patches)
        ga = torch.nn.functional.normalize(raw_embeddings[a].mean(dim=0), dim=0)
        gb = torch.nn.functional.normalize(raw_embeddings[b].mean(dim=0), dim=0)
        return global_cosine(ga, gb)

    ckpt = torch.load(os.path.join(split_dir, "projection_head.pt"))
    head = ProjectionHead(ckpt["in_dim"], ckpt["out_dim"])
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    with torch.no_grad():
        projected = {p: head(raw_embeddings[p]) for p in gallery}

    def trained_meanmax(a, b):
        return symmetric_maxsim(projected[a], projected[b]).similarity

    print("\nComputing Recall@K (this is O(n^2) over the gallery, may take a bit)...")
    results = {}
    for name, fn in [("zero_shot_global", zs_global), ("zero_shot_meanmax", zs_meanmax),
                      ("trained_meanmax", trained_meanmax)]:
        r = recall_at_k(gallery, fn)
        results[name] = r
        print(f"{name:20s}  R@1={r[1]:.3f}  R@5={r[5]:.3f}")

    print("\nFor reference, published SOP full-test-set Recall@1 (fully fine-tuned backbones, "
          "60,502-image gallery): Clustering 67.0%, Proxy-NCA 73.7%, Margin 72.7%, MS 78.2%, "
          "SoftTriple 78.3%, Proxy-Anchor 79.1-80.3%")
    print("Our numbers use a ~600-image gallery (300 classes x 2 photos) -- NOT directly comparable "
          "in difficulty, shown for scale/context only.")

    out_path = os.path.join(split_dir, "recall_at_k_results.json")
    with open(out_path, "w") as f:
        json.dump({"gallery_size": len(gallery), "n_classes": len(set(gallery.values())),
                   "results": results}, f, indent=2)
    print(f"\nWrote results to {out_path}")


if __name__ == "__main__":
    main()
