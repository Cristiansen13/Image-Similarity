"""Does our trained method fix the score-uniformity/permissiveness problem
seen in "consecrated" off-the-shelf similarity methods?

Compares CLIP global cosine (the most common off-the-shelf similarity
baseline in the wild), zero-shot DINOv2 (global + patch MaxSim), and our
trained projection head, on the SAME set of pairs: all 300 known positive
pairs from the SOP test gallery, plus a random sample of negative pairs.
Reports distribution spread (not just mean/std) and a histogram, since
"uniformity" is about how scores are spread across the whole 0-1 range, not
just separation between two group means.
"""
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from maxsim import symmetric_maxsim, global_cosine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_DIR = os.path.join(ROOT, "data", "sop_split2")
N_NEGATIVE_SAMPLES = 300
SEED = 0
HIST_BINS = np.arange(-0.1, 1.05, 0.1)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        return torch.nn.functional.normalize(self.proj(x), dim=-1)


def build_gallery(test_triplets):
    gallery = {}
    for t in test_triplets:
        gallery[t["anchor"]] = t["class_id"]
        gallery[t["positive"]] = t["class_id"]
    return gallery


def sample_pairs(gallery, rng):
    paths = list(gallery.keys())
    by_class = {}
    for p, c in gallery.items():
        by_class.setdefault(c, []).append(p)

    positive_pairs = [(v[0], v[1]) for v in by_class.values() if len(v) == 2]

    negative_pairs = []
    while len(negative_pairs) < N_NEGATIVE_SAMPLES:
        a, b = rng.sample(paths, 2)
        if gallery[a] != gallery[b]:
            negative_pairs.append((a, b))
    return positive_pairs, negative_pairs


def report(name, pos_scores, neg_scores):
    pos_a, neg_a = np.array(pos_scores), np.array(neg_scores)
    all_scores = np.concatenate([pos_a, neg_a])
    hist, _ = np.histogram(all_scores, bins=HIST_BINS)
    hist_frac = hist / hist.sum()
    nonzero = hist_frac[hist_frac > 0]
    entropy = -np.sum(nonzero * np.log(nonzero))  # higher = more spread across bins
    max_entropy = np.log(len(HIST_BINS) - 1)

    print(f"\n{name}")
    print(f"  positive pairs: mean={pos_a.mean():.3f} std={pos_a.std():.3f}")
    print(f"  negative pairs: mean={neg_a.mean():.3f} std={neg_a.std():.3f}  "
          f"[p10={np.percentile(neg_a,10):.3f} p50={np.percentile(neg_a,50):.3f} p90={np.percentile(neg_a,90):.3f}]")
    print(f"  gap (pos-neg mean): {pos_a.mean()-neg_a.mean():.3f}")
    print(f"  histogram entropy (uniformity, max={max_entropy:.2f}): {entropy:.3f}")
    bin_labels = [f"{HIST_BINS[i]:.1f}-{HIST_BINS[i+1]:.1f}" for i in range(len(HIST_BINS)-1)]
    print("  histogram: " + " ".join(f"{l}:{c}" for l, c in zip(bin_labels, hist) if c > 0))
    return {"pos_mean": float(pos_a.mean()), "neg_mean": float(neg_a.mean()),
            "gap": float(pos_a.mean() - neg_a.mean()), "entropy": float(entropy),
            "max_entropy": float(max_entropy)}


def main():
    rng = random.Random(SEED)
    with open(os.path.join(SPLIT_DIR, "test_triplets.json")) as f:
        test_triplets = json.load(f)
    gallery = build_gallery(test_triplets)
    positive_pairs, negative_pairs = sample_pairs(gallery, rng)
    print(f"{len(positive_pairs)} positive pairs, {len(negative_pairs)} sampled negative pairs")

    raw_embeddings = torch.load(os.path.join(SPLIT_DIR, "patch_embeddings.pt"))
    ckpt = torch.load(os.path.join(SPLIT_DIR, "projection_head.pt"))
    head = ProjectionHead(ckpt["in_dim"], ckpt["out_dim"])
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    with torch.no_grad():
        projected = {p: head(raw_embeddings[p]) for p in gallery}

    print("Loading CLIP and encoding gallery images...")
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    clip_embeddings = {}
    with torch.no_grad():
        for i, path in enumerate(gallery):
            image = Image.open(path).convert("RGB")
            inputs = clip_proc(images=image, return_tensors="pt")
            feat = clip_model.get_image_features(**inputs)
            clip_embeddings[path] = torch.nn.functional.normalize(feat[0], dim=0)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(gallery)}")

    def scores_for(fn, pairs):
        return [fn(a, b) for a, b in pairs]

    def clip_score(a, b):
        return global_cosine(clip_embeddings[a], clip_embeddings[b])

    def zs_global(a, b):
        ga = torch.nn.functional.normalize(raw_embeddings[a].mean(dim=0), dim=0)
        gb = torch.nn.functional.normalize(raw_embeddings[b].mean(dim=0), dim=0)
        return global_cosine(ga, gb)

    def zs_meanmax(a, b):
        return symmetric_maxsim(raw_embeddings[a], raw_embeddings[b]).similarity

    def trained_meanmax(a, b):
        return symmetric_maxsim(projected[a], projected[b]).similarity

    results = {}
    for name, fn in [("CLIP (consecrated baseline)", clip_score),
                      ("zero-shot DINOv2 global", zs_global),
                      ("zero-shot DINOv2 meanmax", zs_meanmax),
                      ("trained meanmax", trained_meanmax)]:
        pos = scores_for(fn, positive_pairs)
        neg = scores_for(fn, negative_pairs)
        results[name] = report(name, pos, neg)

    out_path = os.path.join(SPLIT_DIR, "uniformity_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote results to {out_path}")


if __name__ == "__main__":
    main()
