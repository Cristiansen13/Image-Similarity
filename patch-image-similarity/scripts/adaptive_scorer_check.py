"""End-to-end check of the complexity-adaptive scoring idea: does blending
global-cosine and patch-MaxSim, weighted by a per-image "patch diversity"
complexity signal, hold up on BOTH the simple-image benchmark (CUB) and the
complex-scene benchmark (COCO structural confusion) at once?

Weight function: w_patch = sigmoid(-k * (diversity - midpoint)), so a
HOMOGENEOUS (simple) query image gets LOW w_patch (trust global), and a
DIVERSE (complex) query image gets HIGH w_patch (trust patch). midpoint/k are
set from the CUB-vs-COCO separation already measured in check_patch_diversity.py
(CUB mean ~0.296, COCO mean ~0.220 -> midpoint ~0.258).

Global and patch scores are z-scored against the CURRENT batch's own
confusable/random or positive/negative score distribution before blending --
raw patch-MaxSim scores run roughly 2x the magnitude of raw global-cosine
scores in zero-shot DINOv2, so blending raw values would let whichever has
bigger numbers dominate regardless of the intended weight.

Success criteria:
  - CUB triplet accuracy: adaptive should stay near the global-only ceiling
    (not regress toward the weaker zero-shot patch number).
  - COCO confusion elevation: adaptive should stay near the patch-only
    calibration (not regress toward the worse zero-shot global number).

Usage: python adaptive_scorer_check.py --cub-dir /path/CUB_200_2011 --n-cub-triplets 150
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from encoders import PatchEncoder
from maxsim import topk_symmetric_maxsim, global_cosine

PATCH_TOPK = 4  # validated win over plain mean-of-max, see check_topk_maxsim.py

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COCO_TRIPLETS_PATH = os.path.join(ROOT, "data", "coco_confusion_triplets.json")

# from check_patch_diversity.py's measured CUB (~0.296) vs COCO (~0.220) separation
MIDPOINT = 0.258
STEEPNESS = 30.0


def mean_patch_similarity(patch_embeddings):
    sim = patch_embeddings @ patch_embeddings.T
    n = sim.shape[0]
    return ((sim.sum() - torch.diagonal(sim).sum()) / (n * (n - 1))).item()


def patch_weight(diversity):
    return 1.0 / (1.0 + np.exp(-(-STEEPNESS * (diversity - MIDPOINT))))


def zscore(values):
    arr = np.array(values)
    return (arr - arr.mean()) / (arr.std() + 1e-8)


def standardized_elevation(confusable, random_):
    """Effect size (elevation / pooled std), comparable across scorers
    regardless of each scorer's native scale -- unlike raw mean-difference
    elevation, which isn't comparable once one side has been z-scored."""
    c, r = np.array(confusable), np.array(random_)
    pooled_std = np.sqrt((c.var() + r.var()) / 2) + 1e-8
    return (c.mean() - r.mean()) / pooled_std


def load_cub_index(cub_dir):
    from collections import defaultdict
    by_class = defaultdict(list)
    image_paths = {}
    with open(os.path.join(cub_dir, "images.txt")) as f:
        for line in f:
            image_id, path = line.split()
            image_paths[image_id] = path
    with open(os.path.join(cub_dir, "image_class_labels.txt")) as f:
        for line in f:
            image_id, class_id = line.split()
            by_class[class_id].append(image_paths[image_id])
    return by_class


def build_cub_triplets(cub_dir, n, seed):
    rng = random.Random(seed)
    by_class = load_cub_index(cub_dir)
    test_classes = [str(i) for i in range(101, 201)]
    images_root = os.path.join(cub_dir, "images")
    triplets = []
    for _ in range(n):
        pos_class = rng.choice(test_classes)
        neg_class = rng.choice([c for c in test_classes if c != pos_class])
        if len(by_class[pos_class]) < 2:
            continue
        anchor, positive = rng.sample(by_class[pos_class], 2)
        negative = rng.choice(by_class[neg_class])
        triplets.append({
            "anchor": os.path.join(images_root, anchor),
            "positive": os.path.join(images_root, positive),
            "negative": os.path.join(images_root, negative),
        })
    return triplets


def encode_cache(encoder, paths):
    cache = {}
    for i, p in enumerate(paths):
        cache[p] = encoder.encode(p)
        if (i + 1) % 50 == 0:
            print(f"  encoded {i + 1}/{len(paths)}")
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cub-dir", required=True)
    ap.add_argument("--n-cub-triplets", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    encoder = PatchEncoder()

    # ---------- CUB (simple images) triplet accuracy ----------
    print("Building CUB triplets...")
    cub_triplets = build_cub_triplets(args.cub_dir, args.n_cub_triplets, args.seed)
    cub_paths = sorted(set(p for t in cub_triplets for p in (t["anchor"], t["positive"], t["negative"])))
    print(f"Encoding {len(cub_paths)} CUB images...")
    cub_cache = encode_cache(encoder, cub_paths)

    def cub_pair_scores(a, b):
        ea, eb = cub_cache[a], cub_cache[b]
        g = global_cosine(ea.global_embedding, eb.global_embedding)
        v = topk_symmetric_maxsim(ea.patch_embeddings, eb.patch_embeddings, k=PATCH_TOPK).similarity
        div_a = mean_patch_similarity(ea.patch_embeddings)
        return g, v, div_a

    global_pos, global_neg, patch_pos, patch_neg, div_all = [], [], [], [], []
    for t in cub_triplets:
        g_p, v_p, div_a = cub_pair_scores(t["anchor"], t["positive"])
        g_n, v_n, _ = cub_pair_scores(t["anchor"], t["negative"])
        global_pos.append(g_p); global_neg.append(g_n)
        patch_pos.append(v_p); patch_neg.append(v_n)
        div_all.append(div_a)

    def acc_from_raw(pos, neg):
        return float(np.mean(np.array(pos) > np.array(neg)))

    # z-score patch/global jointly (pos+neg combined) so the blend weighting is fair
    g_all = zscore(global_pos + global_neg)
    v_all = zscore(patch_pos + patch_neg)
    n = len(cub_triplets)
    g_pos_z, g_neg_z = g_all[:n], g_all[n:]
    v_pos_z, v_neg_z = v_all[:n], v_all[n:]
    w = np.array([patch_weight(d) for d in div_all])
    adaptive_pos = w * v_pos_z + (1 - w) * g_pos_z
    adaptive_neg = w * v_neg_z + (1 - w) * g_neg_z

    print("\n=== CUB (simple images) triplet accuracy ===")
    print(f"  global-only:   {acc_from_raw(global_pos, global_neg):.4f}")
    print(f"  patch-only:    {acc_from_raw(patch_pos, patch_neg):.4f}")
    print(f"  adaptive:      {acc_from_raw(adaptive_pos, adaptive_neg):.4f}")
    print(f"  mean patch_weight used: {w.mean():.3f} (0=all global, 1=all patch)")

    # ---------- COCO confusion (complex scenes) elevation ----------
    print("\nLoading COCO confusion triplets...")
    with open(COCO_TRIPLETS_PATH) as f:
        coco_triplets = json.load(f)
    anchors = [t["anchor"] for t in coco_triplets]
    positives = [t["positive"] for t in coco_triplets]
    confusable_negs = [t["negative"] for t in coco_triplets]
    rng = random.Random(args.seed)
    shuffled = confusable_negs[:]
    rng.shuffle(shuffled)
    for i in range(len(shuffled)):
        if shuffled[i] == confusable_negs[i]:
            shuffled[i], shuffled[(i + 1) % len(shuffled)] = shuffled[(i + 1) % len(shuffled)], shuffled[i]
    random_negs = shuffled

    all_coco_paths = sorted(set(anchors) | set(positives) | set(confusable_negs))
    print(f"Encoding {len(all_coco_paths)} COCO images...")
    coco_cache = encode_cache(encoder, all_coco_paths)

    def coco_pair_scores(a, b):
        ea, eb = coco_cache[a], coco_cache[b]
        g = global_cosine(ea.global_embedding, eb.global_embedding)
        v = topk_symmetric_maxsim(ea.patch_embeddings, eb.patch_embeddings, k=PATCH_TOPK).similarity
        div_a = mean_patch_similarity(ea.patch_embeddings)
        return g, v, div_a

    g_pos, v_pos, div_pos = [], [], []
    g_conf, g_rand, v_conf, v_rand, div_conf, div_rand = [], [], [], [], [], []
    for a, p, cn, rn in zip(anchors, positives, confusable_negs, random_negs):
        g_p, v_p, d_p = coco_pair_scores(a, p)
        g_c, v_c, d_c = coco_pair_scores(a, cn)
        g_r, v_r, d_r = coco_pair_scores(a, rn)
        g_pos.append(g_p); v_pos.append(v_p); div_pos.append(d_p)
        g_conf.append(g_c); g_rand.append(g_r)
        v_conf.append(v_c); v_rand.append(v_r)
        div_conf.append(d_c); div_rand.append(d_r)

    # z-score pos+conf+rand together per scorer so all three land on the same blended scale
    g_all2 = zscore(g_pos + g_conf + g_rand)
    v_all2 = zscore(v_pos + v_conf + v_rand)
    m = len(coco_triplets)
    g_pos_z, g_conf_z, g_rand_z = g_all2[:m], g_all2[m:2 * m], g_all2[2 * m:]
    v_pos_z, v_conf_z, v_rand_z = v_all2[:m], v_all2[m:2 * m], v_all2[2 * m:]
    w_pos = np.array([patch_weight(d) for d in div_pos])
    w_conf = np.array([patch_weight(d) for d in div_conf])
    w_rand = np.array([patch_weight(d) for d in div_rand])
    adaptive_pos = w_pos * v_pos_z + (1 - w_pos) * g_pos_z
    adaptive_conf = w_conf * v_conf_z + (1 - w_conf) * g_conf_z
    adaptive_rand = w_rand * v_rand_z + (1 - w_rand) * g_rand_z

    def pos_vs_conf_acc(pos, conf):
        return float(np.mean(np.array(pos) > np.array(conf)))

    print("\n=== COCO confusion (complex scenes) ===")
    print("  standardized elevation (effect size, lower=better); pos_vs_conf_acc = does the true")
    print("  augmented-positive still beat the confusable negative (guards against saturation)")
    print(f"  global-only:  effect_size={standardized_elevation(g_conf, g_rand):.4f}  "
          f"pos_vs_conf_acc={pos_vs_conf_acc(g_pos, g_conf):.4f}  "
          f"(raw elevation={np.mean(g_conf) - np.mean(g_rand):.4f})")
    print(f"  patch-only:   effect_size={standardized_elevation(v_conf, v_rand):.4f}  "
          f"pos_vs_conf_acc={pos_vs_conf_acc(v_pos, v_conf):.4f}  "
          f"(raw elevation={np.mean(v_conf) - np.mean(v_rand):.4f})")
    print(f"  adaptive:     effect_size={standardized_elevation(adaptive_conf, adaptive_rand):.4f}  "
          f"pos_vs_conf_acc={pos_vs_conf_acc(adaptive_pos, adaptive_conf):.4f}")
    print(f"  mean patch_weight used: {np.concatenate([w_conf, w_rand]).mean():.3f}")


if __name__ == "__main__":
    main()
