"""Two-stage retrieve-and-rerank evaluation (ColBERT/ColPali-style late interaction):

  Stage 1 (fast): mean-pooled global embedding (cheap, single matmul per query
                   against the whole gallery) finds the top-K candidates.
  Stage 2 (exact): full symmetric patch-level MaxSim reranks ONLY those K
                   candidates to pick the final top-1.

Reuses the patch-embedding cache already written by eval_full_test.py /
eval_cub_test.py (all_embeddings_cache.pt / cub_embeddings_cache.pt) -- no
re-encoding needed, this is a pure post-hoc reranking evaluation.

Reports both the achieved Recall@1 (to compare against the brute-force
full_test_recall.json baseline from the SAME checkpoint) and a stage-1
"ceiling" recall (fraction of queries where a same-class candidate exists
anywhere in the top-K) so a low final score can be attributed to stage 1
missing the candidate vs. stage 2 picking the wrong one.

Usage:
  python two_stage_retrieval.py --dataset sop --labels-path Ebay_test.txt \
      --embeddings-cache checkpoints/all_embeddings_cache.pt --top-k 100
  python two_stage_retrieval.py --dataset cub --labels-path /path/CUB_200_2011 \
      --embeddings-cache checkpoints_cub/cub_embeddings_cache.pt --top-k 100
"""
import argparse
import json
import os
import time
from collections import defaultdict

import torch


def load_ebay_test(ebay_test_path):
    classes = []
    with open(ebay_test_path) as f:
        next(f)
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                classes.append(parts[1])
    return classes


def load_cub_test(cub_dir):
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
    test_classes = [str(i) for i in range(101, 201)]
    classes = []
    for c in test_classes:
        classes.extend([c] * len(by_class[c]))
    return classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["sop", "cub"], required=True)
    ap.add_argument("--labels-path", required=True,
                     help="Ebay_test.txt for sop, CUB_200_2011 dir for cub")
    ap.add_argument("--embeddings-cache", required=True)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--query-chunk", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading labels ({args.dataset})...")
    classes = load_ebay_test(args.labels_path) if args.dataset == "sop" else load_cub_test(args.labels_path)
    print(f"Loading cached patch embeddings from {args.embeddings_cache}...")
    all_embeddings = torch.load(args.embeddings_cache).to(device)  # (N, P, D) bf16, L2-normalized per patch
    N = all_embeddings.shape[0]
    assert N == len(classes), f"embedding count {N} != label count {len(classes)}"
    classes_t = torch.tensor([hash(c) for c in classes], device=device)

    print("Building stage-1 global embeddings (mean-pooled patches, re-normalized)...")
    t_stage1_build = time.time()
    global_embs = torch.nn.functional.normalize(all_embeddings.float().mean(dim=1), dim=-1)  # (N, D)
    stage1_build_time = time.time() - t_stage1_build

    K = args.top_k
    hits_1 = 0
    stage1_ceiling_hits = 0
    t0 = time.time()

    for q_start in range(0, N, args.query_chunk):
        q_end = min(N, q_start + args.query_chunk)
        Qb = q_end - q_start

        # --- Stage 1: fast global retrieval ---
        sims = global_embs[q_start:q_end] @ global_embs.T  # (Qb, N)
        idx_range = torch.arange(q_start, q_end, device=device)
        sims[torch.arange(Qb, device=device), idx_range] = -1e9  # mask self
        topk_vals, topk_idx = sims.topk(K, dim=1)  # (Qb, K)

        # stage-1 ceiling: does a same-class candidate exist anywhere in top-K?
        cand_classes = classes_t[topk_idx]  # (Qb, K)
        query_classes = classes_t[q_start:q_end].unsqueeze(1)  # (Qb, 1)
        stage1_ceiling_hits += (cand_classes == query_classes).any(dim=1).sum().item()

        # --- Stage 2: exact MaxSim rerank over the K candidates only ---
        q_patches = all_embeddings[q_start:q_end]  # (Qb, P, D)
        cand_patches = all_embeddings[topk_idx]  # (Qb, K, P, D)
        sim2 = torch.einsum("qpd,qkrd->qkpr", q_patches.float(), cand_patches.float())  # (Qb,K,P,P)
        a_to_b = sim2.max(dim=3).values.mean(dim=2)  # (Qb, K)
        b_to_a = sim2.max(dim=2).values.mean(dim=2)  # (Qb, K)
        scores = 0.5 * (a_to_b + b_to_a)
        best_within_k = scores.argmax(dim=1)  # (Qb,)
        final_idx = topk_idx[torch.arange(Qb, device=device), best_within_k]

        pred_classes = classes_t[final_idx]
        hits_1 += (pred_classes == classes_t[q_start:q_end]).sum().item()

        if q_start % (args.query_chunk * 50) == 0:
            elapsed = time.time() - t0
            print(f"  {q_end}/{N} queries  ({elapsed:.0f}s elapsed)")

    total_time = time.time() - t0 + stage1_build_time
    recall_1 = hits_1 / N
    stage1_ceiling = stage1_ceiling_hits / N

    print(f"\nStage-1 build time: {stage1_build_time:.1f}s")
    print(f"Total two-stage eval time ({N} queries, K={K}): {total_time:.1f}s ({N / total_time:.1f} queries/s)")
    print(f"Stage-1 ceiling (same-class candidate present in top-{K}): {stage1_ceiling:.4f}")
    print(f"Final two-stage Recall@1: {recall_1:.4f} ({hits_1}/{N})")

    out_path = args.out or os.path.join(os.path.dirname(args.embeddings_cache), f"two_stage_recall_k{K}.json")
    with open(out_path, "w") as f:
        json.dump({
            "dataset": args.dataset,
            "test_size": N,
            "top_k": K,
            "recall_1": recall_1,
            "hits": hits_1,
            "stage1_ceiling_recall": stage1_ceiling,
            "total_time_sec": total_time,
            "queries_per_sec": N / total_time,
        }, f, indent=2)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
