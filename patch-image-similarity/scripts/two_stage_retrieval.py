"""Two-stage retrieve-and-rerank evaluation (ColBERT/ColPali-style late interaction):

  Stage 1 (fast): mean-pooled global embedding (cheap, single matmul per query
                   against the whole gallery) finds the top-K candidates.
  Stage 2 (exact): full symmetric patch-level MaxSim reranks those K
                   candidates, producing a full ranked list.

Reuses the patch-embedding cache already written by eval_full_test.py /
eval_cub_test.py / eval_cars_test.py -- no re-encoding needed, this is a pure
post-hoc reranking evaluation.

Reports Recall@1/2/4/8 and MAP@R (per Musgrave et al. "A Metric Learning
Reality Check" 2020 -- R = number of same-class gallery items excluding the
query, capped at --top-k since that's the retrieval budget; this slightly
UNDER-measures MAP@R for any class with more same-class items than --top-k,
which is a real but small approximation given these datasets' modest
per-class counts). Also reports a stage-1 "ceiling" recall (fraction of
queries where a same-class candidate exists anywhere in the top-K) so a low
score can be attributed to stage 1 missing the candidate vs. stage 2 ranking
it wrong.

Usage:
  python two_stage_retrieval.py --dataset sop --labels-path Ebay_test.txt \
      --embeddings-cache checkpoints/all_embeddings_cache.pt --top-k 100
  python two_stage_retrieval.py --dataset cub --labels-path /path/CUB_200_2011 \
      --embeddings-cache checkpoints_cub/cub_embeddings_cache.pt --top-k 100
  python two_stage_retrieval.py --dataset cars --labels-path /path/cars_raw \
      --embeddings-cache checkpoints_cars/cars_embeddings_cache.pt --top-k 100
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


def load_cars_test(cars_dir):
    names_path = os.path.join(cars_dir, "names.csv")
    with open(names_path) as f:
        class_names = [line.strip() for line in f if line.strip()]
    name_to_id = {}
    for i, name in enumerate(class_names):
        name_to_id[name] = str(i + 1)
        name_to_id[name.replace("/", "-")] = str(i + 1)

    by_class = defaultdict(list)
    car_data_root = os.path.join(cars_dir, "car_data", "car_data")
    for split in ("train", "test"):
        split_dir = os.path.join(car_data_root, split)
        for class_name in os.listdir(split_dir):
            class_id = name_to_id[class_name]
            class_dir = os.path.join(split_dir, class_name)
            for fname in os.listdir(class_dir):
                by_class[class_id].append(fname)
    test_classes = [str(i) for i in range(99, 197)]
    classes = []
    for c in test_classes:
        classes.extend([c] * len(by_class[c]))
    return classes


LOADERS = {"sop": load_ebay_test, "cub": load_cub_test, "cars": load_cars_test}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["sop", "cub", "cars"], required=True)
    ap.add_argument("--labels-path", required=True,
                     help="Ebay_test.txt for sop, CUB_200_2011 dir for cub, cars_raw dir for cars")
    ap.add_argument("--embeddings-cache", required=True)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--query-chunk", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading labels ({args.dataset})...")
    classes = LOADERS[args.dataset](args.labels_path)
    print(f"Loading cached patch embeddings from {args.embeddings_cache}...")
    # Kept on CPU -- at (N, 256, 768) bf16 this is ~22GB for SOP's 60502 images,
    # far too big to keep resident on a 24GB GPU alongside working buffers. Only
    # small per-chunk slices are moved to GPU below, same pattern as eval_full_test.py.
    all_embeddings = torch.load(args.embeddings_cache)  # (N, P, D) bf16, L2-normalized per patch
    N = all_embeddings.shape[0]
    assert N == len(classes), f"embedding count {N} != label count {len(classes)}"
    classes_t = torch.tensor([hash(c) for c in classes], device=device)

    # R for MAP@R = count of same-class gallery items excluding the query itself.
    class_counts = defaultdict(int)
    for c in classes:
        class_counts[c] += 1
    R_per_query = torch.tensor([class_counts[c] - 1 for c in classes], device=device).clamp(min=1)

    print("Building stage-1 global embeddings (mean-pooled patches, re-normalized)...")
    t_stage1_build = time.time()
    # (N, D) is tiny (~185MB for SOP) -- fine to keep resident on GPU for fast stage-1 matmuls.
    global_embs = torch.nn.functional.normalize(all_embeddings.float().mean(dim=1), dim=-1).to(device)
    stage1_build_time = time.time() - t_stage1_build

    K = args.top_k
    recall_hits = {r: 0 for r in (1, 2, 4, 8)}
    ap_sum = 0.0
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
        P = all_embeddings.shape[1]
        q_patches = all_embeddings[q_start:q_end].to(device).float()  # (Qb, P, D)
        cand_patches = all_embeddings[topk_idx.cpu()].to(device).float()  # (Qb, K, P, D)
        # Explicit bmm (not einsum) to guarantee the standard batched-matmul kernel is
        # used -- einsum's automatic contraction-path selection on this shared-batch +
        # extra-free-dim pattern was materializing a huge (Qb,K,P,P,D) intermediate
        # instead of a proper bmm, causing a 44GB allocation attempt.
        cand_flat = cand_patches.reshape(Qb, -1, cand_patches.shape[-1])  # (Qb, K*P, D)
        sim_flat = torch.bmm(q_patches, cand_flat.transpose(1, 2))  # (Qb, P, K*P)
        sim2 = sim_flat.view(Qb, P, K, P).permute(0, 2, 1, 3)  # (Qb, K, P, P)
        a_to_b = sim2.max(dim=3).values.mean(dim=2)  # (Qb, K)
        b_to_a = sim2.max(dim=2).values.mean(dim=2)  # (Qb, K)
        scores = 0.5 * (a_to_b + b_to_a)  # (Qb, K)

        # Full ranking of the K candidates (not just argmax) to get R@1/2/4/8 + MAP@R.
        rank_order = scores.argsort(dim=1, descending=True)  # (Qb, K) indices into the K-dim
        ranked_global_idx = torch.gather(topk_idx, 1, rank_order)  # (Qb, K) actual gallery indices, ranked
        ranked_classes = classes_t[ranked_global_idx]  # (Qb, K)
        query_classes_col = classes_t[q_start:q_end].unsqueeze(1)  # (Qb, 1)
        relevant = (ranked_classes == query_classes_col).float()  # (Qb, K), 1 where same class

        for r in recall_hits:
            recall_hits[r] += relevant[:, :r].any(dim=1).sum().item()

        # MAP@R: mean over queries of (1/R) * sum_{rank=1..R} precision@rank * relevant@rank.
        # R capped at K (the retrieval budget) -- see module docstring.
        cum_relevant = relevant.cumsum(dim=1)  # (Qb, K)
        ranks = torch.arange(1, K + 1, device=device, dtype=torch.float32).unsqueeze(0)  # (1, K)
        precision_at_rank = cum_relevant / ranks  # (Qb, K)
        R_batch = R_per_query[q_start:q_end].clamp(max=K)  # (Qb,)
        rank_mask = (torch.arange(1, K + 1, device=device).unsqueeze(0) <= R_batch.unsqueeze(1)).float()
        ap = (precision_at_rank * relevant * rank_mask).sum(dim=1) / R_batch.float()
        ap_sum += ap.sum().item()

        if q_start % (args.query_chunk * 50) == 0:
            elapsed = time.time() - t0
            print(f"  {q_end}/{N} queries  ({elapsed:.0f}s elapsed)")

    total_time = time.time() - t0 + stage1_build_time
    recall_at = {r: recall_hits[r] / N for r in recall_hits}
    map_at_r = ap_sum / N
    stage1_ceiling = stage1_ceiling_hits / N

    print(f"\nStage-1 build time: {stage1_build_time:.1f}s")
    print(f"Total two-stage eval time ({N} queries, K={K}): {total_time:.1f}s ({N / total_time:.1f} queries/s)")
    print(f"Stage-1 ceiling (same-class candidate present in top-{K}): {stage1_ceiling:.4f}")
    for r in sorted(recall_at):
        print(f"Recall@{r}: {recall_at[r]:.4f}")
    print(f"MAP@R: {map_at_r:.4f}")

    out_path = args.out or os.path.join(os.path.dirname(args.embeddings_cache), f"two_stage_metrics_k{K}.json")
    with open(out_path, "w") as f:
        json.dump({
            "dataset": args.dataset,
            "test_size": N,
            "top_k": K,
            "recall_at_1": recall_at[1],
            "recall_at_2": recall_at[2],
            "recall_at_4": recall_at[4],
            "recall_at_8": recall_at[8],
            "map_at_r": map_at_r,
            "stage1_ceiling_recall": stage1_ceiling,
            "total_time_sec": total_time,
            "queries_per_sec": N / total_time,
        }, f, indent=2)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
