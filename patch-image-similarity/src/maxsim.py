"""Symmetric MaxSim (Chamfer similarity) over two sets of normalized vectors.

For patch sets A = {a_1 ... a_n} and B = {b_1 ... b_m} (already L2-normalized,
so the dot product is cosine similarity):

    S[i, j] = a_i . b_j
    score(A->B) = mean_i( max_j S[i, j] )   # each patch in A finds its best match in B
    score(B->A) = mean_j( max_i S[i, j] )   # symmetric
    similarity(A, B) = 0.5 * (score(A->B) + score(B->A))

Symmetric because this is image<->image comparison, not query->document
retrieval, and an asymmetric score would bias results by which image has more
"clutter".
"""
from dataclasses import dataclass

import torch


@dataclass
class MaxSimResult:
    similarity: float
    score_a_to_b: float
    score_b_to_a: float
    best_match_a_to_b: torch.Tensor  # (n,) index into B of each A patch's best match
    best_match_b_to_a: torch.Tensor  # (m,) index into A of each B patch's best match
    similarity_matrix: torch.Tensor  # (n, m)


def symmetric_maxsim(a: torch.Tensor, b: torch.Tensor) -> MaxSimResult:
    """a: (n, d), b: (m, d), both L2-normalized along dim=-1."""
    sim_matrix = a @ b.T  # (n, m)

    max_a_to_b, argmax_a_to_b = sim_matrix.max(dim=1)  # best match in B for each A patch
    max_b_to_a, argmax_b_to_a = sim_matrix.max(dim=0)  # best match in A for each B patch

    score_a_to_b = max_a_to_b.mean().item()
    score_b_to_a = max_b_to_a.mean().item()
    similarity = 0.5 * (score_a_to_b + score_b_to_a)

    return MaxSimResult(
        similarity=similarity,
        score_a_to_b=score_a_to_b,
        score_b_to_a=score_b_to_a,
        best_match_a_to_b=argmax_a_to_b,
        best_match_b_to_a=argmax_b_to_a,
        similarity_matrix=sim_matrix,
    )


def topk_symmetric_maxsim(a: torch.Tensor, b: torch.Tensor, k: int = 32) -> MaxSimResult:
    """Like symmetric_maxsim, but averages only the top-K per-patch best-match
    scores instead of all N. Two known weaknesses of plain mean-of-max
    motivate this: (1) background patches dilute the aggregate with matches
    that have nothing to do with the actual object (confirmed on tape --
    the SOP explanation demo found a positive pair scored low because patches
    matched on "background is light gray" instead of product features), and
    (2) averaging over all N patches is a smoothing operation that compresses
    score variance relative to global cosine's single-vector comparison,
    which is itself what made the elevation-based calibration comparison
    ambiguous. Restricting to the K strongest correspondences targets both:
    background rarely produces the *strongest* matches for a real object
    correspondence, and an extreme-order statistic over fewer values has more
    dynamic range than a mean over all 256. k is a tunable hyperparameter --
    no principled value is known yet, this is exploratory."""
    sim_matrix = a @ b.T
    max_a_to_b, argmax_a_to_b = sim_matrix.max(dim=1)
    max_b_to_a, argmax_b_to_a = sim_matrix.max(dim=0)

    k_a = min(k, max_a_to_b.shape[0])
    k_b = min(k, max_b_to_a.shape[0])
    score_a_to_b = max_a_to_b.topk(k_a).values.mean().item()
    score_b_to_a = max_b_to_a.topk(k_b).values.mean().item()
    similarity = 0.5 * (score_a_to_b + score_b_to_a)

    return MaxSimResult(
        similarity=similarity,
        score_a_to_b=score_a_to_b,
        score_b_to_a=score_b_to_a,
        best_match_a_to_b=argmax_a_to_b,
        best_match_b_to_a=argmax_b_to_a,
        similarity_matrix=sim_matrix,
    )


def global_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """a, b: (d,), both L2-normalized."""
    return (a @ b).item()


def threshold_hit_rate(a: torch.Tensor, b: torch.Tensor, threshold: float) -> float:
    """Symmetric hit-rate variant of MaxSim: instead of averaging the graded
    best-match cosine value (mean-of-max), score by the *fraction* of patches
    whose best match clears `threshold`. Motivated by generic/repetitive
    caption embeddings inflating mean-of-max for unrelated images -- a hit-rate
    only rewards matches confident enough to cross the bar, rather than
    letting middling-but-nonzero cosine values accumulate into a high score.
    """
    sim_matrix = a @ b.T
    max_a_to_b, _ = sim_matrix.max(dim=1)
    max_b_to_a, _ = sim_matrix.max(dim=0)
    hit_a_to_b = (max_a_to_b > threshold).float().mean().item()
    hit_b_to_a = (max_b_to_a > threshold).float().mean().item()
    return 0.5 * (hit_a_to_b + hit_b_to_a)
