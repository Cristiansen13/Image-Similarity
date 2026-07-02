"""Triplet accuracy harness (spec section 4).

For each (anchor, positive, negative) triplet, checks whether
score(anchor, positive) > score(anchor, negative). Accuracy is the fraction
of triplets where that holds, reported per scorer and broken down by
triplet type (synthetic / semantic).
"""
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable


@dataclass
class TripletEvalResult:
    accuracy_by_type: dict  # type -> accuracy
    accuracy_overall: float
    n_by_type: dict
    failures: list  # triplets where positive did not outscore negative


def triplet_accuracy(triplets: list[dict], score_fn: Callable[[str, str], float]) -> TripletEvalResult:
    """triplets: list of {"anchor", "positive", "negative", "type", ...}.
    score_fn(path_a, path_b) -> similarity score (higher = more similar).
    """
    correct_by_type = defaultdict(int)
    total_by_type = defaultdict(int)
    failures = []

    for triplet in triplets:
        pos_score = score_fn(triplet["anchor"], triplet["positive"])
        neg_score = score_fn(triplet["anchor"], triplet["negative"])
        is_correct = pos_score > neg_score

        t_type = triplet.get("type", "unknown")
        total_by_type[t_type] += 1
        if is_correct:
            correct_by_type[t_type] += 1
        else:
            failures.append({**triplet, "pos_score": pos_score, "neg_score": neg_score})

    accuracy_by_type = {
        t: correct_by_type[t] / total_by_type[t] for t in total_by_type
    }
    total_correct = sum(correct_by_type.values())
    total = sum(total_by_type.values())

    return TripletEvalResult(
        accuracy_by_type=accuracy_by_type,
        accuracy_overall=total_correct / total if total else 0.0,
        n_by_type=dict(total_by_type),
        failures=failures,
    )
