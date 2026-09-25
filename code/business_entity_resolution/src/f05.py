"""Macro-averaged F_0.5 scorer used during validation.

Per the challenge spec:

    F_0.5 = (1.25 * P * R) / (0.25 * P + R)

computed per Source 1 entity then averaged across ALL Source 1 entities.
Singletons matter: correctly predicting empty for a true singleton gives
1.0; any false-positive match gives 0.0.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Set

import numpy as np


def f05_single(pred_set: Set[str], true_set: Set[str]) -> float:
    """F_0.5 for one entity."""
    if not true_set:
        # True singleton: 1.0 if we predicted empty, else 0.0
        return 1.0 if not pred_set else 0.0
    if not pred_set:
        # Predicted singleton but it has matches
        return 0.0
    tp = len(pred_set & true_set)
    p = tp / max(1, len(pred_set))
    r = tp / max(1, len(true_set))
    if p == 0.0 and r == 0.0:
        return 0.0
    return (1.25 * p * r) / (0.25 * p + r)


def macro_f05(preds: Iterable[Set[str]], trues: Iterable[Set[str]]) -> float:
    """Macro-averaged F_0.5 across all entities."""
    return float(np.mean([f05_single(p, t) for p, t in zip(preds, trues)]))


def parse_id_list(s: str) -> Set[str]:
    """Parse a comma-separated matched_entity_ids cell. Empty string -> empty set."""
    if not s:
        return set()
    return {x.strip() for x in s.split(",") if x.strip()}


def bucket_scores(
    preds: Sequence[Set[str]],
    trues: Sequence[Set[str]],
    countries: Sequence[str],
    sources: Sequence[str],
) -> dict:
    """Per-(country, source) bucket macro F_0.5; also singleton P/R/F_0.5."""
    buckets: dict[tuple[str, str], list[float]] = {}
    singleton_pred_pos: list[int] = []
    singleton_true_pos: list[int] = []
    for pred, true, c, s in zip(preds, trues, countries, sources):
        score = f05_single(pred, true)
        key = (c, s)
        buckets.setdefault(key, []).append(score)
        # Singleton bookkeeping
        if not true:
            singleton_pred_pos.append(1 if pred else 0)
            singleton_true_pos.append(1)
        else:
            singleton_pred_pos.append(0 if not pred else 1)
            singleton_true_pos.append(0)

    out = {
        "per_bucket": {
            f"{c}|{s}": float(np.mean(scores))
            for (c, s), scores in buckets.items()
        },
        "overall": macro_f05(preds, trues),
        "singleton_count": int(sum(singleton_true_pos)),
        "singleton_correct": int(
            sum(1 for p, t in zip(singleton_pred_pos, singleton_true_pos) if p == 1 and t == 1)
        ),
        "singleton_accuracy": (
            float(np.mean([p == t for p, t in zip(singleton_pred_pos, singleton_true_pos)]))
            if singleton_pred_pos
            else 0.0
        ),
    }
    return out
