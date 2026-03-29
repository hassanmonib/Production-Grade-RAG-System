"""Standard IR metrics for labeled chunk IDs."""

from __future__ import annotations


def precision_recall_at_k(retrieved_ids: list[str], relevant: set[str], k: int) -> tuple[float, float]:
    if k <= 0:
        return 0.0, 0.0
    top = retrieved_ids[:k]
    hits = sum(1 for x in top if x in relevant)
    prec = hits / len(top) if top else 0.0
    rec = hits / len(relevant) if relevant else 0.0
    return prec, rec


def mrr_at_k(retrieved_ids: list[str], relevant: set[str], k: int) -> float:
    for rank, cid in enumerate(retrieved_ids[:k], start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0
