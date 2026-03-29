"""Score normalization and weighted fusion for hybrid retrieval."""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)


def min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    vals = np.array(list(scores.values()), dtype=np.float64)
    lo, hi = float(vals.min()), float(vals.max())
    if hi - lo < 1e-12:
        return {k: 1.0 for k in scores}
    out = {}
    for k, v in scores.items():
        out[k] = float((v - lo) / (hi - lo))
    return out


def fuse_scores(
    vector_norm: dict[str, float],
    bm25_norm: dict[str, float],
    alpha: float,
    chunk_ids: Iterable[str],
) -> dict[str, float]:
    """Fused = alpha * vector + (1 - alpha) * bm25. Missing branch scores as 0."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    fused: dict[str, float] = {}
    for cid in chunk_ids:
        v = vector_norm.get(cid, 0.0)
        b = bm25_norm.get(cid, 0.0)
        fused[cid] = alpha * v + (1.0 - alpha) * b
    return fused
