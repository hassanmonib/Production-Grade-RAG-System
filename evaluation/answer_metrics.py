"""Answer-level proxies for hallucination / grounding."""

from __future__ import annotations

import re
from typing import Sequence


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def answer_supported_by_context(answer: str, context_texts: Sequence[str]) -> float:
    """Max Jaccard overlap between answer tokens and any context (0..1)."""
    a = _tokens(answer)
    if not a:
        return 0.0
    best = 0.0
    for ctx in context_texts:
        b = _tokens(ctx)
        if not b:
            continue
        inter = len(a & b)
        union = len(a | b)
        best = max(best, inter / union if union else 0.0)
    return best


def hallucination_rate(
    answers: Sequence[str],
    context_texts_per_answer: Sequence[Sequence[str]],
    threshold: float = 0.12,
) -> float:
    """Fraction of answers with weak lexical overlap with retrieved contexts."""
    if not answers:
        return 0.0
    bad = 0
    for ans, ctxs in zip(answers, context_texts_per_answer, strict=True):
        if answer_supported_by_context(ans, ctxs) < threshold:
            bad += 1
    return bad / len(answers)
