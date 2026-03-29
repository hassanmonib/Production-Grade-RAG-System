"""Grounded answer generation with confidence heuristics."""

from __future__ import annotations

import logging
import re
from typing import Sequence

import numpy as np

from generation.citation_formatter import format_context_blocks
from generation.llm import OpenAIClient
from retrieval.hybrid_retriever import RetrievedChunk

logger = logging.getLogger(__name__)


def _tokenize_simple(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _answer_grounding_score(answer: str, contexts: Sequence[RetrievedChunk]) -> float:
    if not answer.strip() or not contexts:
        return 0.0
    a = _tokenize_simple(answer)
    if not a:
        return 0.0
    scores = []
    for c in contexts:
        b = _tokenize_simple(c.chunk.text)
        if not b:
            continue
        inter = len(a & b)
        union = len(a | b)
        scores.append(inter / union if union else 0.0)
    return float(max(scores)) if scores else 0.0


def _rerank_confidence(contexts: Sequence[RetrievedChunk]) -> float:
    vals = [c.rerank_score for c in contexts if c.rerank_score is not None]
    if not vals:
        return 0.5
    arr = np.array(vals, dtype=np.float64)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-9:
        return 0.75
    norm = [(float(v) - lo) / (hi - lo) for v in vals]
    return float(np.mean(norm))


def compute_confidence(answer: str, contexts: Sequence[RetrievedChunk]) -> float:
    r = _rerank_confidence(contexts)
    g = _answer_grounding_score(answer, contexts)
    # Agreement: how well the answer is supported by at least one retrieved chunk.
    score = 0.45 * r + 0.35 * g + 0.2 * min(1.0, len(contexts) / 5.0)
    return float(np.clip(score, 0.0, 1.0))


class AnswerGenerator:
    def __init__(self, llm: OpenAIClient | None = None) -> None:
        self.llm = llm or OpenAIClient()

    def generate(self, question: str, contexts: list[RetrievedChunk]) -> tuple[str, float]:
        if not contexts:
            return (
                "No relevant passages were retrieved. Upload documents or broaden the query.",
                0.0,
            )
        ctx = format_context_blocks(contexts)
        system = (
            "You are a precise assistant for internal documentation. "
            "Answer ONLY using the numbered context passages. "
            "If the context is insufficient, say what is missing. "
            "Do not invent facts. End with a line: Sources: listing chunk indices used."
        )
        user = f"Question: {question}\n\nContext:\n{ctx}"
        try:
            answer = self.llm.complete(system, user)
        except Exception as e:  # noqa: BLE001
            logger.exception("LLM generation failed: %s", e)
            return (f"Generation error: {e}", 0.0)
        conf = compute_confidence(answer, contexts)
        return answer, conf
