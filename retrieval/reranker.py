"""Cross-encoder re-ranking."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

from config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    chunk_id: str
    score: float
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str | None = None) -> None:
        s = get_settings()
        self.model_name = model_name or s.reranker_model
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading cross-encoder %s", self.model_name)
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self, query: str, chunk_ids: Sequence[str], id_to_text: dict[str, str]
    ) -> list[RerankResult]:
        valid_ids = [cid for cid in chunk_ids if cid in id_to_text]
        pairs = [(query, id_to_text[cid]) for cid in valid_ids]
        if not pairs:
            return []
        raw = self.model.predict(pairs, show_progress_bar=False)
        scores_list = raw.tolist() if hasattr(raw, "tolist") else list(raw)
        ranked = sorted(
            zip(valid_ids, scores_list, strict=True),
            key=lambda x: x[1],
            reverse=True,
        )
        out: list[RerankResult] = []
        for rank, (cid, sc) in enumerate(ranked, start=1):
            out.append(RerankResult(chunk_id=cid, score=float(sc), rank=rank))
        return out
