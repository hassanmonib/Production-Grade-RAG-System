"""BM25 lexical index over chunk texts."""

from __future__ import annotations

import logging
import re
from typing import Sequence

from rank_bm25 import BM25Okapi

from models.schemas import ChunkRecord

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25SearchIndex:
    def __init__(self) -> None:
        self._chunks: list[ChunkRecord] = []
        self._bm25: BM25Okapi | None = None
        self._tokenized: list[list[str]] = []

    def rebuild(self, chunks: Sequence[ChunkRecord]) -> None:
        self._chunks = list(chunks)
        self._tokenized = [_tokenize(c.text) for c in self._chunks]
        if not self._tokenized:
            self._bm25 = None
            return
        self._bm25 = BM25Okapi(self._tokenized)
        logger.info("BM25 index rebuilt: %s chunks", len(self._chunks))

    @property
    def chunks(self) -> list[ChunkRecord]:
        return self._chunks

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if not self._bm25 or not self._chunks:
            return []
        q = _tokenize(query)
        scores = self._bm25.get_scores(q)
        pairs = list(zip([c.chunk_id for c in self._chunks], scores, strict=True))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:top_k]
