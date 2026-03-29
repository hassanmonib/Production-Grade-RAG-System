"""FAISS vector store with optional Pinecone stub."""

from __future__ import annotations

import logging
from typing import Sequence

import faiss
import numpy as np

from config.settings import get_settings
from models.schemas import ChunkRecord

logger = logging.getLogger(__name__)


class FaissVectorStore:
    def __init__(self) -> None:
        self._chunks: list[ChunkRecord] = []
        self._embeddings: np.ndarray | None = None
        self._index: faiss.Index | None = None
        self._dim: int | None = None

    def rebuild(self, chunks: Sequence[ChunkRecord], embeddings: np.ndarray) -> None:
        self._chunks = list(chunks)
        if embeddings.size == 0:
            self._embeddings = None
            self._index = None
            self._dim = None
            return
        self._embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
        self._dim = self._embeddings.shape[1]
        index = faiss.IndexFlatIP(self._dim)
        faiss.normalize_L2(self._embeddings)
        index.add(self._embeddings)
        self._index = index
        logger.info("FAISS index rebuilt: %s vectors dim=%s", len(self._chunks), self._dim)

    @property
    def chunks(self) -> list[ChunkRecord]:
        return self._chunks

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        if self._index is None or not self._chunks:
            return []
        q = np.ascontiguousarray(query_embedding.reshape(1, -1).astype(np.float32))
        faiss.normalize_L2(q)
        k = min(top_k, len(self._chunks))
        scores, idx = self._index.search(q, k)
        out: list[tuple[str, float]] = []
        for i, s in zip(idx[0], scores[0], strict=True):
            if i < 0:
                continue
            out.append((self._chunks[i].chunk_id, float(s)))
        return out


def pinecone_available() -> bool:
    s = get_settings()
    return bool(s.pinecone_api_key and s.vector_backend.lower() == "pinecone")
