"""Hybrid BM25 + dense retrieval with optional cross-encoder re-ranking."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from models.schemas import ChunkRecord
from retrieval.bm25_index import BM25SearchIndex
from retrieval.reranker import CrossEncoderReranker, RerankResult
from retrieval.score_fusion import fuse_scores, min_max_normalize
from retrieval.vector_store import FaissVectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    chunk: ChunkRecord
    vector_score: float | None = None
    bm25_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None
    rerank_rank: int | None = None


@dataclass
class HybridRetrievalResult:
    before_rerank: list[RetrievedChunk] = field(default_factory=list)
    after_rerank: list[RetrievedChunk] = field(default_factory=list)
    context_chunks: list[RetrievedChunk] = field(default_factory=list)


class HybridRetriever:
    def __init__(
        self,
        vector_store: FaissVectorStore,
        bm25: BM25SearchIndex,
        embedder,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.bm25 = bm25
        self.embedder = embedder
        self.reranker = reranker or CrossEncoderReranker()

    def _chunk_map(self) -> dict[str, ChunkRecord]:
        m: dict[str, ChunkRecord] = {}
        for c in self.vector_store.chunks:
            m[c.chunk_id] = c
        return m

    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        vector_top_k: int,
        bm25_top_k: int,
        alpha: float,
        use_reranker: bool,
        rerank_top_n: int,
        llm_top_k: int,
    ) -> HybridRetrievalResult:
        cm = self._chunk_map()
        vec_hits = self.vector_store.search(query_embedding, vector_top_k)
        bm_hits = self.bm25.search(query, bm25_top_k)

        vec_scores = {cid: s for cid, s in vec_hits}
        bm_scores = {cid: s for cid, s in bm_hits}
        all_ids = list(dict.fromkeys(list(vec_scores) + list(bm_scores)))

        v_norm = min_max_normalize(vec_scores)
        b_norm = min_max_normalize(bm_scores)
        fused = fuse_scores(v_norm, b_norm, alpha, all_ids)
        fused_sorted = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        fused_order = [cid for cid, _ in fused_sorted[: max(rerank_top_n, llm_top_k)]]

        before: list[RetrievedChunk] = []
        for cid in fused_order:
            ch = cm.get(cid)
            if not ch:
                continue
            before.append(
                RetrievedChunk(
                    chunk=ch,
                    vector_score=vec_scores.get(cid),
                    bm25_score=bm_scores.get(cid),
                    fused_score=fused.get(cid),
                )
            )

        rerank_slice = before[:rerank_top_n]
        id_to_text = {r.chunk.chunk_id: r.chunk.text for r in rerank_slice}
        ids_for_rerank = [r.chunk.chunk_id for r in rerank_slice]
        orig_by_id = {r.chunk.chunk_id: r for r in rerank_slice}

        if use_reranker and ids_for_rerank:
            rr = self.reranker.rerank(query, ids_for_rerank, id_to_text)
            after_sorted: list[RetrievedChunk] = []
            for res in rr:
                o = orig_by_id.get(res.chunk_id)
                if not o:
                    continue
                after_sorted.append(
                    RetrievedChunk(
                        chunk=o.chunk,
                        vector_score=o.vector_score,
                        bm25_score=o.bm25_score,
                        fused_score=o.fused_score,
                        rerank_score=res.score,
                        rerank_rank=res.rank,
                    )
                )
            context = after_sorted[:llm_top_k]
        else:
            after_sorted = [
                RetrievedChunk(
                    chunk=r.chunk,
                    vector_score=r.vector_score,
                    bm25_score=r.bm25_score,
                    fused_score=r.fused_score,
                    rerank_score=r.fused_score,
                    rerank_rank=i,
                )
                for i, r in enumerate(rerank_slice, start=1)
            ]
            context = after_sorted[:llm_top_k]

        return HybridRetrievalResult(
            before_rerank=before,
            after_rerank=after_sorted,
            context_chunks=context,
        )
