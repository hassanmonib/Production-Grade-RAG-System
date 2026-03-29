"""Shared RAG application state: ingestion, retrieval, generation."""

from __future__ import annotations

import json
import logging
import threading
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from config.settings import get_settings
from generation.answer_generator import AnswerGenerator
from generation.llm import OpenAIClient
from ingestion.chunker import Chunker
from ingestion.cleaner import clean_text
from ingestion.embedder import Embedder
from ingestion.loader import load_bytes
from models.schemas import (
    ChunkRecord,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    RetrievalDebug,
    RetrievalDebugEntry,
    SourceRef,
)
from retrieval.bm25_index import BM25SearchIndex
from retrieval.hybrid_retriever import HybridRetriever, RetrievedChunk
from retrieval.reranker import CrossEncoderReranker
from retrieval.vector_store import FaissVectorStore

logger = logging.getLogger(__name__)


class RagService:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.settings = get_settings()
        self.embedder = Embedder()
        self.reranker = CrossEncoderReranker()
        self.vector_store = FaissVectorStore()
        self.bm25 = BM25SearchIndex()
        self.hybrid = HybridRetriever(self.vector_store, self.bm25, self.embedder, self.reranker)
        self.answer_gen = AnswerGenerator(OpenAIClient())
        self._chunks: list[ChunkRecord] = []
        self._load_persisted()

    def _persist_paths(self) -> tuple[Path, Path]:
        s = self.settings
        return s.chunks_store_path, s.faiss_index_path

    def _load_persisted(self) -> None:
        path, _ = self._persist_paths()
        if not path.exists():
            return
        loaded: list[ChunkRecord] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            loaded.append(ChunkRecord.model_validate_json(line))
        if not loaded:
            return
        self._rebuild_indexes(loaded)
        logger.info("Loaded %s chunks from %s", len(loaded), path)

    def _save_chunks(self) -> None:
        path, _ = self._persist_paths()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for c in self._chunks:
                f.write(c.model_dump_json() + "\n")

    def _rebuild_indexes(self, chunks: list[ChunkRecord]) -> None:
        self._chunks = chunks
        if not chunks:
            self.vector_store.rebuild([], np.array([], dtype=np.float32))
            self.bm25.rebuild([])
            return
        texts = [c.text for c in chunks]
        emb = self.embedder.encode(texts)
        self.vector_store.rebuild(chunks, emb)
        self.bm25.rebuild(chunks)

    def ingest_bytes(
        self,
        filename: str,
        data: bytes,
        chunking_strategy: str = "fixed",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        doc_id: str | None = None,
    ) -> IngestResponse:
        with self._lock:
            doc_id = (doc_id or "").strip() or str(uuid.uuid4())
            raw = load_bytes(filename, data)
            text = clean_text(raw)
            chunker = Chunker(
                strategy=chunking_strategy,  # type: ignore[arg-type]
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            new_chunks = chunker.chunk(doc_id, text)
            if not new_chunks:
                return IngestResponse(
                    doc_id=doc_id,
                    num_chunks=0,
                    chunking_strategy=chunking_strategy,
                    embedding_model=self.embedder.model_name,
                    message="No chunks produced (empty document?)",
                )
            merged = self._chunks + new_chunks
            self._rebuild_indexes(merged)
            self._save_chunks()
            return IngestResponse(
                doc_id=doc_id,
                num_chunks=len(new_chunks),
                chunking_strategy=chunking_strategy,
                embedding_model=self.embedder.model_name,
            )

    def _debug_from_result(
        self, before: list[RetrievedChunk], after: list[RetrievedChunk]
    ) -> RetrievalDebug:
        after_ids = {r.chunk.chunk_id for r in after}
        entries: list[RetrievalDebugEntry] = []
        vector_scores: list[float] = []
        bm25_scores: list[float] = []
        reranker_scores: list[float] = []

        rank_map = {r.chunk.chunk_id: r for r in after}
        for r in before:
            cid = r.chunk.chunk_id
            rv = r.vector_score
            rb = r.bm25_score
            if rv is not None:
                vector_scores.append(float(rv))
            if rb is not None:
                bm25_scores.append(float(rb))
            ar = rank_map.get(cid)
            rs = ar.rerank_score if ar else None
            rr = ar.rerank_rank if ar else None
            if rs is not None:
                reranker_scores.append(float(rs))
            entries.append(
                RetrievalDebugEntry(
                    chunk_id=cid,
                    doc_id=r.chunk.doc_id,
                    text_preview=r.chunk.text[:240] + ("…" if len(r.chunk.text) > 240 else ""),
                    vector_score=rv,
                    bm25_score=rb,
                    fused_score=r.fused_score,
                    rerank_score=rs,
                    rerank_rank=rr,
                )
            )

        # Reranked order table: sort by rerank_rank
        entries_sorted = sorted(
            entries,
            key=lambda e: (e.rerank_rank is None, e.rerank_rank or 9999),
        )
        return RetrievalDebug(
            vector_scores=vector_scores,
            bm25_scores=bm25_scores,
            reranker_scores=reranker_scores,
            entries=entries_sorted,
        )

    def query(self, req: QueryRequest) -> QueryResponse:
        s = self.settings
        v_k = req.top_k_retrieval or s.hybrid_vector_top_k
        b_k = req.top_k_retrieval or s.hybrid_bm25_top_k
        alpha = req.alpha if req.alpha is not None else s.fusion_alpha
        rerank_n = s.rerank_top_n
        llm_k = req.llm_top_k or s.llm_context_top_k

        emb = self.embedder.encode([req.question])[0]
        res = self.hybrid.retrieve(
            req.question,
            emb,
            vector_top_k=v_k,
            bm25_top_k=b_k,
            alpha=alpha,
            use_reranker=req.use_reranker,
            rerank_top_n=rerank_n,
            llm_top_k=llm_k,
        )
        answer, conf = self.answer_gen.generate(req.question, res.context_chunks)
        sources = [
            SourceRef(doc_id=c.chunk.doc_id, chunk_id=c.chunk.chunk_id)
            for c in res.context_chunks
        ]
        debug = self._debug_from_result(res.before_rerank, res.after_rerank)
        return QueryResponse(
            answer=answer,
            sources=sources,
            confidence_score=conf,
            retrieval_debug=debug,
        )


_service: RagService | None = None


def get_rag_service() -> RagService:
    global _service
    if _service is None:
        _service = RagService()
    return _service
