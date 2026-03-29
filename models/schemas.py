"""Pydantic schemas for API and internal payloads."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SourceRef(BaseModel):
    doc_id: str
    chunk_id: str


class RetrievalDebugEntry(BaseModel):
    chunk_id: str
    doc_id: str
    text_preview: str = ""
    vector_score: float | None = None
    bm25_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None
    rerank_rank: int | None = None


class RetrievalDebug(BaseModel):
    vector_scores: list[float] = Field(default_factory=list)
    bm25_scores: list[float] = Field(default_factory=list)
    reranker_scores: list[float] = Field(default_factory=list)
    entries: list[RetrievalDebugEntry] = Field(default_factory=list)


class QueryRequest(BaseModel):
    question: str
    top_k_retrieval: int | None = Field(default=None, ge=1, le=100)
    alpha: float | None = Field(default=None, ge=0.0, le=1.0)
    use_reranker: bool = True
    chunk_size: int | None = Field(
        default=None,
        ge=64,
        le=8192,
        description="If set, re-chunking is not applied at query time; used for display/config only.",
    )
    llm_top_k: int | None = Field(default=None, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceRef]
    confidence_score: float
    retrieval_debug: RetrievalDebug


class IngestResponse(BaseModel):
    doc_id: str
    num_chunks: int
    chunking_strategy: str
    embedding_model: str
    message: str = "ok"


class EvaluationResultRow(BaseModel):
    method: str
    precision_at_k: float
    recall_at_k: float
    mrr: float


class EvaluateResponse(BaseModel):
    precision_at_k: float
    recall_at_k: float
    mrr: float
    hallucination_rate: float
    k: int
    comparison: list[EvaluationResultRow]
    details: dict[str, Any] = Field(default_factory=dict)


class LabeledQAPair(BaseModel):
    query: str
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    relevant_doc_ids: list[str] = Field(default_factory=list)
    gold_answer_snippet: str | None = None


class ChunkRecord(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


ChunkingStrategy = Literal["fixed", "sentence", "paragraph"]
