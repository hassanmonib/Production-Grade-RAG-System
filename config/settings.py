"""Application settings loaded from environment and defaults."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    pinecone_api_key: str = Field(default="", alias="PINECONE_API_KEY")
    pinecone_environment: str = Field(default="", alias="PINECONE_ENVIRONMENT")
    pinecone_index_name: str = Field(default="hybrid-rag", alias="PINECONE_INDEX_NAME")
    elasticsearch_url: str = Field(default="", alias="ELASTICSEARCH_URL")

    vector_backend: str = Field(default="faiss", alias="VECTOR_BACKEND")  # faiss | pinecone
    data_dir: Path = Field(default=PROJECT_ROOT / "data", alias="DATA_DIR")
    faiss_index_path: Path = Field(
        default=PROJECT_ROOT / "data" / "faiss.index",
        alias="FAISS_INDEX_PATH",
    )
    chunks_store_path: Path = Field(
        default=PROJECT_ROOT / "data" / "chunks.jsonl",
        alias="CHUNKS_STORE_PATH",
    )

    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        alias="RERANKER_MODEL",
    )
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")

    hybrid_vector_top_k: int = Field(default=20, alias="HYBRID_VECTOR_TOP_K")
    hybrid_bm25_top_k: int = Field(default=20, alias="HYBRID_BM25_TOP_K")
    rerank_top_n: int = Field(default=20, alias="RERANK_TOP_N")
    llm_context_top_k: int = Field(default=5, alias="LLM_CONTEXT_TOP_K")
    fusion_alpha: float = Field(default=0.5, alias="FUSION_ALPHA")

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


@lru_cache
def get_settings() -> Settings:
    return Settings()


def setup_logging() -> None:
    s = get_settings()
    logging.basicConfig(
        level=getattr(logging, s.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
