"""Structured chunking strategies."""

from __future__ import annotations

import logging
import uuid
from typing import Literal

from langchain_text_splitters import RecursiveCharacterTextSplitter

from models.schemas import ChunkRecord

logger = logging.getLogger(__name__)

ChunkingStrategy = Literal["fixed", "sentence", "paragraph"]


class Chunker:
    def __init__(
        self,
        strategy: ChunkingStrategy = "fixed",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> None:
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, doc_id: str, text: str) -> list[ChunkRecord]:
        if self.strategy == "paragraph":
            return _chunk_paragraphs(doc_id, text, self.chunk_size, self.chunk_overlap)
        if self.strategy == "sentence":
            return _chunk_sentences(doc_id, text, self.chunk_size, self.chunk_overlap)
        return _chunk_fixed(doc_id, text, self.chunk_size, self.chunk_overlap)


def chunk_text(
    doc_id: str,
    text: str,
    strategy: ChunkingStrategy = "fixed",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[ChunkRecord]:
    return Chunker(strategy, chunk_size, chunk_overlap).chunk(doc_id, text)


def _chunk_fixed(
    doc_id: str, text: str, chunk_size: int, chunk_overlap: int
) -> list[ChunkRecord]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    docs = splitter.split_text(text)
    return _to_records(doc_id, docs, {"strategy": "fixed"})


def _chunk_paragraphs(
    doc_id: str, text: str, chunk_size: int, chunk_overlap: int
) -> list[ChunkRecord]:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paras:
        return []
    merged: list[str] = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 2 <= chunk_size:
            buf = f"{buf}\n\n{p}" if buf else p
        else:
            if buf:
                merged.append(buf)
            buf = p if len(p) <= chunk_size else p[:chunk_size]
    if buf:
        merged.append(buf)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    out: list[str] = []
    for m in merged:
        out.extend(splitter.split_text(m))
    return _to_records(doc_id, out, {"strategy": "paragraph"})


def _chunk_sentences(
    doc_id: str, text: str, chunk_size: int, chunk_overlap: int
) -> list[ChunkRecord]:
    # Prefer sentence boundaries without requiring tiktoken.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[". ", "?\n", "!\n", "\n\n", "\n", " ", ""],
    )
    docs = splitter.split_text(text)
    return _to_records(doc_id, docs, {"strategy": "sentence"})


def _to_records(doc_id: str, pieces: list[str], meta: dict) -> list[ChunkRecord]:
    return [
        ChunkRecord(
            chunk_id=str(uuid.uuid4()),
            doc_id=doc_id,
            text=piece,
            metadata={**meta},
        )
        for piece in pieces
        if piece.strip()
    ]
