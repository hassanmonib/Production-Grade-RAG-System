"""Format citations for prompts and UI."""

from __future__ import annotations

from retrieval.hybrid_retriever import RetrievedChunk


def format_context_blocks(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, r in enumerate(chunks, start=1):
        cid = r.chunk.chunk_id
        did = r.chunk.doc_id
        parts.append(f"[{i}] (doc_id={did}, chunk_id={cid})\n{r.chunk.text}")
    return "\n\n".join(parts)


def format_sources_line(chunks: list[RetrievedChunk]) -> str:
    return ", ".join(f"{c.chunk.doc_id}:{c.chunk.chunk_id[:8]}" for c in chunks)
