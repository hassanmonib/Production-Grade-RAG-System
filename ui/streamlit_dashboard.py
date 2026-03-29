"""
Retrieval engineering dashboard (not a chat UI).
Run: streamlit run ui/streamlit_dashboard.py
Requires FastAPI backend: uvicorn api.main:app --reload

Evaluation is implemented on the API only: POST /evaluate (see api/main.py).
To set a stable doc_id at ingest, use POST /ingest with form field `doc_id` (e.g. curl or Postman).
"""

from __future__ import annotations

import os

import httpx
import pandas as pd
import streamlit as st

DEFAULT_API = os.environ.get("RAG_API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Hybrid RAG — Retrieval Lab",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Hybrid RAG — Retrieval engineering dashboard")
st.caption("Inspect hybrid retrieval, fusion, and re-ranking — not a conversational UI.")


def _base() -> str:
    return str(st.session_state.get("api_url", DEFAULT_API)).rstrip("/")


def api_post(path: str, **kwargs):
    with httpx.Client(timeout=300.0) as client:
        r = client.post(f"{_base()}{path}", **kwargs)
        r.raise_for_status()
        return r.json()


with st.sidebar:
    st.subheader("Backend")
    st.text_input("API base URL", value=DEFAULT_API, key="api_url")
    if st.button("Health check"):
        try:
            with httpx.Client(timeout=10.0) as c:
                r = c.get(f"{_base()}/health")
            st.success(r.json())
        except Exception as e:  # noqa: BLE001
            st.error(str(e))

# --- 1 Ingestion ---
st.header("1. Document ingestion")
with st.expander("Advanced chunking", expanded=False):
    st.caption("Defaults are fine for most uploads. Open here to change strategy, size, or overlap.")
    ic1, ic2, ic3 = st.columns(3)
    with ic1:
        strat = st.selectbox(
            "Chunking strategy",
            ["fixed", "sentence", "paragraph"],
            key="ingest_chunk_strategy",
        )
    with ic2:
        ch_size = st.slider("Chunk size", 256, 1024, 512, 64, key="ingest_chunk_size")
    with ic3:
        ch_overlap = st.slider("Chunk overlap", 0, 256, 64, 16, key="ingest_chunk_overlap")

st.caption(
    f"**Chunking in use:** `{strat}` · chunk size **{ch_size}** · overlap **{ch_overlap}**"
)
upload = st.file_uploader("PDF, Markdown, or text", type=["pdf", "md", "txt", "markdown"])

if upload and st.button("Ingest document"):
    files = {"file": (upload.name, upload.getvalue(), upload.type or "application/octet-stream")}
    data = {
        "chunking_strategy": strat,
        "chunk_size": str(ch_size),
        "chunk_overlap": str(ch_overlap),
    }
    try:
        with httpx.Client(timeout=600.0) as client:
            r = client.post(f"{_base()}/ingest", files=files, data=data)
            r.raise_for_status()
            out = r.json()
        st.success(
            f"Chunks: **{out['num_chunks']}** | Strategy: **{out['chunking_strategy']}** | "
            f"Embeddings: **{out['embedding_model']}** | doc_id: **{out['doc_id']}**"
        )
    except Exception as e:  # noqa: BLE001
        st.error(str(e))

st.divider()

# --- 2 Query ---
st.header("2. Query & retrieval configuration")
question = st.text_area("Question", height=100, placeholder="Ask about your ingested corpus…")

with st.expander("Advanced retrieval", expanded=False):
    st.caption(
        "Defaults match the API. Change top-k, vector/BM25 balance (α), re-rank, or the display-only chunk hint."
    )
    qa, qb = st.columns(2)
    with qa:
        top_k = st.slider(
            "Top-k (vector & BM25 each)",
            5,
            40,
            20,
            key="query_top_k",
        )
        alpha = st.slider(
            "Fusion α (vector weight)",
            0.0,
            1.0,
            0.5,
            0.05,
            key="query_alpha",
        )
    with qb:
        use_rr = st.toggle("Cross-encoder re-rank", value=True, key="query_use_rerank")
        display_chunk_cfg = st.slider(
            "Chunk size (display only)",
            256,
            1024,
            512,
            64,
            key="query_chunk_display",
        )

rr_note = "on" if use_rr else "off"
st.caption(
    f"**Retrieval in use:** top-k **{top_k}** · fusion α **{alpha:.2f}** · "
    f"re-rank **{rr_note}** · chunk size (display) **{display_chunk_cfg}**"
)

run_query = st.button("Run retrieval + generation", type="primary")

if run_query and question.strip():
    try:
        payload = {
            "question": question.strip(),
            "top_k_retrieval": top_k,
            "alpha": alpha,
            "use_reranker": use_rr,
            "chunk_size": display_chunk_cfg,
        }
        out = api_post("/query", json=payload)
        st.session_state["last_query"] = out
    except Exception as e:  # noqa: BLE001
        st.error(str(e))

st.divider()

# --- 3 Retrieval debug ---
st.header("3. Retrieval debug")
if "last_query" in st.session_state:
    dbg = st.session_state["last_query"]["retrieval_debug"]
    rows = []
    for e in dbg.get("entries", []):
        rows.append(
            {
                "Chunk preview": e.get("text_preview", ""),
                "Vector": e.get("vector_score"),
                "BM25": e.get("bm25_score"),
                "Fused": e.get("fused_score"),
                "Rerank score": e.get("rerank_score"),
                "Rerank rank": e.get("rerank_rank"),
                "chunk_id": e.get("chunk_id", ""),
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    with st.expander("Raw score vectors"):
        st.json(
            {
                "vector_scores": dbg.get("vector_scores"),
                "bm25_scores": dbg.get("bm25_scores"),
                "reranker_scores": dbg.get("reranker_scores"),
            }
        )
else:
    st.info("Run a query to populate the retrieval debug table.")

st.divider()

# --- 4 Answer ---
st.header("4. Grounded answer")
if "last_query" in st.session_state:
    o = st.session_state["last_query"]
    st.metric("Confidence score", f"{o['confidence_score']:.3f}")
    st.markdown("### Answer")
    st.markdown(o["answer"])
    st.markdown("### Sources")
    for s in o["sources"]:
        st.markdown(f"- `doc_id={s['doc_id']}` · `chunk_id={s['chunk_id']}`")
    st.markdown("### Context chunks used (highlights)")
    src_ids = {s["chunk_id"] for s in o["sources"]}
    previews = [
        e["text_preview"]
        for e in o["retrieval_debug"]["entries"]
        if e.get("chunk_id") in src_ids
    ]
    for i, p in enumerate(previews, start=1):
        st.markdown(f"**[{i}]** {p}")
