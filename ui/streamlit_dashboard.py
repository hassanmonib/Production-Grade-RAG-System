"""
Retrieval engineering dashboard (not a chat UI).
Run: streamlit run ui/streamlit_dashboard.py
Requires FastAPI backend: uvicorn api.main:app --reload
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
st.caption("Inspect hybrid retrieval, fusion, re-ranking, and evaluation — not a conversational UI.")


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
c1, c2, c3 = st.columns(3)
with c1:
    strat = st.selectbox("Chunking strategy", ["fixed", "sentence", "paragraph"])
with c2:
    ch_size = st.slider("Chunk size", 256, 1024, 512, 64)
with c3:
    ch_overlap = st.slider("Chunk overlap", 0, 256, 64, 16)
doc_id_opt = st.text_input(
    "Optional stable doc_id (for evaluation labels)",
    placeholder="e.g. employee-handbook",
    help="If set, all chunks from this upload share this doc_id.",
)
upload = st.file_uploader("PDF, Markdown, or text", type=["pdf", "md", "txt", "markdown"])

if upload and st.button("Ingest document"):
    files = {"file": (upload.name, upload.getvalue(), upload.type or "application/octet-stream")}
    data = {
        "chunking_strategy": strat,
        "chunk_size": str(ch_size),
        "chunk_overlap": str(ch_overlap),
    }
    if doc_id_opt.strip():
        data["doc_id"] = doc_id_opt.strip()
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
qcol1, qcol2 = st.columns([2, 1])
with qcol1:
    question = st.text_area("Question", height=100, placeholder="Ask about your ingested corpus…")
with qcol2:
    top_k = st.slider("Top-k (vector & BM25 each)", 5, 40, 20)
    alpha = st.slider("Fusion α (vector weight)", 0.0, 1.0, 0.5, 0.05)
    use_rr = st.toggle("Cross-encoder re-rank", value=True)
    display_chunk_cfg = st.slider("Chunk size (display only)", 256, 1024, 512, 64)

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

st.divider()

# --- 5 Evaluation ---
st.header("5. Evaluation")
st.caption("Uses `data/eval_dataset.json` and **relevant_doc_ids** matching your stable doc_id labels.")
if st.button("Run evaluation"):
    try:
        with httpx.Client(timeout=600.0) as client:
            r = client.post(f"{_base()}/evaluate?k=5")
            r.raise_for_status()
            ev = r.json()
        st.session_state["last_eval"] = ev
    except Exception as e:  # noqa: BLE001
        st.error(str(e))

if "last_eval" in st.session_state:
    ev = st.session_state["last_eval"]
    det = ev.get("details") or {}
    if det.get("evaluated") == 0:
        st.warning(
            det.get("hint")
            or "No queries were evaluated — labels did not match any ingested `doc_id`."
        )
        if det.get("doc_ids_in_index"):
            st.caption(f"**doc_id values currently in the index:** {det['doc_ids_in_index']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision@k (hybrid+rerank)", f"{ev['precision_at_k']:.3f}")
    c2.metric("Recall@k", f"{ev['recall_at_k']:.3f}")
    c3.metric("MRR", f"{ev['mrr']:.3f}")
    c4.metric("Hallucination rate (proxy)", f"{ev['hallucination_rate']:.3f}")

    comp = pd.DataFrame(ev["comparison"])
    st.subheader("Vector vs hybrid vs hybrid + re-rank")
    st.bar_chart(comp.set_index("method")[["precision_at_k", "recall_at_k", "mrr"]])

    st.dataframe(comp, use_container_width=True, hide_index=True)
    st.json(ev.get("details", {}))
