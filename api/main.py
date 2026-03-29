"""FastAPI application: ingest, query, evaluate."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from config.settings import setup_logging
from evaluation.evaluation_runner import EvaluationRunner
from models.schemas import EvaluateResponse, IngestResponse, QueryRequest, QueryResponse

from api.rag_service import RagService, get_rag_service

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="Hybrid RAG API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile = File(...),
    chunking_strategy: str = Form("fixed"),
    chunk_size: int = Form(512),
    chunk_overlap: int = Form(64),
    doc_id: str | None = Form(None),
    svc: RagService = Depends(get_rag_service),
) -> IngestResponse:
    if chunking_strategy not in ("fixed", "sentence", "paragraph"):
        raise HTTPException(status_code=400, detail="Invalid chunking_strategy")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    return svc.ingest_bytes(
        file.filename or "upload",
        data,
        chunking_strategy=chunking_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        doc_id=doc_id,
    )


@app.post("/query", response_model=QueryResponse)
def query_endpoint(
    body: QueryRequest,
    svc: RagService = Depends(get_rag_service),
) -> QueryResponse:
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")
    return svc.query(body)


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate_endpoint(
    k: int = 5,
    svc: RagService = Depends(get_rag_service),
) -> EvaluateResponse:
    root = Path(__file__).resolve().parent.parent
    dataset = root / "data" / "eval_dataset.json"
    if not dataset.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Evaluation dataset not found at {dataset}",
        )
    runner = EvaluationRunner(svc.hybrid)
    return runner.run(dataset, k=k)
