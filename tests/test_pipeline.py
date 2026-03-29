"""Unit tests for ingestion, hybrid fusion, reranking, metrics, and API schema."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.rag_service import get_rag_service
from evaluation.answer_metrics import hallucination_rate
from evaluation.retrieval_metrics import mrr_at_k, precision_recall_at_k
from ingestion.cleaner import clean_text
from ingestion.chunker import Chunker
from models.schemas import QueryResponse, RetrievalDebug, SourceRef
from retrieval.reranker import CrossEncoderReranker
from retrieval.score_fusion import fuse_scores, min_max_normalize


def test_cleaner_normalizes_whitespace():
    assert "  " not in clean_text("hello   world\n\n\nfoo")


def test_chunker_produces_records():
    c = Chunker("fixed", chunk_size=80, chunk_overlap=10)
    chunks = c.chunk("doc-test", "paragraph one.\n\n" * 20)
    assert all(x.doc_id == "doc-test" for x in chunks)
    assert len(chunks) >= 1


def test_min_max_and_fusion():
    v = {"a": 10.0, "b": 20.0}
    b = {"a": 1.0, "c": 3.0}
    vn = min_max_normalize(v)
    bn = min_max_normalize(b)
    assert vn["a"] == 0.0 and vn["b"] == 1.0
    fused = fuse_scores(vn, bn, 0.5, ["a", "b", "c"])
    assert set(fused) == {"a", "b", "c"}


def test_reranker_changes_order():
    r = CrossEncoderReranker(model_name="dummy")
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.1, 0.9, 0.2])
    r._model = mock_model  # noqa: SLF001
    out = r.rerank(
        "q",
        ["c1", "c2", "c3"],
        {"c1": "a", "c2": "b", "c3": "d"},
    )
    order = [x.chunk_id for x in out]
    assert order[0] == "c2"


def test_retrieval_metrics():
    rel = {"x", "y"}
    ids = ["a", "x", "y", "z"]
    p, r = precision_recall_at_k(ids, rel, k=3)
    assert p == pytest.approx(2 / 3)
    assert r == pytest.approx(2 / 2)
    assert mrr_at_k(ids, rel, k=4) == pytest.approx(1 / 2)


def test_hallucination_rate_basic():
    ans = ["the portal requires ten days", "completely unrelated gibberish xyz"]
    ctxs = [
        ["request paid time off through the hr portal at least ten business days"],
        ["some other context without overlap"],
    ]
    rate = hallucination_rate(ans, ctxs, threshold=0.08)
    assert 0.0 <= rate <= 1.0


def test_query_response_schema_via_api():
    mock_resp = QueryResponse(
        answer="test",
        sources=[SourceRef(doc_id="d", chunk_id="c")],
        confidence_score=0.5,
        retrieval_debug=RetrievalDebug(),
    )
    mock_svc = MagicMock()
    mock_svc.query.return_value = mock_resp

    app.dependency_overrides[get_rag_service] = lambda: mock_svc
    try:
        client = TestClient(app)
        r = client.post("/query", json={"question": "hello"})
        assert r.status_code == 200
        data = r.json()
        assert data["answer"] == "test"
        assert data["confidence_score"] == 0.5
        assert "retrieval_debug" in data
        assert "vector_scores" in data["retrieval_debug"]
    finally:
        app.dependency_overrides.clear()
