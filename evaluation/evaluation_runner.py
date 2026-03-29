"""Run retrieval / answer evaluation over a labeled JSON dataset."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.answer_metrics import hallucination_rate
from evaluation.retrieval_metrics import mrr_at_k, precision_recall_at_k
from models.schemas import EvaluateResponse, EvaluationResultRow, LabeledQAPair
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.score_fusion import fuse_scores, min_max_normalize

logger = logging.getLogger(__name__)


def load_labeled_dataset(path: Path) -> list[LabeledQAPair]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [LabeledQAPair.model_validate(x) for x in raw]


class EvaluationRunner:
    def __init__(self, retriever: HybridRetriever) -> None:
        self.retriever = retriever

    def _chunk_map(self) -> dict[str, Any]:
        return {c.chunk_id: c for c in self.retriever.vector_store.chunks}

    def _retrieve_ids_vector_only(self, query: str, emb: np.ndarray, k: int) -> list[str]:
        hits = self.retriever.vector_store.search(emb, k)
        return [cid for cid, _ in hits]

    def _retrieve_ids_hybrid_no_rerank(
        self, query: str, emb: np.ndarray, k: int, alpha: float
    ) -> list[str]:
        cm = self._chunk_map()
        vec = dict(self.retriever.vector_store.search(emb, k))
        bm = dict(self.retriever.bm25.search(query, k))
        all_ids = list(dict.fromkeys(list(vec) + list(bm)))
        v_n = min_max_normalize(vec)
        b_n = min_max_normalize(bm)
        fused = fuse_scores(v_n, b_n, alpha, all_ids)
        return sorted(fused, key=lambda x: fused[x], reverse=True)[:k]

    def _retrieve_ids_hybrid_rerank(
        self, query: str, emb: np.ndarray, k: int, alpha: float, rerank_n: int
    ) -> list[str]:
        res = self.retriever.retrieve(
            query,
            emb,
            vector_top_k=k,
            bm25_top_k=k,
            alpha=alpha,
            use_reranker=True,
            rerank_top_n=rerank_n,
            llm_top_k=k,
        )
        return [r.chunk.chunk_id for r in res.after_rerank[:k]]

    def run(
        self,
        dataset_path: Path,
        k: int = 5,
        alpha: float = 0.5,
    ) -> EvaluateResponse:
        pairs = load_labeled_dataset(dataset_path)
        embedder = self.retriever.embedder

        methods = {
            "vector": [],
            "hybrid": [],
            "hybrid_rerank": [],
        }
        answers: list[str] = []
        ctx_for_hallu: list[list[str]] = []

        cm = self._chunk_map()

        for p in pairs:
            rel = set(p.relevant_chunk_ids)
            for did in p.relevant_doc_ids:
                rel.update(
                    cid for cid, rec in cm.items() if getattr(rec, "doc_id", None) == did
                )
            if not rel:
                logger.warning("Skipping query with no relevant ids: %s", p.query[:80])
                continue
            emb = embedder.encode([p.query])[0]

            v_ids = self._retrieve_ids_vector_only(p.query, emb, k)
            h_ids = self._retrieve_ids_hybrid_no_rerank(p.query, emb, k, alpha)
            hr_ids = self._retrieve_ids_hybrid_rerank(p.query, emb, k, alpha, max(20, k))

            for name, ids in (
                ("vector", v_ids),
                ("hybrid", h_ids),
                ("hybrid_rerank", hr_ids),
            ):
                prec, rec = precision_recall_at_k(ids, rel, k)
                mrr = mrr_at_k(ids, rel, k)
                methods[name].append((prec, rec, mrr))

            res = self.retriever.retrieve(
                p.query,
                emb,
                vector_top_k=20,
                bm25_top_k=20,
                alpha=alpha,
                use_reranker=True,
                rerank_top_n=20,
                llm_top_k=5,
            )
            from generation.answer_generator import AnswerGenerator

            gen = AnswerGenerator()
            ans, _ = gen.generate(p.query, res.context_chunks)
            answers.append(ans)
            ctx_for_hallu.append([c.chunk.text for c in res.context_chunks])

        def aggregate(rows: list[tuple[float, float, float]]) -> tuple[float, float, float]:
            if not rows:
                return 0.0, 0.0, 0.0
            arr = np.array(rows, dtype=np.float64)
            return float(arr[:, 0].mean()), float(arr[:, 1].mean()), float(arr[:, 2].mean())

        v_p, v_r, v_m = aggregate(methods["vector"])
        h_p, h_r, h_m = aggregate(methods["hybrid"])
        hr_p, hr_r, hr_m = aggregate(methods["hybrid_rerank"])

        hallu = hallucination_rate(answers, ctx_for_hallu) if answers else 0.0

        comparison = [
            EvaluationResultRow(method="vector", precision_at_k=v_p, recall_at_k=v_r, mrr=v_m),
            EvaluationResultRow(method="hybrid", precision_at_k=h_p, recall_at_k=h_r, mrr=h_m),
            EvaluationResultRow(
                method="hybrid+rerank",
                precision_at_k=hr_p,
                recall_at_k=hr_r,
                mrr=hr_m,
            ),
        ]

        n_eval = len(methods["vector"])
        details: dict = {"num_queries": len(pairs), "evaluated": n_eval}
        if n_eval == 0:
            doc_ids_in_index = sorted({c.doc_id for c in self.retriever.vector_store.chunks})
            details["hint"] = (
                "No chunks matched `relevant_doc_ids` in the eval dataset. "
                "Re-ingest using the optional **doc_id** field so it matches `data/eval_dataset.json` "
                "(e.g. `employee-handbook`, `security-overview`)."
            )
            details["doc_ids_in_index"] = doc_ids_in_index

        return EvaluateResponse(
            precision_at_k=hr_p,
            recall_at_k=hr_r,
            mrr=hr_m,
            hallucination_rate=hallu,
            k=k,
            comparison=comparison,
            details=details,
        )
