"""Dense embeddings via sentence-transformers."""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from config.settings import get_settings

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_name: str | None = None) -> None:
        s = get_settings()
        self.model_name = model_name or s.embedding_model
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)
        emb = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return emb.astype(np.float32)
