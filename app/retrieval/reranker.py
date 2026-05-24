from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from app.settings import Settings


def build_document_text(hit: dict[str, Any]) -> str:
    return hit.get("full_text") or f"{hit.get('title', '')}\n{hit.get('content', '')}"


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]: ...


class BiEncoderReranker:
    def __init__(self, settings: Settings, embedder: Any | None = None) -> None:
        self.settings = settings
        self.embedder = embedder or self._create_embedder()

    def _create_embedder(self) -> Any:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(
            self.settings.sentence_transformer_model,
            local_files_only=self.settings.sentence_transformer_local_files_only,
        )

    def rerank(self, query: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not hits:
            return []

        texts = [query] + [build_document_text(hit) for hit in hits]
        embeddings = self.embedder.encode(texts, normalize_embeddings=True)
        query_vector = np.array(embeddings[0], dtype=float)
        doc_vectors = [np.array(vector, dtype=float) for vector in embeddings[1:]]

        scored_hits: list[dict[str, Any]] = []
        for hit, doc_vector in zip(hits, doc_vectors):
            item = dict(hit)
            item["rerank_score"] = float(np.dot(query_vector, doc_vector))
            scored_hits.append(item)

        scored_hits.sort(key=lambda item: item["rerank_score"], reverse=True)
        return scored_hits[: self.settings.rerank_top_k]


class CrossEncoderReranker:
    def __init__(self, settings: Settings, model: Any | None = None) -> None:
        self.settings = settings
        self.model = model or self._create_model()

    def _create_model(self) -> Any:
        from sentence_transformers import CrossEncoder

        automodel_args: dict[str, Any] = {}
        if self.settings.reranker_local_files_only:
            automodel_args["local_files_only"] = True

        return CrossEncoder(
            self.settings.reranker_model,
            max_length=self.settings.reranker_max_length,
            automodel_args=automodel_args or None,
        )

    def rerank(self, query: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not hits:
            return []

        pairs = [[query, build_document_text(hit)] for hit in hits]
        scores = self.model.predict(
            pairs,
            batch_size=self.settings.reranker_batch_size,
            show_progress_bar=False,
        )

        scored_hits: list[dict[str, Any]] = []
        for hit, score in zip(hits, scores):
            item = dict(hit)
            item["rerank_score"] = float(score)
            scored_hits.append(item)

        scored_hits.sort(key=lambda item: item["rerank_score"], reverse=True)
        return scored_hits[: self.settings.rerank_top_k]
