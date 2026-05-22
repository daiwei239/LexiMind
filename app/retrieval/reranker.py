from __future__ import annotations

from typing import Any

import numpy as np

from app.settings import Settings


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

        texts = [query] + [hit.get("full_text") or f"{hit.get('title', '')}\n{hit.get('content', '')}" for hit in hits]
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
