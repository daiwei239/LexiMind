from __future__ import annotations

import json
from typing import Any

from app.settings import Settings


class DenseRetriever:
    def __init__(
        self,
        settings: Settings,
        embedder: Any | None = None,
        milvus_client: Any | None = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder or self._create_embedder()
        self.milvus_client = milvus_client or self._create_milvus_client()

    def _create_embedder(self) -> Any:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(
            self.settings.sentence_transformer_model,
            local_files_only=self.settings.sentence_transformer_local_files_only,
        )

    def _create_milvus_client(self) -> Any:
        from pymilvus import MilvusClient

        token = self.settings.milvus_token or None
        return MilvusClient(uri=self.settings.milvus_uri, token=token, db_name=self.settings.milvus_database)

    def search(self, query: str) -> list[dict[str, Any]]:
        vectors = self.embedder.encode([query], normalize_embeddings=True)
        results = self.milvus_client.search(
            collection_name=self.settings.milvus_collection,
            data=vectors,
            limit=self.settings.recall_top_k,
            output_fields=["chunk_id", "title", "content", "source_layer", "metadata_json"],
        )
        return [self._normalize_hit(item) for item in results[0]]

    def _normalize_hit(self, item: Any) -> dict[str, Any]:
        entity = item.get("entity", item.get("_source", {})) if isinstance(item, dict) else {}
        metadata = self._load_metadata(entity.get("metadata_json"))
        title = entity.get("title") or metadata.get("title", "")
        content = entity.get("content", "")
        law_title = metadata.get("law_title", "")
        article_no = metadata.get("article_no", "")
        parent_article_id = entity.get("parent_article_id")
        if not parent_article_id and law_title and article_no:
            parent_article_id = f"{law_title}_{article_no}"
        full_text = entity.get("full_text")
        if not full_text:
            full_text = f"{title}\n{content}".strip()
        return {
            "chunk_id": entity.get("chunk_id", item.get("id")),
            "parent_article_id": parent_article_id,
            "title": title,
            "content": content,
            "full_text": full_text,
            "dense_score": float(item.get("distance", 0.0)),
            "sources": ["dense"],
        }

    @staticmethod
    def _load_metadata(raw: Any) -> dict[str, Any]:
        if not raw:
            return {}
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {}
        return {}
