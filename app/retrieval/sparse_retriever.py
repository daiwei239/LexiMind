from __future__ import annotations

from typing import Any

from app.settings import Settings


class SparseRetriever:
    def __init__(self, settings: Settings, es_client: Any | None = None) -> None:
        self.settings = settings
        self.es_client = es_client or self._create_es_client()

    def _create_es_client(self) -> Any:
        from elasticsearch import Elasticsearch

        auth = None
        if self.settings.elasticsearch_user:
            auth = (self.settings.elasticsearch_user, self.settings.elasticsearch_password)
        return Elasticsearch(
            self.settings.elasticsearch_url,
            basic_auth=auth,
            verify_certs=self.settings.elasticsearch_verify_certs,
            request_timeout=self.settings.elasticsearch_request_timeout,
        )

    def search(self, query: str) -> list[dict[str, Any]]:
        body = {
            "size": self.settings.recall_top_k,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["title^3", "content", "full_text"],
                }
            },
        }
        response = self.es_client.search(
            index=self.settings.milvus_collection,
            body=body,
            request_timeout=self.settings.elasticsearch_request_timeout,
        )
        hits = response.get("hits", {}).get("hits", [])
        return [self._normalize_hit(item) for item in hits]

    def _normalize_hit(self, item: dict[str, Any]) -> dict[str, Any]:
        source = item.get("_source", {})
        return {
            "chunk_id": source.get("chunk_id"),
            "parent_article_id": source.get("parent_article_id"),
            "title": source.get("title", ""),
            "content": source.get("content", ""),
            "full_text": source.get("full_text", ""),
            "sparse_score": float(item.get("_score", 0.0)),
            "sources": ["bm25"],
        }
