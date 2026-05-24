from __future__ import annotations

from typing import Any, TypedDict


class RetrievalHit(TypedDict, total=False):
    chunk_id: str
    parent_article_id: str
    title: str
    content: str
    full_text: str
    dense_score: float
    sparse_score: float
    rerank_score: float
    rrf_score: float
    sources: list[str]


class LegalRAGState(TypedDict, total=False):
    session_id: str
    user_question: str
    chat_history: list[dict[str, str]]
    retrieval_query: str
    dense_hits: list[RetrievalHit]
    sparse_hits: list[RetrievalHit]
    candidates: list[RetrievalHit]
    reranked_hits: list[RetrievalHit]
    answer: str
    sources: list[dict[str, Any]]
    prompt_messages: list[dict[str, str]]
    metadata: dict[str, Any]
