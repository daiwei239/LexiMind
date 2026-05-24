from __future__ import annotations

from app.settings import Settings
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.reranker import BiEncoderReranker, CrossEncoderReranker, Reranker
from app.retrieval.sparse_retriever import SparseRetriever


def build_retrievers(settings: Settings) -> tuple[DenseRetriever, SparseRetriever]:
    return DenseRetriever(settings=settings), SparseRetriever(settings=settings)


def build_reranker(settings: Settings) -> Reranker:
    backend = settings.reranker_backend.strip().lower()
    if backend == "cross_encoder":
        return CrossEncoderReranker(settings=settings)
    if backend == "bi_encoder":
        return BiEncoderReranker(settings=settings)
    raise ValueError(
        f"Unsupported reranker backend: {settings.reranker_backend!r}. "
        "Use 'cross_encoder' or 'bi_encoder'."
    )
