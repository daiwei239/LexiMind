from __future__ import annotations

from app.settings import Settings
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.reranker import BiEncoderReranker
from app.retrieval.sparse_retriever import SparseRetriever


def build_retrievers(settings: Settings) -> tuple[DenseRetriever, SparseRetriever]:
    return DenseRetriever(settings=settings), SparseRetriever(settings=settings)


def build_reranker(settings: Settings) -> BiEncoderReranker:
    return BiEncoderReranker(settings=settings)
