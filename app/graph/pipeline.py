from __future__ import annotations

from app.graph.nodes import HybridRetrieveNode, RRFNode, RerankNode, build_query_node
from app.graph.state import LegalRAGState


def run_retrieval_pipeline(
    state: LegalRAGState,
    *,
    dense_retriever: object,
    sparse_retriever: object,
    reranker: object,
    rrf_k: int = 60,
) -> LegalRAGState:
    merged: LegalRAGState = dict(state)
    merged.update(build_query_node(merged))
    merged.update(
        HybridRetrieveNode(
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
        )(merged)
    )
    merged.update(RRFNode(k=rrf_k)(merged))
    merged.update(RerankNode(reranker=reranker)(merged))
    return merged


def graph_thread_config(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}
