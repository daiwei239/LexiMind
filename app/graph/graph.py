from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.generation.answer_agent import DeepSeekAnswerAgent
from app.graph.nodes import GenerateAnswerNode, HybridRetrieveNode, RRFNode, RerankNode, build_query_node
from app.graph.state import LegalRAGState


class _PassthroughReranker:
    def rerank(self, query: str, hits: list[dict]):
        return list(hits)


def build_simple_rag_graph(
    dense_retriever: object,
    sparse_retriever: object,
    reranker: object | None = None,
    answer_agent: object | None = None,
    checkpointer: object | None = None,
    rrf_k: int = 60,
):
    graph = StateGraph(LegalRAGState)
    graph.add_node("build_query", build_query_node)
    graph.add_node(
        "hybrid_retrieve",
        HybridRetrieveNode(
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
        ),
    )
    graph.add_node("rrf_fusion", RRFNode(k=rrf_k))
    graph.add_node("rerank", RerankNode(reranker=reranker or _PassthroughReranker()))
    graph.add_node(
        "generate_answer",
        GenerateAnswerNode(answer_agent=answer_agent or DeepSeekAnswerAgent()),
    )
    graph.add_edge(START, "build_query")
    graph.add_edge("build_query", "hybrid_retrieve")
    graph.add_edge("hybrid_retrieve", "rrf_fusion")
    graph.add_edge("rrf_fusion", "rerank")
    graph.add_edge("rerank", "generate_answer")
    graph.add_edge("generate_answer", END)
    if checkpointer is None:
        checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
