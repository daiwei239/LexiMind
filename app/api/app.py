from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from app.settings import Settings
from app.graph.graph import build_simple_rag_graph
from app.retrieval.factory import build_reranker, build_retrievers


class ChatRequest(BaseModel):
    session_id: str
    question: str


def create_app(graph=None) -> FastAPI:
    app = FastAPI(title="LexRAG Online")

    if graph is None:
        settings = Settings.from_env()
        dense_retriever, sparse_retriever = build_retrievers(settings)
        reranker = build_reranker(settings)
        graph = build_simple_rag_graph(
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            reranker=reranker,
        )

    @app.post("/chat")
    def chat(payload: ChatRequest) -> dict:
        result = graph.invoke(
            {
                "session_id": payload.session_id,
                "user_question": payload.question,
            }
        )
        return {
            "session_id": payload.session_id,
            "question": payload.question,
            "retrieval_query": result.get("retrieval_query", ""),
            "dense_hits": result.get("dense_hits", []),
            "sparse_hits": result.get("sparse_hits", []),
            "candidates": result.get("candidates", []),
            "reranked_hits": result.get("reranked_hits", []),
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "prompt_messages": result.get("prompt_messages", []),
        }

    return app
