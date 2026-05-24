from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from app.api.runtime import RagRuntime
from app.api.sse import format_sse
from app.generation.answer_agent import DeepSeekAnswerAgent
from app.graph.graph import build_simple_rag_graph
from app.graph.pipeline import graph_thread_config, run_retrieval_pipeline
from app.retrieval.factory import build_reranker, build_retrievers
from app.settings import Settings

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"


class ChatRequest(BaseModel):
    session_id: str
    question: str


def _chat_response(payload: ChatRequest, result: dict[str, Any]) -> dict[str, Any]:
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
        "chat_history": result.get("chat_history", []),
        "prompt_messages": result.get("prompt_messages", []),
    }


def create_app(graph=None, runtime: RagRuntime | None = None) -> FastAPI:
    app = FastAPI(title="LexRAG Online")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if runtime is None and graph is None:
        settings = Settings.from_env()
        dense_retriever, sparse_retriever = build_retrievers(settings)
        reranker = build_reranker(settings)
        answer_agent = DeepSeekAnswerAgent(settings=settings)
        checkpointer = MemorySaver()
        graph = build_simple_rag_graph(
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            reranker=reranker,
            answer_agent=answer_agent,
            checkpointer=checkpointer,
            rrf_k=settings.rrf_k,
        )
        runtime = RagRuntime(
            graph=graph,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            reranker=reranker,
            answer_agent=answer_agent,
            rrf_k=settings.rrf_k,
        )
    elif runtime is None:
        runtime = None

    app.state.runtime = runtime
    if graph is None and runtime is not None:
        graph = runtime.graph

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/chat")
    def chat(payload: ChatRequest) -> dict[str, Any]:
        result = graph.invoke(
            {
                "session_id": payload.session_id,
                "user_question": payload.question,
            },
            config=graph_thread_config(payload.session_id),
        )
        return _chat_response(payload, result)

    @app.post("/chat/stream")
    def chat_stream(payload: ChatRequest) -> StreamingResponse:
        if runtime is None:
            return StreamingResponse(
                iter([format_sse("error", {"message": "流式接口未配置"})]),
                media_type="text/event-stream",
            )

        config = graph_thread_config(payload.session_id)
        snapshot = runtime.graph.get_state(config)
        prior = snapshot.values if snapshot else {}
        chat_history = list(prior.get("chat_history", []))

        def event_generator():
            try:
                state = run_retrieval_pipeline(
                    {
                        "session_id": payload.session_id,
                        "user_question": payload.question,
                        "chat_history": chat_history,
                    },
                    dense_retriever=runtime.dense_retriever,
                    sparse_retriever=runtime.sparse_retriever,
                    reranker=runtime.reranker,
                    rrf_k=runtime.rrf_k,
                )
                evidence = list(state.get("reranked_hits", []))[: runtime.answer_agent.max_evidence]
                sources = runtime.answer_agent.build_sources(evidence)

                yield format_sse(
                    "meta",
                    {
                        "session_id": payload.session_id,
                        "question": payload.question,
                        "retrieval_query": state.get("retrieval_query", ""),
                        "sources": sources,
                    },
                )

                chunks: list[str] = []
                for token in runtime.answer_agent.generate_stream(
                    payload.question,
                    evidence,
                    chat_history=chat_history,
                ):
                    chunks.append(token)
                    yield format_sse("token", {"text": token})

                answer = "".join(chunks)
                prompt_messages = runtime.answer_agent.build_messages(
                    payload.question,
                    evidence,
                    chat_history=chat_history,
                )
                updated_history = runtime.answer_agent.append_turn_history(
                    chat_history,
                    payload.question,
                    answer,
                )
                runtime.graph.update_state(
                    config,
                    {
                        **state,
                        "answer": answer,
                        "sources": sources,
                        "prompt_messages": prompt_messages,
                        "chat_history": updated_history,
                    },
                )
                yield format_sse("done", {"answer": answer})
            except Exception as exc:  # pragma: no cover - surfaced to client
                yield format_sse("error", {"message": str(exc)})

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/", include_in_schema=False)
    def serve_frontend() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")

    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")

    return app
