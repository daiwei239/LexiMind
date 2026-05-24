from fastapi.testclient import TestClient

from app.api.app import create_app
from app.api.runtime import RagRuntime
from app.graph.graph import build_simple_rag_graph


class StubGraph:
    def invoke(self, state, config=None):
        return {
            **state,
            "retrieval_query": state["user_question"],
            "dense_hits": [],
            "sparse_hits": [],
            "candidates": [
                {
                    "chunk_id": "civil_code_563_p1",
                    "parent_article_id": "civil_code_563",
                    "title": "中华人民共和国民法典 第五百六十三条",
                    "content": "有下列情形之一的，当事人可以解除合同。",
                    "full_text": "中华人民共和国民法典 第五百六十三条\n有下列情形之一的，当事人可以解除合同。",
                    "sources": ["dense"],
                }
            ],
            "answer": "示例回答",
            "sources": [],
            "chat_history": [],
            "prompt_messages": [],
        }


class StreamDenseRetriever:
    def search(self, query: str):
        return [
            {
                "chunk_id": "dense-1",
                "title": "法条A",
                "content": "内容A",
                "sources": ["dense"],
            }
        ]


class StreamSparseRetriever:
    def search(self, query: str):
        return []


class StreamAnswerAgent:
    max_evidence = 2

    def build_sources(self, evidence):
        return [{"chunk_id": item["chunk_id"], "title": item["title"], "content": item["content"]} for item in evidence]

    def build_messages(self, question, evidence, chat_history=None):
        return [{"role": "user", "content": question}]

    def append_turn_history(self, chat_history, question, answer):
        updated = list(chat_history or [])
        updated.append({"role": "user", "content": question})
        updated.append({"role": "assistant", "content": answer})
        return updated

    def generate_stream(self, question, evidence, chat_history=None):
        yield "流式"
        yield "回答"


def test_chat_endpoint_returns_graph_results():
    app = create_app(graph=StubGraph())
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={"session_id": "session-1", "question": "合同解除条件是什么？"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == "session-1"
    assert payload["retrieval_query"] == "合同解除条件是什么？"
    assert payload["candidates"][0]["chunk_id"] == "civil_code_563_p1"


def test_frontend_routes_are_available():
    app = create_app(graph=StubGraph())
    client = TestClient(app)

    assert client.get("/").status_code == 200
    assert client.get("/assets/app.js").status_code == 200
    assert client.get("/health").json() == {"status": "ok"}


def test_chat_stream_endpoint_returns_sse_tokens():
    graph = build_simple_rag_graph(
        dense_retriever=StreamDenseRetriever(),
        sparse_retriever=StreamSparseRetriever(),
        answer_agent=StreamAnswerAgent(),
    )
    runtime = RagRuntime(
        graph=graph,
        dense_retriever=StreamDenseRetriever(),
        sparse_retriever=StreamSparseRetriever(),
        reranker=type("R", (), {"rerank": lambda self, q, hits: hits})(),
        answer_agent=StreamAnswerAgent(),
        rrf_k=60,
    )
    app = create_app(runtime=runtime)
    client = TestClient(app)

    with client.stream(
        "POST",
        "/chat/stream",
        json={"session_id": "stream-1", "question": "合同问题"},
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    assert "event: meta" in body
    assert "event: token" in body
    assert "流式" in body
    assert "event: done" in body
    assert "流式回答" in body
