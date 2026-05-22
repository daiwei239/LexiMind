from fastapi.testclient import TestClient

from app.api.app import create_app


class StubGraph:
    def invoke(self, state):
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
        }


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
