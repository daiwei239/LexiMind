from app.graph.graph import build_simple_rag_graph
from app.graph.pipeline import graph_thread_config


class StubDenseRetriever:
    def search(self, query: str):
        return [
            {
                "chunk_id": "dense-1",
                "parent_article_id": "article-1",
                "title": "中华人民共和国民法典 第一条",
                "content": "为了保护民事主体。",
                "full_text": "中华人民共和国民法典 第一条\n为了保护民事主体。",
                "dense_score": 0.91,
                "sources": ["dense"],
            }
        ]


class StubSparseRetriever:
    def search(self, query: str):
        return [
            {
                "chunk_id": "sparse-1",
                "parent_article_id": "article-2",
                "title": "中华人民共和国民法典 第二条",
                "content": "民法调整平等主体之间的人身关系。",
                "full_text": "中华人民共和国民法典 第二条\n民法调整平等主体之间的人身关系。",
                "sparse_score": 12.6,
                "sources": ["bm25"],
            }
        ]


def test_simple_rag_graph_builds_query_and_merges_hits():
    graph = build_simple_rag_graph(
        dense_retriever=StubDenseRetriever(),
        sparse_retriever=StubSparseRetriever(),
    )

    assert hasattr(graph, "invoke")
    assert hasattr(graph, "get_graph")
    assert graph.__class__.__name__ == "CompiledStateGraph"

    result = graph.invoke(
        {
            "session_id": "session-1",
            "user_question": "合同解除条件是什么？",
        },
        config=graph_thread_config("session-1"),
    )

    assert result["retrieval_query"] == "合同解除条件是什么？"
    assert len(result["dense_hits"]) == 1
    assert len(result["sparse_hits"]) == 1
    assert [hit["chunk_id"] for hit in result["candidates"]] == ["dense-1", "sparse-1"]
    assert [hit["chunk_id"] for hit in result["reranked_hits"]] == ["dense-1", "sparse-1"]
    assert result["answer"]
    assert len(result["sources"]) == 2
    assert len(result["prompt_messages"]) == 3


def test_simple_rag_graph_deduplicates_hybrid_hits():
    class DuplicateDenseRetriever:
        def search(self, query: str):
            return [
                {
                    "chunk_id": "shared-1",
                    "parent_article_id": "article-1",
                    "title": "中华人民共和国民法典 第五百六十三条",
                    "content": "有下列情形之一的，当事人可以解除合同。",
                    "full_text": "中华人民共和国民法典 第五百六十三条\n有下列情形之一的，当事人可以解除合同。",
                    "dense_score": 0.88,
                    "sources": ["dense"],
                }
            ]

    class DuplicateSparseRetriever:
        def search(self, query: str):
            return [
                {
                    "chunk_id": "shared-1",
                    "parent_article_id": "article-1",
                    "title": "中华人民共和国民法典 第五百六十三条",
                    "content": "有下列情形之一的，当事人可以解除合同。",
                    "full_text": "中华人民共和国民法典 第五百六十三条\n有下列情形之一的，当事人可以解除合同。",
                    "sparse_score": 14.3,
                    "sources": ["bm25"],
                }
            ]

    graph = build_simple_rag_graph(
        dense_retriever=DuplicateDenseRetriever(),
        sparse_retriever=DuplicateSparseRetriever(),
    )

    result = graph.invoke(
        {
            "session_id": "session-2",
            "user_question": "不可抗力能解除合同吗？",
        },
        config=graph_thread_config("session-2"),
    )

    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["chunk_id"] == "shared-1"
    assert set(result["candidates"][0]["sources"]) == {"dense", "bm25"}


def test_simple_rag_graph_applies_reranker_scores():
    class DenseRetriever:
        def search(self, query: str):
            return [
                {
                    "chunk_id": "dense-low",
                    "parent_article_id": "article-low",
                    "title": "合同一般规则",
                    "content": "一般规则。",
                    "full_text": "合同一般规则\n一般规则。",
                    "dense_score": 0.3,
                    "sources": ["dense"],
                }
            ]

    class SparseRetriever:
        def search(self, query: str):
            return [
                {
                    "chunk_id": "sparse-high",
                    "parent_article_id": "article-high",
                    "title": "合同解除",
                    "content": "因不可抗力致使不能实现合同目的。",
                    "full_text": "合同解除\n因不可抗力致使不能实现合同目的。",
                    "sparse_score": 3.0,
                    "sources": ["bm25"],
                }
            ]

    class StubReranker:
        def rerank(self, query: str, hits: list[dict]):
            boosted = []
            for hit in hits:
                item = dict(hit)
                item["rerank_score"] = 0.9 if item["chunk_id"] == "sparse-high" else 0.1
                boosted.append(item)
            return sorted(boosted, key=lambda item: item["rerank_score"], reverse=True)

    graph = build_simple_rag_graph(
        dense_retriever=DenseRetriever(),
        sparse_retriever=SparseRetriever(),
        reranker=StubReranker(),
    )

    result = graph.invoke(
        {
            "session_id": "session-3",
            "user_question": "不可抗力能解除合同吗？",
        },
        config=graph_thread_config("session-3"),
    )

    assert [hit["chunk_id"] for hit in result["reranked_hits"]] == ["sparse-high", "dense-low"]


def test_simple_rag_graph_generates_answer_from_top_two_reranked_hits():
    class DenseRetriever:
        def search(self, query: str):
            return [
                {
                    "chunk_id": "a",
                    "parent_article_id": "article-a",
                    "title": "法条A",
                    "content": "内容A",
                    "full_text": "法条A\n内容A",
                    "dense_score": 0.9,
                    "sources": ["dense"],
                }
            ]

    class SparseRetriever:
        def search(self, query: str):
            return [
                {
                    "chunk_id": "b",
                    "parent_article_id": "article-b",
                    "title": "法条B",
                    "content": "内容B",
                    "full_text": "法条B\n内容B",
                    "sparse_score": 9.0,
                    "sources": ["bm25"],
                },
                {
                    "chunk_id": "c",
                    "parent_article_id": "article-c",
                    "title": "法条C",
                    "content": "内容C",
                    "full_text": "法条C\n内容C",
                    "sparse_score": 8.0,
                    "sources": ["bm25"],
                },
            ]

    class StubReranker:
        def rerank(self, query: str, hits: list[dict]):
            score_map = {"b": 0.95, "c": 0.8, "a": 0.2}
            output = []
            for hit in hits:
                item = dict(hit)
                item["rerank_score"] = score_map[item["chunk_id"]]
                output.append(item)
            return sorted(output, key=lambda item: item["rerank_score"], reverse=True)

    class StubAnswerAgent:
        def generate(self, question: str, evidence: list[dict], chat_history=None):
            return {
                "answer": f"{question} -> {','.join(item['chunk_id'] for item in evidence)}",
                "sources": [{"chunk_id": item["chunk_id"], "title": item["title"]} for item in evidence],
                "prompt_messages": [{"role": "system", "content": "stub"}],
            }

        def append_turn_history(self, chat_history, question, answer):
            updated = list(chat_history or [])
            updated.append({"role": "user", "content": question})
            updated.append({"role": "assistant", "content": answer})
            return updated

    graph = build_simple_rag_graph(
        dense_retriever=DenseRetriever(),
        sparse_retriever=SparseRetriever(),
        reranker=StubReranker(),
        answer_agent=StubAnswerAgent(),
    )

    result = graph.invoke(
        {
            "session_id": "session-4",
            "user_question": "请总结相关法条",
        },
        config=graph_thread_config("session-4"),
    )

    assert result["answer"] == "请总结相关法条 -> b,c"
    assert [item["chunk_id"] for item in result["sources"]] == ["b", "c"]


def test_simple_rag_graph_persists_chat_history_across_turns():
    class StubAnswerAgent:
        def generate(self, question: str, evidence: list[dict], chat_history=None):
            prior = len(chat_history or [])
            return {
                "answer": f"{question}:{prior}",
                "sources": [],
                "prompt_messages": [],
            }

        def append_turn_history(self, chat_history, question, answer):
            updated = list(chat_history or [])
            updated.append({"role": "user", "content": question})
            updated.append({"role": "assistant", "content": answer})
            return updated

    graph = build_simple_rag_graph(
        dense_retriever=StubDenseRetriever(),
        sparse_retriever=StubSparseRetriever(),
        answer_agent=StubAnswerAgent(),
    )
    config = graph_thread_config("session-multi")

    first = graph.invoke(
        {"session_id": "session-multi", "user_question": "第一轮问题"},
        config=config,
    )
    second = graph.invoke(
        {"session_id": "session-multi", "user_question": "那第二轮呢？"},
        config=config,
    )

    assert first["answer"] == "第一轮问题:0"
    assert len(first["chat_history"]) == 2
    assert second["answer"] == "那第二轮呢？:2"
    assert len(second["chat_history"]) == 4
    assert second["chat_history"][0]["content"] == "第一轮问题"
