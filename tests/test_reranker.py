from app.settings import Settings
from app.retrieval.reranker import BiEncoderReranker


class StubEmbedder:
    def encode(self, texts, normalize_embeddings=True):
        mapping = {
            "不可抗力能解除合同吗？": [1.0, 0.0],
            "合同解除\n因不可抗力致使不能实现合同目的。": [0.9, 0.1],
            "合同一般规则\n一般规则。": [0.1, 0.9],
        }
        return [mapping[text] for text in texts]


def test_bi_encoder_reranker_sorts_hits_by_similarity():
    settings = Settings(rerank_top_k=2)
    reranker = BiEncoderReranker(settings=settings, embedder=StubEmbedder())
    hits = [
        {
            "chunk_id": "low",
            "title": "合同一般规则",
            "content": "一般规则。",
            "full_text": "合同一般规则\n一般规则。",
        },
        {
            "chunk_id": "high",
            "title": "合同解除",
            "content": "因不可抗力致使不能实现合同目的。",
            "full_text": "合同解除\n因不可抗力致使不能实现合同目的。",
        },
    ]

    reranked = reranker.rerank("不可抗力能解除合同吗？", hits)

    assert [item["chunk_id"] for item in reranked] == ["high", "low"]
    assert reranked[0]["rerank_score"] > reranked[1]["rerank_score"]
