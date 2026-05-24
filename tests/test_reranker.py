from unittest.mock import patch

from app.settings import Settings
from app.retrieval.factory import build_reranker
from app.retrieval.reranker import BiEncoderReranker, CrossEncoderReranker


class StubEmbedder:
    def encode(self, texts, normalize_embeddings=True):
        mapping = {
            "不可抗力能解除合同吗？": [1.0, 0.0],
            "合同解除\n因不可抗力致使不能实现合同目的。": [0.9, 0.1],
            "合同一般规则\n一般规则。": [0.1, 0.9],
        }
        return [mapping[text] for text in texts]


class StubCrossEncoder:
    def predict(self, pairs, batch_size=16, show_progress_bar=False):
        scores = []
        for query, doc in pairs:
            if "不可抗力" in doc:
                scores.append(0.95)
            else:
                scores.append(0.15)
        return scores


def test_bi_encoder_reranker_sorts_hits_by_similarity():
    settings = Settings(rerank_top_k=2, reranker_backend="bi_encoder")
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


def test_cross_encoder_reranker_sorts_hits_by_relevance():
    settings = Settings(rerank_top_k=2, reranker_backend="cross_encoder")
    reranker = CrossEncoderReranker(settings=settings, model=StubCrossEncoder())
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


def test_build_reranker_selects_cross_encoder_by_default():
    settings = Settings()

    with patch.object(CrossEncoderReranker, "_create_model", return_value=StubCrossEncoder()):
        reranker = build_reranker(settings)

    assert isinstance(reranker, CrossEncoderReranker)


def test_build_reranker_selects_bi_encoder_when_configured():
    settings = Settings(reranker_backend="bi_encoder")

    with patch.object(BiEncoderReranker, "_create_embedder", return_value=StubEmbedder()):
        reranker = build_reranker(settings)

    assert isinstance(reranker, BiEncoderReranker)
