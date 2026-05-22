from app.settings import Settings
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.sparse_retriever import SparseRetriever


class StubEmbedder:
    def encode(self, texts, normalize_embeddings=True):
        assert texts == ["合同解除条件是什么？"]
        assert normalize_embeddings is True
        return [[0.1, 0.2, 0.3]]


class StubMilvusClient:
    def search(self, **kwargs):
        assert kwargs["collection_name"] == "civil_code"
        assert kwargs["limit"] == 3
        return [
            [
                {
                    "id": "civil_code_563_p1",
                    "distance": 0.91,
                    "entity": {
                        "chunk_id": "civil_code_563_p1",
                        "parent_article_id": "civil_code_563",
                        "title": "中华人民共和国民法典 第五百六十三条",
                        "content": "有下列情形之一的，当事人可以解除合同。",
                        "full_text": "中华人民共和国民法典 第五百六十三条\n有下列情形之一的，当事人可以解除合同。",
                    },
                }
            ]
        ]


class StubElasticsearchClient:
    def search(self, index, body, request_timeout):
        assert index == "civil_code"
        assert body["size"] == 3
        assert request_timeout == 30
        return {
            "hits": {
                "hits": [
                    {
                        "_score": 12.8,
                        "_source": {
                            "chunk_id": "civil_code_563_p2",
                            "parent_article_id": "civil_code_563",
                            "title": "中华人民共和国民法典 第五百六十三条",
                            "content": "因不可抗力致使不能实现合同目的。",
                            "full_text": "中华人民共和国民法典 第五百六十三条\n因不可抗力致使不能实现合同目的。",
                        },
                    }
                ]
            }
        }


def test_dense_retriever_returns_normalized_hits():
    settings = Settings(recall_top_k=3, milvus_collection="civil_code")
    retriever = DenseRetriever(settings=settings, embedder=StubEmbedder(), milvus_client=StubMilvusClient())

    hits = retriever.search("合同解除条件是什么？")

    assert hits == [
        {
            "chunk_id": "civil_code_563_p1",
            "parent_article_id": "civil_code_563",
            "title": "中华人民共和国民法典 第五百六十三条",
            "content": "有下列情形之一的，当事人可以解除合同。",
            "full_text": "中华人民共和国民法典 第五百六十三条\n有下列情形之一的，当事人可以解除合同。",
            "dense_score": 0.91,
            "sources": ["dense"],
        }
    ]


def test_dense_retriever_falls_back_to_metadata_json_fields():
    class MetadataMilvusClient:
        def search(self, **kwargs):
            return [
                [
                    {
                        "id": "normative-1",
                        "distance": 0.77,
                        "entity": {
                            "chunk_id": "normative-1",
                            "title": "《中华人民共和国宪法》第十条",
                            "content": "城市的土地属于国家所有。",
                            "source_layer": "normative",
                            "metadata_json": (
                                '{"law_title":"中华人民共和国宪法",'
                                '"article_no":"十",'
                                '"title":"《中华人民共和国宪法》第十条"}'
                            ),
                        },
                    }
                ]
            ]

    settings = Settings(recall_top_k=3, milvus_collection="civil_code")
    retriever = DenseRetriever(
        settings=settings,
        embedder=StubEmbedder(),
        milvus_client=MetadataMilvusClient(),
    )

    hits = retriever.search("合同解除条件是什么？")

    assert hits == [
        {
            "chunk_id": "normative-1",
            "parent_article_id": "中华人民共和国宪法_十",
            "title": "《中华人民共和国宪法》第十条",
            "content": "城市的土地属于国家所有。",
            "full_text": "《中华人民共和国宪法》第十条\n城市的土地属于国家所有。",
            "dense_score": 0.77,
            "sources": ["dense"],
        }
    ]


def test_sparse_retriever_returns_bm25_hits():
    settings = Settings(
        recall_top_k=3,
        elasticsearch_url="http://127.0.0.1:9200",
        elasticsearch_request_timeout=30,
        milvus_collection="civil_code",
    )
    retriever = SparseRetriever(settings=settings, es_client=StubElasticsearchClient())

    hits = retriever.search("合同解除条件是什么？")

    assert hits == [
        {
            "chunk_id": "civil_code_563_p2",
            "parent_article_id": "civil_code_563",
            "title": "中华人民共和国民法典 第五百六十三条",
            "content": "因不可抗力致使不能实现合同目的。",
            "full_text": "中华人民共和国民法典 第五百六十三条\n因不可抗力致使不能实现合同目的。",
            "sparse_score": 12.8,
            "sources": ["bm25"],
        }
    ]
