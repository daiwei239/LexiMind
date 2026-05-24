from app.graph.rrf import reciprocal_rank_fusion


def _hit(chunk_id: str, source: str, rank_hint: float = 0.0) -> dict:
    return {
        "chunk_id": chunk_id,
        "title": chunk_id,
        "content": chunk_id,
        "sources": [source],
        "score_hint": rank_hint,
    }


def test_reciprocal_rank_fusion_boosts_overlap():
    dense_hits = [_hit("a", "dense"), _hit("b", "dense")]
    sparse_hits = [_hit("b", "bm25"), _hit("c", "bm25")]

    fused = reciprocal_rank_fusion([dense_hits, sparse_hits], k=60)

    assert [hit["chunk_id"] for hit in fused] == ["b", "a", "c"]
    assert fused[0]["rrf_score"] > fused[1]["rrf_score"]
    assert set(fused[0]["sources"]) == {"dense", "bm25"}


def test_reciprocal_rank_fusion_deduplicates_shared_chunk():
    shared = {
        "chunk_id": "shared-1",
        "title": "法条",
        "content": "内容",
        "dense_score": 0.9,
        "sources": ["dense"],
    }
    sparse = {
        "chunk_id": "shared-1",
        "title": "法条",
        "content": "内容",
        "sparse_score": 12.0,
        "sources": ["bm25"],
    }

    fused = reciprocal_rank_fusion([[shared], [sparse]], k=60)

    assert len(fused) == 1
    assert fused[0]["chunk_id"] == "shared-1"
    assert set(fused[0]["sources"]) == {"dense", "bm25"}
    assert fused[0]["rrf_score"] == 2 / 61
