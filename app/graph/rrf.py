from __future__ import annotations

from app.graph.state import RetrievalHit


def merge_hit_metadata(existing: RetrievalHit, incoming: RetrievalHit) -> RetrievalHit:
    merged = dict(existing)
    for key, value in incoming.items():
        if key == "sources":
            continue
        if key not in merged:
            merged[key] = value
    existing_sources = set(merged.get("sources", []))
    new_sources = set(incoming.get("sources", []))
    merged["sources"] = list(existing_sources | new_sources)
    return merged


def reciprocal_rank_fusion(
    rank_lists: list[list[RetrievalHit]],
    *,
    k: int = 60,
) -> list[RetrievalHit]:
    scores: dict[str, float] = {}
    hits_by_id: dict[str, RetrievalHit] = {}

    for hits in rank_lists:
        for rank, hit in enumerate(hits, start=1):
            chunk_id = hit["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            if chunk_id not in hits_by_id:
                stored = dict(hit)
                stored["sources"] = list(hit.get("sources", []))
                hits_by_id[chunk_id] = stored
            else:
                hits_by_id[chunk_id] = merge_hit_metadata(hits_by_id[chunk_id], hit)

    ordered_ids = sorted(
        scores.keys(),
        key=lambda chunk_id: (-scores[chunk_id], chunk_id),
    )
    fused: list[RetrievalHit] = []
    for chunk_id in ordered_ids:
        item = dict(hits_by_id[chunk_id])
        item["rrf_score"] = scores[chunk_id]
        fused.append(item)
    return fused
