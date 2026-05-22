from __future__ import annotations

from dataclasses import dataclass

from app.graph.state import LegalRAGState, RetrievalHit


def build_query_node(state: LegalRAGState) -> LegalRAGState:
    question = state.get("user_question", "").strip()
    return {"retrieval_query": question}


@dataclass
class HybridRetrieveNode:
    dense_retriever: object
    sparse_retriever: object

    def __call__(self, state: LegalRAGState) -> LegalRAGState:
        query = state.get("retrieval_query", "").strip()
        dense_hits = list(self.dense_retriever.search(query))
        sparse_hits = list(self.sparse_retriever.search(query))
        candidates = self._merge_hits(dense_hits, sparse_hits)
        return {
            "dense_hits": dense_hits,
            "sparse_hits": sparse_hits,
            "candidates": candidates,
        }

    def _merge_hits(
        self,
        dense_hits: list[RetrievalHit],
        sparse_hits: list[RetrievalHit],
    ) -> list[RetrievalHit]:
        merged: dict[str, RetrievalHit] = {}
        for hit in [*dense_hits, *sparse_hits]:
            chunk_id = hit["chunk_id"]
            if chunk_id not in merged:
                merged[chunk_id] = dict(hit)
                merged[chunk_id]["sources"] = list(hit.get("sources", []))
                continue

            existing = merged[chunk_id]
            for key, value in hit.items():
                if key == "sources":
                    continue
                if key not in existing:
                    existing[key] = value
            existing_sources = set(existing.get("sources", []))
            new_sources = set(hit.get("sources", []))
            existing["sources"] = list(existing_sources | new_sources)
        return list(merged.values())


@dataclass
class RerankNode:
    reranker: object

    def __call__(self, state: LegalRAGState) -> LegalRAGState:
        query = state.get("retrieval_query", "").strip()
        candidates = list(state.get("candidates", []))
        reranked_hits = list(self.reranker.rerank(query, candidates))
        return {"reranked_hits": reranked_hits}


@dataclass
class GenerateAnswerNode:
    answer_agent: object
    max_evidence: int = 2

    def __call__(self, state: LegalRAGState) -> LegalRAGState:
        question = state.get("user_question", "").strip()
        reranked_hits = list(state.get("reranked_hits", []))
        evidence = reranked_hits[: self.max_evidence]
        result = self.answer_agent.generate(question, evidence)
        return {
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "prompt_messages": result.get("prompt_messages", []),
        }
