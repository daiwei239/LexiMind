from __future__ import annotations

from dataclasses import dataclass

from app.graph.rrf import reciprocal_rank_fusion
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
        return {
            "dense_hits": dense_hits,
            "sparse_hits": sparse_hits,
        }


@dataclass
class RRFNode:
    k: int = 60

    def __call__(self, state: LegalRAGState) -> LegalRAGState:
        dense_hits = list(state.get("dense_hits", []))
        sparse_hits = list(state.get("sparse_hits", []))
        candidates = reciprocal_rank_fusion(
            [dense_hits, sparse_hits],
            k=self.k,
        )
        return {"candidates": candidates}


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
        chat_history = list(state.get("chat_history", []))
        evidence = reranked_hits[: self.max_evidence]
        result = self.answer_agent.generate(question, evidence, chat_history=chat_history)
        answer = result.get("answer", "")
        return {
            "answer": answer,
            "sources": result.get("sources", []),
            "prompt_messages": result.get("prompt_messages", []),
            "chat_history": self.answer_agent.append_turn_history(
                chat_history,
                question,
                answer,
            ),
        }
