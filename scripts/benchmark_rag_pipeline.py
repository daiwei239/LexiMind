"""Benchmark RAG pipeline step timings (retrieval + optional generation)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.generation.answer_agent import DeepSeekAnswerAgent
from app.graph.nodes import HybridRetrieveNode, RerankNode, build_query_node
from app.retrieval.factory import build_reranker, build_retrievers
from app.settings import Settings


# 依据 data/samples 中「分期车/保险/扣车」类咨询生成的测试问题
TEST_QUESTION = (
    "分期付款买车后，第二年保险没有在分期公司购买，但月供一直按时还，"
    "分期公司有权扣车吗？如果强行扣车应该怎么维权？"
)


def ms(seconds: float) -> float:
    return round(seconds * 1000, 2)


def main() -> None:
    settings = Settings.from_env(ROOT / ".env")
    print("=" * 60)
    print("LexRAG 流水线耗时测试")
    print("=" * 60)
    print(f"MILVUS_COLLECTION: {settings.milvus_collection}")
    print(f"RECALL_TOP_K: {settings.recall_top_k}  RERANK_TOP_K: {settings.rerank_top_k}")
    print(f"EVIDENCE_BUDGET / max evidence: {settings.evidence_budget}")
    print(f"测试问题:\n  {TEST_QUESTION}\n")

    timings: dict[str, float] = {}

    # --- 冷启动：加载模型与客户端 ---
    t0 = time.perf_counter()
    dense, sparse = build_retrievers(settings)
    reranker = build_reranker(settings)
    timings["init_retrievers_reranker"] = time.perf_counter() - t0

    # --- 1. build_query ---
    state = {"user_question": TEST_QUESTION}
    t0 = time.perf_counter()
    state.update(build_query_node(state))
    timings["build_query"] = time.perf_counter() - t0
    query = state["retrieval_query"]

    # --- 2a. dense: embedding ---
    t0 = time.perf_counter()
    vectors = dense.embedder.encode([query], normalize_embeddings=True)
    timings["dense_embed"] = time.perf_counter() - t0

    # --- 2b. dense: milvus search ---
    t0 = time.perf_counter()
    milvus_results = dense.milvus_client.search(
        collection_name=settings.milvus_collection,
        data=vectors,
        limit=settings.recall_top_k,
        output_fields=["chunk_id", "title", "content", "source_layer", "metadata_json"],
    )
    timings["dense_milvus_search"] = time.perf_counter() - t0
    dense_hits = [dense._normalize_hit(item) for item in milvus_results[0]]

    # --- 3. sparse: elasticsearch ---
    t0 = time.perf_counter()
    sparse_hits = sparse.search(query)
    timings["sparse_elasticsearch"] = time.perf_counter() - t0

    # --- 4. merge ---
    hybrid = HybridRetrieveNode(dense_retriever=dense, sparse_retriever=sparse)
    t0 = time.perf_counter()
    candidates = hybrid._merge_hits(dense_hits, sparse_hits)
    timings["merge_candidates"] = time.perf_counter() - t0

    # --- 5. rerank ---
    t0 = time.perf_counter()
    reranked = reranker.rerank(query, candidates)
    timings["bi_encoder_rerank"] = time.perf_counter() - t0
    evidence = reranked[: settings.evidence_budget]

    retrieval_total = (
        timings["build_query"]
        + timings["dense_embed"]
        + timings["dense_milvus_search"]
        + timings["sparse_elasticsearch"]
        + timings["merge_candidates"]
        + timings["bi_encoder_rerank"]
    )

    print("-" * 60)
    print("检索子流程 (ms)")
    print("-" * 60)
    for key in [
        "init_retrievers_reranker",
        "build_query",
        "dense_embed",
        "dense_milvus_search",
        "sparse_elasticsearch",
        "merge_candidates",
        "bi_encoder_rerank",
    ]:
        label = key.replace("_", " ")
        print(f"  {label:<28} {ms(timings[key]):>10} ms")
    print(f"  {'检索合计(不含冷启动)':<28} {ms(retrieval_total):>10} ms")

    print("-" * 60)
    print(f"候选合并后: {len(candidates)} 条 | 重排后: {len(reranked)} 条 | 证据: {len(evidence)} 条")
    for i, hit in enumerate(evidence, 1):
        title = (hit.get("title") or "")[:60]
        score = hit.get("rerank_score", hit.get("dense_score", hit.get("sparse_score")))
        print(f"  [{i}] score={score:.4f} | {title}")

    # --- 6. DeepSeek 生成（可选）---
    if not settings.deepseek_api_key:
        print("\n[跳过] DEEPSEEK_API_KEY 未配置，不测试生成阶段")
        return

    agent = DeepSeekAnswerAgent(settings=settings)
    t0 = time.perf_counter()
    t_build = time.perf_counter()
    messages = agent.build_messages(TEST_QUESTION, evidence)
    timings["build_prompt"] = time.perf_counter() - t_build

    t_llm = time.perf_counter()
    answer = agent.client.generate(messages)
    timings["deepseek_generate"] = time.perf_counter() - t_llm
    timings["generate_total"] = time.perf_counter() - t0

    full_total = retrieval_total + timings["generate_total"]

    print("-" * 60)
    print("生成子流程 (ms)")
    print("-" * 60)
    print(f"  {'build_prompt':<28} {ms(timings['build_prompt']):>10} ms")
    print(f"  {'deepseek_generate':<28} {ms(timings['deepseek_generate']):>10} ms")
    print(f"  {'generate_total':<28} {ms(timings['generate_total']):>10} ms")
    print("-" * 60)
    print(f"  {'全流程(不含冷启动)':<28} {ms(full_total):>10} ms")
    print(f"  {'含冷启动':<28} {ms(full_total + timings['init_retrievers_reranker']):>10} ms")
    print("-" * 60)
    print("回答摘要 (前 300 字):")
    print(answer[:300] + ("..." if len(answer) > 300 else ""))

    report = {
        "question": TEST_QUESTION,
        "timings_ms": {k: ms(v) for k, v in timings.items()},
        "retrieval_total_ms": ms(retrieval_total),
        "evidence_count": len(evidence),
        "evidence_titles": [h.get("title") for h in evidence],
    }
    out = ROOT / "scripts" / "benchmark_last_run.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已写入: {out}")


if __name__ == "__main__":
    main()
