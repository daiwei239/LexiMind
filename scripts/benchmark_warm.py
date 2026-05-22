"""Warm-run benchmark: models already loaded, multiple iterations."""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.generation.answer_agent import DeepSeekAnswerAgent
from app.graph.nodes import HybridRetrieveNode, build_query_node
from app.retrieval.factory import build_reranker, build_retrievers
from app.settings import Settings

TEST_QUESTION = (
    "分期付款买车后，第二年保险没有在分期公司购买，但月供一直按时还，"
    "分期公司有权扣车吗？如果强行扣车应该怎么维权？"
)
WARMUP_ROUNDS = 1
BENCH_ROUNDS = 3


def ms(seconds: float) -> float:
    return round(seconds * 1000, 2)


def run_retrieval_once(dense, sparse, reranker, hybrid, settings, query: str) -> dict[str, float]:
    t: dict[str, float] = {}

    t0 = time.perf_counter()
    vectors = dense.embedder.encode([query], normalize_embeddings=True)
    t["dense_embed"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    results = dense.milvus_client.search(
        collection_name=settings.milvus_collection,
        data=vectors,
        limit=settings.recall_top_k,
        output_fields=["chunk_id", "title", "content", "source_layer", "metadata_json"],
    )
    t["dense_milvus_search"] = time.perf_counter() - t0
    dense_hits = [dense._normalize_hit(item) for item in results[0]]

    t0 = time.perf_counter()
    sparse_hits = sparse.search(query)
    t["sparse_elasticsearch"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    candidates = hybrid._merge_hits(dense_hits, sparse_hits)
    t["merge_candidates"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    reranked = reranker.rerank(query, candidates)
    t["bi_encoder_rerank"] = time.perf_counter() - t0

    t["retrieval_total"] = sum(
        t[k]
        for k in (
            "dense_embed",
            "dense_milvus_search",
            "sparse_elasticsearch",
            "merge_candidates",
            "bi_encoder_rerank",
        )
    )
    t["_evidence"] = reranked[: settings.evidence_budget]
    return t


def main() -> None:
    settings = Settings.from_env(ROOT / ".env")
    print("=" * 60)
    print("热加载后 RAG 耗时测试")
    print("=" * 60)
    print(f"collection: {settings.milvus_collection}")
    print(f"问题: {TEST_QUESTION}\n")

    print("正在加载模型（仅此一次）...")
    t_load = time.perf_counter()
    dense, sparse = build_retrievers(settings)
    reranker = build_reranker(settings)
    hybrid = HybridRetrieveNode(dense_retriever=dense, sparse_retriever=sparse)
    load_ms = ms(time.perf_counter() - t_load)
    print(f"加载完成: {load_ms} ms\n")

    state = {"user_question": TEST_QUESTION}
    state.update(build_query_node(state))
    query = state["retrieval_query"]

    for i in range(WARMUP_ROUNDS):
        run_retrieval_once(dense, sparse, reranker, hybrid, settings, query)
    print(f"预热 {WARMUP_ROUNDS} 轮完成，开始正式计时 ({BENCH_ROUNDS} 轮)...\n")

    keys = [
        "dense_embed",
        "dense_milvus_search",
        "sparse_elasticsearch",
        "merge_candidates",
        "bi_encoder_rerank",
        "retrieval_total",
    ]
    samples: dict[str, list[float]] = {k: [] for k in keys}
    last_evidence = []

    for i in range(BENCH_ROUNDS):
        row = run_retrieval_once(dense, sparse, reranker, hybrid, settings, query)
        last_evidence = row.pop("_evidence", [])
        for k in keys:
            samples[k].append(row[k])
        print(f"  第{i+1}轮检索合计: {ms(row['retrieval_total'])} ms")

    print("\n" + "-" * 60)
    print(f"检索耗时 (ms) — {BENCH_ROUNDS} 轮平均 / 最小 / 最大")
    print("-" * 60)
    for k in keys:
        vals = [ms(v) for v in samples[k]]
        avg = statistics.mean(vals)
        print(
            f"  {k.replace('_', ' '):<26} "
            f"avg={avg:>8.2f}  min={min(vals):>8.2f}  max={max(vals):>8.2f}"
        )

    print("-" * 60)
    print("证据 (最后一轮 Top-2):")
    for i, hit in enumerate(last_evidence, 1):
        title = (hit.get("title") or "")[:55]
        print(f"  [{i}] rerank={hit.get('rerank_score', 0):.4f} | {title}")

    gen_ms = None
    if settings.deepseek_api_key:
        agent = DeepSeekAnswerAgent(settings=settings)
        evidence = last_evidence
        gen_times = []
        for i in range(BENCH_ROUNDS):
            t0 = time.perf_counter()
            messages = agent.build_messages(TEST_QUESTION, evidence)
            t_build = time.perf_counter() - t0
            t1 = time.perf_counter()
            agent.client.generate(messages)
            t_llm = time.perf_counter() - t1
            total = t_build + t_llm
            gen_times.append(total)
            print(f"  第{i+1}轮生成: {ms(total)} ms (prompt {ms(t_build)} + llm {ms(t_llm)})")
        gen_avg = statistics.mean([ms(t) for t in gen_times])
        gen_ms = gen_avg
        print(f"\n生成平均: {gen_avg:.2f} ms")
        ret_avg = statistics.mean([ms(v) for v in samples["retrieval_total"]])
        print(f"全流程平均(检索+生成): {ret_avg + gen_avg:.2f} ms")
    else:
        print("\n[跳过] 未配置 DEEPSEEK_API_KEY")

    report = {
        "mode": "warm",
        "bench_rounds": BENCH_ROUNDS,
        "load_ms": load_ms,
        "retrieval_ms": {
            k: {
                "avg": round(statistics.mean(samples[k]) * 1000, 2),
                "min": round(min(samples[k]) * 1000, 2),
                "max": round(max(samples[k]) * 1000, 2),
            }
            for k in keys
        },
        "generate_avg_ms": gen_ms,
    }
    out = ROOT / "scripts" / "benchmark_warm_last_run.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告: {out}")


if __name__ == "__main__":
    main()
