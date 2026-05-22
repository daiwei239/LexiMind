"""Ping DeepSeek with a short message (thinking disabled)."""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.generation.answer_agent import DeepSeekClient
from app.settings import Settings


def main() -> None:
    settings = Settings.from_env(ROOT / ".env")
    client = DeepSeekClient(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        model=settings.deepseek_model,
        thinking_enabled=settings.deepseek_thinking_enabled,
    )
    print(f"model: {settings.deepseek_model}")
    print(f"thinking_enabled: {settings.deepseek_thinking_enabled}")
    print("--- 你好 x3 ---")
    times: list[float] = []
    for i in range(3):
        t0 = time.perf_counter()
        answer = client.generate([{"role": "user", "content": "你好"}])
        elapsed_ms = (time.perf_counter() - t0) * 1000
        times.append(elapsed_ms)
        preview = answer.replace("\n", " ")[:100]
        print(f"  第{i + 1}轮: {elapsed_ms:.0f} ms | {preview}")
    print(
        f"平均: {statistics.mean(times):.0f} ms  "
        f"min={min(times):.0f} max={max(times):.0f}"
    )


if __name__ == "__main__":
    main()
