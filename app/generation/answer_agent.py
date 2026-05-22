from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from app.settings import Settings


class DeepSeekClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        thinking_enabled: bool = False,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.thinking_enabled = thinking_enabled
        self.http_client = http_client or httpx.Client(timeout=60.0)

    def generate(self, messages: list[dict[str, str]]) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "stream": False,
        }
        if not self.thinking_enabled:
            payload["thinking"] = {"type": "disabled"}

        response = self.http_client.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        payload = response.json()
        return payload["choices"][0]["message"]["content"].strip()


@dataclass
class DeepSeekAnswerAgent:
    client: Any | None = None
    max_evidence: int = 2
    settings: Settings | None = None

    def __post_init__(self) -> None:
        self.settings = self.settings or Settings.from_env()
        self.client = self.client or DeepSeekClient(
            api_key=self.settings.deepseek_api_key,
            base_url=self.settings.deepseek_base_url,
            model=self.settings.deepseek_model,
            thinking_enabled=self.settings.deepseek_thinking_enabled,
        )

    def build_messages(self, question: str, evidence: list[dict[str, Any]]) -> list[dict[str, str]]:
        selected = list(evidence[: self.max_evidence])
        evidence_lines = []
        for index, item in enumerate(selected, start=1):
            title = item.get("title", "未命名法条")
            content = item.get("content", "")
            evidence_lines.append(
                f"[证据{index}]\n标题：{title}\n内容：{content}"
            )

        if not evidence_lines:
            evidence_block = "当前没有可用的检索证据，请明确说明依据不足。"
        else:
            evidence_block = "以下是检索到的法条依据：\n\n" + "\n\n".join(evidence_lines)

        return [
            {
                "role": "system",
                "content": (
                    "你是一个严谨的法律问答助手。请优先依据提供的法条证据回答，"
                    "不要编造未提供的法律依据；如果证据不足，请明确说明依据不足。"
                ),
            },
            {"role": "system", "content": evidence_block},
            {"role": "user", "content": question},
        ]

    def generate(self, question: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
        selected = list(evidence[: self.max_evidence])
        prompt_messages = self.build_messages(question, selected)
        answer = self.client.generate(prompt_messages)
        sources = [
            {
                "chunk_id": item.get("chunk_id"),
                "parent_article_id": item.get("parent_article_id"),
                "title": item.get("title"),
                "content": item.get("content"),
            }
            for item in selected
        ]
        return {
            "answer": answer,
            "sources": sources,
            "prompt_messages": prompt_messages,
        }
