from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterator

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

    def generate_stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "stream": True,
        }
        if not self.thinking_enabled:
            payload["thinking"] = {"type": "disabled"}

        with self.http_client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode("utf-8")
                if not line.startswith("data:"):
                    continue
                chunk = line[len("data:") :].strip()
                if not chunk or chunk == "[DONE]":
                    if chunk == "[DONE]":
                        break
                    continue
                data = json.loads(chunk)
                delta = data["choices"][0].get("delta", {})
                content = delta.get("content")
                if content:
                    yield content


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

    def build_messages(
        self,
        question: str,
        evidence: list[dict[str, Any]],
        chat_history: list[dict[str, str]] | None = None,
        max_history_messages: int = 6,
    ) -> list[dict[str, str]]:
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

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个严谨的法律问答助手。请优先依据提供的法条证据回答，"
                    "不要编造未提供的法律依据；如果证据不足，请明确说明依据不足。"
                    "若用户问题与上文相关，请结合对话上下文理解指代和追问。"
                ),
            },
            {"role": "system", "content": evidence_block},
        ]

        for item in list(chat_history or [])[-max_history_messages:]:
            role = item.get("role")
            content = item.get("content", "").strip()
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": question})
        return messages

    def build_sources(self, evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
        selected = list(evidence[: self.max_evidence])
        return [
            {
                "chunk_id": item.get("chunk_id"),
                "parent_article_id": item.get("parent_article_id"),
                "title": item.get("title"),
                "content": item.get("content"),
            }
            for item in selected
        ]

    @staticmethod
    def append_turn_history(
        chat_history: list[dict[str, str]] | None,
        question: str,
        answer: str,
    ) -> list[dict[str, str]]:
        updated = list(chat_history or [])
        updated.append({"role": "user", "content": question})
        updated.append({"role": "assistant", "content": answer})
        return updated

    def generate(
        self,
        question: str,
        evidence: list[dict[str, Any]],
        chat_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        selected = list(evidence[: self.max_evidence])
        prompt_messages = self.build_messages(question, selected, chat_history)
        answer = self.client.generate(prompt_messages)
        return {
            "answer": answer,
            "sources": self.build_sources(selected),
            "prompt_messages": prompt_messages,
        }

    def generate_stream(
        self,
        question: str,
        evidence: list[dict[str, Any]],
        chat_history: list[dict[str, str]] | None = None,
    ) -> Iterator[str]:
        selected = list(evidence[: self.max_evidence])
        prompt_messages = self.build_messages(question, selected, chat_history)
        yield from self.client.generate_stream(prompt_messages)
