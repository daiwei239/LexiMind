from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RagRuntime:
    graph: Any
    dense_retriever: Any
    sparse_retriever: Any
    reranker: Any
    answer_agent: Any
