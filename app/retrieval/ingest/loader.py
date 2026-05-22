from __future__ import annotations

import json
from pathlib import Path

from app.models.chunk import LawDocument


def load_law_documents(path: str | Path) -> list[LawDocument]:
    documents: list[LawDocument] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            documents.append(
                LawDocument(
                    doc_id=str(payload["id"]),
                    title=payload["name"].strip(),
                    content=payload["content"].strip(),
                )
            )
    return documents
