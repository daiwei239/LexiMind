from __future__ import annotations

from dataclasses import dataclass

from app.models.chunk import Chunk, LawDocument


def _split_title(title: str) -> tuple[str, str]:
    parts = title.rsplit(" ", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return title, title


@dataclass
class LawChunker:
    long_article_threshold: int = 600
    max_chars: int = 800

    def chunk_document(self, document: LawDocument) -> list[Chunk]:
        content = document.content.strip()
        if len(content) <= self.long_article_threshold:
            return [self._make_chunk(document, content, "article", None, 1, document.doc_id)]

        paragraphs = [segment.strip() for segment in content.split("\n\n") if segment.strip()]
        if len(paragraphs) <= 1:
            return self._split_long_text(document)

        chunks: list[Chunk] = []
        for index, paragraph in enumerate(paragraphs, start=1):
            for sub_index, text in enumerate(self._wrap_text(paragraph), start=1):
                paragraph_id = f"{document.doc_id}_p{index}"
                chunk_id = paragraph_id if sub_index == 1 else f"{paragraph_id}_{sub_index}"
                chunks.append(
                    self._make_chunk(
                        document,
                        text,
                        "paragraph",
                        index,
                        len(paragraphs),
                        chunk_id,
                    )
                )
        return chunks

    def _split_long_text(self, document: LawDocument) -> list[Chunk]:
        return [
            self._make_chunk(document, text, "paragraph", index, None, f"{document.doc_id}_p{index}")
            for index, text in enumerate(self._wrap_text(document.content.strip()), start=1)
        ]

    def _wrap_text(self, text: str) -> list[str]:
        if len(text) <= self.max_chars:
            return [text]
        return [text[i : i + self.max_chars] for i in range(0, len(text), self.max_chars)]

    def _make_chunk(
        self,
        document: LawDocument,
        content: str,
        chunk_level: str,
        paragraph_index: int | None,
        total_paragraphs: int | None,
        chunk_id: str,
    ) -> Chunk:
        law_name, article_no = _split_title(document.title)
        return Chunk(
            chunk_id=chunk_id,
            parent_article_id=document.doc_id,
            source_type="law_article",
            law_name=law_name,
            article_no=article_no,
            title=document.title,
            content=content,
            full_text=f"{document.title}\n{content}",
            chunk_level=chunk_level,
            paragraph_index=paragraph_index,
            total_paragraphs=total_paragraphs,
            char_count=len(content),
        )
