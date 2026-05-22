from pathlib import Path

from app.models.chunk import LawDocument
from app.retrieval.ingest.chunker import LawChunker
from app.retrieval.ingest.loader import load_law_documents


def test_load_law_documents_reads_jsonl():
    temp_dir = Path("tests/.tmp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = temp_dir / "law_library.jsonl"
    corpus_path.write_text(
        '{"id": 1, "name": "中华人民共和国民法典 第一条", "content": "为了保护民事主体。"}\n',
        encoding="utf-8",
    )

    try:
        documents = load_law_documents(corpus_path)

        assert documents == [
            LawDocument(
                doc_id="1",
                title="中华人民共和国民法典 第一条",
                content="为了保护民事主体。",
            )
        ]
    finally:
        corpus_path.unlink(missing_ok=True)


def test_chunker_keeps_short_article_as_single_chunk():
    document = LawDocument(
        doc_id="civil_code_1",
        title="中华人民共和国民法典 第一条",
        content="为了保护民事主体的合法权益，制定本法。",
    )
    chunker = LawChunker(long_article_threshold=80, max_chars=120)

    chunks = chunker.chunk_document(document)

    assert len(chunks) == 1
    assert chunks[0].chunk_level == "article"
    assert chunks[0].chunk_id == "civil_code_1"
    assert chunks[0].parent_article_id == "civil_code_1"
    assert chunks[0].full_text.startswith("中华人民共和国民法典 第一条")


def test_chunker_splits_long_article_by_paragraph():
    document = LawDocument(
        doc_id="civil_code_563",
        title="中华人民共和国民法典 第五百六十三条",
        content=(
            "有下列情形之一的，当事人可以解除合同：\n\n"
            "（一）因不可抗力致使不能实现合同目的；\n\n"
            "（二）在履行期限届满前，当事人一方明确表示不履行主要债务。"
        ),
    )
    chunker = LawChunker(long_article_threshold=20, max_chars=40)

    chunks = chunker.chunk_document(document)

    assert [chunk.chunk_level for chunk in chunks] == ["paragraph", "paragraph", "paragraph"]
    assert chunks[0].chunk_id == "civil_code_563_p1"
    assert chunks[1].paragraph_index == 2
    assert all(chunk.parent_article_id == "civil_code_563" for chunk in chunks)
    assert all(chunk.char_count <= 40 for chunk in chunks)
