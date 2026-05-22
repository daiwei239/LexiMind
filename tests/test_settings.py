from pathlib import Path

from app.settings import Settings


def test_settings_reads_chunking_and_retrieval_env(monkeypatch):
    monkeypatch.setenv("MILVUS_COLLECTION", "civil_code")
    monkeypatch.setenv("RECALL_TOP_K", "7")
    monkeypatch.setenv("EVIDENCE_BUDGET", "3")
    monkeypatch.setenv("CHUNK_MAX_CHARS", "120")
    monkeypatch.setenv("CHUNK_LONG_ARTICLE_THRESHOLD", "80")

    settings = Settings.from_env()

    assert settings.milvus_collection == "civil_code"
    assert settings.recall_top_k == 7
    assert settings.evidence_budget == 3
    assert settings.chunk_max_chars == 120
    assert settings.chunk_long_article_threshold == 80


def test_settings_loads_env_file(monkeypatch):
    temp_dir = Path("tests/.tmp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    env_path = temp_dir / ".env"
    env_path.write_text(
        "\n".join(
            [
                "EMBEDDING_BACKEND=sentence_transformers",
                "SENTENCE_TRANSFORMER_MODEL=bge-base-zh-v1.5",
                "SENTENCE_TRANSFORMER_LOCAL_FILES_ONLY=true",
                "DEEPSEEK_MODEL=deepseek-v4-flash",
                "MILVUS_URI=http://127.0.0.1:19530",
                "ELASTICSEARCH_ENABLED=true",
                "HYDE_ENABLED=true",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)

    try:
        settings = Settings.from_env(env_file=env_path)

        assert settings.embedding_backend == "sentence_transformers"
        assert settings.sentence_transformer_model == "bge-base-zh-v1.5"
        assert settings.sentence_transformer_local_files_only is True
        assert settings.deepseek_model == "deepseek-v4-flash"
        assert settings.milvus_uri == "http://127.0.0.1:19530"
        assert settings.elasticsearch_enabled is True
        assert settings.hyde_enabled is True
    finally:
        env_path.unlink(missing_ok=True)
