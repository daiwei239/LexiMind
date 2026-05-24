from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _parse_bool(value: str | bool | None, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(data: dict[str, str], name: str, default: int) -> int:
    value = data.get(name)
    return int(value) if value is not None and value != "" else default


def _env_str(data: dict[str, str], name: str, default: str) -> str:
    value = data.get(name)
    return value if value is not None and value != "" else default


def _env_bool(data: dict[str, str], name: str, default: bool) -> bool:
    return _parse_bool(data.get(name), default)


def load_env_file(env_file: str | Path | None = None) -> dict[str, str]:
    path = Path(env_file or ".env")
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        values[key] = value
    return values


@dataclass
class Settings:
    embedding_backend: str = "sentence_transformers"
    sentence_transformer_model: str = "bge-base-zh-v1.5"
    sentence_transformer_local_files_only: bool = True
    sentence_transformer_batch_size: int = 64
    index_batch_size: int = 500
    normative_cache_min_lines: int = 870000

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"
    deepseek_thinking_enabled: bool = False
    tavily_api_key: str = ""

    vector_backend: str = "milvus"
    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_token: str = ""
    milvus_database: str = "default"
    milvus_collection: str = "civil_code"
    milvus_dimension: int = 768
    milvus_index_type: str = "HNSW"
    milvus_metric_type: str = "COSINE"

    top_k: int = 2
    recall_top_k: int = 3
    rerank_top_k: int = 6
    rrf_k: int = 60
    evidence_budget: int = 2

    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_local_files_only: bool = True
    reranker_max_length: int = 512
    reranker_batch_size: int = 16

    elasticsearch_url: str = "http://127.0.0.1:9200"
    elasticsearch_enabled: bool = True
    elasticsearch_user: str = ""
    elasticsearch_password: str = ""
    elasticsearch_verify_certs: bool = False
    elasticsearch_request_timeout: int = 30
    sparse_retrieval_backend: str = "elasticsearch"

    reranker_backend: str = "cross_encoder"
    hyde_enabled: bool = True
    hyde_max_chars: int = 512

    chunk_max_chars: int = 800
    chunk_long_article_threshold: int = 600

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> "Settings":
        file_values = load_env_file(env_file)
        data = {**file_values, **os.environ}
        return cls(
            embedding_backend=_env_str(data, "EMBEDDING_BACKEND", cls.embedding_backend),
            sentence_transformer_model=_env_str(
                data,
                "SENTENCE_TRANSFORMER_MODEL",
                cls.sentence_transformer_model,
            ),
            sentence_transformer_local_files_only=_env_bool(
                data,
                "SENTENCE_TRANSFORMER_LOCAL_FILES_ONLY",
                cls.sentence_transformer_local_files_only,
            ),
            sentence_transformer_batch_size=_env_int(
                data,
                "SENTENCE_TRANSFORMER_BATCH_SIZE",
                cls.sentence_transformer_batch_size,
            ),
            index_batch_size=_env_int(data, "INDEX_BATCH_SIZE", cls.index_batch_size),
            normative_cache_min_lines=_env_int(
                data,
                "NORMATIVE_CACHE_MIN_LINES",
                cls.normative_cache_min_lines,
            ),
            deepseek_api_key=_env_str(data, "DEEPSEEK_API_KEY", cls.deepseek_api_key),
            deepseek_base_url=_env_str(data, "DEEPSEEK_BASE_URL", cls.deepseek_base_url),
            deepseek_model=_env_str(data, "DEEPSEEK_MODEL", cls.deepseek_model),
            deepseek_thinking_enabled=_env_bool(
                data,
                "DEEPSEEK_THINKING_ENABLED",
                cls.deepseek_thinking_enabled,
            ),
            tavily_api_key=_env_str(data, "TAVILY_API_KEY", cls.tavily_api_key),
            vector_backend=_env_str(data, "VECTOR_BACKEND", cls.vector_backend),
            milvus_uri=_env_str(data, "MILVUS_URI", cls.milvus_uri),
            milvus_token=_env_str(data, "MILVUS_TOKEN", cls.milvus_token),
            milvus_database=_env_str(data, "MILVUS_DATABASE", cls.milvus_database),
            milvus_collection=_env_str(data, "MILVUS_COLLECTION", cls.milvus_collection),
            milvus_dimension=_env_int(data, "MILVUS_DIMENSION", cls.milvus_dimension),
            milvus_index_type=_env_str(data, "MILVUS_INDEX_TYPE", cls.milvus_index_type),
            milvus_metric_type=_env_str(data, "MILVUS_METRIC_TYPE", cls.milvus_metric_type),
            top_k=_env_int(data, "TOP_K", cls.top_k),
            recall_top_k=_env_int(data, "RECALL_TOP_K", cls.recall_top_k),
            rerank_top_k=_env_int(data, "RERANK_TOP_K", cls.rerank_top_k),
            rrf_k=_env_int(data, "RRF_K", cls.rrf_k),
            evidence_budget=_env_int(data, "EVIDENCE_BUDGET", cls.evidence_budget),
            elasticsearch_url=_env_str(
                data,
                "ELASTICSEARCH_URL",
                cls.elasticsearch_url,
            ),
            elasticsearch_enabled=_env_bool(
                data,
                "ELASTICSEARCH_ENABLED",
                cls.elasticsearch_enabled,
            ),
            elasticsearch_user=_env_str(
                data,
                "ELASTICSEARCH_USER",
                cls.elasticsearch_user,
            ),
            elasticsearch_password=_env_str(
                data,
                "ELASTICSEARCH_PASSWORD",
                cls.elasticsearch_password,
            ),
            elasticsearch_verify_certs=_env_bool(
                data,
                "ELASTICSEARCH_VERIFY_CERTS",
                cls.elasticsearch_verify_certs,
            ),
            elasticsearch_request_timeout=_env_int(
                data,
                "ELASTICSEARCH_REQUEST_TIMEOUT",
                cls.elasticsearch_request_timeout,
            ),
            sparse_retrieval_backend=_env_str(
                data,
                "SPARSE_RETRIEVAL_BACKEND",
                cls.sparse_retrieval_backend,
            ),
            reranker_backend=_env_str(data, "RERANKER_BACKEND", cls.reranker_backend),
            reranker_model=_env_str(data, "RERANKER_MODEL", cls.reranker_model),
            reranker_local_files_only=_env_bool(
                data,
                "RERANKER_LOCAL_FILES_ONLY",
                cls.reranker_local_files_only,
            ),
            reranker_max_length=_env_int(data, "RERANKER_MAX_LENGTH", cls.reranker_max_length),
            reranker_batch_size=_env_int(data, "RERANKER_BATCH_SIZE", cls.reranker_batch_size),
            hyde_enabled=_env_bool(data, "HYDE_ENABLED", cls.hyde_enabled),
            hyde_max_chars=_env_int(data, "HYDE_MAX_CHARS", cls.hyde_max_chars),
            chunk_max_chars=_env_int(data, "CHUNK_MAX_CHARS", cls.chunk_max_chars),
            chunk_long_article_threshold=_env_int(
                data,
                "CHUNK_LONG_ARTICLE_THRESHOLD",
                cls.chunk_long_article_threshold,
            ),
        )
