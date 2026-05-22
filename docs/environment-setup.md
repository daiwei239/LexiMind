# Environment Setup

This project targets the `torch_gpu` conda environment for local embedding and reranking.

## 1. Activate the Conda Environment

```powershell
conda activate torch_gpu
```

## 2. Install Python Dependencies

```powershell
pip install -r requirements.txt
```

If your local `torch_gpu` environment already contains `torch` and CUDA support, keep those packages as-is and only install the missing application dependencies.

## 3. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your secrets locally:

```powershell
Copy-Item .env.example .env
```

Keep API keys in `.env` only. Do not commit real credentials.

The local sentence-transformers embedding model is expected at:

```text
models/bge-base-zh-v1.5
```

Set `SENTENCE_TRANSFORMER_MODEL` to that local path when running with `SENTENCE_TRANSFORMER_LOCAL_FILES_ONLY=true`.

## 4. External Services

Start the retrieval backends before running the app:

- Milvus at `http://127.0.0.1:19530`
- Elasticsearch at `http://127.0.0.1:9200`

## 5. Verify Configuration Loading

You can smoke-test the config loader with:

```powershell
python -c "from app.core.settings import Settings; print(Settings.from_env())"
```
