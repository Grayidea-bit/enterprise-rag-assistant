# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An enterprise RAG backend (FastAPI + PostgreSQL/pgvector + NVIDIA NIM via pydantic-ai for both chat `meta/llama-3.3-70b-instruct` and embeddings `baai/bge-m3`, through NIM's OpenAI-compatible endpoint). **Early-stage skeleton**: config, DB layer, vector schema, model wiring, and Docker are done. The HTTP API (ingest/chat/search), the agent's retrieval tools, and the chat UI are **not implemented** — `GET /` only serves a placeholder `index.html`. See the Roadmap in `README.md` before assuming an endpoint exists.

## Commands

```bash
# Local dev
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
createdb enterprise_rag
psql enterprise_rag -f database/sql/schema.sql   # apply schema
uvicorn server:app --reload                       # run on :8000

psql enterprise_rag -f database/sql/reset.sql     # drop tables (dev only)

# Docker (starts pgvector DB, auto-applies schema, builds app)
export LLM_API_KEY=nvapi-...
docker compose up --build
```

No test suite, linter, or formatter is configured yet. Both chat and embeddings call NVIDIA NIM's hosted OpenAI-compatible endpoint (`https://integrate.api.nvidia.com/v1`), so no local model server is needed — only `LLM_API_KEY` must be set.

## Architecture

Two-phase RAG. Ingest: `document → chunk → embed (NIM bge-m3, 1024-dim) → store (pgvector)`. Query: `question → embed → vector search (HNSW, cosine) → top-k chunks → NIM Llama agent → answer`. The DB helpers backing both phases already exist in `database/func.py` (`insert_document`, `insert_chunk`, `search_chunks` using the pgvector `<=>` cosine operator) but are **not wired to any endpoint or agent tool**.

Module roles:
- `config.py` — two settings objects. `env_settings` (`EnvSettings`) holds secrets from `.env` (`LLM_URL`, `LLM_API_KEY`, `DATABASE_URL`). `app_settings` (`AppConfig`) holds app settings from `system.yaml` (`llm_model_name`, `embedding_model_name`, `embedding_dim`, agent `prompt`). Both are module-level singletons imported elsewhere.
- `core/llm.py` — builds the NIM chat `model` (pydantic-ai `OpenAIChatModel` over NIM's OpenAI-compatible endpoint, `base_url=env_settings.LLM_URL`).
- `core/agent.py` — builds the pydantic-ai `agent`. `tools = []` is currently empty; retrieval tools go here.
- `database/conn.py` — `psycopg_pool.ConnectionPool` with `configure=register_vector` so every connection knows the pgvector type. Created with `open=False`.
- `database/__init__.py` — `db_startup()` / `db_shutdown()` open and close the pool; called from the FastAPI lifespan in `server.py`.

## Things that will bite you

- **Lazy LLM/agent construction.** `core/llm.py` exposes `get_model()` and `core/agent.py` exposes `get_agent()` (both `@cache`d). The NIM model is built — and `RuntimeError("Missing the api key of LLM (NVIDIA NIM)")` raised if `LLM_API_KEY` is empty — only on first call, **not at import**. So `core.llm` / `core.agent` can be imported without a key (tests/scripts); call `get_agent()` to get the singleton agent. Note `config.py` still instantiates `env_settings`/`app_settings` on import.
- **Settings precedence** (highest first): constructor args → `system.yaml` → process env → `.env`. This is non-standard — YAML overrides environment variables — and is set explicitly in `AppConfig.settings_customise_sources`. Note `AppConfig` reads from YAML, `EnvSettings` reads from `.env`; they are separate classes.
- **Embedding dimension is hard-coded to 1024** (`VECTOR(1024)` in `schema.sql`) to match `bge-m3`. Changing the embedding model means changing both the schema and `embedding_dim` in `system.yaml`.
- **`llm_model_name` / `embedding_model_name` default to `None`** in `config.py` (the `system.yaml` values supply them); don't assume a default model name from the code.
- **NIM `base_url` must end in `/v1`.** The OpenAI-compatible client appends `/chat/completions`, so `LLM_URL=https://integrate.api.nvidia.com/v1` resolves to `/v1/chat/completions`. A bare host (no `/v1`) 404s.
- Inline comments are in Traditional Chinese — keep new comments consistent with surrounding style.
- Default DSNs/passwords (`postgres:secret`, the `graytsao@localhost` DSN) are dev-only defaults.
