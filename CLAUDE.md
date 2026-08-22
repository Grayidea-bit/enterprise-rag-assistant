# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An enterprise RAG backend (FastAPI + PostgreSQL/pgvector + **any OpenAI-compatible
LLM endpoint** via pydantic-ai, for both chat and embeddings). **Early-stage
skeleton**: config, the LLM/embedding layer, DB layer, vector schema, and Docker
are done and verified by `scripts/smoke_llm.py`. The HTTP API (ingest/chat/search),
chunking, the agent's retrieval tools, `tenant_id`, and the chat UI are **not
implemented** — `GET /` only serves a placeholder `index.html`. See the Roadmap in
`README.md` before assuming an endpoint exists.

Currently verified against a self-hosted Ollama (`qwen3.8:27b` chat, `bge-m3`
embeddings). Provider choice is entirely an `.env` concern — no code knows or cares
which vendor is behind the endpoint.

## Commands

```bash
# Local dev
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                              # then fill it in
python scripts/smoke_llm.py                       # verify the LLM endpoint first
createdb enterprise_rag
psql enterprise_rag -f database/sql/schema.sql   # apply schema
uvicorn server:app --reload                       # run on :8000

psql enterprise_rag -f database/sql/reset.sql     # drop tables (dev only)

# Docker (starts pgvector DB, auto-applies schema, builds app)
docker compose up --build                         # reads ${...} from .env

# Smoke test with a different chat model, without editing .env
python scripts/smoke_llm.py --model qwen3:8b
```

No test suite, linter, or formatter is configured yet. `scripts/smoke_llm.py` is
the only automated verification: it checks config → embedding → chat → tool calling
and exits non-zero on failure. Run it after touching anything in `core/` or `config.py`.

## Architecture

Two-phase RAG. Ingest: `document → chunk → embed (1024-dim) → store (pgvector)`.
Query: `question → embed → vector search (HNSW, cosine) → top-k chunks → agent → answer`.
The DB helpers backing both phases exist in `database/func.py` (`insert_document`,
`insert_chunk`, `search_chunks` using the pgvector `<=>` cosine operator) but are
**not wired to any endpoint or agent tool**. Chunking does not exist yet.

Module roles:
- `config.py` — two settings objects. `env_settings` (`EnvSettings`) holds **all
  connection config** from `.env`: `CHAT_BASE_URL` / `CHAT_API_KEY` / `CHAT_MODEL`,
  the optional `EMBEDDING_*` counterparts, `EMBEDDING_DIM`, and `DATABASE_URL`.
  `app_settings` (`AppConfig`) holds app-only settings from `system.yaml` (just the
  agent `prompt`). Both are module-level singletons imported elsewhere.
- `core/llm.py` — `build_provider()` (shared by chat and embeddings), `compat_profile()`,
  and the cached `get_model()`.
- `core/embedding.py` — cached `get_embedder()` plus `embed_query()` / `embed_documents()`,
  which validate the returned dimension against `EMBEDDING_DIM`.
- `core/agent.py` — builds the pydantic-ai `agent`. `tools = []` is currently empty;
  retrieval tools go here.
- `database/conn.py` — `psycopg_pool.ConnectionPool` with `configure=register_vector`
  so every connection knows the pgvector type. Created with `open=False`.
- `database/__init__.py` — `db_startup()` / `db_shutdown()` open and close the pool;
  called from the FastAPI lifespan in `server.py`.
- `scripts/smoke_llm.py` — the four-step endpoint verification described above.

## Things that will bite you

- **Empty-string API keys are not the same as `None`.** `OpenAIProvider` only
  substitutes its `'api-key-not-set'` placeholder when `api_key is None`; passing
  `""` sends a genuinely empty key and breaks key-less endpoints like Ollama. This
  is why `build_provider()` does `api_key or None` and why `chat_target` /
  `embedding_target` normalize empty strings. Don't "simplify" that away.
- **The `profile` argument replaces, it does not merge.** `OpenAIChatModel.__init__`
  does `profile or provider.model_profile`, so passing a `ModelProfile` instance
  discards everything the provider inferred from the model name. `compat_profile`
  is therefore a **callable** that starts from `openai_model_profile(model_name)`
  and overrides only two fields.
- **Those two overrides are both load-bearing.** `openai_supports_strict_tool_definition`
  defaults to `True` but self-hosted servers don't implement strict mode — leaving it
  on breaks tool calling, which is exactly what RAG retrieval needs.
  `openai_chat_thinking_field` defaults to `None`, but Qwen3 and other thinking models
  put their output in a `reasoning` field — leaving it unset yields empty responses.
- **Settings precedence differs between the two classes.** `EnvSettings` uses the
  pydantic-settings default (process env > `.env`), so `EMBEDDING_DIM=768 python …`
  overrides the file. `AppConfig` still puts YAML above env via
  `settings_customise_sources` — harmless now that it only holds `prompt`, but don't
  move connection settings back into it.
- **Embedding keys never cross hosts.** `embedding_target` inherits `CHAT_API_KEY`
  only when `EMBEDDING_BASE_URL` is empty (same host). Setting an explicit embedding
  endpoint means it gets its own key or none at all.
- **Embedding dimension must match in three places**: `EMBEDDING_DIM` in `.env`,
  `VECTOR(1024)` in `schema.sql`, and whatever the model actually returns.
  `core/embedding.py` validates the third against the first at runtime; the schema is
  on you. Do **not** pass `EmbeddingSettings.dimensions` — some compatible endpoints
  reject it and fixed-size models ignore it.
- **`input_type` is not sent to the API.** `OpenAIEmbeddingModel.embed()` records
  `input_type` in the `EmbeddingResult` but never puts it in the request. Providers
  that need a `query`/`passage` distinction (e.g. NVIDIA NIM) require
  `EmbeddingSettings.extra_body`.
- **Lazy LLM/agent construction.** `get_model()`, `get_embedder()`, and `get_agent()`
  are all `@cache`d and build nothing at import time, so `core.*` can be imported
  without a live endpoint. Note `config.py` still instantiates `env_settings` /
  `app_settings` on import, and `CHAT_BASE_URL` / `CHAT_MODEL` / `EMBEDDING_MODEL`
  are **required** — importing `config` without them raises a pydantic
  `ValidationError` naming the missing fields.
- **Base URLs usually need `/v1`.** The OpenAI-compatible client appends
  `/chat/completions`, so a bare host 404s. `scripts/smoke_llm.py` warns about this.
- **First calls can be slow.** Self-hosted servers load models on demand; one cold
  start measured ~3 minutes, while warm calls to a 27B returned in ~8s. The eventual
  chat endpoint will need streaming and generous timeouts.
- **Docker containers can't reach your `localhost`.** Use `host.docker.internal` or a
  LAN address in `CHAT_BASE_URL` when running under compose.
- Inline comments are in Traditional Chinese — keep new comments consistent with
  surrounding style.
- Default DSNs/passwords (`postgres:secret`) are dev-only defaults.
