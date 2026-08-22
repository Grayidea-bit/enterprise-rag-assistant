# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An enterprise RAG backend (FastAPI + PostgreSQL/pgvector + **any OpenAI-compatible
LLM endpoint** via pydantic-ai, for both chat and embeddings). **The full RAG loop
works**: config, the LLM/embedding layer, chunking, the async DB layer, `tenant_id`
isolation, `POST /documents`, and the query side (`/search`, `/chat`,
`/chat/stream`) are all implemented and verified by the three smoke scripts.
**Still missing**: authentication, a UI (`GET /` is a placeholder), PDF/docx
parsing, conversation history, and schema migrations. See the Roadmap in
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
docker compose up -d db                           # DB on localhost:5433 (not 5432)
uvicorn server:app --reload                       # run on :8000

# Verify the whole ingest pipeline against a real DB
DATABASE_URL=postgresql://postgres:secret@localhost:5433/enterprise_rag \
    python scripts/smoke_ingest.py

psql enterprise_rag -f database/sql/reset.sql     # drop tables (dev only)

# Docker (starts pgvector DB, auto-applies schema, builds app)
docker compose up --build                         # reads ${...} from .env

# Smoke test with a different chat model, without editing .env
python scripts/smoke_llm.py --model qwen3:8b
```

No test suite, linter, or formatter is configured yet. Three smoke scripts are the
only automated verification; all exit non-zero on failure:

| Script | Covers | Needs |
| --- | --- | --- |
| `scripts/smoke_llm.py` | config → embedding → chat → tool calling | LLM endpoint |
| `scripts/smoke_ingest.py` | upload → chunk → embed → store → isolation → errors | + DB |
| `scripts/smoke_chat.py` | search → cited answer → refusal → SSE → validation | + DB |

Run the relevant one after touching `core/`, `config.py`, `database/`, or `api/`.

## Architecture

Two-phase RAG, both halves implemented. Ingest: `document → chunk → embed (1024-dim)
→ store (pgvector)` via `POST /documents`. Query: the agent calls
`search_knowledge_base` (`core/tools.py`), which embeds the query and runs
`search_chunks()`; `POST /chat` returns the answer plus the chunks that were actually
retrieved.

Every `database/func.py` function takes `tenant_id` as its first argument — that is
where tenant isolation is enforced. The tenant comes from the `X-Tenant-Id` header
via `resolve_tenant()` in `api/upload_files.py`, with **no authentication**: the
header is trusted as-is. That split is intentional (tenant columns are painful to
retrofit; auth is not), but it means this cannot be exposed as-is.

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
- `core/chunking.py` — `split_text()`, a recursive character splitter (paragraph →
  line → sentence → space → hard cut). Deliberately no tokenizer dependency.
- `core/tools.py` — `search_knowledge_base` (the agent's retrieval tool) and
  `RagDeps`, the per-request dependency object carrying `tenant_id`, `limit`,
  `max_distance`, and the `retrieved` list.
- `core/agent.py` — builds the pydantic-ai `agent` with `deps_type=RagDeps` and the
  retrieval tool registered.
- `api/deps.py` — `resolve_tenant()`, shared by every router.
- `api/upload_files.py` — `POST /documents` (ingest) and `GET /documents` (list).
- `api/chat.py` — `POST /search` (no LLM), `POST /chat`, `POST /chat/stream` (SSE).
- `database/conn.py` — `psycopg_pool.AsyncConnectionPool` with
  `configure=register_vector_async`. Created with `open=False`.
- `database/__init__.py` — `db_startup()` / `db_shutdown()` are **async**; awaited
  from the FastAPI lifespan in `server.py`.
- `database/func.py` — all async, all tenant-scoped.
- `scripts/smoke_llm.py`, `scripts/smoke_ingest.py` — the two verification scripts.

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
- **`RagDeps.retrieved` is a deliberate side channel.** A tool's return value goes to
  the *model*; the HTTP caller also needs to know which chunks were cited, so
  `search_knowledge_base` appends its hits to `ctx.deps.retrieved` and the endpoint
  reads them afterwards. `unique_sources()` dedupes because the model often searches
  several times with reworded queries. An empty `sources` in a `/chat` response means
  the tool was never called — the answer came from the model's own knowledge.
- **The system prompt is load-bearing.** `system.yaml` explicitly orders the model to
  call `search_knowledge_base` before answering and to say "找不到相關資料" when
  retrieval is empty. Weaken that text and the model starts answering from memory,
  which is exactly the failure a RAG system exists to prevent. `smoke_chat.py` guards
  this by asking a tenant with no documents and asserting a refusal.
- **In `/chat/stream`, `sources` is emitted before any `delta`.** By the time
  `run_stream()` yields, tool calls have already completed, so `deps.retrieved` is
  populated. Errors raised after the response has started can only be reported as an
  SSE `error` event — the status code is long gone.
- **A plain `list[float]` is not a vector.** psycopg adapts it to
  `double precision[]`. That silently *works* on `INSERT` (PostgreSQL applies an
  assignment cast to the `VECTOR(1024)` column) but **fails on `<=>`**, which has no
  `vector <-> float8[]` overload. Always wrap with `pgvector.Vector(...)` — see
  `database/func.py`. This asymmetry means ingest can look perfectly healthy while
  every search 500s.
- **pgvector's GUCs only exist after its library loads.** `SHOW hnsw.iterative_scan`
  raises `unrecognized configuration parameter` on a connection that has not yet
  touched a vector type, even on pgvector 0.8. `_supports_iterative_scan()` runs
  `SELECT '[1]'::vector` first for exactly this reason — don't "simplify" that line
  away or the recall fix silently turns itself off forever.
- **Filtered HNSW under-returns.** The index walks the graph before `WHERE
  tenant_id = …` is applied, so a sparse tenant can get fewer than `k` rows.
  `hnsw.iterative_scan` (pgvector 0.8+) is set per-transaction via `SET LOCAL` —
  never plain `SET`, which would leak into other users of the pooled connection.
- **`schema.sql` only runs on a fresh volume.** It is mounted into
  `docker-entrypoint-initdb.d`, so schema changes need `docker compose down -v` (dev)
  or a real migration (anything else). There is no migration tool yet.
- **The dev DB is on port 5433, not 5432**, set in `docker-compose.override.yaml` to
  avoid colliding with other local Postgres instances.
- **Docker containers can't reach your `localhost`.** Use `host.docker.internal` or a
  LAN address in `CHAT_BASE_URL` when running under compose.
- Inline comments are in Traditional Chinese — keep new comments consistent with
  surrounding style.
- Default DSNs/passwords (`postgres:secret`) are dev-only defaults.
