# Enterprise RAG Assistant

A retrieval-augmented generation (RAG) backend for enterprise knowledge bases,
built on **FastAPI**, **PostgreSQL + pgvector**, and **any OpenAI-compatible LLM
endpoint** (via [pydantic-ai](https://ai.pydantic.dev/)) for both chat and
embeddings.

The LLM layer is deliberately **provider-agnostic**: chat and embeddings both go
through the OpenAI-compatible interface, and the endpoint, key, and model names
all live in `.env`. Switching between a self-hosted Ollama, NVIDIA NIM, vLLM, or
OpenAI itself is a one-file change — no code edits.

> **Status: early-stage skeleton.** Configuration, the LLM/embedding layer, the
> database layer, the vector schema, and Docker are in place and verified
> end-to-end by a smoke test. The HTTP API (ingest / chat / search), the agent's
> retrieval tools, and the chat UI are **not implemented yet** — see the
> [Roadmap](#roadmap). Right now the server only exposes `GET /`, which serves a
> placeholder page.

---

## Architecture

The system follows a standard two-phase RAG design:

```
Ingest phase
  document ──► chunk ──► embed (1024-dim) ──► store (pgvector)

Query phase
  question ──► embed ──► vector search (HNSW, cosine)
            ──► top-k chunks ──► chat model (pydantic-ai Agent) ──► answer
```

What exists today:

- [`core/llm.py`](core/llm.py) — chat model over any OpenAI-compatible endpoint
- [`core/embedding.py`](core/embedding.py) — `embed_query()` / `embed_documents()`,
  sharing the same provider abstraction, with runtime dimension validation
- [`database/func.py`](database/func.py) — `insert_document`, `insert_chunk`,
  `search_chunks` (cosine distance via the pgvector `<=>` operator)

The chunking step, the HTTP endpoints, and the agent's retrieval tool are the
missing links between them.

## Tech stack

| Layer         | Choice                                                   |
| ------------- | -------------------------------------------------------- |
| Web framework | FastAPI + uvicorn (ASGI)                                 |
| LLM framework | pydantic-ai                                              |
| LLM interface | OpenAI-compatible (`/v1/chat/completions`, `/v1/embeddings`) |
| Vector store  | PostgreSQL 18 + pgvector (HNSW index, cosine)            |
| DB driver     | psycopg 3 with a connection pool (`psycopg-pool`)        |
| Config        | pydantic-settings (`.env` for connections, YAML for app) |
| Runtime       | Python 3.13                                              |

Verified working against a self-hosted **Ollama** (`qwen3.8:27b` for chat,
`bge-m3` for embeddings). Any other OpenAI-compatible endpoint should work by
changing `.env` alone.

## Project structure

```
.
├── server.py              # FastAPI app + lifespan (opens/closes the DB pool)
├── config.py              # EnvSettings (.env) + AppConfig (system.yaml)
├── system.yaml            # Agent system prompt (app settings only)
├── .env.example           # Copy to .env and fill in
├── index.html             # Placeholder page served at GET /
├── requirements.txt
├── core/
│   ├── llm.py             # Chat model + shared OpenAI-compatible provider
│   ├── embedding.py       # embed_query / embed_documents + dim validation
│   └── agent.py           # pydantic-ai Agent (tools list currently empty)
├── database/
│   ├── __init__.py        # db_startup() / db_shutdown()
│   ├── conn.py            # pgvector-aware connection pool
│   ├── func.py            # insert_document / insert_chunk / search_chunks
│   └── sql/
│       ├── schema.sql     # documents + chunks tables, HNSW index
│       └── reset.sql      # drops tables (dev only)
├── scripts/
│   └── smoke_llm.py       # Verifies config → embedding → chat → tool calling
├── Dockerfile
├── docker-compose.yaml    # app + pgvector/pgvector:pg18
└── .dockerignore
```

## Prerequisites

- **Python 3.13**
- Access to an **OpenAI-compatible LLM endpoint** serving both a chat model and
  an embedding model. Options:
  - **Ollama** (self-hosted, no API key): `ollama pull qwen3:8b && ollama pull bge-m3`
  - **NVIDIA NIM**: an `nvapi-…` key from https://build.nvidia.com/
  - **OpenAI**, **vLLM**, **LM Studio**, or anything else speaking the same protocol
- **Docker** + Docker Compose (optional, for the containerized setup)

## Configuration

Two sources, deliberately separated:

- **`.env`** — everything about *connections*: endpoints, keys, model names.
- **`system.yaml`** — application settings only (currently just the agent prompt).

Start from the template:

```bash
cp .env.example .env
```

### `.env`

| Key                  | Required | Description                                              |
| -------------------- | :------: | -------------------------------------------------------- |
| `CHAT_BASE_URL`      |    ✅    | OpenAI-compatible chat endpoint (usually ends in `/v1`)   |
| `CHAT_API_KEY`       |          | Leave empty for self-hosted services that need no key     |
| `CHAT_MODEL`         |    ✅    | Chat model name                                           |
| `EMBEDDING_BASE_URL` |          | Falls back to `CHAT_BASE_URL` when empty                  |
| `EMBEDDING_API_KEY`  |          | See the fallback rule below                               |
| `EMBEDDING_MODEL`    |    ✅    | Embedding model name                                      |
| `EMBEDDING_DIM`      |          | Default `1024`; must match `VECTOR(n)` in `schema.sql`    |
| `DATABASE_URL`       |          | PostgreSQL DSN                                            |

**Key fallback rule.** The embedding key is inherited from `CHAT_API_KEY` *only*
when the embedding endpoint is also inherited — i.e. when both point at the same
host. Once you set an explicit `EMBEDDING_BASE_URL`, the chat key is never sent
to that different host.

Example — self-hosted Ollama serving both models:

```dotenv
CHAT_BASE_URL=http://localhost:11434/v1
CHAT_API_KEY=
CHAT_MODEL=qwen3:8b
EMBEDDING_BASE_URL=
EMBEDDING_API_KEY=
EMBEDDING_MODEL=bge-m3
EMBEDDING_DIM=1024
```

Example — chat on NVIDIA NIM, embeddings on a local Ollama:

```dotenv
CHAT_BASE_URL=https://integrate.api.nvidia.com/v1
CHAT_API_KEY=nvapi-your-key-here
CHAT_MODEL=meta/llama-3.3-70b-instruct
EMBEDDING_BASE_URL=http://localhost:11434/v1
EMBEDDING_API_KEY=
EMBEDDING_MODEL=bge-m3
EMBEDDING_DIM=1024
```

> Most OpenAI-compatible clients append `/chat/completions` to the base URL, so it
> generally **must end in `/v1`**. A bare host will 404. The smoke test warns about
> this.

### `system.yaml`

| Key      | Description             |
| -------- | ----------------------- |
| `prompt` | The agent system prompt |

## Verifying your setup

Before wiring anything else up, confirm the endpoint actually works:

```bash
python scripts/smoke_llm.py
```

It checks four things in order and reports each with timing:

1. Config parses; keys are masked; embedding fallback resolves as expected
2. `embed_query()` / `embed_documents()` return vectors of `EMBEDDING_DIM`
3. The chat model returns a non-empty answer
4. The agent **actually calls a tool** — the prerequisite for RAG retrieval

To try a smaller/faster model without editing `.env`:

```bash
python scripts/smoke_llm.py --model qwen3:8b
```

> **The first call can be slow.** A self-hosted server loads the model into memory
> on demand; one cold start was measured at ~3 minutes, while warm calls to a 27B
> model returned in ~8 seconds. Budget generous timeouts, but don't assume every
> request is slow.

## Getting started — Docker Compose

Compose starts a `pgvector/pgvector:pg18` database, auto-applies the schema, then
builds and runs the app. It reads `${...}` values from your `.env` automatically.

```bash
cp .env.example .env    # then edit it
docker compose up --build
```

- The app listens on **http://localhost:8000**.
- `database/sql/schema.sql` is mounted into the Postgres init directory and runs
  automatically **on first start only** (an existing volume will not re-run it).
- Inside the compose network the app reaches the DB at
  `postgresql://postgres:secret@db:5432/enterprise_rag`.

> ⚠️ `localhost` inside the container is *not* your machine. If your LLM runs
> locally, use `host.docker.internal` or a LAN address in `CHAT_BASE_URL`.

## Getting started — local development

```bash
# 1. Create a virtualenv and install dependencies
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Create the database and apply the schema
createdb enterprise_rag
psql enterprise_rag -f database/sql/schema.sql

# 3. Configure .env, verify the LLM endpoint, then run the server
cp .env.example .env
python scripts/smoke_llm.py
uvicorn server:app --reload
```

The server opens the connection pool on startup and closes it on shutdown via the
FastAPI lifespan in [`server.py`](server.py).

To reset the schema during development:

```bash
psql enterprise_rag -f database/sql/reset.sql   # drops tables — dev only
```

## Database schema

Defined in [`database/sql/schema.sql`](database/sql/schema.sql):

- **`documents`** — one row per source document (`title`, `source`, `metadata`
  JSONB, `created_at`).
- **`chunks`** — one row per text chunk: `content`, a `VECTOR(1024)` `embedding`,
  `chunk_index`, and a `document_id` FK (`ON DELETE CASCADE`).
- An **HNSW** index on `embedding` using `vector_cosine_ops` for approximate
  nearest-neighbor search.

> The vector dimension is fixed at **1024** to match `bge-m3`. If you change the
> embedding model, update both `VECTOR(1024)` and `EMBEDDING_DIM`. The mismatch is
> caught at runtime by `core/embedding.py` with a readable error.

## Roadmap

- [x] Provider-agnostic OpenAI-compatible LLM layer, driven entirely from `.env`
- [x] Embedding client with runtime dimension validation
- [x] Smoke test covering config → embedding → chat → tool calling
- [ ] Chunking strategy
- [ ] `tenant_id` in the schema + tenant-filtered retrieval
- [ ] `POST` ingest endpoint (upload → chunk → embed → store)
- [ ] Agent retrieval tool wired into the pydantic-ai `Agent` (`tools` is empty)
- [ ] Chat / query endpoint (embed → vector search → answer)
- [ ] Async DB layer (`AsyncConnectionPool`) so slow LLM calls don't block
- [ ] Real chat UI in `index.html` (currently a placeholder)

## Security notes

- `.env` is gitignored — never commit API keys or DSNs. Use `.env.example` as the
  committed template.
- The embedding key is never inherited across hosts (see the fallback rule above).
- The `POSTGRES_PASSWORD: secret` in `docker-compose.yaml` and the matching DSN
  are **development defaults only**. Use a strong, secret-managed password and a
  hardened Postgres configuration in production.
- There is **no authentication or tenant isolation yet**. Every document in the
  database is visible to every query. This is a deliberate, tracked gap — see the
  Roadmap.
