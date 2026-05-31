# Enterprise RAG Assistant

A retrieval-augmented generation (RAG) backend for enterprise knowledge bases,
built on **FastAPI**, **PostgreSQL + pgvector**, and **NVIDIA NIM** (via
[pydantic-ai](https://ai.pydantic.dev/)) for both chat (`meta/llama-3.3-70b-instruct`)
and embeddings (`baai/bge-m3`), through NIM's OpenAI-compatible endpoint.

> **Status: early-stage skeleton.** The configuration layer, database layer,
> vector schema, model wiring, and Docker setup are in place. The HTTP API
> (ingest / chat / search), the agent's retrieval tools, and the chat UI are
> **not implemented yet** — see the [Roadmap](#roadmap). Right now the server
> only exposes `GET /`, which serves a placeholder page.

---

## Architecture

The system follows a standard two-phase RAG design:

```
Ingest phase
  document ──► chunk ──► embed (NIM bge-m3, 1024-dim) ──► store (pgvector)

Query phase
  question ──► embed (NIM bge-m3) ──► vector search (HNSW, cosine)
            ──► top-k chunks ──► NIM Llama 3.3 (pydantic-ai Agent) ──► answer
```

The database helpers that back both phases already exist in
[`database/func.py`](database/func.py) — `insert_document`, `insert_chunk`, and
`search_chunks` (cosine distance via the pgvector `<=>` operator). They are not
yet wired to HTTP endpoints or to an agent tool.

## Tech stack

| Layer        | Choice                                              |
| ------------ | --------------------------------------------------- |
| Web framework | FastAPI + uvicorn (ASGI)                            |
| LLM framework | pydantic-ai                                         |
| LLM provider  | NVIDIA NIM `meta/llama-3.3-70b-instruct` (OpenAI-compatible) |
| Embeddings    | NVIDIA NIM `baai/bge-m3` (1024-dim, multilingual)   |
| Vector store  | PostgreSQL 18 + pgvector (HNSW index, cosine)       |
| DB driver     | psycopg 3 with a connection pool (`psycopg-pool`)   |
| Config        | pydantic-settings (layered: init → YAML → env)      |
| Runtime       | Python 3.13                                         |

## Project structure

```
.
├── server.py              # FastAPI app + lifespan (opens/closes the DB pool)
├── config.py              # EnvSettings (.env) + AppConfig (system.yaml)
├── system.yaml            # NVIDIA NIM model names / prompt settings
├── index.html             # Placeholder page served at GET /
├── requirements.txt
├── core/
│   ├── llm.py             # NIM Llama model via pydantic-ai OpenAIChatModel
│   └── agent.py           # pydantic-ai Agent (tools list currently empty)
├── database/
│   ├── __init__.py        # db_startup() / db_shutdown()
│   ├── conn.py            # pgvector-aware connection pool
│   ├── func.py            # insert_document / insert_chunk / search_chunks
│   └── sql/
│       ├── schema.sql     # documents + chunks tables, HNSW index
│       └── reset.sql      # drops tables (dev only)
├── Dockerfile
├── docker-compose.yaml    # app + pgvector/pgvector:pg18
└── .dockerignore
```

## Prerequisites

- **Python 3.13**
- An **NVIDIA NIM API key** (`nvapi-…`, from https://build.nvidia.com/) — used for
  both chat and embeddings
- **Docker** + Docker Compose (optional, for the containerized setup)

## Configuration

Settings come from two sources, resolved with the following precedence (highest
first): constructor args → `system.yaml` → process environment → `.env`. See
[`config.py`](config.py).

### `.env` (secrets / connection)

| Key            | Default                                                | Description                         |
| -------------- | ------------------------------------------------------ | ----------------------------------- |
| `LLM_URL`      | `https://integrate.api.nvidia.com/v1`                  | NIM OpenAI-compatible base URL (`/v1`) |
| `LLM_API_KEY`  | _(empty)_                                              | NIM API key (`nvapi-…`)             |
| `DATABASE_URL` | `postgresql://graytsao@localhost:5432/enterprise_rag`  | PostgreSQL DSN                      |

Example `.env`:

```dotenv
LLM_URL=https://integrate.api.nvidia.com/v1
LLM_API_KEY=nvapi-your-key-here
DATABASE_URL=postgresql://postgres:secret@localhost:5432/enterprise_rag
```

> The OpenAI-compatible client appends `/chat/completions` to `LLM_URL`, so the
> base URL **must end in `/v1`** (it resolves to `/v1/chat/completions`).

### `system.yaml` (application settings)

| Key                    | Default                          | Description                                   |
| ---------------------- | -------------------------------- | --------------------------------------------- |
| `llm_model_name`       | `meta/llama-3.3-70b-instruct`    | NIM chat model                                |
| `embedding_model_name` | `baai/bge-m3`                    | NIM embedding model                           |
| `embedding_dim`        | `1024`                           | Vector dimension (must match `VECTOR(n)`)     |
| `prompt`               | `TESTING` _(placeholder)_        | Agent system prompt                           |

## Getting started — Docker Compose

This is the easiest path. Compose starts a `pgvector/pgvector:pg18` database and
auto-applies the schema, then builds and runs the app.

```bash
# 1. Provide your NIM key (read from the environment by compose)
export LLM_API_KEY=nvapi-your-key-here

# 2. Start everything
docker compose up --build
```

- The app listens on **http://localhost:8000**.
- `database/sql/schema.sql` is mounted into the Postgres init directory and runs
  automatically on first start.
- Inside the compose network the app reaches the DB at
  `postgresql://postgres:secret@db:5432/enterprise_rag`.

> Both chat and embeddings call NVIDIA NIM's hosted endpoint, so no local model
> server is needed — just set `LLM_API_KEY` (and optionally `LLM_URL`).

## Getting started — local development

```bash
# 1. Create a virtualenv and install dependencies
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Create the database and apply the schema
createdb enterprise_rag
psql enterprise_rag -f database/sql/schema.sql

# 3. Configure .env (see Configuration above), then run the server
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
> embedding model, update both `VECTOR(1024)` and `embedding_dim` accordingly.

## Roadmap

- [ ] `POST` ingest endpoint (upload → chunk → embed → store)
- [ ] Chat / query endpoint (embed → vector search → NIM Llama answer)
- [ ] Vector search endpoint
- [ ] Agent retrieval tool wired into the pydantic-ai `Agent` (`tools` is empty)
- [ ] Real chat UI in `index.html` (currently a placeholder)
- [ ] A real system prompt in `system.yaml` (currently `TESTING`)

## Security notes

- `.env` is gitignored — never commit API keys or DSNs.
- The `POSTGRES_PASSWORD: secret` in `docker-compose.yaml` and the matching DSN
  are **development defaults only**. Use a strong, secret-managed password and a
  hardened Postgres configuration in production.
