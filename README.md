# Enterprise RAG Assistant

A retrieval-augmented generation (RAG) backend for enterprise knowledge bases,
built on **FastAPI**, **PostgreSQL + pgvector**, and **any OpenAI-compatible LLM
endpoint** (via [pydantic-ai](https://ai.pydantic.dev/)) for both chat and
embeddings.

The LLM layer is deliberately **provider-agnostic**: chat and embeddings both go
through the OpenAI-compatible interface, and the endpoint, key, and model names
all live in `.env`. Switching between a self-hosted Ollama, NVIDIA NIM, vLLM, or
OpenAI itself is a one-file change — no code edits.

> **Status: usable end to end, with a UI and a retrieval benchmark.** Ingest, query,
> multi-turn conversations, hybrid retrieval, and a chat interface at `GET /` are
> implemented, verified by four smoke tests, and measured by an evaluation harness
> with a 20-document / 36-question ground-truth set. What's still missing:
> **authentication**, PDF/docx parsing, request timeouts, and schema migrations. See
> the [Roadmap](#roadmap).

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

Both phases are implemented: `POST /documents` runs the ingest chain, and
`POST /chat` runs the query chain — the agent decides when to call the retrieval
tool, and the endpoint returns the answer alongside the chunks it actually cited.
Turns are persisted, so follow-up questions ("and after three years?") resolve
against the earlier context.

- [`core/llm.py`](core/llm.py) — chat model over any OpenAI-compatible endpoint
- [`core/embedding.py`](core/embedding.py) — `embed_query()` / `embed_documents()`,
  sharing the same provider abstraction, with runtime dimension validation
- [`core/chunking.py`](core/chunking.py) — recursive character splitter, no tokenizer
  dependency
- [`core/tools.py`](core/tools.py) — `search_knowledge_base`, the agent's retrieval
  tool, plus the request-scoped `RagDeps`
- [`api/upload_files.py`](api/upload_files.py) — the ingest endpoints
- [`api/chat.py`](api/chat.py) — search, chat, and SSE streaming endpoints
- [`database/func.py`](database/func.py) — async, tenant-scoped: `upsert_document`,
  `insert_chunks`, `delete_chunks`, `search_chunks`, `list_documents`

### Tenant isolation

Every row in `documents` and `chunks` carries a `tenant_id`, and every function in
`database/func.py` requires one. The tenant comes from the `X-Tenant-Id` request
header, falling back to `DEFAULT_TENANT_ID`.

> ⚠️ **There is no authentication.** What is isolated is the *data model and the
> retrieval path*, not identity — any caller can claim any tenant. Adding real
> authentication in front of `resolve_tenant()` is a prerequisite for any real
> deployment. This split is deliberate: tenant columns are painful to retrofit,
> auth is not.

## Tech stack

| Layer         | Choice                                                   |
| ------------- | -------------------------------------------------------- |
| Web framework | FastAPI + uvicorn (ASGI)                                 |
| LLM framework | pydantic-ai                                              |
| LLM interface | OpenAI-compatible (`/v1/chat/completions`, `/v1/embeddings`) |
| Vector store  | PostgreSQL 18 + pgvector (HNSW index, cosine)            |
| DB driver     | psycopg 3, async connection pool (`psycopg-pool`)        |
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
├── index.html             # Single-file chat UI served at GET / (no build step)
├── requirements.txt
├── api/
│   ├── deps.py            # resolve_tenant (X-Tenant-Id)
│   ├── upload_files.py    # POST /documents (ingest), GET /documents (list)
│   ├── conversations.py   # Conversation CRUD + history reconstruction
│   └── chat.py            # POST /search, /chat, /chat/stream
├── core/
│   ├── llm.py             # Chat model + shared OpenAI-compatible provider
│   ├── embedding.py       # embed_query / embed_documents + dim validation
│   ├── chunking.py        # Recursive character splitter
│   ├── tools.py           # search_knowledge_base tool + RagDeps
│   └── agent.py           # pydantic-ai Agent wired to the retrieval tool
├── database/
│   ├── __init__.py        # db_startup() / db_shutdown()
│   ├── conn.py            # pgvector-aware async connection pool
│   ├── func.py            # Async, tenant-scoped document/chunk/search helpers
│   └── sql/
│       ├── schema.sql     # documents + chunks tables, HNSW index
│       └── reset.sql      # drops tables (dev only)
├── eval/
│   └── dataset.json       # 20 documents, 36 questions with ground truth
├── scripts/
│   ├── eval_retrieval.py  # recall@k / MRR, vector vs hybrid
│   ├── smoke_llm.py       # Verifies config → embedding → chat → tool calling
│   ├── smoke_ingest.py    # Verifies upload → chunk → embed → store → isolation
│   ├── smoke_chat.py      # Verifies search → cited answer → refusal → streaming
│   └── smoke_conversation.py  # Verifies multi-turn history, isolation, cascade
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
| `RETRIEVAL_MODE`     |          | `hybrid` (default) or `vector` — see [Retrieval](#retrieval) |
| `DEFAULT_TENANT_ID`  |          | Tenant used when no `X-Tenant-Id` header is sent           |
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

## Retrieval

Two modes, switchable with `RETRIEVAL_MODE` in `.env` or per-request via `"mode"` on
`POST /search`:

- **`vector`** — cosine kNN over the HNSW index.
- **`hybrid`** (default) — the vector ranking fused with a trigram lexical ranking
  using **Reciprocal Rank Fusion** (`score = Σ 1/(60 + rank)`). RRF compares only
  ranks, so the two arms' wildly different score scales never have to be reconciled.

### Why trigram and not `tsvector`

PostgreSQL's full-text search does not segment Chinese. The entire sentence becomes a
single token:

```sql
SELECT to_tsvector('simple', '國內出差住宿費核實報支上限為新臺幣二千八百元');
-- → '國內出差住宿費核實報支上限為新臺幣二千八百元':1     ← one token, useless

SELECT to_tsvector('english', 'lodging expenses are reimbursed up to 2800 dollars');
-- → '2800':7 'dollar':8 'expens':2 'lodg':1 'reimburs':4  ← works fine
```

So the lexical arm uses `pg_trgm` character trigrams instead, which are
language-agnostic. On Chinese they behave close to exact substring matching —
measured `word_similarity` was 0.50–0.71 for a matching chunk and **exactly 0.0** for
everything else. That sparseness is why the lexical arm filters to non-zero matches
before fusion; feeding a list of arbitrarily-ordered zero-score rows into RRF would
just inject noise.

`pg_trgm`'s default `word_similarity_threshold` of 0.6 is too strict here (a correct
match measured 0.50 and would be discarded), so `search_chunks_hybrid` lowers it to
0.25 with `SET LOCAL`, scoped to the transaction.

### Measured results

```bash
python scripts/eval_retrieval.py --chunk-size 150 --overlap 30
```

20 documents (6 with answers, 14 topically-adjacent distractors sharing vocabulary),
36 questions in three flavours — `semantic` (paraphrased), `lexical` (shares wording),
and `exact` (hinges on a figure or form code, dense retrieval's classic weak spot).
Ground truth is a `must_contain` substring rather than a chunk id, so the eval set
survives changes to the chunking strategy.

| Mode | recall@1 | recall@3 | recall@5 | MRR | ms/query |
| --- | --- | --- | --- | --- | --- |
| vector | 94.4% | 100% | 100% | 0.972 | 7 |
| **hybrid** | **97.2%** | 100% | 100% | **0.986** | 9 |

**Read this honestly: the win is small and the benchmark is saturated.** `bge-m3` is
a strong multilingual retriever and already handles exact identifiers well — a direct
probe of `IR-001`, `千分之一`, `百分之四十` found the right chunk at rank 1–2 with
vector alone. Hybrid moved one question from rank 2 to rank 1; recall@3 and recall@5
were already at ceiling for both modes, so only recall@1 and MRR carry any signal at
this corpus size (31 chunks).

The benchmark's job right now is to be a **regression guard and a harness**, not proof
that hybrid is dramatically better. Making it discriminate properly needs a corpus an
order of magnitude larger — that's tracked in the Roadmap.

## The chat UI

`GET /` serves a single-file interface — no build step, no npm, no framework. It
covers the whole demo loop: upload a document, ask about it, watch the answer stream
in with inline `[n]` citations and expandable source cards, and flip the tenant field
to watch the same knowledge base become invisible.

```bash
docker compose up -d db
uvicorn server:app --reload
open http://localhost:8000
```

The tenant field in the sidebar is the demo's point: change it and the documents and
conversations vanish, because isolation happens in SQL, not in the UI.

## API

### `POST /documents` — ingest

Upload a plain-text or Markdown file. The request runs the full ingest chain
synchronously and returns once everything is committed.

```bash
curl -X POST http://localhost:8000/documents \
  -H "X-Tenant-Id: acme" \
  -F "file=@handbook.md" \
  -F "title=Employee Handbook"
```

| Field        | Type | Default | Notes                                     |
| ------------ | ---- | ------- | ----------------------------------------- |
| `file`       | file | —       | `.txt`, `.md`, `.markdown`, `.text`; UTF-8; ≤ 2 MB |
| `title`      | form | filename | Display title                            |
| `chunk_size` | form | `800`   | Characters per chunk                      |
| `overlap`    | form | `100`   | Characters carried between chunks         |

```json
{
  "document_id": 13, "tenant_id": "acme", "source": "handbook.md",
  "title": "Employee Handbook", "chunks": 5, "replaced": false
}
```

**Re-uploading the same filename replaces the document** rather than duplicating
it — `(tenant_id, source)` is unique, old chunks are deleted and rewritten, and
`replaced` tells you which happened.

Errors: `415` unsupported extension · `413` too large · `400` not valid UTF-8 or
no chunkable content · `422` invalid `chunk_size`/`overlap`.

> Embeddings are computed **before** anything is written, so a failure at the LLM
> endpoint leaves no half-ingested document behind.

### `GET /documents` — list

```bash
curl http://localhost:8000/documents -H "X-Tenant-Id: acme"
```

Returns that tenant's documents with a `chunk_count` each. Only ever returns rows
matching the header's tenant.

### Conversations

`POST /chat` accepts an optional `conversation_id` and always returns one. Omit it to
start a new conversation; pass it back to continue. Every turn is persisted, so the
follow-up below resolves against the previous answer:

```bash
# → {"answer": "…十四天…", "conversation_id": 7, "sources": [...]}
curl -X POST localhost:8000/chat -H "X-Tenant-Id: acme" -H "Content-Type: application/json" \
  -d '{"question": "特休有幾天?"}'

# 「那滿三年之後呢?」 — no subject; only works because history is replayed
curl -X POST localhost:8000/chat -H "X-Tenant-Id: acme" -H "Content-Type: application/json" \
  -d '{"question": "那滿三年之後呢?", "conversation_id": 7}'
```

| Endpoint | Purpose |
| --- | --- |
| `POST /conversations` | Create an empty conversation |
| `GET /conversations` | List this tenant's conversations, most recent first |
| `GET /conversations/{id}` | Full message history, with stored sources |
| `DELETE /conversations/{id}` | Delete it; messages go with it via `ON DELETE CASCADE` |

Only the last 20 messages are replayed to the model (`MAX_HISTORY_MESSAGES`) so
context can't grow without bound. Conversations belonging to another tenant return
**404, not 403** — a 403 would confirm the conversation exists.

> **What is stored is user/assistant text turns, not pydantic-ai's internal message
> objects.** That format is library-internal, so persisting it would pin the database
> schema to a library version; and replaying old tool calls and retrieved chunks into
> context just burns tokens without helping the model.

### `POST /search` — retrieval only

Vector search with no LLM involved. Useful for judging retrieval quality on its own,
before blaming the model for a bad answer.

```bash
curl -X POST http://localhost:8000/search \
  -H "X-Tenant-Id: acme" -H "Content-Type: application/json" \
  -d '{"query": "how much can I claim for lodging?", "limit": 5}'
```

`limit` is 1–50 (default 5). `max_distance` optionally drops hits above a cosine
distance — without it you always get `limit` rows back, however irrelevant. In hybrid
mode `max_distance` constrains **only the vector arm**: a chunk found purely by literal
match can legitimately sit far away in embedding space, and filtering on distance would
silently delete the lexical arm's whole contribution.

`"mode": "vector" | "hybrid"` overrides `RETRIEVAL_MODE` for a single request, which is
how the evaluation harness compares the two.

### `POST /chat` — retrieval-augmented answer

```bash
curl -X POST http://localhost:8000/chat \
  -H "X-Tenant-Id: acme" -H "Content-Type: application/json" \
  -d '{"question": "國內出差住宿費上限是多少?"}'
```

```json
{
  "answer": "根據差旅費用報支規定，國內出差住宿費核實報支，上限為新臺幣 2,800 元[1]。",
  "sources": [{"source": "差旅規定.md", "title": "…", "distance": 0.36, "excerpt": "…"}]
}
```

The agent decides when to call `search_knowledge_base`; `sources` reports the chunks
it actually retrieved, deduplicated and sorted by distance. An empty `sources` array
means the tool was never called — worth noticing, because it means the answer came
from the model's own knowledge rather than your documents.

When retrieval comes back empty the system prompt requires the model to say so
rather than improvise. This is covered by a smoke test that asks a tenant with no
documents and asserts a refusal.

### `POST /chat/stream` — same, as SSE

Same request body. Emits `sources` first (retrieval finishes before the model starts
writing), then a series of `delta` events, then `done`. Failures after the stream has
opened arrive as an `error` event, since the HTTP status is already sent.

```
event: sources
data: {"sources": [...]}

event: delta
data: {"text": "根據差旅"}

event: done
data: {}
```

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

Then verify the ingest pipeline against a real database:

```bash
docker compose up -d db
DATABASE_URL=postgresql://postgres:secret@localhost:5433/enterprise_rag \
    python scripts/smoke_ingest.py
```

It uploads documents for two tenants and checks seven things: multi-chunk ingest
with contiguous `chunk_index`, upsert-on-reupload, tenant isolation in both the
listing and the vector search, the `max_distance` threshold rejecting irrelevant
hits, and the four error paths. It cleans up its own test data.

Finally, verify the query half:

```bash
python scripts/smoke_chat.py
```

Six checks: retrieval ordering and tenant scoping, a cited answer that quotes a real
figure from the document, a **refusal** from a tenant with no documents, the SSE
event sequence, and request validation.

And the conversation layer:

```bash
python scripts/smoke_conversation.py
```

Seven checks, the important one being a **subjectless follow-up** ("那滿三年之後呢?")
that can only be answered correctly if history is actually being replayed. Also
covers persistence of sources and titles, cross-tenant 404s on read/continue/delete,
streamed turns reaching the database, and `ON DELETE CASCADE`.

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
- `docker-compose.override.yaml` exposes the DB to the host on **port 5433**, not
  5432, so it doesn't collide with any other Postgres you have running.

> ⚠️ `localhost` inside the container is *not* your machine. If your LLM runs
> locally, use `host.docker.internal` or a LAN address in `CHAT_BASE_URL`.

## Getting started — local development

```bash
# 1. Create a virtualenv and install dependencies
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Start the database (or use your own Postgres with pgvector)
docker compose up -d db          # exposed on localhost:5433, schema auto-applied
# — or locally:
# createdb enterprise_rag && psql enterprise_rag -f database/sql/schema.sql

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

- **`documents`** — one row per source document: `tenant_id`, `title`, `source`,
  `metadata` JSONB, `created_at`, `updated_at`, with `UNIQUE (tenant_id, source)`
  so re-uploading a filename replaces rather than duplicates.
- **`chunks`** — one row per text chunk: `content`, a `VECTOR(1024)` `embedding`,
  `chunk_index`, a `document_id` FK (`ON DELETE CASCADE`), and a **denormalized
  `tenant_id`** so vector search can filter without joining `documents`.
- An **HNSW** index on `embedding` using `vector_cosine_ops`, plus btree indexes on
  `chunks(tenant_id)` and `chunks(document_id, chunk_index)`.

**A note on filtered HNSW search.** HNSW walks the vector graph *first* and applies
`WHERE tenant_id = …` after, so a tenant holding a small slice of the table can come
back with fewer than `k` results — or none. `search_chunks()` turns on pgvector 0.8's
`hnsw.iterative_scan` (via `SET LOCAL`, so it stays inside the transaction) to widen
the search until the limit is met, and degrades gracefully on older pgvector.

> The vector dimension is fixed at **1024** to match `bge-m3`. If you change the
> embedding model, update both `VECTOR(1024)` and `EMBEDDING_DIM`. The mismatch is
> caught at runtime by `core/embedding.py` with a readable error.

## Roadmap

- [x] Provider-agnostic OpenAI-compatible LLM layer, driven entirely from `.env`
- [x] Embedding client with runtime dimension validation
- [x] Chunking strategy (recursive character splitter, no tokenizer dependency)
- [x] `tenant_id` in the schema + tenant-filtered retrieval
- [x] `POST /documents` ingest endpoint (upload → chunk → embed → store)
- [x] Async DB layer (`AsyncConnectionPool`) so slow LLM calls don't block
- [x] Smoke tests for both the LLM layer and the ingest pipeline
- [x] Agent retrieval tool wired into the pydantic-ai `Agent`
- [x] `POST /search`, `POST /chat`, and SSE streaming at `POST /chat/stream`
- [x] Multi-turn conversations with persisted history and sources
- [x] Single-file chat UI at `GET /` — streaming, citations, tenant switcher
- [ ] Authentication in front of `resolve_tenant()`
- [x] Retrieval evaluation harness (recall@k, MRR, split by question type)
- [x] Hybrid retrieval: vector + trigram lexical, fused with RRF
- [ ] A corpus large enough for the benchmark to discriminate above recall@1
- [ ] Reranking (no cross-encoder available on the current endpoint)
- [ ] Timeouts and `UsageLimits` on `/chat`
- [ ] PDF / docx parsing (currently plain text and Markdown only)
- [ ] Schema migrations (`schema.sql` only runs on a fresh volume)

## Security notes

- `.env` is gitignored — never commit API keys or DSNs. Use `.env.example` as the
  committed template.
- The embedding key is never inherited across hosts (see the fallback rule above).
- The `POSTGRES_PASSWORD: secret` in `docker-compose.yaml` and the matching DSN
  are **development defaults only**. Use a strong, secret-managed password and a
  hardened Postgres configuration in production.
- Tenant isolation exists in the data model and is enforced by every function in
  `database/func.py`, but **there is no authentication** — the `X-Tenant-Id` header
  is taken at face value, so any caller can read any tenant. Put real auth in front
  of `resolve_tenant()` before exposing this anywhere. Tracked in the Roadmap.
- Uploads are capped at 2 MB and restricted to UTF-8 text; nothing is executed or
  rendered from uploaded content.
