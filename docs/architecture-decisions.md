# Architecture decisions

Every entry below is a decision that was made deliberately, could reasonably have gone
the other way, and would be expensive or confusing to reverse. Each records what forced
the choice, what it costs, and what was rejected.

Where a decision was settled by measurement rather than judgement, the measurement is
included. Where it wasn't, that's stated too.

Written in English to match the README; inline code comments are in Traditional Chinese.

---

## 1. The LLM is reached through a generic OpenAI-compatible interface, configured entirely from `.env`

**Context.** The first version hard-coded NVIDIA NIM: model names lived in `system.yaml`,
endpoint and key in `.env`, and `AppConfig` resolved YAML *above* environment variables.
Changing provider meant editing two files, and a containerised deployment could not
override the model name at all.

**Decision.** Chat and embeddings both go through `pydantic-ai`'s generic
`OpenAIProvider`. Endpoint, key, and model name are all environment variables. Nothing in
the codebase knows which vendor is behind the URL.

**Consequences.**

- Switching between a self-hosted Ollama, NVIDIA NIM, vLLM, or OpenAI is a `.env` edit.
- Two library defaults must be overridden for self-hosted servers to work at all — see
  decision 2.
- `system.yaml` now holds only the agent prompt. The unusual YAML-over-environment
  precedence remains, but it no longer governs anything a deployment needs to change.

**Rejected.** `pydantic-ai` ships a dedicated `OllamaProvider` with better defaults. Using
it would have solved decision 2 for free but re-created the vendor coupling the change
existed to remove.

---

## 2. The model profile overrides `openai_supports_strict_tool_definition` and `openai_chat_thinking_field`

**Context.** With the generic provider, `pydantic-ai` infers a model profile from the
model *name*, which for an unrecognised name yields OpenAI's defaults: strict tool
definitions supported, no thinking field.

Both are wrong for self-hosted servers. Strict tool definitions are an OpenAI-proprietary
extension that Ollama does not implement — leaving it on breaks tool calling, which is
exactly the mechanism RAG retrieval depends on. And Qwen3 models put their output in a
`reasoning` field; with `openai_chat_thinking_field` unset the agent receives an empty
string.

**Decision.** `core/llm.py` passes a **callable** profile that starts from
`openai_model_profile(model_name)` and overrides those two fields.

**Consequences.** A callable is required, not a `ModelProfile` instance:
`OpenAIChatModel.__init__` does `profile or provider.model_profile`, so passing an
instance discards everything inferred from the model name.

**Evidence.** With either override missing, the tool-calling check in
`scripts/smoke_llm.py` fails.

---

## 3. The tenant comes from the API key and never from a request header

**Context.** Tenant isolation was built into the schema before authentication existed,
with the tenant supplied by an `X-Tenant-Id` header. Adding keys raised the question of
what to do with that header.

**Decision.** Under `AUTH_MODE=api_key`, `X-Tenant-Id` is ignored entirely. The tenant is
whatever the key maps to.

**Consequences.** If a valid key could be combined with an arbitrary tenant header, the
key would authenticate the caller without constraining them, and every `tenant_id` column
in the database would be decorative. `tests/integration/test_auth_api.py` asserts this
exact attack returns an empty result rather than another tenant's documents.

Cross-tenant access to a known id returns **404, not 403** — a 403 confirms the resource
exists.

`AUTH_MODE=disabled` restores header trust for local development. The server prints a
warning on every startup in that mode.

**Ordering note.** Tenant columns were added before authentication on purpose: retrofitting
a tenant column across a populated database means a migration, a backfill, and touching
every query. Adding authentication in front of a single dependency does not.

---

## 4. API keys are stored as SHA-256 hashes, not bcrypt or argon2

**Context.** "Never store secrets with a fast hash" is good advice that does not apply
here.

**Decision.** Keys are `secrets.token_urlsafe(32)` — 256 bits of entropy — and are stored
as plain SHA-256.

**Consequences.** There is no dictionary or rule-set to attack a 256-bit random value
with, so the work factor a slow hash buys is protecting against an attack that cannot
happen. It would instead add latency to **every authenticated request**, since the hash is
computed on each call.

This reasoning depends entirely on the secret being high-entropy and machine-generated. If
user-chosen passwords are ever added, they need argon2 or bcrypt — the opposite case, and
the distinction is the whole point.

---

## 5. Conversation history is stored as user/assistant text turns, not as the library's message objects

**Context.** `pydantic-ai` can serialise its own `ModelMessage` list, which would preserve
tool calls and their results verbatim across turns.

**Decision.** The `messages` table holds plain `role` / `content` rows. `to_history()`
rebuilds `ModelRequest` / `ModelResponse` objects from them when a conversation continues.

**Consequences.**

- The database schema is not coupled to a library's internal representation, which can
  change between versions.
- Old tool calls and retrieved chunks are not replayed into context. The model does not
  need to see what the previous turn retrieved; re-sending it costs tokens for nothing.
- History is capped at `MAX_HISTORY_MESSAGES` (20) so context cannot grow without bound.
- Rendering the UI is trivial, because the stored shape is already the shape the UI wants.

**Cost.** Multi-step tool reasoning from an earlier turn is not visible to the model in
later turns. For a question-answering assistant that is the right trade; for an agent
carrying out long multi-turn tasks it would not be.

---

## 6. Chunking is character-based, with no tokenizer dependency

**Context.** Precise chunk sizing wants a tokenizer. `bge-m3` accepts 8192 tokens.

**Decision.** A recursive character splitter — paragraph → line → sentence → space → hard
cut — at 800 characters with 100 characters of overlap. No `transformers`, no `tiktoken`.

**Consequences.** 800 characters of Chinese is roughly 600–800 tokens, an order of
magnitude below the model's limit, so precise token accounting buys nothing. Adding
`transformers` for it would be one of the heaviest dependencies in the project.

If the embedding model is ever swapped for one with a small context window, this decision
has to be revisited — the safety margin is doing the work here, not the algorithm.

---

## 7. Lexical retrieval uses `pg_trgm` trigrams, not PostgreSQL full-text search

**Context.** Hybrid retrieval needs a lexical arm. The obvious choice is `tsvector` +
`ts_rank`.

**It does not work for Chinese.** PostgreSQL does not segment Chinese, so an entire
sentence becomes a single token:

```sql
SELECT to_tsvector('simple', '國內出差住宿費核實報支上限為新臺幣二千八百元');
-- → '國內出差住宿費核實報支上限為新臺幣二千八百元':1     ← one token

SELECT to_tsvector('english', 'lodging expenses are reimbursed up to 2800 dollars');
-- → '2800':7 'dollar':8 'expens':2 'lodg':1 'reimburs':4  ← works
```

**Decision.** Character trigrams via `pg_trgm`, which are language-agnostic.

**Consequences.** On Chinese, trigram similarity behaves close to exact substring
matching. Measured `word_similarity` for a query against the chunk containing its answer
was 0.50–0.71; against every other chunk it was **exactly 0.0**.

That sparseness has two direct implications:

- The lexical arm must filter to non-zero matches before fusion. Feeding a list of
  arbitrarily-ordered zero-score rows into RRF injects pure noise.
- `pg_trgm`'s default `word_similarity_threshold` of 0.6 discards correct matches (one
  measured 0.50). `search_chunks_hybrid` lowers it to 0.25 via `SET LOCAL`, scoped to the
  transaction.

The arm is precise and brittle: it finds literal figures, form codes, and proper nouns
that embeddings blur, and finds nothing at all for a paraphrase. That is precisely the
complement dense retrieval needs. A query of `特休天數` against a chunk saying
`特別休假` scores zero — and the vector arm handles it.

---

## 8. The two retrieval arms are combined with Reciprocal Rank Fusion

**Decision.** `score = Σ 1/(60 + rank)` over each arm's ranked list, `k = 60` from the
original paper.

**Why not score fusion.** Cosine distance and trigram similarity have unrelated scales and
distributions; combining them requires normalisation that is arbitrary and needs retuning
whenever either side changes. RRF reads only ranks, so the scales never have to be
reconciled.

**Consequence.** `max_distance` applies to the vector arm only. A chunk found purely by
literal match can legitimately sit far away in embedding space; filtering the fused result
on distance would silently delete the lexical arm's entire contribution.

**Measured result.** Against a 20-document / 36-question ground-truth set (9 lexical, 15
paraphrased, 12 exact-figure):

| Mode | recall@1 | recall@3 | recall@5 | MRR | ms/query |
| --- | --- | --- | --- | --- | --- |
| vector | 91.7% | 100% | 100% | 0.958 | ~5 |
| hybrid | 88.9% | 100% | 100% | 0.944 | ~7 |

Re-measured 2026-08-25 against `eval/dataset.json` as it stands in this repository
(21 chunks at `chunk_size=300 overlap=60`), on `pgvector/pgvector:0.8.6-pg18` with
`bge-m3`. **An earlier revision of this document reported 94.4% / 97.2% over a
31-chunk corpus. Those numbers could not be reproduced** — see "Reproducing this"
below — so they have been replaced rather than explained away.

**Read honestly, this is a small loss on a saturated benchmark.** `bge-m3` is a strong
multilingual retriever that already handles exact identifiers well: every `exact`
question (n=12) is answered at recall@5 by *both* modes, and vector alone puts them at
rank 1 often enough to score 91.7% overall. Hybrid moves exactly one question the wrong
way — 「超過三十萬的採購案要幾家報價?」 falls from rank 1 to rank 2 — which on 36
questions is the entire −2.8% gap. With 21 chunks in the corpus, recall@3 and recall@5
are at ceiling for both modes, so only recall@1 and MRR carry any signal at all.

**Hybrid is kept, but it is no longer the default.** `RETRIEVAL_MODE` now defaults to
`vector` (changed 2026-08-25). The reason to keep hybrid available is unchanged: it costs
~2ms and covers a failure mode dense retrieval is known to have. What can no longer be
said is that it is "never worse" — on this corpus it is measurably, if marginally, worse,
and defaulting to the mode that measures worse on the only evidence available is not
defensible. Switching it on remains one line of `.env`, or one field on a single request.

This cuts both ways, and the reversal condition should be stated plainly: 21 chunks is
far too small a corpus to exercise what the lexical arm is for. A corpus an order of
magnitude larger — or a domain heavy in identifiers, part numbers, and form codes — could
easily flip this, and the honest move then is to re-run the benchmark and switch the
default back, not to argue from first principles either way.

**Reproducing this.** The gap between the old and current numbers is not environmental
drift; three candidate causes were tested and eliminated on 2026-08-25:

| Hypothesis | Test | Result |
| --- | --- | --- |
| The embedding model changed | `bge-m3` on the endpoint was last modified 2026-08-21, before the benchmark commit (2026-08-22) | Not the cause |
| HNSW graph order varies per build | `docker compose down -v`, rebuild, re-run | Identical to 3 decimal places |
| PostgreSQL / `pg_trgm` version | pg 18.6 vs pg 17.11, same pgvector | Identical |

Nor can the 31-chunk figure be recovered: `split_text()` over the current dataset yields
21 chunks at `chunk_size=300`, and 27 even at 200. The old numbers were therefore measured
against a corpus that differs from the one committed here. The image is now pinned to
`0.8.6-pg18` instead of the floating `pg18` tag so that this class of doubt is cheaper to
settle next time. Making the benchmark actually discriminate still needs a corpus an order
of magnitude larger.

---

## 9. Migrations are numbered SQL files run over their own connection

**Context.** The schema originally lived in a single `schema.sql` mounted into Postgres's
init directory, which runs only on a fresh volume. There was no path for schema evolution.

**Decision.** Numbered SQL under `database/migrations/`, tracked in a `schema_migrations`
table, applied by a ~90-line runner.

**Why not Alembic.** This project is hand-written SQL with no ORM. Alembic's autogenerate
and model-diffing — the reason its dependency is usually worth taking — have no models to
work from. Numbered SQL plus a version table is what `dbmate` and `golang-migrate` do.

**Each file runs in its own transaction** and is recorded only on success, so a failure
leaves neither half-applied DDL nor a false "applied" record. Concurrent instances are
serialised with a Postgres advisory lock.

**Migrations use their own connection, not the application pool.** This is not a style
choice — it is a fix for a chicken-and-egg deadlock. The pool's `configure` hook registers
the pgvector type on every connection it hands out, which fails on a database where the
extension does not exist yet. So the migration that creates the extension could never run:
the pool cannot connect until the extension exists, and the extension does not exist until
the migration runs. `db_startup()` therefore migrates *before* opening the pool.

This only manifests on a genuinely fresh database. It was invisible locally for the entire
life of the project and surfaced the first time CI ran. `tests/test_migrate_isolation.py`
now guards it structurally.

**`AUTO_MIGRATE` defaults to false.** Compose sets it true so `docker compose up` works out
of the box, but a real deployment should treat migration as a deliberate, reviewable step
rather than a side effect of a container restart.

---

## 10. Two dangerous library defaults are overridden explicitly

Neither of these is a preference. Both are defaults that will hurt a real deployment.

**The OpenAI SDK's read timeout is 600 seconds.** A dead endpoint leaves the caller hanging
for ten minutes with no signal. `build_provider()` supplies its own `httpx` client with
`LLM_TIMEOUT_SECONDS` (120) for read/write and 10s for connect — an unreachable host should
fail immediately, not after two minutes.

**`pydantic-ai`'s `tool_calls_limit` is unlimited** and `request_limit` defaults to 50. A
model that keeps re-searching can loop until the token budget is gone. `get_usage_limits()`
caps a run at 6 model calls and 4 tool calls.

`_as_http_error()` in `api/chat.py` is the single place model failures become status codes:
**429** for usage limits, **504** for timeouts, **502** for endpoint errors. It re-raises
anything it does not recognise rather than flattening it to a 500, so new failure modes
stay visible instead of being silently absorbed.

---

## 11. Verification is split into layers by what each layer can actually prove

**Decision.** Three layers, with an explicit boundary between them.

| Layer | Count | Needs | Proves |
| --- | --- | --- | --- |
| Unit | 70 | nothing | pure logic: chunking, key handling, extraction, config, history |
| Integration | 44 | PostgreSQL | the real app end to end, with stubbed models |
| Smoke | 5 suites | a live LLM | answer quality, citation, refusal |

Integration tests drive the real FastAPI app against a real PostgreSQL, but replace the
models with `pydantic-ai`'s `TestModel` and `TestEmbeddingModel`. **CI runs the first two
layers — 114 tests, no external services.**

**What the stubs cannot prove.** `TestEmbeddingModel` returns the same all-ones vector for
every input, so every chunk sits at distance 0 and ranking is meaningless. No integration
test asserts anything about retrieval quality; that belongs to `scripts/eval_retrieval.py`.
Nor can they prove the model refuses to answer when retrieval comes back empty — that is a
smoke test against a real endpoint.

Being explicit about this boundary matters more than the test count. A suite that appears
to test ranking but is actually testing a constant would be worse than no test at all.

**Consequence.** Integration tests share one event loop for the whole session
(`asyncio_default_*_loop_scope = "session"`), because the connection pool is a module-level
singleton and a per-test loop would leave its connections bound to a dead loop.

---

## 12. A PDF with no text layer is rejected, not ingested empty

**Decision.** If `pypdf` extracts fewer than 20 characters, the upload fails with an
explicit message saying the file appears to be scanned and that the system does not do OCR.

**Why.** Silently storing zero chunks is the worst available outcome: the user believes the
upload succeeded, and later the assistant says it cannot find anything about a document
that is visibly listed in the sidebar. A loud failure at upload time is recoverable; a
silent one is not.

A single page that fails to parse is skipped and logged rather than failing the whole file.

---

## 13. The UI is one file with no build step

**Decision.** `index.html` is vanilla HTML, CSS, and JavaScript, served directly by
FastAPI. No npm, no bundler, no framework.

**Why.** This is a Python backend. A build toolchain for a single chat page would add more
operational surface than the page itself. The one place it shows is SSE parsing: the
browser's `EventSource` is GET-only, so the streaming endpoint is read by hand from
`fetch().body.getReader()`.

If the UI grows past a handful of views this decision should be revisited — it is sized to
what the project actually needs today, not to what a larger frontend would.

---

## What is deliberately not decided yet

- **Rate limiting.** One key can issue unlimited requests, each of which costs a real LLM
  call. This is the only remaining gap that blocks exposing the service.
- **Reranking.** No cross-encoder is available on the current endpoint.
- **A benchmark corpus large enough to discriminate.** The current one saturates above
  recall@1, which is stated in the results rather than hidden by them.
- **`.docx` and OCR.**
