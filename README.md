# Community Voices

A small web app that generates a **Community Voices Document** for an online
community — what it discussed in the past week and what it will likely discuss
next week — powered by **RAG** over an automatically-ingested vector store, with
a flattened embedding visualization, retrieval stats, and an **A/B comparison**
of LLM generation with vs. without RAG.

**Community:** [r/japanweather](https://www.reddit.com/r/japanweather/). Weather is
an unusually good fit for this brief: it has a **strong weekly/seasonal cadence**
(typhoon tracks, the rainy season (tsuyu), sakura forecasts, heat-stroke alerts,
first snow) which makes "what will they discuss *next* week" genuinely predictable
rather than hand-waved. It's text-rich and active. The app is **not hardcoded to
it** — retrieval is data-derived, so any text-rich subreddit works via `SUBREDDIT`.

> **For reviewers:** this implements the brief's full stack — **pgvector** (the
> vector DB: `db/schema.sql`, `src/store_postgres.py`, `docker-compose.yml`) and
> **real Claude generation** (`src/generate.py`, `claude-opus-4-8`). Both are
> complete and one env var away. The app *defaults* to a zero-setup in-memory store
> and simulated generation so it runs with **no Docker and no API key**; set
> `VECTOR_BACKEND=postgres` and `ANTHROPIC_API_KEY` for the production path. See the
> requirement → file map below.

## Quick start (no Docker, no database)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # one command; first run pulls torch (a few min)
cp .env.example .env                      # optional: set ANTHROPIC_API_KEY for real LLM text
python -m src.verify                      # one-command end-to-end check; writes DEMO_OUTPUT.md
streamlit run app.py
```

That's it — **no Docker, no database, and no API key required to run.** The default
vector store is in-memory, and with no key the app uses **simulated generation**
(documents stitched from the retrieved context, clearly labeled in the UI) so the
full ingest → retrieve → A/B → map → stats flow works out of the box. Add
`ANTHROPIC_API_KEY` to `.env` for real Claude output (embeddings always run
locally). Tip: `GEN_MODEL=claude-haiku-4-5` makes real generation ~free.

In the browser: **Ingest this week** (or **Load sample data** if Reddit blocks your
network) → **Documents (A/B)** → **Generate both** → **Embedding map** → **Render**
→ **Retrieval stats** → **Refresh**.

## Vector store: in-memory (default) or pgvector

Both backends implement the same interface (`src/store.py` dispatches on
`VECTOR_BACKEND`) with identical cosine-search + retrieval-count semantics:

| `VECTOR_BACKEND` | What it is | Setup |
|---|---|---|
| `memory` (default) | numpy vector store — zero infrastructure | none |
| `postgres` | **pgvector** (Postgres + `vector`, HNSW index) — the brief's vector DB | `docker compose up -d db`, **or** native Postgres + `db/schema.sql` |

The pgvector implementation is real and complete (`db/schema.sql`,
`src/store_postgres.py`, `docker-compose.yml`); the in-memory backend is the
default purely so the app runs anywhere with one command. To use pgvector:

```bash
# Option A — Docker:
docker compose up -d db
# Option B — native Postgres (macOS, no Docker):
brew install postgresql@16 pgvector && brew services start postgresql@16
createdb community_voices && psql community_voices -f db/schema.sql
# then in .env:
VECTOR_BACKEND=postgres
# (Option B also needs DATABASE_URL=postgresql://localhost:5432/community_voices)
```

## Architecture

```
Reddit JSON ──► ingest.py ──► chunk + dedup + caps
  (or sample corpus, offline)     │
                          sentence-transformers (all-MiniLM-L6-v2, local)
                                          │
                       vector store  ◄── retrieval_count (stats)
              (in-memory default, or pgvector — src/store.py dispatch)
                                          │
            ┌─────────────────────────────┼─────────────────────────────┐
   data-derived RAG retrieval         embedding map                A/B vs no-RAG
   (cluster centroids as queries)     (UMAP+KMeans)                     │
            └──────────────────► Claude (claude-opus-4-8) ◄──────────────┘
                                          │
                                   Streamlit UI
```

| Brief requirement | Where |
|---|---|
| Active community, weekly discussion | r/japanweather, `t=week` ingest (`src/ingest.py`) |
| Community Voices Document (past + next week) | `src/generate.py` |
| RAG + vector DB | pgvector (`db/schema.sql`, `src/store_postgres.py`) + in-memory fallback |
| Flattened embedding visualization | `src/viz.py` |
| Retrieval stats | `retrieval_count`, incremented per search (both backends) |
| Automated ingestion | `src/ingest.py` |
| Get around overly large data | post/comment caps + chunking + dedup (`src/config.py`, `src/ingest.py`) |
| A/B: LLM with vs without RAG | `generate_rag()` vs `generate_norag()`, side by side |

## How retrieval works (and why it generalizes)

The RAG context is **not** built from hardcoded topic queries. After ingestion,
`retrieve_context()` (`src/generate.py`) **clusters the stored embeddings** (the
same clusters the map shows) and uses each **cluster centroid as a query**, plus a
couple of community-neutral intent probes (questions, complaints). So the context
automatically spans the week's actual topics for *any* community. Every retrieval
increments `retrieval_count`, feeding the stats panel and the map's point sizes.

## Tests

Stdlib `unittest` (no extra deps), forced onto the in-memory backend:

```bash
python -m unittest discover -s tests -v
```

Covers seeding, ingest parsing + error handling, the store (dedup, search,
retrieval-count, stats), data-derived retrieval, generation guards (empty store,
thinking-only responses, citation counting), and the visualization.

## Demo output

`python -m src.verify` writes **[`DEMO_OUTPUT.md`](DEMO_OUTPUT.md)** — the actual
captured No-RAG vs RAG documents plus a metrics table — so you can see the result
without running the app. Expect a stark contrast: the **No-RAG** baseline is fluent
but generic with **~0 citations**, while **RAG** grounds each point in retrieved
posts with **many real permalink citations** across several unique posts.

## Reddit access (important)

Ingestion uses Reddit's **unauthenticated public JSON** with a compliant
User-Agent — no Reddit credentials. But Reddit **403s these endpoints from some
networks** (datacenter IPs, some VPNs) regardless of User-Agent. If that happens:

- **Load sample data (offline)** — a fixed, deterministic sample corpus (below).
  The full RAG / A-B / map / stats demo runs identically on it.
- **PRAW (authenticated)** — drop in `praw` with free Reddit app credentials in
  `src/ingest.py`; authenticated requests aren't subject to the blanket 403. The
  chunk → embed → upsert pipeline is unchanged.

### Offline sample corpus

A fixed, fabricated **sample corpus** of ~41 realistic japanweather-style chunks
(typhoons, tsuyu, sakura, heat, Hokkaido snow, cloud ID, forecasts) lets the full
pipeline run with no network. Clearly labeled sample data, not a live pull.

- In the app: sidebar → **Load sample data (offline)**.
- From the CLI: `python -m src.seed`

## Handling "overly large data" (req #4b)

Ingestion is bounded by design (`src/config.py`): top **60** posts/week, top **15**
comments/post by score, long text **chunked** (~350 words, 50 overlap), and
everything **deduped** by content hash before embedding.

## Notes

- **Embeddings are local** (`sentence-transformers`), so generation is the only
  thing that needs an API key. Anthropic has no embeddings endpoint — that's why
  the embedding model is separate from the Claude generation model.
- **Generation backend** (`GEN_BACKEND`): `auto` (default — real Claude when a key
  is set, else simulated), `anthropic` (force real), `stub` (force simulated, no
  key/network). Real generation uses `claude-opus-4-8` with adaptive thinking +
  streaming (~$0.15–0.30 per A/B run; `GEN_MODEL=claude-haiku-4-5` for near-free).
  Simulated mode stitches the document from retrieved context so the app and the
  A/B contrast work with zero setup.
- **Security**: put your real key only in `.env` (gitignored). Never in
  `.env.example` (committed). If a key has ever touched a tracked file, rotate it.
- `requirements.txt` includes `psycopg`/`pgvector`; they're only used when
  `VECTOR_BACKEND=postgres`.
