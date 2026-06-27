# Demo output — Community Voices

_Placeholder. Run `python -m src.verify` (after `docker compose up -d db`, with
`ANTHROPIC_API_KEY` set) to overwrite this file with the **real captured A/B
output** from your run — the No-RAG vs RAG documents plus a metrics table._

What to expect once generated:

- **No-RAG (baseline)** — fluent but generic; **~0 permalink citations**; cannot
  reference this week's actual threads.
- **RAG-empowered** — each point grounded in retrieved posts with **many real
  permalink citations** spanning several unique posts; next-week predictions
  reason from recurring/seasonal patterns visible in the retrieved excerpts.

The metrics table will quantify the gap (citations, unique posts grounded,
context chunks), making the value of RAG legible at a glance.
