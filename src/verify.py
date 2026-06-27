"""One-command live smoke test of the full pipeline (the acceptance gate).

Run AFTER `docker compose up -d db`, with ANTHROPIC_API_KEY set:

    python -m src.verify

Tip: set GEN_MODEL=claude-haiku-4-5 for a near-free run (the two generation steps
make real Claude calls). Prints PASS/FAIL per stage; exits non-zero on any failure.
On success it also writes DEMO_OUTPUT.md (the captured No-RAG vs RAG A/B), so a
reviewer can see the result without running anything.
"""
import sys
import traceback
from pathlib import Path

from .config import GEN_MODEL, SUBREDDIT

_results = {}  # collected across steps for the DEMO_OUTPUT.md capture


def _step(name, fn):
    try:
        result = fn()
        print(f"  PASS  {name}" + (f" — {result}" if result else ""))
        return True
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        traceback.print_exc()
        return False


def _rag_summary(res):
    return (f"{res['citations']} citations · {res['unique_posts']} unique posts · "
            f"{res['context_chunks']} context chunks")


def _write_demo_output():
    norag, rag = _results.get("norag"), _results.get("rag")
    if not (norag and rag):
        return
    md = (
        f"# Demo output — Community Voices (r/{SUBREDDIT})\n\n"
        f"_Captured by `python -m src.verify` using model `{GEN_MODEL}`. "
        f"This is the actual A/B output; regenerate any time._\n\n"
        f"## Metrics\n\n"
        f"| | No-RAG (baseline) | RAG-empowered |\n|---|---|---|\n"
        f"| permalink citations | {norag['citations']} | {rag['citations']} |\n"
        f"| unique posts grounded | 0 | {rag['unique_posts']} |\n"
        f"| context chunks | 0 | {rag['context_chunks']} |\n\n"
        f"## ⬜ No-RAG (prior knowledge only)\n\n{norag['document']}\n\n"
        f"## 🟩 RAG-empowered (grounded in this week's retrieved posts)\n\n{rag['document']}\n"
    )
    Path("DEMO_OUTPUT.md").write_text(md)
    print("  wrote DEMO_OUTPUT.md")


def main():
    import os

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set. Copy .env.example to .env and set your key.")
        sys.exit(1)

    from . import generate, ingest, seed, store, viz
    from .embeddings import embed

    print(f"Community Voices smoke test — r/{SUBREDDIT} · model {GEN_MODEL}\n")
    ok = True

    ok &= _step(
        "schema (ensure_schema)",
        lambda: store.ensure_schema() or "vector extension + chunks table",
    )

    def _populate():
        src = f"live r/{SUBREDDIT}"
        try:
            rows = ingest.fetch_chunks()
        except Exception as e:
            print(f"        (live ingest unavailable: {e}\n         -> falling back to sample corpus)")
            rows = []
        if not rows:
            rows, src = seed.build_rows(), "sample corpus"
        added = store.upsert_chunks(rows, embed([r["text"] for r in rows]))
        return f"{len(rows)} chunks via {src} ({added} new)"

    ok &= _step("ingest + embed + upsert", _populate)
    ok &= _step(
        "vector search (+retrieval_count)",
        lambda: f"{len(store.search(embed(['weather forecast'])[0], k=5))} hits",
    )

    def _norag():
        _results["norag"] = generate.generate_norag()
        return f"{len(_results['norag']['document'])} chars, {_results['norag']['citations']} citations"

    def _rag():
        _results["rag"] = generate.generate_rag()
        return _rag_summary(_results["rag"])

    ok &= _step("generate_norag (Claude call)", _norag)
    ok &= _step("generate_rag (retrieval + Claude)", _rag)
    ok &= _step("embedding viz (build_figure)", lambda: viz.build_figure()[1])
    ok &= _step("retrieval stats", lambda: f"{len(store.top_retrieved())} retrieved chunks")

    if ok:
        _write_demo_output()

    print("\n" + ("ALL PASS — pipeline works end to end ✅" if ok else "SOME STAGES FAILED ❌"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
