"""Community Voices Document generation — the A/B core (req #2, #5).

Two paths over the same task:
  - generate_norag(): Claude writes from prior knowledge only (baseline).
  - generate_rag(): retrieve grounded chunks from pgvector, cite them, predict
    next week from the actual current-week evidence.

Retrieval increments retrieval_count in the store, feeding req #3c.
"""
import datetime
import re

import anthropic

from . import store
from .config import GEN_BACKEND, GEN_MODEL, HAS_API_KEY, SUBREDDIT
from .embeddings import embed


def _use_stub() -> bool:
    """Simulated generation when forced, or in 'auto' mode with no API key."""
    if GEN_BACKEND == "stub":
        return True
    if GEN_BACKEND == "anthropic":
        return False
    return not HAS_API_KEY  # "auto"

# Community-neutral intent probes — coverage of asks/complaints regardless of topic.
# (The bulk of retrieval is data-derived via clustering; see retrieve_context.)
INTENT_PROBES = [
    "questions, help requests, and advice",
    "problems, complaints, and frustrations",
]

_PERMALINK_RE = re.compile(r"/r/[A-Za-z0-9_]+/comments/|reddit\.com", re.IGNORECASE)


def _today() -> str:
    return datetime.date.today().isoformat()


def _client():
    return anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env


def _complete(system: str, user: str, max_tokens: int = 16000) -> str:
    """Single Claude turn. Streaming + get_final_message avoids HTTP timeouts on
    long documents; adaptive thinking per the claude-api skill guidance."""
    with _client().messages.stream(
        model=GEN_MODEL,
        max_tokens=max_tokens,
        thinking={"type": "adaptive"},
        system=system,
        messages=[{"role": "user", "content": user}],
    ) as stream:
        msg = stream.get_final_message()
    text = "".join(b.text for b in msg.content if b.type == "text").strip()
    if not text:
        # Adaptive thinking can, in edge cases, return only thinking blocks.
        raise RuntimeError(
            "Model returned no text (only thinking). Try regenerating, or check the model/key."
        )
    return text


def retrieve_context(k: int = 4, max_clusters: int = 6):
    """Community-AGNOSTIC retrieval. Themes are derived from the data, not hardcoded:
    cluster the stored embeddings (same clusters the map shows), then use each cluster
    CENTROID as a query — so context spans the week's actual topics for any subreddit.
    Add a couple of neutral intent probes for asks/complaints. Union, dedup by id.
    Every store.search() increments retrieval_count (req #3c)."""
    by_id = {}

    data = store.all_embeddings()
    embs = data["embeddings"]
    if embs:
        import numpy as np
        from sklearn.cluster import KMeans

        X = np.vstack(embs)
        n_clusters = min(max_clusters, len(X))
        centroids = KMeans(
            n_clusters=n_clusters, n_init=10, random_state=42
        ).fit(X).cluster_centers_
        # The mean of unit vectors isn't unit-norm; re-normalize so each centroid is a
        # proper cosine query (matches the normalized embeddings in the store).
        centroids = centroids / np.clip(
            np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12, None
        )
        for centroid in centroids:
            for row in store.search(centroid, k=k):
                by_id.setdefault(row["id"], row)

    for probe in INTENT_PROBES:
        qv = embed([probe])[0]
        for row in store.search(qv, k=k):
            by_id.setdefault(row["id"], row)

    return list(by_id.values())


def _format_context(chunks) -> str:
    blocks = []
    for c in chunks:
        snippet = c["text"][:600]
        url = f"https://www.reddit.com{c['permalink']}" if c.get("permalink") else "(no link)"
        blocks.append(
            f"- [{c['kind']}] \"{c.get('title') or ''}\" (score {c.get('score', 0)})\n"
            f"  {snippet}\n  source: {url}"
        )
    return "\n".join(blocks)


def count_citations(doc: str) -> int:
    return len(_PERMALINK_RE.findall(doc))


def _stub_norag() -> str:
    """Simulated baseline — no retrieval, no LLM."""
    return (
        f"## What the community discussed this past week\n\n"
        f"_Simulated baseline (no retrieval, no LLM — set `ANTHROPIC_API_KEY` or "
        f"`GEN_BACKEND=anthropic` for real model output)._\n\n"
        f"- General, non-time-specific discussion typical of r/{SUBREDDIT}.\n"
        f"- No reference to this week's actual threads — that's the point of the baseline.\n\n"
        f"## What the community will likely discuss next week\n\n"
        f"- Generic continuation of the community's usual topics.\n"
    )


def _stub_rag(chunks) -> str:
    """Simulated RAG — stitch the document straight from retrieved context (real
    titles + permalinks), so the A/B mechanics work with no LLM call."""
    seen, items = set(), []
    for c in chunks:
        if c["post_id"] in seen:
            continue
        seen.add(c["post_id"])
        url = f"https://www.reddit.com{c['permalink']}" if c.get("permalink") else ""
        items.append((c.get("title") or c["text"][:70], url))
    past = "\n".join(f"- **{t}** ({u})" for t, u in items[:8])
    nxt = "\n".join(f"- Likely continued discussion of: {t}" for t, _ in items[:4])
    return (
        f"## What the community discussed this past week\n\n"
        f"_Simulated generation (`GEN_BACKEND=stub`): stitched from {len(chunks)} "
        f"retrieved chunks across {len(seen)} posts — no LLM. Set a key for real output._\n\n"
        f"{past}\n\n"
        f"## What the community will likely discuss next week\n\n{nxt}\n"
    )


_BASE_TASK = (
    "Write a 'Community Voices Document' for the subreddit r/{sub}. "
    "Today is {today}. Use exactly two sections:\n"
    "1. **What the community discussed this past week** — the main themes, notable "
    "threads, and the overall mood.\n"
    "2. **What the community will likely discuss next week** — concrete predictions "
    "with brief reasoning.\n"
    "Use clear Markdown with headers and bullet points."
)


def generate_norag():
    """Baseline: no retrieval, model prior knowledge only."""
    if _use_stub():
        doc = _stub_norag()
    else:
        task = _BASE_TASK.format(sub=SUBREDDIT, today=_today())
        system = (
            "You are an analyst summarizing an online community. You have NO access to "
            "this week's posts — rely only on your general knowledge of the community. "
            "Do not invent specific posts, usernames, or links you cannot verify."
        )
        doc = _complete(system, task)
    return {"document": doc, "mode": "no-rag", "citations": count_citations(doc), "context_chunks": 0}


def generate_rag(k: int = 4):
    """RAG: retrieve grounded chunks, cite them, predict from real evidence."""
    chunks = retrieve_context(k=k)
    if not chunks:
        return {
            "document": "_The vector store is empty — click **Ingest this week** or "
            "**Load sample data** first, then generate._",
            "mode": "rag", "citations": 0, "context_chunks": 0,
            "unique_posts": 0, "retrieved": [],
        }
    if _use_stub():
        doc = _stub_rag(chunks)
    else:
        context = _format_context(chunks)
        task = _BASE_TASK.format(sub=SUBREDDIT, today=_today())
        system = (
            "You are an analyst summarizing an online community. You are given a set of "
            "REAL excerpts retrieved from this past week's posts and comments. Cite the "
            "source link in parentheses after EVERY claim. Assert nothing that is not "
            "directly supported by the excerpts — do not invent threads, users, or links. "
            "For next-week predictions, reason only from recurring/seasonal patterns "
            "visible in the excerpts."
        )
        user = f"{task}\n\nRetrieved excerpts from this past week:\n{context}"
        doc = _complete(system, user)
    return {
        "document": doc,
        "mode": "rag",
        "citations": count_citations(doc),
        "context_chunks": len(chunks),
        "unique_posts": len({c["post_id"] for c in chunks}),
        "retrieved": chunks,
    }
