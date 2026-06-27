"""In-memory vector backend — zero-setup fallback (no Docker, no Postgres).

Selected with VECTOR_BACKEND=memory. Implements the SAME interface and semantics as
the pgvector backend (cosine search, retrieval_count, dedup by content_hash) using
numpy. Embeddings are L2-normalized, so cosine == dot product.

State lives in module globals — it persists across Streamlit reruns within one server
process, and resets when the process restarts (just re-ingest). pgvector remains the
primary store for the brief; this exists so the app runs anywhere.
"""
import numpy as np

_ROWS = []           # list of dicts (id, content_hash, post_id, kind, ...)
_VECS = []           # list of np.float32 arrays aligned to _ROWS
_BY_HASH = set()     # dedup
_NEXT_ID = [1]


def ensure_schema():
    return "in-memory store ready (no Postgres)"


def count_chunks() -> int:
    return len(_ROWS)


def upsert_chunks(rows, vectors) -> int:
    if not rows:
        return 0
    inserted = 0
    for row, vec in zip(rows, vectors):
        h = row["content_hash"]
        if h in _BY_HASH:
            continue
        _BY_HASH.add(h)
        r = dict(row)
        r["id"] = _NEXT_ID[0]
        r["retrieval_count"] = 0
        _NEXT_ID[0] += 1
        _ROWS.append(r)
        _VECS.append(np.asarray(vec, dtype=np.float32))
        inserted += 1
    return inserted


def search(query_vec, k: int = 5):
    """Cosine-nearest k chunks; increments retrieval_count on every hit (req #3c)."""
    if not _VECS:
        return []
    M = np.vstack(_VECS)                       # rows are normalized embeddings
    q = np.asarray(query_vec, dtype=np.float32)
    q = q / (np.linalg.norm(q) or 1.0)         # normalize (centroids may not be unit)
    sims = M @ q
    idx = np.argsort(-sims)[: min(k, len(_ROWS))]
    out = []
    for i in idx:
        _ROWS[i]["retrieval_count"] += 1
        r = _ROWS[i]
        out.append({
            "id": r["id"], "post_id": r["post_id"], "kind": r["kind"],
            "author": r.get("author"), "score": r.get("score", 0),
            "permalink": r.get("permalink"), "title": r.get("title"),
            "text": r["text"], "similarity": float(sims[i]),
        })
    return out


def top_retrieved(limit: int = 15):
    hits = [r for r in _ROWS if r["retrieval_count"] > 0]
    hits.sort(key=lambda r: (r["retrieval_count"], r.get("score", 0)), reverse=True)
    return [
        {"kind": r["kind"], "title": r.get("title"), "text": r["text"],
         "permalink": r.get("permalink"), "retrieval_count": r["retrieval_count"],
         "score": r.get("score", 0)}
        for r in hits[:limit]
    ]


def all_embeddings():
    return {
        "ids": [r["id"] for r in _ROWS],
        "kinds": [r["kind"] for r in _ROWS],
        "titles": [r.get("title") for r in _ROWS],
        "texts": [r["text"] for r in _ROWS],
        "retrieval_count": [r["retrieval_count"] for r in _ROWS],
        "embeddings": [np.asarray(v, dtype=np.float32) for v in _VECS],
    }
