"""pgvector backend: schema, upsert, cosine search, retrieval stats.

This is the default/primary store (the brief's "vectorized table in a relational DB").
Selected when VECTOR_BACKEND=postgres (the default). See src/store.py for the dispatch
and src/store_memory.py for the zero-setup fallback.
"""
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from .config import DATABASE_URL

_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "db" / "schema.sql"


def ensure_schema():
    """Apply db/schema.sql (idempotent). Lets the app self-heal if the DB is fresh."""
    sql = _SCHEMA_PATH.read_text()
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        for stmt in (s.strip() for s in sql.split(";")):
            if stmt:
                conn.execute(stmt)
    return None


def _connect():
    conn = psycopg.connect(DATABASE_URL, autocommit=True)
    try:
        register_vector(conn)  # requires the `vector` extension to already exist
    except Exception as e:
        conn.close()
        raise RuntimeError(
            "pgvector not registered — is the `vector` extension installed? "
            "Run ensure_schema() (the app does this at startup) or apply db/schema.sql."
        ) from e
    return conn


def count_chunks() -> int:
    with _connect() as conn:
        return conn.execute("SELECT count(*) FROM chunks").fetchone()[0]


def upsert_chunks(rows, vectors: np.ndarray) -> int:
    """Insert chunk rows with their embeddings; skip duplicates by content_hash.

    `rows` is a list of dicts (see ingest.fetch_chunks); `vectors` is (n, 384)
    aligned to `rows`. Returns the number of newly inserted rows.
    """
    if not rows:
        return 0
    inserted = 0
    with _connect() as conn:
        for row, vec in zip(rows, vectors):
            cur = conn.execute(
                """
                INSERT INTO chunks
                    (content_hash, post_id, kind, author, created_utc,
                     score, permalink, title, text, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (content_hash) DO NOTHING
                """,
                (
                    row["content_hash"],
                    row["post_id"],
                    row["kind"],
                    row.get("author"),
                    row.get("created_utc"),
                    row.get("score", 0),
                    row.get("permalink"),
                    row.get("title"),
                    row["text"],
                    np.asarray(vec, dtype=np.float32),
                ),
            )
            inserted += cur.rowcount
    return inserted


_SEARCH_COLS = ["id", "post_id", "kind", "author", "score", "permalink", "title", "text", "similarity"]


def search(query_vec: np.ndarray, k: int = 5):
    """Cosine-nearest k chunks. Increments retrieval_count on every hit (req #3c)."""
    qv = np.asarray(query_vec, dtype=np.float32)
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, post_id, kind, author, score, permalink, title, text,
                   1 - (embedding <=> %s) AS similarity
            FROM chunks
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (qv, qv, k),
        ).fetchall()
        ids = [r[0] for r in rows]
        if ids:
            conn.execute(
                "UPDATE chunks SET retrieval_count = retrieval_count + 1 WHERE id = ANY(%s)",
                (ids,),
            )
    return [dict(zip(_SEARCH_COLS, r)) for r in rows]


def top_retrieved(limit: int = 15):
    cols = ["kind", "title", "text", "permalink", "retrieval_count", "score"]
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT kind, title, text, permalink, retrieval_count, score
            FROM chunks
            WHERE retrieval_count > 0
            ORDER BY retrieval_count DESC, score DESC
            LIMIT %s
            """,
            (limit,),
        ).fetchall()
    return [dict(zip(cols, r)) for r in rows]


def all_embeddings():
    """Everything needed for the 2D map: vectors + labels + retrieval counts."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, kind, title, text, retrieval_count, embedding FROM chunks"
        ).fetchall()
    return {
        "ids": [r[0] for r in rows],
        "kinds": [r[1] for r in rows],
        "titles": [r[2] for r in rows],
        "texts": [r[3] for r in rows],
        "retrieval_count": [r[4] for r in rows],
        "embeddings": [np.asarray(r[5], dtype=np.float32) for r in rows],
    }
