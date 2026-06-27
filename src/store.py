"""Vector store dispatcher.

VECTOR_BACKEND selects the implementation:
  - "postgres" (default): pgvector — the brief's vector DB (src/store_postgres.py)
  - "memory": zero-setup numpy fallback, no Docker/Postgres (src/store_memory.py)

The chosen backend is imported lazily, so `memory` mode never imports psycopg and
`postgres` mode never needs the in-memory module. All callers just use `store.*`.
"""
from .config import VECTOR_BACKEND

if VECTOR_BACKEND == "memory":
    from .store_memory import (  # noqa: F401
        all_embeddings, count_chunks, ensure_schema, search, top_retrieved, upsert_chunks,
    )
else:
    from .store_postgres import (  # noqa: F401
        all_embeddings, count_chunks, ensure_schema, search, top_retrieved, upsert_chunks,
    )
