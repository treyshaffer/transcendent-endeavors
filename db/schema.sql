-- Community Voices vector store.
-- Idempotent: safe to run on every app startup (ensure_schema) and as a
-- docker-entrypoint-initdb.d script on first container init.

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per text chunk (post title, post body chunk, or comment).
-- 384 dims == all-MiniLM-L6-v2.
CREATE TABLE IF NOT EXISTS chunks (
    id              BIGSERIAL PRIMARY KEY,
    content_hash    TEXT UNIQUE NOT NULL,            -- dedup key
    post_id         TEXT NOT NULL,
    kind            TEXT NOT NULL,                   -- post_title | post_body | comment
    author          TEXT,
    created_utc     DOUBLE PRECISION,
    score           INTEGER DEFAULT 0,
    permalink       TEXT,                            -- citation target
    title           TEXT,                            -- parent post title for context
    text            TEXT NOT NULL,
    embedding       vector(384),
    retrieval_count INTEGER NOT NULL DEFAULT 0,      -- req #3c: how often retrieved
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Cosine-distance ANN index (HNSW needs no training data, unlike ivfflat).
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS chunks_retrieval_idx ON chunks (retrieval_count DESC);
