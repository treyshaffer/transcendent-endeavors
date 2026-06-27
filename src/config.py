"""Central configuration. Loads .env so ANTHROPIC_API_KEY / DATABASE_URL are available."""
import os

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://cv:cv@localhost:5432/community_voices"
)

# Vector store backend. Default "memory" = zero-setup (no Docker/Postgres) so the app
# runs with the least possible setup. Set "postgres" to use pgvector (the brief's
# vector DB; needs `docker compose up -d db` or a native Postgres + db/schema.sql).
VECTOR_BACKEND = os.environ.get("VECTOR_BACKEND", "memory").lower()

SUBREDDIT = os.environ.get("SUBREDDIT", "japanweather")

# Generation needs a key; ingest/seed/viz/stats don't — so warn, don't hard-fail.
HAS_API_KEY = bool(os.environ.get("ANTHROPIC_API_KEY"))

# Generation model. Default to Opus for the demo; override via env for cheap
# iteration, e.g. GEN_MODEL=claude-haiku-4-5 (exact ids, no date suffix).
GEN_MODEL = os.environ.get("GEN_MODEL", "claude-opus-4-8")

# Local embedding model + its dimensionality (must match db/schema.sql vector(384)).
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384

# Large-data controls (req #4b): keep a week's ingest bounded.
MAX_POSTS = 60
MAX_COMMENTS_PER_POST = 15
CHUNK_MAX_WORDS = 350
CHUNK_OVERLAP_WORDS = 50
