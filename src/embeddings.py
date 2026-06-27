"""Local embeddings via sentence-transformers (no API key required)."""
from functools import lru_cache

import numpy as np

from .config import EMBED_DIM, EMBED_MODEL


@lru_cache(maxsize=1)
def _model():
    # Imported lazily so importing this module is cheap until embeddings are needed.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL)


def embed(texts) -> np.ndarray:
    """Embed a list of strings -> (n, 384) float32, L2-normalized for cosine."""
    texts = list(texts)
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    vecs = _model().encode(
        texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    )
    # Fail fast if the model dim drifts from the DB column (vector(384)).
    if vecs.shape[1] != EMBED_DIM:
        raise ValueError(
            f"Embedding dim {vecs.shape[1]} != EMBED_DIM {EMBED_DIM}; "
            f"update db/schema.sql vector(...) and config.EMBED_DIM to match {EMBED_MODEL}."
        )
    return vecs.astype(np.float32)
