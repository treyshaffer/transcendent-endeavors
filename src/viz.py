"""Flattened 2D visualization of the stored embeddings (req #3b).

UMAP -> 2D (PCA fallback), KMeans for cluster color. Point size scales with
retrieval_count so the viz visibly ties to the retrieval stats (req #3c).
"""
import numpy as np

from . import store
from .config import SUBREDDIT


def build_figure():
    """Return (matplotlib Figure, method_str) or (None, reason_str)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = store.all_embeddings()
    embs = data["embeddings"]
    if len(embs) < 3:
        return None, f"Need >=3 chunks to plot (have {len(embs)}). Ingest first."

    X = np.vstack(embs)

    # 2D projection
    try:
        import umap

        reducer = umap.UMAP(
            n_components=2, random_state=42, n_neighbors=min(15, len(X) - 1)
        )
        XY = reducer.fit_transform(X)
        method = "UMAP"
    except Exception:
        from sklearn.decomposition import PCA

        XY = PCA(n_components=2).fit_transform(X)
        method = "PCA (UMAP unavailable)"

    # Cluster for color
    from sklearn.cluster import KMeans

    k = min(6, len(X))
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)

    counts = np.array(data["retrieval_count"], dtype=float)
    sizes = 30 + counts * 45  # retrieved points pop out

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        XY[:, 0], XY[:, 1], c=labels, cmap="tab10", s=sizes,
        alpha=0.75, edgecolors="white", linewidths=0.5,
    )
    ax.set_title(
        f"r/{SUBREDDIT} embeddings ({method})\n"
        f"{len(X)} chunks · color = cluster · size = times retrieved"
    )
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    legend = ax.legend(*sc.legend_elements(), title="cluster", loc="best", fontsize=8)
    ax.add_artist(legend)
    fig.tight_layout()
    return fig, method
