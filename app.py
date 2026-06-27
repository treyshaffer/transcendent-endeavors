"""Community Voices — Streamlit app.

Flow: Ingest this week -> Generate (No-RAG vs RAG side by side) -> Embedding map
-> Retrieval stats. One screen, four tabs.
"""
import streamlit as st

from src import generate, ingest, seed, store, viz
from src.config import GEN_MODEL, SUBREDDIT, VECTOR_BACKEND
from src.embeddings import embed

st.set_page_config(page_title="Community Voices", layout="wide")


@st.cache_resource
def _bootstrap():
    """Ensure the vector schema exists once per server process. Tolerant of a
    down DB so the UI still renders a helpful message instead of a stack trace."""
    try:
        store.ensure_schema()
        return None
    except Exception as e:
        return str(e)


_boot_err = _bootstrap()

st.title("🏚️ Community Voices")
_store_label = "pgvector" if VECTOR_BACKEND == "postgres" else "in-memory (no DB)"
st.caption(
    f"RAG-powered weekly digest for **r/{SUBREDDIT}** · store: `{_store_label}` · "
    f"generation by `{GEN_MODEL}`"
)
if _boot_err:
    st.error(
        f"Couldn't reach Postgres / apply schema: {_boot_err}\n\n"
        "Start the DB with `docker compose up -d db`, or set `VECTOR_BACKEND=memory` "
        "in `.env` to run with no database, then reload."
    )

def _embed_and_store(rows):
    """Shared pipeline for both live ingest and the synthetic seed."""
    vecs = embed([r["text"] for r in rows])
    return store.upsert_chunks(rows, vecs)


# ---- Sidebar: ingestion + corpus state -------------------------------------
with st.sidebar:
    st.header("1 · Ingest")
    st.write(f"Pull this past week of **r/{SUBREDDIT}** into the pgvector store.")
    if st.button("Ingest this week", type="primary", use_container_width=True):
        with st.spinner("Fetching Reddit, chunking, embedding…"):
            try:
                rows = ingest.fetch_chunks()
                if rows:
                    added = _embed_and_store(rows)
                    st.success(f"Fetched {len(rows)} chunks · {added} new inserted.")
                else:
                    st.warning(
                        "No chunks returned (Reddit may have rate-limited). "
                        "Use **Load sample data** below to run the demo offline."
                    )
            except Exception as e:  # surface fetch/DB issues plainly
                st.error(f"Ingestion failed: {e}\nTry **Load sample data** below.")

    st.caption("No Reddit access? Throttle-proof fallback (fixed sample corpus):")
    if st.button("Load sample data (offline)", use_container_width=True):
        with st.spinner("Embedding sample corpus…"):
            try:
                added = _embed_and_store(seed.build_rows())
                st.success(f"Seeded sample chunks · {added} new inserted.")
            except Exception as e:
                st.error(f"Seeding failed: {e}")

    try:
        total = store.count_chunks()
    except Exception as e:
        total = None
        st.error(f"DB not reachable: {e}\nIs `docker compose up -d db` running?")
    if total is not None:
        st.metric("Chunks in store", total)

    if generate._use_stub():
        st.info("🧪 Simulated generation (no API key) — documents are stitched from "
                "retrieved context, not a live LLM. Set `ANTHROPIC_API_KEY` in `.env` "
                "(or `GEN_BACKEND=anthropic`) for real model output.")

# ---- Tabs ------------------------------------------------------------------
tab_docs, tab_map, tab_stats = st.tabs(
    ["📄 Documents (A/B)", "🗺️ Embedding map", "📊 Retrieval stats"]
)

with tab_docs:
    st.subheader("2 · Generate the Community Voices Document")
    st.write(
        "Same task, two ways. **No-RAG** uses Claude's prior knowledge only; "
        "**RAG** grounds the document in this week's retrieved posts and cites them."
    )
    if st.button("Generate both (A/B)", type="primary"):
        if total is None:
            st.error("Database not reachable — start it with `docker compose up -d db`.")
        elif not total:
            st.warning("Ingest some posts or load sample data first (sidebar).")
        else:
            try:
                with st.spinner("Asking Claude (no-RAG baseline)…"):
                    st.session_state["norag"] = generate.generate_norag()
                with st.spinner("Retrieving context + asking Claude (RAG)…"):
                    st.session_state["rag"] = generate.generate_rag()
            except Exception as e:
                st.error(f"Generation failed: {e}\nCheck ANTHROPIC_API_KEY / model and retry.")

    norag = st.session_state.get("norag")
    rag = st.session_state.get("rag")
    if norag or rag:
        left, right = st.columns(2)
        with left:
            st.markdown("### ⬜ No-RAG (baseline)")
            if norag:
                st.caption(f"permalink citations: {norag['citations']} · context chunks: 0")
                st.markdown(norag["document"])
        with right:
            st.markdown("### 🟩 RAG-empowered")
            if rag:
                st.caption(
                    f"permalink citations: {rag['citations']} · "
                    f"context chunks retrieved: {rag['context_chunks']} · "
                    f"from {rag.get('unique_posts', 0)} unique posts"
                )
                st.markdown(rag["document"])

        if norag and rag:
            st.divider()
            st.subheader("A/B comparison (req #5)")
            c1, c2, c3 = st.columns(3)
            c1.metric("No-RAG citations", norag["citations"])
            c2.metric("RAG citations", rag["citations"], delta=rag["citations"] - norag["citations"])
            c3.metric("RAG: unique posts grounded", rag.get("unique_posts", 0))
            st.info(
                "RAG grounds claims in real, retrieved posts (more concrete citations, "
                "recency, less hallucination); the baseline is fluent but generic and "
                "cannot reference this week's actual threads. Retrieval is data-derived "
                "(embeddings clustered, cluster centroids used as queries), so it adapts "
                "to whatever community is configured — no hardcoded topics."
            )

with tab_map:
    st.subheader("Flattened embedding map (req #3b)")
    st.write("UMAP → 2D, colored by KMeans cluster. Point size = how often a chunk was retrieved.")
    if st.button("Render embedding map"):
        try:
            with st.spinner("Projecting embeddings…"):
                fig, method = viz.build_figure()
            if fig is None:
                st.warning(method)
            else:
                st.pyplot(fig)
                st.caption(f"Projection method: {method}")
        except Exception as e:
            st.error(f"Could not build the map: {e}")

with tab_stats:
    st.subheader("Retrieval stats (req #3c)")
    st.write("Which chunks RAG pulls most. `retrieval_count` is incremented on every search hit.")
    if st.button("Refresh stats"):
        try:
            rows = store.top_retrieved(limit=20)
        except Exception as e:
            rows = None
            st.error(f"DB not reachable: {e}")
        if rows is None:
            pass
        elif not rows:
            st.info("Nothing retrieved yet — run an RAG generation first.")
        else:
            st.dataframe(
                [
                    {
                        "retrieved": r["retrieval_count"],
                        "kind": r["kind"],
                        "title": (r["title"] or "")[:60],
                        "text": r["text"][:80],
                        "score": r["score"],
                    }
                    for r in rows
                ],
                use_container_width=True,
                hide_index=True,
            )
