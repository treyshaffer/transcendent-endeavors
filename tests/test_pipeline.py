"""Fast unit/integration tests for the Community Voices pipeline.

Uses the in-memory backend and deterministic fake embeddings (no torch, no DB, no
network, no API calls), so the whole suite runs in well under a second.
"""
import unittest

import numpy as np

from src import generate, ingest, seed
from src import store_memory as mem


# ---------- helpers ----------------------------------------------------------
def reset_memory():
    mem._ROWS.clear()
    mem._VECS.clear()
    mem._BY_HASH.clear()
    mem._NEXT_ID[0] = 1


_RNG = np.random.default_rng(0)


def fake_embed(texts):
    """Deterministic topic-clustered unit vectors (same topic -> similar)."""
    out = []
    for t in texts:
        base = sum(ord(c) for c in t) % 6
        v = _RNG.normal(size=384) * 0.1
        v[base * 60:(base + 1) * 60] += 1.0
        v = v / np.linalg.norm(v)
        out.append(v.astype(np.float32))
    return np.vstack(out)


class MockClaude:
    """Stands in for anthropic.Anthropic()."""
    def __init__(self, blocks):
        self._blocks = blocks

    class _Block:
        def __init__(self, type_, text=""):
            self.type, self.text = type_, text

    class _Msg:
        def __init__(self, blocks):
            self.content = blocks

    class _Stream:
        def __init__(self, blocks):
            self._m = MockClaude._Msg(blocks)
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def get_final_message(self):
            return self._m

    @property
    def messages(self):
        outer = self
        class _M:
            def stream(self_inner, **kw):
                return MockClaude._Stream(outer._blocks)
        return _M()


def text_block(s):
    return MockClaude._Block("text", s)


def thinking_block():
    return MockClaude._Block("thinking")


# ---------- seed -------------------------------------------------------------
class TestSeed(unittest.TestCase):
    def test_shape_and_themes(self):
        rows = seed.build_rows()
        self.assertEqual(len(rows), 41)
        self.assertEqual(len({r["post_id"] for r in rows}), 10)
        required = {"content_hash", "post_id", "kind", "text", "permalink", "title", "score"}
        self.assertTrue(all(required <= set(r) for r in rows))
        # permalinks follow the configured subreddit
        self.assertTrue(all(r["permalink"].startswith("/r/japanweather/") for r in rows))
        # weather vocabulary present (drives clean clusters)
        blob = " ".join(r["text"].lower() for r in rows)
        for kw in ["typhoon", "tsuyu", "sakura", "heatstroke", "hokkaido", "jma"]:
            self.assertIn(kw, blob)

    def test_hashes_unique(self):
        rows = seed.build_rows()
        hashes = [r["content_hash"] for r in rows]
        self.assertEqual(len(hashes), len(set(hashes)))


# ---------- ingest parsing ---------------------------------------------------
class TestIngest(unittest.TestCase):
    def test_chunking(self):
        short = ingest._chunk("a b c")
        self.assertEqual(short, ["a b c"])
        long = ingest._chunk("word " * 800)
        self.assertGreater(len(long), 1)  # long text splits into multiple chunks

    def test_403_message_is_honest(self):
        class FakeResp:
            status_code = 403
            def json(self_):
                return {}
            def raise_for_status(self_):
                pass
        class FakeClient:
            def get(self_, *a, **k):
                return FakeResp()
        with self.assertRaises(ValueError) as ctx:
            ingest._fetch_json(FakeClient(), "https://x")
        msg = str(ctx.exception).lower()
        self.assertIn("403", msg)
        self.assertIn("sample data", msg)  # points to the offline fallback

    def test_404_message(self):
        class FakeResp:
            status_code = 404
            def raise_for_status(self_):
                pass
        class FakeClient:
            def get(self_, *a, **k):
                return FakeResp()
        with self.assertRaises(ValueError) as ctx:
            ingest._fetch_json(FakeClient(), "https://x")
        self.assertIn("not found", str(ctx.exception).lower())


# ---------- in-memory store --------------------------------------------------
class TestMemoryStore(unittest.TestCase):
    def setUp(self):
        reset_memory()
        self.rows = seed.build_rows()
        self.vecs = fake_embed([r["text"] for r in self.rows])

    def test_upsert_and_dedup(self):
        added = mem.upsert_chunks(self.rows, self.vecs)
        self.assertEqual(added, len(self.rows))
        self.assertEqual(mem.count_chunks(), len(self.rows))
        # re-inserting the same rows inserts nothing (dedup by content_hash)
        again = mem.upsert_chunks(self.rows, self.vecs)
        self.assertEqual(again, 0)
        self.assertEqual(mem.count_chunks(), len(self.rows))

    def test_search_increments_retrieval_count(self):
        mem.upsert_chunks(self.rows, self.vecs)
        before = sum(r["retrieval_count"] for r in mem._ROWS)
        hits = mem.search(self.vecs[0], k=5)
        self.assertEqual(len(hits), 5)
        self.assertIn("similarity", hits[0])
        after = sum(r["retrieval_count"] for r in mem._ROWS)
        self.assertEqual(after - before, 5)

    def test_top_retrieved_and_all_embeddings(self):
        mem.upsert_chunks(self.rows, self.vecs)
        mem.search(self.vecs[0], k=3)
        top = mem.top_retrieved(limit=10)
        self.assertTrue(top and top[0]["retrieval_count"] >= 1)
        data = mem.all_embeddings()
        self.assertEqual(len(data["embeddings"]), len(self.rows))
        self.assertEqual(data["embeddings"][0].shape, (384,))

    def test_search_empty_store(self):
        self.assertEqual(mem.search(self.vecs[0], k=5), [])


# ---------- data-derived retrieval ------------------------------------------
class TestRetrieval(unittest.TestCase):
    def setUp(self):
        reset_memory()
        rows = seed.build_rows()
        mem.upsert_chunks(rows, fake_embed([r["text"] for r in rows]))
        self._orig_embed = generate.embed
        generate.embed = fake_embed  # intent probes use this

    def tearDown(self):
        generate.embed = self._orig_embed

    def test_clustering_retrieval_is_diverse(self):
        ctx = generate.retrieve_context(k=4)
        self.assertGreaterEqual(len(ctx), 6)
        # spans many of the 10 posts -> data-derived, not single-topic
        self.assertGreaterEqual(len({c["post_id"] for c in ctx}), 5)
        # retrieval bumped the counts (req #3c)
        self.assertGreater(sum(r["retrieval_count"] for r in mem._ROWS), 0)

    def test_retrieval_on_empty_store(self):
        reset_memory()
        self.assertEqual(generate.retrieve_context(k=4), [])


# ---------- generation guards & metrics -------------------------------------
class TestGeneration(unittest.TestCase):
    def setUp(self):
        reset_memory()
        rows = seed.build_rows()
        mem.upsert_chunks(rows, fake_embed([r["text"] for r in rows]))
        self._orig_embed = generate.embed
        self._orig_client = generate._client
        generate.embed = fake_embed

    def tearDown(self):
        generate.embed = self._orig_embed
        generate._client = self._orig_client

    def test_rag_counts_citations_and_posts(self):
        generate._client = lambda: MockClaude([
            thinking_block(),
            text_block("- A (https://www.reddit.com/r/japanweather/comments/x/)\n"
                       "- B (reddit.com/r/japanweather/comments/y/)"),
        ])
        res = generate.generate_rag(k=4)
        self.assertEqual(res["mode"], "rag")
        self.assertGreaterEqual(res["citations"], 2)
        self.assertGreaterEqual(res["unique_posts"], 1)
        self.assertGreater(res["context_chunks"], 0)

    def test_norag_does_not_touch_store(self):
        generate._client = lambda: MockClaude([text_block("Generic summary, no links.")])
        res = generate.generate_norag()
        self.assertEqual(res["mode"], "no-rag")
        self.assertEqual(res["context_chunks"], 0)

    def test_empty_text_guard_raises(self):
        generate._client = lambda: MockClaude([thinking_block()])
        with self.assertRaises(RuntimeError):
            generate._complete("sys", "user")

    def test_empty_store_guard(self):
        reset_memory()
        res = generate.generate_rag(k=4)
        self.assertEqual(res["context_chunks"], 0)
        self.assertIn("empty", res["document"].lower())

    def test_citation_counter(self):
        self.assertEqual(generate.count_citations("no links here"), 0)
        self.assertGreaterEqual(
            generate.count_citations("see https://www.reddit.com/r/x/comments/1/"), 1)


# ---------- visualization ----------------------------------------------------
class TestViz(unittest.TestCase):
    def setUp(self):
        reset_memory()
        from src import viz
        self.viz = viz

    def test_small_n_message(self):
        rows = seed.build_rows()[:2]
        mem.upsert_chunks(rows, fake_embed([r["text"] for r in rows]))
        fig, msg = self.viz.build_figure()
        self.assertIsNone(fig)
        self.assertIn("ingest", msg.lower())

    def test_builds_figure(self):
        rows = seed.build_rows()
        mem.upsert_chunks(rows, fake_embed([r["text"] for r in rows]))
        fig, method = self.viz.build_figure()
        self.assertIsNotNone(fig)


# ---------- postgres backend still imports & fails gracefully ----------------
class TestPostgresBackendImportable(unittest.TestCase):
    def test_import_and_public_api(self):
        from src import store_postgres
        for fn in ("ensure_schema", "upsert_chunks", "search", "top_retrieved",
                   "all_embeddings", "count_chunks"):
            self.assertTrue(callable(getattr(store_postgres, fn)))


if __name__ == "__main__":
    unittest.main()
