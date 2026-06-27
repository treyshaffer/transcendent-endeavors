"""Automated weekly ingestion from a subreddit (req #4).

Uses Reddit's public, unauthenticated JSON endpoints so the evaluator needs no
Reddit credentials. Large-data controls (req #4b): cap posts, take top-N comments
per post, chunk long text, and dedup by content hash.

Fallback if Reddit rate-limits the public JSON: swap to PRAW (authenticated) by
setting REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET — see README. The rest of the
pipeline (chunk -> embed -> upsert) is unchanged.
"""
import hashlib
import time

import httpx

from .config import (
    CHUNK_MAX_WORDS,
    CHUNK_OVERLAP_WORDS,
    MAX_COMMENTS_PER_POST,
    MAX_POSTS,
    SUBREDDIT,
)

# Reddit wants a descriptive UA in the form <platform>:<app>:<version> (by /u/<user>).
# Even so, the unauthenticated JSON endpoint 403s some networks (datacenter IPs/VPNs);
# that's why the offline sample corpus exists and the README documents the PRAW path.
_HEADERS = {"User-Agent": "python:community-voices-takehome:0.1 (by /u/community-voices-demo)"}
_SKIP_AUTHORS = {"AutoModerator", None}
_SKIP_BODIES = {"[deleted]", "[removed]", "", None}


def _hash(kind: str, text: str) -> str:
    return hashlib.sha256(f"{kind}\x00{text}".encode("utf-8")).hexdigest()


def _chunk(text: str):
    """Split on words into ~CHUNK_MAX_WORDS windows with overlap."""
    words = text.split()
    if len(words) <= CHUNK_MAX_WORDS:
        return [text.strip()] if text.strip() else []
    step = CHUNK_MAX_WORDS - CHUNK_OVERLAP_WORDS
    out = []
    for start in range(0, len(words), step):
        piece = " ".join(words[start : start + CHUNK_MAX_WORDS]).strip()
        if piece:
            out.append(piece)
        if start + CHUNK_MAX_WORDS >= len(words):
            break
    return out


def _fetch_json(client: httpx.Client, url: str):
    resp = client.get(url, headers=_HEADERS, timeout=30, follow_redirects=True)
    # Translate the common cases into clear, actionable messages.
    if resp.status_code == 404:
        raise ValueError(f"Subreddit not found (404). Check the SUBREDDIT name. [{url}]")
    if resp.status_code == 403:
        raise ValueError(
            "Reddit blocked this request (403). On the unauthenticated JSON endpoint this "
            "usually means the network (datacenter IP/VPN) is blocked — not necessarily a "
            "private subreddit. Use 'Load sample data (offline)' for a deterministic demo, "
            "or set up PRAW with Reddit API credentials (see README)."
        )
    if resp.status_code == 429:
        raise ValueError(
            "Reddit rate-limited this request (429). Wait a minute, or use "
            "'Load sample data (offline)'."
        )
    resp.raise_for_status()
    return resp.json()


def _fetch_comments(client: httpx.Client, permalink: str):
    """Top-scoring comments for a post, capped at MAX_COMMENTS_PER_POST."""
    url = f"https://www.reddit.com{permalink}.json?limit=50&sort=top"
    try:
        payload = _fetch_json(client, url)
    except Exception:
        return []
    if not isinstance(payload, list) or len(payload) < 2:
        return []
    children = payload[1].get("data", {}).get("children", [])
    comments = []
    for child in children:
        if child.get("kind") != "t1":
            continue  # skip "more" / non-comment nodes
        d = child.get("data", {})
        body = d.get("body")
        if body in _SKIP_BODIES or d.get("author") in _SKIP_AUTHORS:
            continue
        comments.append(
            {
                "author": d.get("author"),
                "score": d.get("score", 0),
                "created_utc": d.get("created_utc"),
                "body": body,
            }
        )
    comments.sort(key=lambda c: c.get("score", 0), reverse=True)
    return comments[:MAX_COMMENTS_PER_POST]


def fetch_chunks(subreddit: str = None):
    """Return a deduped list of chunk dicts for the past week.

    Each dict: content_hash, post_id, kind, author, created_utc, score,
    permalink, title, text.
    """
    subreddit = subreddit or SUBREDDIT
    seen_hashes = set()
    rows = []

    with httpx.Client() as client:
        listing = _fetch_json(
            client,
            f"https://www.reddit.com/r/{subreddit}/top.json?t=week&limit=100",
        )
        posts = listing.get("data", {}).get("children", [])[:MAX_POSTS]

        for child in posts:
            d = child.get("data", {})
            if d.get("stickied"):
                continue
            post_id = d.get("id")
            title = d.get("title", "")
            permalink = d.get("permalink")
            base = {
                "post_id": post_id,
                "author": d.get("author"),
                "created_utc": d.get("created_utc"),
                "score": d.get("score", 0),
                "permalink": permalink,
                "title": title,
            }

            def _add(kind, text, score=None, author=None, created=None):
                text = (text or "").strip()
                if not text:
                    return
                h = _hash(kind, text)
                if h in seen_hashes:
                    return
                seen_hashes.add(h)
                row = dict(base)
                row.update(
                    kind=kind,
                    text=text,
                    content_hash=h,
                    score=base["score"] if score is None else score,
                    author=base["author"] if author is None else author,
                    created_utc=base["created_utc"] if created is None else created,
                )
                rows.append(row)

            _add("post_title", title)
            for piece in _chunk(d.get("selftext", "")):
                _add("post_body", piece)

            for c in _fetch_comments(client, permalink):
                for piece in _chunk(c["body"]):
                    _add(
                        "comment",
                        piece,
                        score=c.get("score", 0),
                        author=c.get("author"),
                        created=c.get("created_utc"),
                    )
            time.sleep(0.5)  # be polite to the public endpoint

    return rows
