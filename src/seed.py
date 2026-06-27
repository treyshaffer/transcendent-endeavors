"""Synthetic, offline SAMPLE corpus for the demo.

A throttle-proof fallback: if Reddit's public JSON rate-limits during a demo, this
seeds a realistic, theme-diverse corpus so the full RAG / A-B / embedding-map /
stats flow still works without the network. The content is *fabricated* sample
data (clearly `seed_*` post ids) themed for a weather community like r/japanweather
(typhoons, rainy season, sakura forecasts, heat, snow, cloud ID, forecasts) so the
embedding clusters and retrieval behave like the real thing.

Output rows match `ingest.fetch_chunks()` exactly, so they flow through the same
embed -> upsert pipeline. Permalinks use the configured SUBREDDIT.
"""
import hashlib

from .config import SUBREDDIT

# Each post: id, title, body, author, score, comments[(body, author, score)].
_POSTS = [
    {
        "id": "seed_typhoon1",
        "title": "Typhoon No. 7 tracking toward Kanto this weekend — latest model runs",
        "body": "JMA has the center making closest approach to Tokyo Saturday night. The "
                "ECMWF and GFS spaghetti are tightening on a track just east of the bay, but "
                "intensity is still uncertain. Expect strong winds and heavy rain bands "
                "Saturday afternoon. Anyone with experience, how reliable is the 3-day cone "
                "this far out?",
        "author": "kanto_skywatch", "score": 248,
        "comments": [
            ("JMA's official track is the one to follow for warnings — the foreign models "
             "are useful for trend but JMA nowcasts the rain bands best.", "jma_first", 121),
            ("Expect keikaku unkyu (planned suspension) on the JR lines if sustained winds "
             "top 25 m/s. Check the railway sites Friday night.", "train_delays", 73),
            ("Tape the big windows and bring in everything from the veranda. Learned that "
             "the hard way in 2019.", "veranda_veteran", 44),
        ],
    },
    {
        "id": "seed_tsuyu1",
        "title": "When will tsuyu (rainy season) actually end in Kanto this year?",
        "body": "Okinawa already declared the end of tsuyu, but Kanto is still stuck under "
                "the seasonal rain front. Humidity has been brutal and nothing dries. Does "
                "anyone have a sense of when JMA usually calls the end for the Tokyo area?",
        "author": "soggy_in_saitama", "score": 187,
        "comments": [
            ("Historically late July for Kanto, but JMA only confirms it retroactively. Watch "
             "for the Pacific High to push north.", "pacific_high", 96),
            ("Mold season. Run the dehumidifier and don't leave laundry out — the guerrilla "
             "downpours come with no warning.", "mold_patrol", 52),
        ],
    },
    {
        "id": "seed_sakura1",
        "title": "2026 sakura (cherry blossom) forecast — first bloom (kaika) dates for Tokyo",
        "body": "First sakura forecasts are out. Tokyo kaika is projected around late March with "
                "full bloom (mankai) about a week later, slightly earlier than average thanks "
                "to a warm February. How much do warm spells vs late cold snaps actually move "
                "these dates?",
        "author": "hanami_planner", "score": 164,
        "comments": [
            ("A late cold snap can delay mankai by days — the forecasts get reliable only "
             "inside ~10 days. Don't book hanami too early.", "bloom_betting", 81),
            ("Rain and wind right at peak bloom is the real risk; the petals don't last.",
             "petal_pessimist", 39),
        ],
    },
    {
        "id": "seed_heat1",
        "title": "Heatstroke alert (necchusho) — 38C and brutal humidity in Osaka today",
        "body": "WBGT index is in the 'danger' band across Kansai this afternoon and the "
                "overnight low barely dropped below 28C. These tropical nights (nettaiya) are "
                "relentless. How is everyone coping without running the AC into the ground?",
        "author": "kansai_swelter", "score": 201,
        "comments": [
            ("Watch the WBGT, not just the air temp — humidity is what gets you. They issued "
             "a heatstroke alert (necchusho keikai) for the whole prefecture.", "wbgt_watch", 110),
            ("Salt tablets, electrolyte drinks, and don't trust 'it's a dry heat' — it never "
             "is here in summer.", "hydrate_hero", 47),
        ],
    },
    {
        "id": "seed_hokkaido1",
        "title": "First snow (hatsuyuki) in Hokkaido — Asahikawa already below freezing",
        "body": "Asahikawa dipped below zero overnight and the peaks around Daisetsuzan have "
                "their first dusting. Feels early this year. Powder season can't come soon "
                "enough — when do the resorts usually open if the early cold holds?",
        "author": "powder_chaser", "score": 142,
        "comments": [
            ("Niseko base usually needs late November/December storms regardless of an early "
             "first snow. One cold night doesn't make a season.", "japow_realist", 88),
            ("Time to swap to winter tires — black ice on the bridges comes before the real "
             "snow does.", "winter_tires", 36),
        ],
    },
    {
        "id": "seed_forecast1",
        "title": "Weekly forecast thread — unsettled with afternoon thunderstorms (yudachi)",
        "body": "Pattern this week: hot and hazy mornings, then atmospheric instability firing "
                "off afternoon thunderstorms (yudachi) inland that drift toward the coast by "
                "evening. Carry an umbrella even on 'sunny' days. Post your local conditions.",
        "author": "weekly_thread_bot", "score": 118,
        "comments": [
            ("These pop-up guerrilla rainstorms (gerira gou) are impossible to time — radar "
             "apps with rain-cloud nowcast are the only thing that helps.", "radar_addict", 70),
            ("Got drenched in Shibuya yesterday under a clear-ish sky 20 minutes earlier. "
             "Always carry the folding umbrella.", "always_umbrella", 33),
        ],
    },
    {
        "id": "seed_cloud1",
        "title": "What is this strange roll cloud over Tokyo Bay this morning?",
        "body": "Saw a long, low, tube-shaped cloud stretching across the bay just after "
                "sunrise, moving toward the city ahead of the rain. Never seen one like it. "
                "Is this a shelf cloud or something rarer? Photo in comments.",
        "author": "cloud_curious", "score": 173,
        "comments": [
            ("That's an arcus/shelf cloud on the leading edge of an outflow boundary — strong "
             "gusts usually arrive right behind it.", "skywarn_jp", 92),
            ("Gorgeous and a little ominous. Means the storm's gust front is about to hit.",
             "gustfront", 41),
        ],
    },
    {
        "id": "seed_quakeweather1",
        "title": "Is 'earthquake weather' (jishin biyori) a real thing? It felt muggy before.",
        "body": "Heard people say muggy, still days precede earthquakes. Felt oppressively "
                "humid right before a jolt last week. Is there any meteorological basis for "
                "this, or is it pure confirmation bias?",
        "author": "skeptical_shaker", "score": 129,
        "comments": [
            ("No physical link — weather and tectonics operate on totally different systems. "
             "JMA has debunked this repeatedly. Classic confirmation bias.", "debunk_dan", 104),
            ("You remember the muggy days that had a quake and forget the thousands that "
             "didn't.", "selective_memory", 38),
        ],
    },
    {
        "id": "seed_prep1",
        "title": "Typhoon prep checklist — what do you ACTUALLY do before landfall?",
        "body": "With the season ramping up, what's on your before-landfall checklist? Mine: "
                "fill water containers, charge battery banks, tape windows, clear the veranda "
                "and balcony drains, and download the offline NHK and JMA warning apps. What "
                "am I missing?",
        "author": "prep_pragmatist", "score": 156,
        "comments": [
            ("Clear the balcony drain — flooding from a blocked drain ruins more apartments "
             "than broken windows do. And know your hazard map (hazaado mappu).", "drain_first", 99),
            ("Cash on hand in case the power and card readers go down, plus a battery radio "
             "for the JMA emergency warnings.", "blackout_ready", 45),
        ],
    },
    {
        "id": "seed_jma1",
        "title": "JMA vs Windy vs tenki.jp — which forecast source do you actually trust?",
        "body": "Everyone has a favorite. I default to JMA for official warnings, Windy for "
                "the model layers when a typhoon is out at sea, and tenki.jp for the hourly "
                "local rain. Curious what mix others rely on and why.",
        "author": "forecast_nerd", "score": 138,
        "comments": [
            ("JMA for anything official — warnings, advisories, the rain-cloud nowcast. The "
             "others are nice-to-have but JMA is the source of record here.", "official_only", 86),
            ("Windy's ECMWF layer is great for typhoon track trends days out; just don't "
             "treat raw model output as a forecast.", "model_vs_forecast", 40),
        ],
    },
]


def _hash(kind: str, text: str) -> str:
    return hashlib.sha256(f"{kind}\x00{text}".encode("utf-8")).hexdigest()


def build_rows():
    """Return chunk rows matching ingest.fetch_chunks() output shape."""
    rows = []
    base_ts = 1_750_000_000.0  # static so the data is deterministic
    for i, post in enumerate(_POSTS):
        permalink = f"/r/{SUBREDDIT}/comments/{post['id']}/{post['id']}/"
        created = base_ts + i * 3600
        common = {
            "post_id": post["id"],
            "permalink": permalink,
            "title": post["title"],
        }

        rows.append({
            **common, "kind": "post_title", "text": post["title"],
            "author": post["author"], "score": post["score"], "created_utc": created,
            "content_hash": _hash("post_title", post["title"]),
        })
        if post.get("body"):
            rows.append({
                **common, "kind": "post_body", "text": post["body"],
                "author": post["author"], "score": post["score"], "created_utc": created,
                "content_hash": _hash("post_body", post["body"]),
            })
        for j, (body, author, score) in enumerate(post["comments"]):
            rows.append({
                **common, "kind": "comment", "text": body,
                "author": author, "score": score, "created_utc": created + j * 60,
                "content_hash": _hash("comment", body),
            })
    return rows


def seed():
    """Embed + upsert the synthetic rows. Returns (n_rows, n_inserted)."""
    from .embeddings import embed
    from . import store

    rows = build_rows()
    vecs = embed([r["text"] for r in rows])
    inserted = store.upsert_chunks(rows, vecs)
    return len(rows), inserted


if __name__ == "__main__":
    from . import store

    store.ensure_schema()
    total, inserted = seed()
    print(f"Seeded {total} sample chunks ({inserted} newly inserted).")
