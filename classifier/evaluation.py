"""Evaluation: rating pools, the rating page, and the quality report.

    python -m classifier.evaluation pools     # build a pool of jobs to rate for each resume in eval/resumes/
    python -m classifier.evaluation report    # NDCG@10 / P@10 of the ranker vs keyword search, train vs test

Resumes (eval/resumes/), pools and ratings (eval/private/) contain personal data and are git-ignored.
Each pool mixes the ranker's top jobs, plain keyword search's top jobs and random same-country jobs,
shuffled, so raters cannot tell which system picked what."""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import random
import sqlite3
import sys
from datetime import datetime, timezone

from fastapi import Depends, Form
from fastapi.responses import HTMLResponse, RedirectResponse

from classifier import store
from classifier.ranker import Preferences, bm25_scores, rank
from classifier.resume import extract_text, parse_resume

RESUMES_DIR = "eval/resumes"
PRIVATE = "eval/private"
POOLS = f"{PRIVATE}/pools.json"
RATINGS = f"{PRIVATE}/ratings.db"
FROM_RANKER, FROM_KEYWORDS, FROM_RANDOM = 12, 8, 5
TEST_SHARE = 1 / 3


def _ratings_db() -> sqlite3.Connection:
    os.makedirs(PRIVATE, exist_ok=True)
    db = sqlite3.connect(RATINGS)
    db.execute(
        "CREATE TABLE IF NOT EXISTS ratings (resume_id TEXT, job_id TEXT, rater TEXT, rating INTEGER, at TEXT, "
        "PRIMARY KEY (resume_id, job_id, rater))"
    )
    return db


def _load_pools() -> dict:
    return json.load(open(POOLS)) if os.path.exists(POOLS) else {}


def resume_id(filename: str) -> str:
    return hashlib.sha1(filename.encode()).hexdigest()[:10]


def split_of(rid: str) -> str:
    """Fixed train/test assignment, so test resumes never influence tuning."""
    return "test" if int(rid, 16) % 1000 < TEST_SHARE * 1000 else "train"


def build_pools() -> None:
    os.makedirs(PRIVATE, exist_ok=True)
    pools = _load_pools()
    db = store.connect()
    files = sorted(f for f in os.listdir(RESUMES_DIR) if f.lower().endswith((".pdf", ".docx", ".txt")))
    for name in files:
        rid = resume_id(name)
        if rid in pools:
            continue
        text = extract_text(open(os.path.join(RESUMES_DIR, name), "rb").read(), name)
        resume = parse_resume(text)
        country = resume.location_country
        prefs = Preferences(countries=(country,) if country else ("US", "GB", "CA"), city=resume.location_city, region=resume.location_region)
        picked: list[str] = []
        for m in rank(resume, prefs, limit=FROM_RANKER, db=db):
            picked.append(m.job_id)
        for jid in bm25_scores(resume, db, limit=60):
            if len(picked) >= FROM_RANKER + FROM_KEYWORDS:
                break
            if jid not in picked:
                picked.append(jid)
        like = f"%,{country},%" if country else "%"
        for (jid,) in db.execute(
            "SELECT id FROM jobs WHERE duplicate_of IS NULL AND countries LIKE ? ORDER BY random() LIMIT ?", (like, FROM_RANDOM * 2)
        ):
            if len(picked) >= FROM_RANKER + FROM_KEYWORDS + FROM_RANDOM:
                break
            if jid not in picked:
                picked.append(jid)
        random.Random(rid).shuffle(picked)
        jobs = {}
        for jid in picked:
            row = db.execute("SELECT * FROM jobs WHERE id = ?", (jid,)).fetchone()
            locs = json.loads(row["locations"])
            jobs[jid] = {
                "title": row["title"], "company": row["company"], "url": row["url"], "remote": row["remote"],
                "location": "; ".join(loc["raw"] for loc in locs[:3]), "description": row["description"][:8000],
            }
        pools[rid] = {"file": name, "text": text, "split": split_of(rid), "order": picked, "jobs": jobs,
                      "built": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        print(f"  {name}: {len(picked)} jobs ({pools[rid]['split']})", flush=True)
        json.dump(pools, open(POOLS, "w"))
    print(f"{len(pools)} resumes pooled")


# ---- Metrics ----

def _dcg(gains: list[float]) -> float:
    return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at(ranked: list[str], rel: dict[str, float], k: int = 10) -> float:
    ideal = _dcg(sorted(rel.values(), reverse=True)[:k])
    return _dcg([rel.get(j, 0.0) for j in ranked[:k]]) / ideal if ideal else math.nan


def precision_at(ranked: list[str], rel: dict[str, float], k: int = 10, good: float = 2.0) -> float:
    top = ranked[:k]
    return sum(1 for j in top if rel.get(j, 0) >= good) / len(top) if top else math.nan


def averaged_ratings(raters: str | None = None) -> dict[str, dict[str, float]]:
    """Mean rating per (resume, job). raters: "all" (default), "ai" or "human" (env EVAL_RATERS)."""
    raters = raters or os.environ.get("EVAL_RATERS", "all")
    where = {"ai": " AND rater LIKE 'ai:%'", "human": " AND rater NOT LIKE 'ai:%'"}.get(raters, "")
    out: dict[str, dict[str, list[int]]] = {}
    for rid, jid, r in _ratings_db().execute("SELECT resume_id, job_id, rating FROM ratings WHERE rating >= 0" + where):
        out.setdefault(rid, {}).setdefault(jid, []).append(r)
    return {rid: {j: sum(v) / len(v) for j, v in jobs.items()} for rid, jobs in out.items()}


def report() -> None:
    pools, ratings = _load_pools(), averaged_ratings()
    db = store.connect()
    results: dict[str, dict[str, list[float]]] = {}
    for rid, pool in pools.items():
        rel = ratings.get(rid, {})
        if len(rel) < 0.8 * len(pool["order"]):
            continue  # only fully rated pools count
        rated = [j for j in pool["order"] if j in rel]
        resume = parse_resume(pool["text"])
        live = [j for j in rated if db.execute("SELECT 1 FROM jobs WHERE id=?", (j,)).fetchone()]
        systems = {
            "ranker": [m.job_id for m in rank(resume, Preferences(), limit=len(live), db=db, only=live, group=False)],
            "keywords": sorted(live, key=lambda j: -bm25_scores(resume, db, limit=len(live), only=live).get(j, 0.0)),
            "random": random.Random(rid).sample(live, len(live)),
        }
        for name, ranked in systems.items():
            r = results.setdefault(f"{pool['split']}:{name}", {"ndcg10": [], "p10": []})
            r["ndcg10"].append(ndcg_at(ranked, rel))
            r["p10"].append(precision_at(ranked, rel))
    if not results:
        print("No fully rated resumes yet.")
        return
    print(f"{'split:system':22} {'resumes':>7} {'NDCG@10':>8} {'P@10':>6}")
    for key in sorted(results):
        vals = results[key]
        nd = [v for v in vals["ndcg10"] if not math.isnan(v)]
        p = [v for v in vals["p10"] if not math.isnan(v)]
        print(f"{key:22} {len(nd):7} {sum(nd) / len(nd):8.3f} {sum(p) / len(p):6.2f}")


# ---- Rating page ----

GUIDE = """<b>3</b> great fit: I'd apply today &nbsp; <b>2</b> good fit: worth applying &nbsp;
<b>1</b> weak: a stretch or a neighbouring field &nbsp; <b>0</b> not a fit. Judge the person, not the score."""

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rate matches</title><style>
:root{{--bg:#fafaf9;--fg:#1c1917;--muted:#57534e;--card:#fff;--edge:#e7e5e4;--accent:#c2410c}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0c0a09;--fg:#f5f5f4;--muted:#a8a29e;--card:#1c1917;--edge:#292524;--accent:#fb923c}}}}
body{{margin:0;font:15px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--fg)}}
header{{padding:12px 16px;border-bottom:1px solid var(--edge);display:flex;gap:16px;align-items:center;flex-wrap:wrap}}
main{{display:grid;grid-template-columns:1fr 1fr;gap:16px;padding:16px}}
@media (max-width:800px){{main{{grid-template-columns:1fr}}}}
section{{background:var(--card);border:1px solid var(--edge);border-radius:12px;padding:16px;max-height:70vh;overflow:auto;white-space:pre-wrap}}
h2{{margin:0 0 4px;font-size:18px}} .muted{{color:var(--muted)}}
form{{display:flex;gap:8px;flex-wrap:wrap}} button{{font:inherit;font-weight:600;padding:10px 18px;border-radius:10px;border:1px solid var(--edge);background:var(--card);color:var(--fg);cursor:pointer}}
button:hover{{border-color:var(--accent)}} .bar{{position:sticky;bottom:0;background:var(--bg);padding:12px 16px;border-top:1px solid var(--edge)}}
</style></head><body>
<header><b>Rate matches</b><span class="muted">{who} · {done}/{total} rated</span><span class="muted">{guide}</span></header>
<main><section><h2>Resume</h2>{resume}</section>
<section><h2>{title}</h2><div class="muted">{company} · {location} · {remote} · <a href="{url}" target="_blank" rel="noopener">posting</a></div><hr>{description}</section></main>
<div class="bar"><form method="post" action="/eval/rate">
<input type="hidden" name="resume_id" value="{rid}"><input type="hidden" name="job_id" value="{jid}">
<button name="rating" value="0">0 not a fit</button><button name="rating" value="1">1 weak</button>
<button name="rating" value="2">2 good</button><button name="rating" value="3">3 great</button>
<button name="rating" value="-1" class="muted">skip</button></form></div>
<script>document.addEventListener('keydown',e=>{{const b=document.querySelector('button[value="'+e.key+'"]');if(b&&!e.target.closest('input,textarea'))b.click();}});</script>
</body></html>"""


def mount(app, rater_dep) -> None:
    @app.get("/eval", response_class=HTMLResponse)
    def next_pair(who: str = Depends(rater_dep)):
        pools = _load_pools()
        db = _ratings_db()
        mine = {(r, j) for r, j in db.execute("SELECT resume_id, job_id FROM ratings WHERE rater=?", (who,))}
        total = sum(len(p["order"]) for p in pools.values())
        # Finish one resume before starting the next, so context carries over.
        for rid, pool in sorted(pools.items(), key=lambda kv: kv[1]["file"]):
            for jid in pool["order"]:
                if (rid, jid) not in mine:
                    job = pool["jobs"][jid]
                    return PAGE.format(
                        who=html.escape(who), done=len(mine), total=total, guide=GUIDE, resume=html.escape(pool["text"]),
                        title=html.escape(job["title"]), company=html.escape(job["company"]), location=html.escape(job["location"] or "—"),
                        remote=job["remote"], url=html.escape(job["url"]), description=html.escape(job["description"]), rid=rid, jid=html.escape(jid),
                    )
        return HTMLResponse("<p style='font:16px system-ui;padding:24px'>All done. Thank you!</p>")

    @app.post("/eval/rate")
    def save(resume_id: str = Form(...), job_id: str = Form(...), rating: int = Form(...), who: str = Depends(rater_dep)):
        if rating in (0, 1, 2, 3, -1):
            db = _ratings_db()
            # -1 is a skip: remembered so the pair is not shown again, ignored by the metrics.
            db.execute("INSERT OR REPLACE INTO ratings VALUES (?,?,?,?,?)",
                       (resume_id, job_id, who, rating, datetime.now(timezone.utc).isoformat(timespec="seconds")))
            db.commit()
        return RedirectResponse("/eval", status_code=303)


if __name__ == "__main__":
    {"pools": build_pools, "report": report}[sys.argv[1]]()
