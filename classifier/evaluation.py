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

# Broad fields for filtering the rating page, from the resume's O*NET occupation.
FIELDS = [
    ("data", ("15-2051", "15-2041", "15-2031", "15-20")),
    ("software", ("15-12", "11-3021", "15-1299")),
    ("sales", ("41-", "11-2022")),
    ("marketing", ("11-2021", "13-1161", "27-3031", "11-2032")),
    ("healthcare", ("29-", "31-", "11-9111", "21-1014")),
    ("finance", ("13-2", "11-3031", "43-3031")),
    ("design", ("27-1", "15-1255")),
    ("people-ops", ("13-1071", "11-3121", "13-1151", "11-1021", "13-1111", "13-1082", "13-1081", "13-1023", "43-")),
    ("education", ("25-",)),
    ("engineering", ("17-",)),
]


def field_of(occupation: str | None) -> str:
    for name, prefixes in FIELDS:
        if occupation and occupation.startswith(prefixes):
            return name
    return "other"
TEST_SHARE = 1 / 3


def _ratings_db() -> sqlite3.Connection:
    os.makedirs(PRIVATE, exist_ok=True)
    # The rating page and the AI judges write at the same time: wait for the lock instead of failing.
    db = sqlite3.connect(RATINGS, timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute(
        "CREATE TABLE IF NOT EXISTS ratings (resume_id TEXT, job_id TEXT, rater TEXT, rating INTEGER, at TEXT, "
        "PRIMARY KEY (resume_id, job_id, rater))"
    )
    db.execute(
        "CREATE TABLE IF NOT EXISTS comments (resume_id TEXT, job_id TEXT, rater TEXT, comment TEXT, at TEXT, "
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
                      "field": field_of(resume.occupation), "title": resume.current_title,
                      "built": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        print(f"  {name}: {pools[rid]['field']} / {resume.current_title} -> {len(picked)} jobs ({pools[rid]['split']})", flush=True)
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
    """The label for each (resume, job). raters (env EVAL_RATERS):
    "human": mean of human ratings; "ai": the cross-checked AI verdict (ai:final), else the mean of
    the AI judges; "all" (default): human ratings where they exist, otherwise the AI label."""
    raters = raters or os.environ.get("EVAL_RATERS", "all")
    human: dict[tuple[str, str], list[int]] = {}
    final: dict[tuple[str, str], int] = {}
    judges: dict[tuple[str, str], list[int]] = {}
    for rid, jid, who, r in _ratings_db().execute("SELECT resume_id, job_id, rater, rating FROM ratings WHERE rating >= 0"):
        if who == "ai:final":
            final[(rid, jid)] = r
        elif who.startswith("ai:"):
            judges.setdefault((rid, jid), []).append(r)
        else:
            human.setdefault((rid, jid), []).append(r)
    labels: dict[tuple[str, str], float] = {}
    if raters in ("ai", "all"):
        labels.update({k: sum(v) / len(v) for k, v in judges.items()})
        labels.update({k: float(v) for k, v in final.items()})
    if raters in ("human", "all"):
        labels.update({k: sum(v) / len(v) for k, v in human.items()})
    out: dict[str, dict[str, float]] = {}
    for (rid, jid), value in labels.items():
        out.setdefault(rid, {})[jid] = value
    return out


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
form{{display:flex;gap:8px;flex-wrap:wrap}} textarea{{flex:1 1 100%;font:inherit;padding:8px 10px;border-radius:10px;border:1px solid var(--edge);background:var(--card);color:var(--fg)}} button{{font:inherit;font-weight:600;padding:10px 18px;border-radius:10px;border:1px solid var(--edge);background:var(--card);color:var(--fg);cursor:pointer}}
button:hover{{border-color:var(--accent)}} .last{{margin:12px 16px 0;padding:12px 16px;border:1px solid var(--edge);border-radius:12px;background:var(--card)}}
.last table,.rev{{border-collapse:collapse;width:100%}} .last td,.rev td,.rev th{{padding:6px 8px;border-top:1px solid var(--edge);vertical-align:top;text-align:left}}
.score{{font-weight:700;font-size:16px;white-space:nowrap}} .ok{{color:#15803d}} .off{{color:#b91c1c}}
@media (prefers-color-scheme:dark){{.ok{{color:#4ade80}} .off{{color:#f87171}}}} .bar{{position:sticky;bottom:0;background:var(--bg);padding:12px 16px;border-top:1px solid var(--edge)}}
</style></head><body>
<header><b>Rate matches</b><span class="muted">{who} · {done}/{total} rated</span><span class="muted">{guide}</span>
<a href="/eval/review" class="muted">Compare with the models →</a></header>
{last}
<main><section><h2>Resume</h2>{resume}</section>
<section><h2>{title}</h2><div class="muted">{company} · {location} · {remote} · <a href="{url}" target="_blank" rel="noopener">posting</a></div><hr>{description}</section></main>
<div class="bar"><form method="post" action="/eval/rate?field={field}&set={set}">
<textarea name="comment" rows="2" placeholder="Your notes or assumptions (optional), e.g. 'needs a CA license', 'title fits but too senior'"></textarea>
<input type="hidden" name="resume_id" value="{rid}"><input type="hidden" name="job_id" value="{jid}">
<button name="rating" value="0">0 not a fit</button><button name="rating" value="1">1 weak</button>
<button name="rating" value="2">2 good</button><button name="rating" value="3">3 great</button>
<button name="rating" value="-1" class="muted">skip</button></form></div>
<script>document.addEventListener('keydown',e=>{{const b=document.querySelector('button[value="'+e.key+'"]');if(b&&!e.target.closest('input,textarea'))b.click();}});</script>
</body></html>"""


MODELS = [("ai:gemini", "Gemini"), ("ai:deepseek", "DeepSeek"), ("ai:final", "Cross-checked final")]


def _model_view(db: sqlite3.Connection, rid: str, jid: str) -> dict[str, tuple[int, str]]:
    out = {}
    for who, rating in db.execute("SELECT rater, rating FROM ratings WHERE resume_id=? AND job_id=? AND rater LIKE 'ai:%'", (rid, jid)):
        reason = db.execute("SELECT reason FROM ai_reasons WHERE resume_id=? AND job_id=? AND rater=?", (rid, jid, who)).fetchone() if _has_table(db, "ai_reasons") else None
        out[who] = (rating, reason[0] if reason else "")
    return out


def _has_table(db: sqlite3.Connection, name: str) -> bool:
    return bool(db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone())


def _cell(rating: int | None, mine: int | None, reason: str = "") -> str:
    if rating is None:
        return "<td class='muted'>not rated yet</td>"
    cls = "" if mine is None else ("ok" if abs(rating - mine) <= 0 else ("" if abs(rating - mine) == 1 else "off"))
    return f"<td><span class='score {cls}'>{rating}</span> <span class='muted'>{html.escape(reason)}</span></td>"


def _last_panel(db: sqlite3.Connection, who: str, pools: dict) -> str:
    row = db.execute("SELECT resume_id, job_id, rating FROM ratings WHERE rater=? AND rating >= 0 ORDER BY at DESC LIMIT 1", (who,)).fetchone()
    if not row:
        return ""
    rid, jid, mine = row
    job = pools.get(rid, {}).get("jobs", {}).get(jid)
    if not job:
        return ""
    models = _model_view(db, rid, jid)
    cells = "".join(f"<tr><td>{name}</td>{_cell(models.get(key, (None, ''))[0], mine, models.get(key, (None, ''))[1])}</tr>" for key, name in MODELS)
    return (f"<div class='last'><b>Your last rating: {mine}</b> <span class='muted'>for {html.escape(job['title'])} at "
            f"{html.escape(job['company'])}</span><table>{cells}</table></div>")


REVIEW = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>You vs the models</title><style>
:root{{--bg:#fafaf9;--fg:#1c1917;--muted:#57534e;--card:#fff;--edge:#e7e5e4}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0c0a09;--fg:#f5f5f4;--muted:#a8a29e;--card:#1c1917;--edge:#292524}}}}
body{{margin:0;font:14px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--fg)}} main{{padding:16px;max-width:1400px;margin:auto}}
.muted{{color:var(--muted)}} .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:12px 0 20px}}
.card{{background:var(--card);border:1px solid var(--edge);border-radius:12px;padding:12px 16px}} .big{{font-size:28px;font-weight:700}}
.wrap{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;background:var(--card)}} td,th{{padding:8px;border-top:1px solid var(--edge);vertical-align:top;text-align:left}}
th{{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}} .score{{font-weight:700;font-size:16px}}
.ok{{color:#15803d}} .off{{color:#b91c1c}} @media (prefers-color-scheme:dark){{.ok{{color:#4ade80}} .off{{color:#f87171}}}}
</style></head><body><main><p><a href="/eval?set=check">← Back to rating</a></p><h1>You vs the models</h1>
<p class="muted">{who}: {n} jobs rated. <b class="ok">Green</b> = same score as you, <b class="off">red</b> = 2+ points apart.</p>
<div class="cards">{cards}</div><div class="wrap"><table><tr><th>Job</th><th>You</th><th>Gemini</th><th>DeepSeek</th><th>Cross-checked final</th></tr>
{rows}</table></div></main></body></html>"""


def mount(app, rater_dep) -> None:
    @app.get("/eval/review", response_class=HTMLResponse)
    def review(who: str = Depends(rater_dep)):
        pools = _load_pools()
        db = _ratings_db()
        mine = db.execute("SELECT resume_id, job_id, rating FROM ratings WHERE rater=? AND rating >= 0 ORDER BY at DESC", (who,)).fetchall()
        notes = {(r, j): c for r, j, c in db.execute("SELECT resume_id, job_id, comment FROM comments WHERE rater=?", (who,))}
        stats = {key: [0, 0, 0] for key, _ in MODELS}  # compared, exact, within 1
        rows = []
        for rid, jid, my in mine:
            job = pools.get(rid, {}).get("jobs", {}).get(jid)
            if not job:
                continue
            models = _model_view(db, rid, jid)
            for key, _ in MODELS:
                if key in models:
                    diff = abs(models[key][0] - my)
                    stats[key][0] += 1
                    stats[key][1] += diff == 0
                    stats[key][2] += diff <= 1
            note = notes.get((rid, jid), "")
            rows.append(
                f"<tr><td><b>{html.escape(job['title'])}</b><br><span class='muted'>{html.escape(job['company'])} · "
                f"resume: {html.escape(pools[rid].get('title') or '?')}</span></td>"
                f"<td><span class='score'>{my}</span><br><span class='muted'>{html.escape(note)}</span></td>"
                + "".join(_cell(models.get(key, (None, ""))[0], my, models.get(key, (None, ""))[1]) for key, _ in MODELS)
                + "</tr>"
            )
        cards = "".join(
            f"<div class='card'><div class='muted'>{name}</div><div class='big'>{(st[1] / st[0] if st[0] else 0):.0%}</div>"
            f"<div class='muted'>same score as you · {(st[2] / st[0] if st[0] else 0):.0%} within 1 point · {st[0]} jobs compared</div></div>"
            for (key, name), st in ((m, stats[m[0]]) for m in MODELS)
        )
        return REVIEW.format(who=html.escape(who), n=len(rows), cards=cards, rows="".join(rows) or "<tr><td colspan=5>Nothing rated yet.</td></tr>")

    @app.get("/eval", response_class=HTMLResponse)
    def next_pair(field: str = "", set: str = "", who: str = Depends(rater_dep)):
        """?field=software,sales shows only those fields; ?set=check only the human check set."""
        wanted = {f.strip() for f in field.split(",") if f.strip()}
        pools = {k: v for k, v in _load_pools().items()
                 if (not wanted or v.get("field") in wanted) and (set != "check" or v.get("check"))}
        db = _ratings_db()
        mine = {(r, j) for r, j in db.execute("SELECT resume_id, job_id FROM ratings WHERE rater=?", (who,))}
        total = sum(len(p["order"]) for p in pools.values())
        # Finish one resume before starting the next, so context carries over.
        for rid, pool in sorted(pools.items(), key=lambda kv: kv[1]["file"]):
            for jid in pool["order"]:
                if (rid, jid) not in mine:
                    job = pool["jobs"][jid]
                    return PAGE.format(last=_last_panel(db, who, _load_pools()),
                        who=html.escape(who), done=len(mine), total=total, guide=GUIDE, resume=html.escape(pool["text"]),
                        title=html.escape(job["title"]), company=html.escape(job["company"]), location=html.escape(job["location"] or "—"),
                        remote=job["remote"], url=html.escape(job["url"]), description=html.escape(job["description"]), rid=rid, jid=html.escape(jid),
                        field=html.escape(field), set=html.escape(set),
                    )
        return HTMLResponse("<p style='font:16px system-ui;padding:24px'>All done. Thank you!</p>")

    @app.post("/eval/rate")
    def save(resume_id: str = Form(...), job_id: str = Form(...), rating: int = Form(...), comment: str = Form(""),
             field: str = "", set: str = "", who: str = Depends(rater_dep)):
        if rating in (0, 1, 2, 3, -1):
            db = _ratings_db()
            # -1 is a skip: remembered so the pair is not shown again, ignored by the metrics.
            stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
            db.execute("INSERT OR REPLACE INTO ratings VALUES (?,?,?,?,?)", (resume_id, job_id, who, rating, stamp))
            if comment.strip():
                db.execute("INSERT OR REPLACE INTO comments VALUES (?,?,?,?,?)", (resume_id, job_id, who, comment.strip()[:2000], stamp))
            db.commit()
        query = "&".join(f"{k}={v}" for k, v in (("field", field), ("set", set)) if v)
        return RedirectResponse("/eval" + (f"?{query}" if query else ""), status_code=303)


def choose_check_set(software: int = 7, sales: int = 3, seed: int = 10) -> None:
    """Marks a fixed random set of resumes for human rating (/eval?set=check); only resumes whose
    work history was read (so the rater is not judging a parsing failure)."""
    pools = _load_pools()
    rng = random.Random(seed)
    chosen = []
    for field, n in (("software", software), ("sales", sales)):
        ok = [rid for rid, p in sorted(pools.items()) if p.get("field") == field and parse_resume(p["text"]).roles]
        chosen += rng.sample(ok, min(n, len(ok)))
    for rid, p in pools.items():
        p["check"] = rid in chosen
    json.dump(pools, open(POOLS, "w"))
    for rid in chosen:
        print(f"  {pools[rid]['field']:9} {pools[rid].get('title')}")
    print(f"{len(chosen)} resumes in the human check set")


if __name__ == "__main__":
    {"pools": build_pools, "report": report, "check-set": choose_check_set}[sys.argv[1]]()
