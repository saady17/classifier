"""The product metric: of the jobs the ranker shows a candidate FIRST (searching all jobs, not a
25-job pool), how many are a good fit?

    python -m classifier.topk build [splits]   # ranker's real top 10 per resume -> eval/private/live.json
    EVAL_POOLS=eval/private/live.json python -m classifier.judge --only=gemini   # rate them
    python -m classifier.topk report [splits]

A job counts as GOOD when the judges' mean rating is at least 2 ("worth applying") and as
ACCEPTABLE at 1 or more. Default splits: test,new (never used for tuning)."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

from classifier import store
from classifier.evaluation import PRIVATE, _load_pools, _ratings_db
from classifier.ranker import Preferences, rank
from classifier.resume import parse_resume

LIVE = f"{PRIVATE}/live.json"
TOP = 10


def build(splits: set[str]) -> None:
    pools = _load_pools()
    db = store.connect()
    live = json.load(open(LIVE)) if os.path.exists(LIVE) else {}
    for rid, pool in pools.items():
        if pool["split"] not in splits:
            continue
        resume = parse_resume(pool["text"])
        country = resume.location_country
        prefs = Preferences(countries=(country,) if country else ("US", "GB", "CA"), city=resume.location_city, region=resume.location_region)
        matches = rank(resume, prefs, limit=TOP, db=db)
        jobs = {}
        for m in matches:
            row = db.execute("SELECT description FROM jobs WHERE id=?", (m.job_id,)).fetchone()
            jobs[m.job_id] = {"title": m.title, "company": m.company, "url": m.url, "remote": m.remote, "location": m.location,
                              "description": row["description"][:8000], "score": m.score}
        live[rid] = {**{k: pool[k] for k in ("file", "text", "split", "field", "title") if k in pool},
                     "order": [m.job_id for m in matches], "jobs": jobs,
                     "built": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    json.dump(live, open(LIVE, "w"))
    print(f"live top-{TOP} built for {sum(1 for v in live.values() if v['split'] in splits)} resumes")


def report(splits: set[str]) -> None:
    live = json.load(open(LIVE))
    db = _ratings_db()
    labels: dict[tuple[str, str], list[int]] = {}
    for rid, jid, who, r in db.execute("SELECT resume_id, job_id, rater, rating FROM ratings WHERE rater IN ('ai:gemini','ai:glm','ai:deepseek') AND rating >= 0"):
        labels.setdefault((rid, jid), []).append(r)
    by_split: dict[str, dict[str, list[float]]] = {}
    for rid, v in live.items():
        if v["split"] not in splits:
            continue
        rated = [(jid, sum(labels[(rid, jid)]) / len(labels[(rid, jid)])) for jid in v["order"] if (rid, jid) in labels]
        if len(rated) < min(5, len(v["order"])) or not rated:
            continue
        stats = by_split.setdefault(v["split"], {"good5": [], "ok5": [], "good10": [], "ok10": [], "great1": []})
        top5, top10 = rated[:5], rated[:10]
        stats["good5"].append(sum(r >= 2 for _, r in top5) / len(top5))
        stats["ok5"].append(sum(r >= 1 for _, r in top5) / len(top5))
        stats["good10"].append(sum(r >= 2 for _, r in top10) / len(top10))
        stats["ok10"].append(sum(r >= 1 for _, r in top10) / len(top10))
        stats["great1"].append(float(rated[0][1] >= 2))
    print(f"{'split':6} {'resumes':>7}  {'#1 good':>8}  {'top-5 good':>10}  {'top-5 ok':>8}  {'top-10 good':>11}  {'top-10 ok':>9}")
    for split, st in sorted(by_split.items()):
        n = len(st["good5"])
        avg = {k: sum(x) / n for k, x in st.items()}
        print(f"{split:6} {n:7}  {avg['great1']:8.0%}  {avg['good5']:10.0%}  {avg['ok5']:8.0%}  {avg['good10']:11.0%}  {avg['ok10']:9.0%}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    splits = set((sys.argv[2] if len(sys.argv) > 2 else "test,new").split(","))
    {"build": build, "report": report}[cmd](splits)
