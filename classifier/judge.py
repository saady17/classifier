"""AI judges for EVALUATION ONLY. They grade the ranker's results like an exam marker; the ranker
itself never calls them. Three independent judges (Gemini; DeepSeek and GLM via OpenRouter) rate each
resume-job pair 0-3 with the same rubric human raters use.

    python -m classifier.judge                 # rate every unrated pair in eval/private/pools.json
    python -m classifier.judge --only=gemini   # one judge
    python -m classifier.judge agreement       # how the judges agree with each other and with humans
    python -m classifier.judge crosscheck      # final verdicts: majority of three; GLM referees three-way splits

Personal details are removed from resumes before anything is sent (redact())."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from datetime import datetime, timezone

import httpx

from classifier.evaluation import _load_pools, _ratings_db

BATCH = 5  # jobs per request, each with the same resume
JOB_CHARS = 5000
JUDGES = {
    "gemini": {"model": os.environ.get("JUDGE_GEMINI_MODEL", "gemini-3.7-flash"), "key": "GEMINI_API_KEY"},
    "deepseek": {"model": os.environ.get("JUDGE_OPENROUTER_MODEL", "deepseek/deepseek-v4.1-flash"), "key": "OPENROUTER_API_KEY"},
    "glm": {"model": os.environ.get("JUDGE_GLM_MODEL", "z-ai/glm-5.3-flash"), "key": "OPENROUTER_API_KEY"},
}
REFEREE = "glm"

RUBRIC = """You are an experienced recruiter judging how well jobs fit one candidate.
Rate each job for THIS candidate on this scale:
3 = great fit: the candidate would likely be shortlisted; title, skills and level all line up.
2 = good fit: worth applying; most requirements met, maybe one gap (a skill, a year or two, a level).
1 = weak fit: a stretch or a neighbouring field; several important requirements missing.
0 = not a fit: a different field, far above or below their level, or a required license/degree they lack.
Judge only the candidate's background against each job's requirements. Ignore company prestige,
salary and location (location is handled separately). Be strict and consistent.
Reply with JSON only: {"ratings": [{"job": <number>, "rating": <0-3>, "reason": "<max 30 words, cite concrete evidence>"}]}"""

CROSSCHECK = """You are a senior recruiter reviewing three colleagues (A, B, C) who rated the same jobs for one
candidate and all disagreed. Use the same scale:
3 great fit, 2 good fit (one gap), 1 weak fit (stretch or neighbouring field), 0 not a fit (different field,
wrong level, or a required license/degree missing).
For each job, re-read the resume and the job yourself, weigh all three colleagues' reasoning, decide who is
right (or none), and give the final rating. Cite concrete evidence: skills, years, level, licenses, degree.
Ignore company prestige, salary and location.
Reply with JSON only: {"ratings": [{"job": <number>, "final_rating": <0-3>, "sided_with": "A"|"B"|"C"|"none",
"reasoning": "<max 40 words>"}]}"""

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(\+?\d[\d\s().-]{7,}\d)")
_URL = re.compile(r"(https?://\S+|www\.\S+|linkedin\.com/\S+|github\.com/\S+)", re.I)
_STREET = re.compile(r"\b\d{1,6}\s+[A-Z][\w.]*(\s+[A-Z][\w.]*){0,3}\s+(street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr|court|ct|way|place|pl)\b\.?", re.I)


def redact(text: str) -> str:
    """Removes names, emails, phone numbers, links and street addresses; keeps city/state."""
    lines = text.splitlines()
    out = []
    for i, line in enumerate(lines):
        if i < 3 and re.fullmatch(r"[A-Z][A-Za-z'’.-]+(\s+[A-Z][A-Za-z'’.-]+){1,3}(,\s*[A-Z]{2,5})?", line.strip()):
            out.append("[NAME]")  # the name line at the top ("Maria Gonzalez, RN")
            continue
        line = _EMAIL.sub("[EMAIL]", line)
        line = _URL.sub("[LINK]", line)
        line = _STREET.sub("[ADDRESS]", line)
        line = _PHONE.sub(lambda m: m.group(0) if re.fullmatch(r"\d{4}\s*[-–]\s*\d{4}", m.group(0).strip()) else "[PHONE]", line)
        out.append(line)
    return "\n".join(out)


def _prompt(resume: str, jobs: list[dict]) -> str:
    parts = [f"CANDIDATE RESUME:\n{resume[:9000]}\n"]
    for n, job in enumerate(jobs, 1):
        parts.append(f"JOB {n}: {job['title']} at {job['company']}\n{job['description'][:JOB_CHARS]}\n")
    return "\n".join(parts)


async def _ask_gemini(client: httpx.AsyncClient, model: str, prompt: str) -> str:
    r = await client.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]},
        json={"systemInstruction": {"parts": [{"text": RUBRIC}]}, "contents": [{"parts": [{"text": prompt}]}],
              "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}},
    )
    r.raise_for_status()
    parts = r.json()["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))


async def _ask_openrouter(client: httpx.AsyncClient, model: str, prompt: str, system: str = RUBRIC) -> str:
    r = await client.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "X-Title": "Jobzhub classifier evaluation"},
        json={"model": model, "temperature": 0, "response_format": {"type": "json_object"},
              # Only providers that neither store nor train on prompts.
              "provider": {"data_collection": "deny"},
              "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]},
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


async def rate_all(only: set[str] | None = None) -> None:
    pools = _load_pools()
    db = _ratings_db()
    judges = {n: j for n, j in JUDGES.items() if (not only or n in only) and os.environ.get(j["key"])}
    if not judges:
        print("No judge has an API key (GEMINI_API_KEY / OPENROUTER_API_KEY).")
        return
    done = {(r, j, who) for r, j, who in db.execute("SELECT resume_id, job_id, rater FROM ratings WHERE rater LIKE 'ai:%'")}
    tasks = []
    for rid, pool in pools.items():
        resume = redact(pool["text"])
        for name in judges:
            todo = [jid for jid in pool["order"] if (rid, jid, f"ai:{name}") not in done]
            for i in range(0, len(todo), BATCH):
                tasks.append((name, rid, resume, todo[i : i + BATCH], pool))
    print(f"{len(tasks)} requests to send ({', '.join(judges)})")
    gate = asyncio.Semaphore(4)
    async with httpx.AsyncClient(timeout=120) as client:

        async def one(name, rid, resume, ids, pool):
            jobs = [pool["jobs"][j] for j in ids]
            ask = _ask_gemini if name == "gemini" else _ask_openrouter  # deepseek and glm go through OpenRouter
            for attempt in range(3):
                try:
                    async with gate:
                        raw = await ask(client, judges[name]["model"], _prompt(resume, jobs))
                    items = json.loads(raw)["ratings"]
                    got = {int(x["job"]): (int(x["rating"]), str(x.get("reason", ""))[:200]) for x in items}
                    if set(got) != set(range(1, len(ids) + 1)) or any(not 0 <= r <= 3 for r, _ in got.values()):
                        raise ValueError("incomplete or out-of-range ratings")
                    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
                    for n, jid in enumerate(ids, 1):
                        rating, reason = got[n]
                        db.execute("INSERT OR REPLACE INTO ratings VALUES (?,?,?,?,?)", (rid, jid, f"ai:{name}", rating, stamp))
                        db.execute("INSERT OR REPLACE INTO ai_reasons VALUES (?,?,?,?)", (rid, jid, f"ai:{name}", reason))
                    db.commit()
                    return
                except Exception as exc:  # retried, then reported
                    err = exc
                    await asyncio.sleep(3 * (attempt + 1))
            print(f"  {name} {pool['file']}: failed ({type(err).__name__}: {str(err)[:120]})")

        db.execute("CREATE TABLE IF NOT EXISTS ai_reasons (resume_id TEXT, job_id TEXT, rater TEXT, reason TEXT, PRIMARY KEY (resume_id, job_id, rater))")
        await asyncio.gather(*(one(*t) for t in tasks))
    print("done")


async def crosscheck() -> None:
    """Final AI verdict per pair (rater ai:final) from three blind judges: if two or three agree, that
    rating wins; if all three differ, the referee re-reads the resume and job, weighs the three
    reasonings and decides, with written reasoning."""
    pools = _load_pools()
    db = _ratings_db()
    db.execute("CREATE TABLE IF NOT EXISTS ai_reasons (resume_id TEXT, job_id TEXT, rater TEXT, reason TEXT, PRIMARY KEY (resume_id, job_id, rater))")
    names = ["gemini", "deepseek", "glm"]
    votes: dict[tuple[str, str], dict[str, tuple[int, str]]] = {}
    for rid, jid, who, rating in db.execute("SELECT resume_id, job_id, rater, rating FROM ratings WHERE rater IN ('ai:gemini','ai:deepseek','ai:glm')"):
        reason = db.execute("SELECT reason FROM ai_reasons WHERE resume_id=? AND job_id=? AND rater=?", (rid, jid, who)).fetchone()
        votes.setdefault((rid, jid), {})[who[3:]] = (rating, reason[0] if reason else "")
    done = {(r, j) for r, j in db.execute("SELECT resume_id, job_id FROM ratings WHERE rater='ai:final'")}
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    disputes: dict[str, list] = {}
    majority = 0
    for (rid, jid), v in votes.items():
        if (rid, jid) in done or len(v) < 3 or rid not in pools:
            continue
        ratings = [v[n][0] for n in names]
        top = max(set(ratings), key=ratings.count)
        if ratings.count(top) >= 2:
            agreeing = [n for n in names if v[n][0] == top]
            db.execute("INSERT OR REPLACE INTO ratings VALUES (?,?,?,?,?)", (rid, jid, "ai:final", top, stamp))
            db.execute("INSERT OR REPLACE INTO ai_reasons VALUES (?,?,?,?)", (rid, jid, "ai:final", f"Majority: {', '.join(a.title() for a in agreeing)} agreed."))
            majority += 1
        else:
            disputes.setdefault(rid, []).append((jid, v))
    db.commit()
    tasks = [(rid, items[i : i + BATCH]) for rid, items in disputes.items() for i in range(0, len(items), BATCH)]
    print(f"{majority} settled by majority; {sum(len(x) for x in disputes.values())} three-way splits go to the referee ({REFEREE}) in {len(tasks)} requests")
    gate = asyncio.Semaphore(4)
    model = JUDGES[REFEREE]["model"]
    async with httpx.AsyncClient(timeout=180) as client:

        async def one(rid, items):
            pool = pools[rid]
            parts = [f"CANDIDATE RESUME:\n{redact(pool['text'])[:9000]}\n"]
            for n, (jid, v) in enumerate(items, 1):
                job = pool["jobs"][jid]
                parts.append(f"JOB {n}: {job['title']} at {job['company']}\n{job['description'][:JOB_CHARS]}\n"
                             + "".join(f"Colleague {label} rated {v[name][0]}: {v[name][1]}\n" for label, name in zip("ABC", names)))
            err = None
            for attempt in range(3):
                try:
                    async with gate:
                        raw = await _ask_openrouter(client, model, "\n".join(parts), system=CROSSCHECK)
                    got = {int(x["job"]): x for x in json.loads(raw)["ratings"]}
                    if set(got) != set(range(1, len(items) + 1)) or any(not 0 <= int(x["final_rating"]) <= 3 for x in got.values()):
                        raise ValueError("incomplete or out-of-range ratings")
                    for n, (jid, _v) in enumerate(items, 1):
                        x = got[n]
                        side = {"A": "Gemini", "B": "DeepSeek", "C": "GLM"}.get(str(x.get("sided_with")), "none of them")
                        db.execute("INSERT OR REPLACE INTO ratings VALUES (?,?,?,?,?)", (rid, jid, "ai:final", int(x["final_rating"]), stamp))
                        db.execute("INSERT OR REPLACE INTO ai_reasons VALUES (?,?,?,?)", (rid, jid, "ai:final", f"Referee sided with {side}. {str(x.get('reasoning', ''))[:400]}"))
                    db.commit()
                    return
                except Exception as exc:  # retried, then reported
                    err = exc
                    await asyncio.sleep(3 * (attempt + 1))
            print(f"  referee {pool['file']}: failed ({type(err).__name__}: {str(err)[:120]})")

        await asyncio.gather(*(one(*t) for t in tasks))
    print("final verdicts done")


def agreement() -> None:
    db = _ratings_db()
    by: dict[tuple[str, str], dict[str, int]] = {}
    for rid, jid, who, r in db.execute("SELECT resume_id, job_id, rater, rating FROM ratings WHERE rating >= 0"):
        by.setdefault((rid, jid), {})[who] = r
    raters = sorted({w for v in by.values() for w in v})
    humans = [w for w in raters if not w.startswith("ai:")]

    def compare(a_name, b_name, a_get, b_get):
        pairs = [(a_get(v), b_get(v)) for v in by.values() if a_get(v) is not None and b_get(v) is not None]
        if not pairs:
            return
        exact = sum(a == b for a, b in pairs) / len(pairs)
        within1 = sum(abs(a - b) <= 1 for a, b in pairs) / len(pairs)
        print(f"{a_name:>14} vs {b_name:<14} pairs {len(pairs):5}  exact {exact:.0%}  within 1 point {within1:.0%}")

    def ai_avg(v):
        if "ai:final" in v:
            return v["ai:final"]
        vals = [r for w, r in v.items() if w.startswith("ai:")]
        return round(sum(vals) / len(vals)) if vals else None

    for a_, b_ in (("ai:gemini", "ai:deepseek"), ("ai:gemini", "ai:glm"), ("ai:deepseek", "ai:glm")):
        compare(a_, b_, lambda v, a_=a_: v.get(a_), lambda v, b_=b_: v.get(b_))
    for h in humans:
        for j in ("ai:gemini", "ai:deepseek", "ai:glm"):
            compare(h, j, lambda v, h=h: v.get(h), lambda v, j=j: v.get(j))
        compare(h, "AI final", lambda v, h=h: v.get(h), ai_avg)
    three = [v for v in by.values() if all(k in v for k in ("ai:gemini", "ai:deepseek", "ai:glm"))]
    unanimous = sum(1 for v in three if v["ai:gemini"] == v["ai:deepseek"] == v["ai:glm"])
    split = sum(1 for v in three if len({v["ai:gemini"], v["ai:deepseek"], v["ai:glm"]}) == 3)
    if three:
        print(f"three judges on {len(three)} pairs: unanimous {unanimous / len(three):.0%}, majority {(len(three) - unanimous - split) / len(three):.0%}, three-way split {split / len(three):.0%}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "agreement":
        agreement()
    elif args and args[0] == "crosscheck":
        asyncio.run(crosscheck())
    else:
        only = {a.split("=", 1)[1] for a in args if a.startswith("--only=")} or None
        asyncio.run(rate_all(only))
