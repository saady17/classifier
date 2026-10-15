"""SQLite job store with an FTS5 (BM25) index for the first-pass candidate search."""

from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone

from classifier.models import Job, Location
from classifier.text import normalize

DB_PATH = os.environ.get("CLASSIFIER_DB", "data/jobs.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  account TEXT NOT NULL,
  company TEXT NOT NULL,
  title TEXT NOT NULL,
  url TEXT NOT NULL,
  description TEXT NOT NULL,
  locations TEXT NOT NULL,          -- JSON list of Location
  countries TEXT NOT NULL,          -- e.g. ",US,CA,"
  remote TEXT NOT NULL,
  posted_at TEXT,
  employment_type TEXT,
  department TEXT,
  salary_min REAL, salary_max REAL, salary_currency TEXT, salary_period TEXT,
  dedup_key TEXT NOT NULL,
  duplicate_of TEXT,
  first_seen TEXT NOT NULL,
  last_seen TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_account ON jobs(source, account);
CREATE INDEX IF NOT EXISTS jobs_dedup ON jobs(dedup_key);
CREATE VIRTUAL TABLE IF NOT EXISTS jobs_fts USING fts5(
  title, company, description, content='jobs', content_rowid='rowid', tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS jobs_ai AFTER INSERT ON jobs BEGIN
  INSERT INTO jobs_fts(rowid, title, company, description) VALUES (new.rowid, new.title, new.company, new.description);
END;
CREATE TRIGGER IF NOT EXISTS jobs_ad AFTER DELETE ON jobs BEGIN
  INSERT INTO jobs_fts(jobs_fts, rowid, title, company, description) VALUES ('delete', old.rowid, old.title, old.company, old.description);
END;
CREATE TRIGGER IF NOT EXISTS jobs_au AFTER UPDATE ON jobs BEGIN
  INSERT INTO jobs_fts(jobs_fts, rowid, title, company, description) VALUES ('delete', old.rowid, old.title, old.company, old.description);
  INSERT INTO jobs_fts(rowid, title, company, description) VALUES (new.rowid, new.title, new.company, new.description);
END;
CREATE TABLE IF NOT EXISTS fetch_log (
  source TEXT, account TEXT, fetched_at TEXT, jobs INTEGER, kept INTEGER, error TEXT,
  PRIMARY KEY (source, account)
);
"""


def connect(path: str = DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript(SCHEMA)
    return db


_SUFFIX = re.compile(r"\b(inc|llc|ltd|limited|corp|corporation|co|gmbh|plc|technologies|labs|hq)\b\.?")


def dedup_key(job: Job) -> str:
    company = _SUFFIX.sub("", normalize(job.company)).strip()
    title = re.sub(r"\s*[\(\[].*?[\)\]]", "", normalize(job.title))
    place = sorted({(loc.country or "") + "/" + normalize(loc.city or "") for loc in job.locations})[:1]
    return f"{company}|{title}|{','.join(place)}"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_account(db: sqlite3.Connection, source: str, account: str, jobs: list[Job]) -> int:
    """Upserts this account's jobs and removes its jobs that are no longer listed (closed)."""
    stamp = now()
    ids = []
    for job in jobs:
        ids.append(job.id)
        countries = "," + ",".join(sorted(job.countries)) + ","
        row = (
            job.id, source, account, job.company, job.title, job.url, job.description,
            json.dumps([asdict(loc) for loc in job.locations]), countries, job.remote, job.posted_at,
            job.employment_type, job.department, job.salary_min, job.salary_max, job.salary_currency,
            job.salary_period, dedup_key(job), stamp, stamp,
        )
        db.execute(
            """INSERT INTO jobs (id, source, account, company, title, url, description, locations, countries, remote,
                 posted_at, employment_type, department, salary_min, salary_max, salary_currency, salary_period,
                 dedup_key, first_seen, last_seen)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET company=excluded.company, title=excluded.title, url=excluded.url,
                 description=excluded.description, locations=excluded.locations, countries=excluded.countries,
                 remote=excluded.remote, posted_at=excluded.posted_at, employment_type=excluded.employment_type,
                 department=excluded.department, salary_min=excluded.salary_min, salary_max=excluded.salary_max,
                 salary_currency=excluded.salary_currency, salary_period=excluded.salary_period,
                 dedup_key=excluded.dedup_key, last_seen=excluded.last_seen""",
            row,
        )
    marks = ",".join("?" * len(ids)) or "''"
    removed = db.execute(f"DELETE FROM jobs WHERE source=? AND account=? AND id NOT IN ({marks})", (source, account, *ids)).rowcount
    db.commit()
    return removed


def mark_duplicates(db: sqlite3.Connection) -> int:
    """Within each dedup key, keeps the posting with the longest description; the rest point to it."""
    db.execute("UPDATE jobs SET duplicate_of = NULL")
    rows = db.execute(
        """SELECT dedup_key, id, length(description) AS n FROM jobs
           WHERE dedup_key IN (SELECT dedup_key FROM jobs GROUP BY dedup_key HAVING count(*) > 1)
           ORDER BY dedup_key, n DESC"""
    ).fetchall()
    keep, marked = {}, 0
    for r in rows:
        if r["dedup_key"] not in keep:
            keep[r["dedup_key"]] = r["id"]
        else:
            db.execute("UPDATE jobs SET duplicate_of=? WHERE id=?", (keep[r["dedup_key"]], r["id"]))
            marked += 1
    db.commit()
    return marked


def row_to_job(row: sqlite3.Row) -> Job:
    return Job(
        source=row["source"], external_id=row["id"].split(":", 1)[1], company=row["company"], title=row["title"],
        url=row["url"], description=row["description"],
        locations=[Location(**loc) for loc in json.loads(row["locations"])], remote=row["remote"],
        posted_at=row["posted_at"], employment_type=row["employment_type"], department=row["department"],
        salary_min=row["salary_min"], salary_max=row["salary_max"], salary_currency=row["salary_currency"],
        salary_period=row["salary_period"],
    )
