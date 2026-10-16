"""Precomputes each job's reading (skills, years, degree, visa, occupation, level) and the skill
rarity table, so ranking a resume never re-reads job posts.

    python -m classifier.features
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

import numpy as np

from classifier import store
from classifier.jobparse import parse_job
from classifier.skills import matcher
from classifier.titles import index as title_index
from classifier.titles import level as title_level

SCHEMA = """
CREATE TABLE IF NOT EXISTS job_features (
  id TEXT PRIMARY KEY,
  desc_hash TEXT NOT NULL,
  occupation TEXT,
  occ_conf REAL,
  level INTEGER,
  data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS job_features_occ ON job_features(occupation);
CREATE TABLE IF NOT EXISTS skill_stats (
  skill TEXT PRIMARY KEY,
  df INTEGER NOT NULL,
  idf REAL NOT NULL
);
"""


# Bump when the job reader's logic changes, so every stored job is read again.
READER_VERSION = "3"


def digest(row) -> str:
    return hashlib.sha1(f"{READER_VERSION}\n{row['title']}\n{row['company']}\n{row['description']}".encode()).hexdigest()


_ENGLISH = {"the", "and", "to", "of", "a", "in", "for", "with", "you", "we", "our", "is", "are", "will", "on", "as", "your", "be", "this", "or"}


def english_share(text: str) -> float:
    words = re.findall(r"[a-zà-ÿ]+", text.lower())[:600]
    return sum(w in _ENGLISH for w in words) / len(words) if words else 0.0


def compute(row: dict) -> tuple:
    job = parse_job(row["description"], row["company"])
    # The title says a lot: skills named in it count as required ("Python Developer").
    for name in matcher().find(row["title"]):
        job.required.setdefault(name, 1)
    code, conf = title_index().occupation(row["title"])
    data = asdict(job)
    data["english"] = english_share(row["description"]) >= 0.12  # English posts sit around 0.2-0.3
    return (row["id"], row["hash"], code, conf, title_level(row["title"]), json.dumps(data))


def run() -> None:
    db = store.connect()
    db.executescript(SCHEMA)
    known = dict(db.execute("SELECT id, desc_hash FROM job_features").fetchall())
    todo = []
    for row in db.execute("SELECT id, company, title, description FROM jobs"):
        h = digest(row)
        if known.get(row["id"]) != h:
            todo.append({"id": row["id"], "company": row["company"], "title": row["title"], "description": row["description"], "hash": h})
    started = time.monotonic()
    if todo:
        with ProcessPoolExecutor(max_workers=2) as pool:
            for n, result in enumerate(pool.map(compute, todo, chunksize=64), 1):
                db.execute("INSERT OR REPLACE INTO job_features VALUES (?,?,?,?,?,?)", result)
                if n % 2000 == 0:
                    db.commit()
                    print(f"  {n}/{len(todo)}", flush=True)
    db.execute("DELETE FROM job_features WHERE id NOT IN (SELECT id FROM jobs)")
    db.commit()
    print(f"features: {len(todo)} jobs (re)computed in {time.monotonic() - started:.0f}s")
    rebuild_skill_stats(db)


def rebuild_skill_stats(db: sqlite3.Connection) -> None:
    """Inverse document frequency of each skill across live jobs: rare skills say more about fit."""
    df: dict[str, int] = {}
    n = 0
    for (data,) in db.execute(
        "SELECT f.data FROM job_features f JOIN jobs j ON j.id = f.id WHERE j.duplicate_of IS NULL"
    ):
        n += 1
        d = json.loads(data)
        for skill in set(d["required"]) | set(d["preferred"]) | set(d["duties"]):
            df[skill] = df.get(skill, 0) + 1
    db.execute("DELETE FROM skill_stats")
    db.executemany(
        "INSERT INTO skill_stats VALUES (?,?,?)",
        [(s, c, math.log((n + 1) / (c + 1)) + 1) for s, c in df.items()],
    )
    db.commit()
    print(f"skill stats: {len(df)} skills over {n} jobs")


TFIDF_PATH = "data/tfidf.pkl"


def build_tfidf(db: sqlite3.Connection) -> None:
    """Classic TF-IDF over job posts (no AI): lets the ranker compare a resume's and a job's whole
    vocabulary, including industry words the skills dictionary does not list."""
    import pickle

    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = [f"{t}\n{d}" for t, d in db.execute("SELECT title, description FROM jobs WHERE duplicate_of IS NULL")]
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=3, max_df=0.4, ngram_range=(1, 2),
                          max_features=150_000, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9+#.]{1,}\b", dtype=np.float32)
    vec.fit(texts)
    with open(TFIDF_PATH, "wb") as f:
        pickle.dump(vec, f)
    print(f"tf-idf: {len(vec.vocabulary_)} terms over {len(texts)} jobs")


if __name__ == "__main__":
    run()
    build_tfidf(store.connect())
