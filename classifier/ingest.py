"""Fetch every account listed in data/sources.yaml, keep US/UK/Canada jobs, store and de-duplicate.

    python -m classifier.ingest                 # all sources
    python -m classifier.ingest greenhouse      # one source
"""

from __future__ import annotations

import asyncio
import importlib
import sys
import time

import httpx
import yaml

from classifier import store
from classifier.locations import TARGET_COUNTRIES
from classifier.models import Job
from classifier.sources import client

SOURCES_FILE = "data/sources.yaml"
CONCURRENCY = 6


def in_market(job: Job) -> bool:
    """US, UK or Canada; remote jobs with no stated country are kept only from remote-only feeds."""
    if job.countries & TARGET_COUNTRIES:
        return True
    return False


async def run(only: set[str] | None = None) -> None:
    with open(SOURCES_FILE) as f:
        config = yaml.safe_load(f)
    db = store.connect()
    gate = asyncio.Semaphore(CONCURRENCY)
    totals: dict[str, list[int]] = {}

    async with client() as http:

        async def one(source: str, account: str):
            module = importlib.import_module(f"classifier.sources.{source}")
            if hasattr(module, "enabled") and not module.enabled():
                return
            async with gate:
                started = time.monotonic()
                jobs, error = [], None
                for attempt in range(3):
                    try:
                        jobs, error = await module.fetch(http, account), None
                        break
                    except httpx.HTTPStatusError as exc:
                        error = f"HTTP {exc.response.status_code}"
                        if exc.response.status_code != 429:
                            break
                        # Rate limited: wait as asked, then retry.
                        await asyncio.sleep(float(exc.response.headers.get("retry-after") or 30 * (attempt + 1)))
                    except Exception as exc:  # one bad account must not stop the run
                        error = f"{type(exc).__name__}: {exc}"[:300]
                        break
            kept = [j for j in jobs if in_market(j) and j.title and j.description]
            if error is None:
                store.save_account(db, source, account, kept)
            db.execute(
                "INSERT OR REPLACE INTO fetch_log VALUES (?,?,?,?,?,?)",
                (source, account, store.now(), len(jobs), len(kept), error),
            )
            db.commit()
            t = totals.setdefault(source, [0, 0, 0])
            t[0] += len(jobs)
            t[1] += len(kept)
            t[2] += 1 if error else 0
            status = f"ERROR {error}" if error else f"{len(kept)}/{len(jobs)} kept"
            print(f"  {source:16} {account:28} {status} ({time.monotonic() - started:.1f}s)", flush=True)

        tasks = [
            one(source, str(account))
            for source, accounts in config.items()
            if not only or source in only
            for account in accounts or []
        ]
        await asyncio.gather(*tasks)

    dupes = store.mark_duplicates(db)
    live = db.execute("SELECT count(*) FROM jobs WHERE duplicate_of IS NULL").fetchone()[0]
    for source, (seen, kept, errors) in sorted(totals.items()):
        print(f"{source:16} fetched {seen:6}  kept {kept:6}  failed accounts {errors}")
    print(f"duplicates marked: {dupes}; unique jobs in store: {live}")


if __name__ == "__main__":
    asyncio.run(run(set(sys.argv[1:]) or None))
