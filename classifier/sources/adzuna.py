"""Adzuna Search API (aggregator). Needs a free key: https://developer.adzuna.com/
Set ADZUNA_APP_ID and ADZUNA_APP_KEY. `account` is "<country>:<keyword>", e.g. "us:data analyst".
Adzuna only returns a snippet of each description, so these jobs rank on less text."""

from __future__ import annotations

import os

import httpx

from classifier.locations import parse_location
from classifier.models import Job
from classifier.text import clean_text

API = "https://api.adzuna.com/v1/api/jobs"
COUNTRY = {"us": "US", "gb": "GB", "ca": "CA"}


def enabled() -> bool:
    return bool(os.environ.get("ADZUNA_APP_ID") and os.environ.get("ADZUNA_APP_KEY"))


async def fetch(client: httpx.AsyncClient, account: str) -> list[Job]:
    if not enabled():
        return []
    country, _, keyword = account.partition(":")
    jobs = []
    for page in range(1, 6):
        resp = await client.get(f"{API}/{country}/search/{page}", params={
            "app_id": os.environ["ADZUNA_APP_ID"], "app_key": os.environ["ADZUNA_APP_KEY"],
            "what": keyword, "results_per_page": 50, "content-type": "application/json"})
        resp.raise_for_status()
        results = resp.json().get("results", [])
        for r in results:
            area = (r.get("location") or {}).get("display_name") or ""
            loc = parse_location(area)
            loc.country = loc.country or COUNTRY.get(country)
            jobs.append(Job(
                source="adzuna", external_id=str(r["id"]), company=(r.get("company") or {}).get("display_name", ""),
                title=r.get("title", ""), url=r.get("redirect_url", ""), description=clean_text(r.get("description", "")),
                locations=[loc], posted_at=r.get("created"), employment_type=r.get("contract_time"),
                salary_min=r.get("salary_min"), salary_max=r.get("salary_max"), salary_period="year" if r.get("salary_min") else None,
            ))
        if len(results) < 50:
            break
    return jobs
