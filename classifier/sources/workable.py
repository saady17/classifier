"""Workable public widget API (the feed behind each company's careers widget)."""

from __future__ import annotations

import httpx

from classifier.locations import parse_location, remote_mode
from classifier.models import Job
from classifier.text import html_to_text

API = "https://apply.workable.com/api/v1/widget/accounts"


async def fetch(client: httpx.AsyncClient, account: str) -> list[Job]:
    resp = await client.get(f"{API}/{account}", params={"details": "true"})
    resp.raise_for_status()
    data = resp.json()
    company = data.get("name") or account
    jobs = []
    for item in data.get("jobs", []):
        raws = [", ".join(x for x in (loc.get("city"), loc.get("region"), loc.get("country")) if x) for loc in item.get("locations") or []]
        if not raws:
            raws = [", ".join(x for x in (item.get("city"), item.get("state"), item.get("country")) if x)]
        jobs.append(
            Job(
                source="workable",
                external_id=f"{account}:{item.get('shortcode')}",
                company=company,
                title=(item.get("title") or "").strip(),
                url=item.get("url") or item.get("shortlink") or "",
                description=html_to_text(item.get("description")),
                locations=[parse_location(r) for r in raws if r],
                remote="remote" if item.get("telecommuting") else remote_mode(item.get("title")),
                posted_at=item.get("published_on") or item.get("created_at"),
                employment_type=item.get("employment_type"),
                department=item.get("department"),
            )
        )
    return jobs
