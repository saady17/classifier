"""Lever Postings API (public): https://github.com/lever/postings-api"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx

from classifier.locations import parse_location, remote_mode
from classifier.models import Job
from classifier.text import clean_text, html_to_text

API = "https://api.lever.co/v0/postings"
_INTERVAL = {"per-year-salary": "year", "per-month-salary": "month", "per-hour-wage": "hour"}


async def fetch(client: httpx.AsyncClient, site: str) -> list[Job]:
    resp = await client.get(f"{API}/{site}", params={"mode": "json"})
    resp.raise_for_status()
    jobs = []
    for item in resp.json():
        cats = item.get("categories") or {}
        raw_locations = cats.get("allLocations") or [cats.get("location") or ""]
        parts = [item.get("descriptionPlain") or html_to_text(item.get("description"))]
        for block in item.get("lists") or []:
            parts.append(block.get("text", ""))
            parts.append(html_to_text(block.get("content")))
        parts.append(item.get("additionalPlain") or "")
        salary = item.get("salaryRange") or {}
        created = item.get("createdAt")
        jobs.append(
            Job(
                source="lever",
                external_id=f"{site}:{item['id']}",
                company=site.replace("-", " ").title(),
                title=(item.get("text") or "").strip(),
                url=item.get("hostedUrl", ""),
                description=clean_text("\n".join(p for p in parts if p)),
                locations=[parse_location(r) for r in raw_locations if r],
                remote=(item.get("workplaceType") or "").replace("unspecified", "") or remote_mode(cats.get("location")),
                posted_at=datetime.fromtimestamp(created / 1000, timezone.utc).isoformat() if created else None,
                employment_type=cats.get("commitment"),
                department=cats.get("team"),
                salary_min=salary.get("min"),
                salary_max=salary.get("max"),
                salary_currency=salary.get("currency"),
                salary_period=_INTERVAL.get(salary.get("interval") or ""),
            )
        )
    return jobs
