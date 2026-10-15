"""Remotive public API (remote jobs). Their terms ask for a link back and no more than a few calls a day."""

from __future__ import annotations

import httpx

from classifier.locations import parse_location
from classifier.models import Job
from classifier.text import html_to_text

API = "https://remotive.com/api/remote-jobs"


async def fetch(client: httpx.AsyncClient, _account: str = "all") -> list[Job]:
    resp = await client.get(API)
    resp.raise_for_status()
    jobs = []
    for item in resp.json().get("jobs", []):
        where = item.get("candidate_required_location") or ""
        locs = [parse_location(part) for part in where.split(",")] if where else []
        if where.strip().lower() in ("worldwide", "anywhere", ""):
            locs = [parse_location("United States"), parse_location("United Kingdom"), parse_location("Canada")]
        jobs.append(
            Job(
                source="remotive",
                external_id=str(item["id"]),
                company=item.get("company_name") or "",
                title=(item.get("title") or "").strip(),
                url=item.get("url", ""),
                description=html_to_text(item.get("description")),
                locations=locs,
                remote="remote",
                posted_at=item.get("publication_date"),
                employment_type=item.get("job_type"),
                department=item.get("category"),
            )
        )
    return jobs
