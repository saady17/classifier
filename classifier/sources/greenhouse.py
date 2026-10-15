"""Greenhouse Job Board API (public): https://developers.greenhouse.io/job-board.html"""

from __future__ import annotations

import httpx

from classifier.locations import parse_location, remote_mode
from classifier.models import Job
from classifier.text import html_to_text

API = "https://boards-api.greenhouse.io/v1/boards"


async def fetch(client: httpx.AsyncClient, board: str) -> list[Job]:
    meta = await client.get(f"{API}/{board}")
    meta.raise_for_status()
    company = meta.json().get("name") or board
    resp = await client.get(f"{API}/{board}/jobs", params={"content": "true"})
    resp.raise_for_status()
    jobs = []
    for item in resp.json().get("jobs", []):
        location = (item.get("location") or {}).get("name") or ""
        offices = [o.get("name") or "" for o in item.get("offices") or []]
        raw_locations = [s.strip() for s in location.split(";") if s.strip()] or offices
        description = html_to_text(item.get("content"))
        jobs.append(
            Job(
                source="greenhouse",
                external_id=f"{board}:{item['id']}",
                company=company,
                title=item.get("title", "").strip(),
                url=item.get("absolute_url", ""),
                description=description,
                locations=[parse_location(r) for r in raw_locations],
                remote=remote_mode(location, item.get("title")),
                posted_at=item.get("first_published") or item.get("updated_at"),
                department=", ".join(d.get("name", "") for d in item.get("departments") or []) or None,
            )
        )
    return jobs
