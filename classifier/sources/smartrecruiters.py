"""SmartRecruiters public Posting API: https://developers.smartrecruiters.com/docs/posting-api"""

from __future__ import annotations

import asyncio

import httpx

from classifier.locations import parse_location
from classifier.models import Job
from classifier.text import clean_text, html_to_text

API = "https://api.smartrecruiters.com/v1/companies"
MAX_POSTINGS = 400  # per company per run; details are one request each


async def fetch(client: httpx.AsyncClient, company_id: str) -> list[Job]:
    listed, offset = [], 0
    while len(listed) < MAX_POSTINGS:
        resp = await client.get(f"{API}/{company_id}/postings", params={"limit": 100, "offset": offset})
        resp.raise_for_status()
        page = resp.json()
        listed += page.get("content", [])
        offset += 100
        if offset >= page.get("totalFound", 0):
            break

    gate = asyncio.Semaphore(4)

    async def detail(item):
        loc = item.get("location") or {}
        raw = ", ".join(x for x in (loc.get("city"), loc.get("region"), loc.get("country", "").upper()) if x)
        parsed = parse_location(raw)
        if parsed.country not in ("US", "GB", "CA") and not loc.get("remote"):
            return None
        async with gate:
            resp = await client.get(f"{API}/{company_id}/postings/{item['id']}")
        if resp.status_code != 200:
            return None
        body = resp.json()
        sections = (body.get("jobAd") or {}).get("sections") or {}
        text = "\n\n".join(
            html_to_text((sections.get(key) or {}).get("text"))
            for key in ("jobDescription", "qualifications", "additionalInformation")
        )
        return Job(
            source="smartrecruiters",
            external_id=f"{company_id}:{item['id']}",
            company=(item.get("company") or {}).get("name") or company_id,
            title=(item.get("name") or "").strip(),
            url=body.get("postingUrl") or body.get("applyUrl") or "",
            description=clean_text(text),
            locations=[parsed],
            remote="remote" if loc.get("remote") else ("hybrid" if loc.get("hybrid") else "onsite"),
            posted_at=item.get("releasedDate"),
            employment_type=(item.get("typeOfEmployment") or {}).get("label"),
            department=(item.get("department") or {}).get("label"),
        )

    results = await asyncio.gather(*(detail(i) for i in listed))
    return [job for job in results if job]
