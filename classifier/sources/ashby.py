"""Ashby public job board API: https://developers.ashbyhq.com/docs/public-job-posting-api"""

from __future__ import annotations

import httpx

from classifier.locations import parse_location, remote_mode
from classifier.models import Job
from classifier.text import clean_text, html_to_text

API = "https://api.ashbyhq.com/posting-api/job-board"


def _salary(comp: dict | None):
    for tier in (comp or {}).get("summaryComponents") or []:
        if tier.get("compensationType") == "Salary":
            return tier.get("minValue"), tier.get("maxValue"), tier.get("currencyCode"), (tier.get("interval") or "").replace("1 ", "").lower() or None
    return None, None, None, None


async def fetch(client: httpx.AsyncClient, board: str) -> list[Job]:
    resp = await client.get(f"{API}/{board}", params={"includeCompensation": "true"})
    resp.raise_for_status()
    data = resp.json()
    company = (data.get("apiVersion") and board) or board
    jobs = []
    for item in data.get("jobs", []):
        if item.get("isListed") is False:
            continue
        raw = [item.get("location") or ""] + [s.get("location", "") for s in item.get("secondaryLocations") or []]
        locations = [parse_location(r) for r in raw if r]
        postal = ((item.get("address") or {}).get("postalAddress") or {})
        if locations and locations[0].country is None and postal.get("addressCountry"):
            locations[0] = parse_location(", ".join(x for x in (postal.get("addressLocality"), postal.get("addressRegion"), postal.get("addressCountry")) if x))
        lo, hi, cur, per = _salary(item.get("compensation"))
        workplace = (item.get("workplaceType") or "").lower()
        jobs.append(
            Job(
                source="ashby",
                external_id=f"{board}:{item['id']}",
                company=company.replace("-", " ").title(),
                title=(item.get("title") or "").strip(),
                url=item.get("jobUrl", ""),
                description=clean_text(item.get("descriptionPlain") or html_to_text(item.get("descriptionHtml"))),
                locations=locations,
                remote="remote" if item.get("isRemote") or workplace == "remote" else ("hybrid" if workplace == "hybrid" else remote_mode(item.get("location"))),
                posted_at=item.get("publishedAt"),
                employment_type=item.get("employmentType"),
                department=item.get("department") or item.get("team"),
                salary_min=lo, salary_max=hi, salary_currency=cur, salary_period=per,
            )
        )
    return jobs
