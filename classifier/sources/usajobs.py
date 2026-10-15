"""USAJOBS Search API (US federal jobs). Needs a free key: https://developer.usajobs.gov/apirequest/
Set USAJOBS_API_KEY and USAJOBS_EMAIL. `account` is a keyword (e.g. "software engineer")."""

from __future__ import annotations

import os

import httpx

from classifier.locations import parse_location, remote_mode
from classifier.models import Job
from classifier.text import clean_text

API = "https://data.usajobs.gov/api/search"


def enabled() -> bool:
    return bool(os.environ.get("USAJOBS_API_KEY") and os.environ.get("USAJOBS_EMAIL"))


async def fetch(client: httpx.AsyncClient, keyword: str) -> list[Job]:
    if not enabled():
        return []
    headers = {"Host": "data.usajobs.gov", "User-Agent": os.environ["USAJOBS_EMAIL"], "Authorization-Key": os.environ["USAJOBS_API_KEY"]}
    jobs = []
    for page in range(1, 6):
        resp = await client.get(API, params={"Keyword": keyword, "ResultsPerPage": 500, "Page": page}, headers=headers)
        resp.raise_for_status()
        items = resp.json().get("SearchResult", {}).get("SearchResultItems", [])
        for wrap in items:
            d = wrap.get("MatchedObjectDescriptor", {})
            details = (d.get("UserArea") or {}).get("Details") or {}
            pay = (d.get("PositionRemuneration") or [{}])[0]
            text = "\n\n".join(x for x in (d.get("QualificationSummary"), details.get("JobSummary"), "\n".join(details.get("MajorDuties") or []), details.get("Education"), details.get("Requirements")) if x)
            jobs.append(Job(
                source="usajobs", external_id=d.get("PositionID") or wrap.get("MatchedObjectId"),
                company=d.get("OrganizationName") or "US Government", title=d.get("PositionTitle", ""),
                url=d.get("PositionURI", ""), description=clean_text(text),
                locations=[parse_location(loc.get("LocationName", "")) for loc in d.get("PositionLocation") or []],
                remote=remote_mode(d.get("PositionLocationDisplay"), details.get("RemoteIndicator") and "remote"),
                posted_at=d.get("PublicationStartDate"),
                salary_min=float(pay["MinimumRange"]) if pay.get("MinimumRange") else None,
                salary_max=float(pay["MaximumRange"]) if pay.get("MaximumRange") else None,
                salary_currency="USD", salary_period={"PA": "year", "PH": "hour"}.get(pay.get("RateIntervalCode")),
            ))
        if len(items) < 500:
            break
    return jobs
