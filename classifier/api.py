"""HTTP API for app.jobzhub.com, plus the internal rating page for evaluation.

    uvicorn classifier.api:app --host 0.0.0.0 --port 8000

Set CLASSIFIER_API_KEY to require an X-API-Key header on /v1/*.
Set RATER_PASSWORD to enable the rating page at /eval (HTTP basic auth, any user name)."""

from __future__ import annotations

import os
import secrets
from dataclasses import asdict

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel

from classifier import store
from classifier.filters import (
    EMPLOYMENT_TYPES,
    LABEL_ORDER,
    SENIORITY_LEVELS,
    WORK_MODES,
)
from classifier.ranker import Preferences, rank
from classifier.resume import extract_text, parse_resume
from classifier.titles import index as title_index

app = FastAPI(title="Jobzhub classifier", version="0.1")
MAX_RESUME_BYTES = 5 * 1024 * 1024


def require_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.environ.get("CLASSIFIER_API_KEY")
    if expected and not (x_api_key and secrets.compare_digest(x_api_key, expected)):
        raise HTTPException(401, "Missing or wrong X-API-Key.")


class Prefs(BaseModel):
    """Who the person is (facts) and what they are looking for (recruiter-style hard filters)."""

    countries: list[str] | None = None  # default: the resume's own country, else US/UK/CA
    remote: str = "any"  # legacy: any | remote_only | no_remote (prefer work_modes)
    needs_sponsorship: bool = False  # hard check against US Department of Labor H-1B data
    has_clearance: bool = False
    willing_to_relocate: bool = True
    work_authorization: str = "unknown"  # citizen | permanent_resident | needs_sponsorship | unknown
    employment_types: list[str] = []  # full_time, part_time, contract, temporary, internship
    work_modes: list[str] = []  # remote, hybrid, onsite
    locations: list[str] = []  # "Austin, TX", "New York", "Ontario", "UK"
    excluded_employers: list[str] = []
    only_employers: list[str] = []
    title_include: list[str] = []
    title_exclude: list[str] = []
    seniority: list[str] = []  # intern, entry, mid, senior, lead, director, executive, or "auto"
    min_salary: float | None = None  # annual, in the job's currency
    include_unlisted_salary: bool = True
    min_label: str | None = None  # "Excellent match" | "Good match" | "Possible match"


class TextRequest(BaseModel):
    resume_text: str
    preferences: Prefs = Prefs()
    limit: int = 30


def _run(text: str, prefs: Prefs, limit: int) -> dict:
    if len(text.strip()) < 80:
        raise HTTPException(422, "The resume has too little readable text. Is it a scanned image?")
    resume = parse_resume(text)
    countries = tuple(c.upper() for c in prefs.countries) if prefs.countries else (
        (resume.location_country,) if resume.location_country else ("US", "GB", "CA")
    )
    for name, values, allowed in (("employment_types", prefs.employment_types, EMPLOYMENT_TYPES),
                                  ("work_modes", prefs.work_modes, WORK_MODES),
                                  ("seniority", prefs.seniority, (*SENIORITY_LEVELS, "auto"))):
        bad = [v for v in values if v not in allowed]
        if bad:
            raise HTTPException(422, f"Unknown {name}: {', '.join(bad)}. Allowed: {', '.join(allowed)}.")
    if prefs.min_label and prefs.min_label not in LABEL_ORDER:
        raise HTTPException(422, f"min_label must be one of: {', '.join(LABEL_ORDER[1:])}.")
    p = Preferences(
        countries=countries, remote=prefs.remote, needs_sponsorship=prefs.needs_sponsorship,
        has_clearance=prefs.has_clearance, willing_to_relocate=prefs.willing_to_relocate,
        work_authorization=prefs.work_authorization,
        city=resume.location_city, region=resume.location_region,
        employment_types=tuple(prefs.employment_types), work_modes=tuple(prefs.work_modes), locations=tuple(prefs.locations),
        excluded_employers=tuple(prefs.excluded_employers), only_employers=tuple(prefs.only_employers),
        title_include=tuple(prefs.title_include), title_exclude=tuple(prefs.title_exclude), seniority=tuple(prefs.seniority),
        min_salary=prefs.min_salary, include_unlisted_salary=prefs.include_unlisted_salary, min_label=prefs.min_label,
    )
    matches = rank(resume, p, limit=max(1, min(limit, 100)))
    return {
        "resume": {
            "current_title": resume.current_title,
            "occupation": title_index().name(resume.occupation),
            "level": resume.level,
            "years": resume.years,
            "degree": resume.degree,
            "location": ", ".join(x for x in (resume.location_city, resume.location_region, resume.location_country) if x) or None,
            "top_skills": [s for s, _ in sorted(resume.skills.items(), key=lambda x: -x[1])[:20]],
        },
        "matches": [{k: v for k, v in asdict(m).items() if k != "features"} for m in matches],
    }


@app.get("/health")
def health() -> dict:
    db = store.connect()
    jobs = db.execute("SELECT count(*) FROM jobs WHERE duplicate_of IS NULL").fetchone()[0]
    last = db.execute("SELECT max(fetched_at) FROM fetch_log").fetchone()[0]
    return {"ok": True, "jobs": jobs, "last_fetch": last}


@app.post("/v1/rank", dependencies=[Depends(require_key)])
async def rank_file(
    file: UploadFile = File(...),
    preferences: str = Form("{}"),  # JSON object with the same fields as Prefs
    limit: int = Form(30),
) -> dict:
    data = await file.read()
    if len(data) > MAX_RESUME_BYTES:
        raise HTTPException(413, "Resume files must be under 5 MB.")
    try:
        prefs = Prefs.model_validate_json(preferences or "{}")
    except ValueError as exc:
        raise HTTPException(422, f"preferences is not valid: {exc}") from exc
    text = extract_text(data, file.filename or "resume.pdf")
    return _run(text, prefs, limit)


@app.get("/v1/filters")
def filter_options() -> dict:
    """The allowed values for each filter, for building a search form."""
    return {"employment_types": list(EMPLOYMENT_TYPES), "work_modes": list(WORK_MODES),
            "seniority": [*SENIORITY_LEVELS, "auto"], "min_label": LABEL_ORDER[1:],
            "work_authorization": ["citizen", "permanent_resident", "needs_sponsorship", "unknown"]}


@app.post("/v1/rank-text", dependencies=[Depends(require_key)])
def rank_text(body: TextRequest) -> dict:
    return _run(body.resume_text, body.preferences, body.limit)


# ---- Internal rating page (evaluation) ----
basic = HTTPBasic(auto_error=True)


def rater(credentials: HTTPBasicCredentials = Depends(basic)) -> str:
    password = os.environ.get("RATER_PASSWORD")
    if not password or not secrets.compare_digest(credentials.password, password):
        raise HTTPException(401, "Wrong password.", headers={"WWW-Authenticate": "Basic"})
    return credentials.username.strip()[:40] or "rater"


from classifier import evaluation  # noqa: E402  (routes need `app` and `rater`)

evaluation.mount(app, rater)
