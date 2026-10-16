"""Recruiter-style hard filters: the criteria a recruiter or job seeker sets before ranking.

A job that fails any filter is not shown at all (it is never "ranked low"). Filters are applied to
data the job reader already extracted; nothing here uses AI.

    employment type   full_time | part_time | contract | temporary | internship
    work mode         remote | hybrid | onsite
    locations         "Austin, TX", "New York", "Ontario", "UK"... (city, state/province or country)
    employers         excluded employers (e.g. the current one); optional allow-list
    title keywords    must contain any of / must not contain any of
    seniority         intern | entry | mid | senior | lead | director | executive, or "auto"
    salary            minimum annual pay (jobs without a stated salary can be kept or dropped)
    match label       only Excellent / Good / Possible matches"""

from __future__ import annotations

import re

from classifier.h1b import name_key
from classifier.locations import parse_location

EMPLOYMENT_TYPES = ("full_time", "part_time", "contract", "temporary", "internship")
WORK_MODES = ("remote", "hybrid", "onsite")
SENIORITY_LEVELS = {
    "intern": {0}, "entry": {1}, "mid": {2}, "senior": {3}, "lead": {4}, "director": {5}, "executive": {6, 7},
}
LABEL_ORDER = ["Stretch", "Possible match", "Good match", "Excellent match"]

_FEED_TYPES = [
    ("internship", r"intern|co-?op|apprentice|trainee"),
    ("part_time", r"part[\s_-]?time|parttime"),
    ("temporary", r"temp|temporary|seasonal|per[\s_-]?diem|prn|fixed[\s_-]?term"),
    ("contract", r"contract|contractor|freelance|consult|1099|c2c|corp[\s-]?to[\s-]?corp"),
    ("full_time", r"full[\s_-]?time|fulltime|permanent|regular|salaried"),
]
_TITLE_TYPES = [
    ("internship", r"\b(intern|internship|co-?op|apprentice(ship)?)\b"),
    ("part_time", r"\bpart[\s-]?time\b"),
    ("temporary", r"\b(temporary|temp|seasonal|per diem|prn|fixed[\s-]term)\b"),
    ("contract", r"\b(contract|contractor|freelance|1099|c2c)\b"),
]
_TEXT_TYPES = [
    ("internship", r"\b(this|the) (is an? )?(paid |summer |fall |spring )?internship\b"),
    ("part_time", r"\b(this|the) (is an? )?(role|position|job)? ?(is )?(a )?part[\s-]?time\b|employment type:?\s*part[\s-]?time|\bpart[\s-]?time (role|position|job|schedule)\b"),
    ("contract", r"employment type:?\s*contract|\b(this|the) (is an? )?(\d+[\s-]month )?(role|position)? ?(is )?(a )?contract\b|\b(\d+[\s-]month )?contract (role|position|engagement)\b|\bw2 contract\b"),
    ("temporary", r"employment type:?\s*(temporary|seasonal)|\b(temporary|seasonal) (role|position|assignment)\b|\b\d+[\s-]month assignment\b"),
]


def employment_type(feed_value: str | None, title: str, description: str) -> str:
    """Normalized employment type from the feed's field, the title, or the opening of the post."""
    v = (feed_value or "").lower()
    for name, pattern in _FEED_TYPES:
        if v and re.search(pattern, v):
            return name
    for name, pattern in _TITLE_TYPES:
        if re.search(pattern, title, re.I):
            return name
    head = description[:1500]
    for name, pattern in _TEXT_TYPES:
        if re.search(pattern, head, re.I):
            return name
    return "full_time"


_AMOUNT = r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?\s?[kK]|\d+(?:\.\d+)?)"
_SALARY = re.compile(
    rf"(US\$|CA\$|C\$|\$|£|USD|CAD|GBP)\s?{_AMOUNT}\s*(?:USD|CAD|GBP)?\s*(?:-|–|—|to)\s*(?:US\$|CA\$|C\$|\$|£)?\s?{_AMOUNT}"
    r"(?:\s*(?:USD|CAD|GBP))?(?:\s*(?:per|/|an?|a)\s*(hour|hr|year|yr|annum|annually|month))?",
    re.I,
)


def _amount(text: str) -> float:
    t = text.replace(",", "").strip().lower()
    return float(t[:-1].strip()) * 1000 if t.endswith("k") else float(t)


def salary_from_text(text: str) -> tuple[float, float, str, str] | None:
    """(min, max, currency, period) from the first pay range in a post, e.g. "$120,000 - $150,000"."""
    for m in _SALARY.finditer(text):
        if re.match(r"\s*(million|billion|mn|bn|m\b|b\b)", text[m.end():], re.I):
            continue  # funding or revenue, not pay
        cur = {"$": "USD", "us$": "USD", "usd": "USD", "ca$": "CAD", "c$": "CAD", "cad": "CAD", "£": "GBP", "gbp": "GBP"}[m.group(1).lower()]
        lo, hi = _amount(m.group(2)), _amount(m.group(3))
        if hi < lo:
            continue
        period = (m.group(4) or "").lower()
        period = "hour" if period in ("hour", "hr") or hi < 300 else ("month" if period == "month" else "year")
        if period == "year" and hi < 15_000:
            continue  # "$5 - $10 million in funding" and similar
        return lo, hi, cur, period
    return None


def annual(value: float | None, period: str | None) -> float | None:
    if value is None:
        return None
    return value * {"hour": 2080, "month": 12}.get(period or "year", 1)


def _location_match(locations: list[dict], wanted: list) -> bool:
    for want in wanted:
        for loc in locations:
            if want.city and (loc.get("city") or "").lower() == want.city.lower():
                return True
            if want.region and not want.city and loc.get("region") == want.region:
                return True
            if want.country and not want.region and not want.city and loc.get("country") == want.country:
                return True
    return False


def check(job_row, data: dict, prefs, resume, label: str | None = None) -> str | None:
    """None when the job passes every filter, otherwise the name of the first filter it fails."""
    import json

    etype = data.get("employment_type") or "full_time"
    if prefs.employment_types and etype not in prefs.employment_types:
        return "employment type"
    mode = job_row["remote"] if job_row["remote"] in WORK_MODES else "onsite"
    if prefs.work_modes and mode not in prefs.work_modes:
        return "work mode"
    if prefs.locations:
        wanted = [parse_location(x) for x in prefs.locations]
        locs = json.loads(job_row["locations"])
        remote_ok = mode == "remote" and "remote" in (prefs.work_modes or WORK_MODES) and any(
            w.country in {loc.get("country") for loc in locs} for w in wanted if w.country)
        if not remote_ok and not _location_match(locs, wanted):
            return "location"
    company = name_key(job_row["company"])
    if prefs.excluded_employers and any(company == name_key(e) for e in prefs.excluded_employers):
        return "excluded employer"
    if prefs.only_employers and not any(company == name_key(e) for e in prefs.only_employers):
        return "employer allow-list"
    title = job_row["title"].lower()
    if prefs.title_include and not any(k.lower() in title for k in prefs.title_include):
        return "title keywords"
    if prefs.title_exclude and any(k.lower() in title for k in prefs.title_exclude):
        return "excluded title keyword"
    if prefs.seniority:
        if "auto" in prefs.seniority:
            allowed = {resume.level - 1, resume.level, resume.level + 1}
        else:
            allowed = set().union(*(SENIORITY_LEVELS[s] for s in prefs.seniority if s in SENIORITY_LEVELS))
        if job_row["job_level"] not in allowed:
            return "seniority"
    if prefs.min_salary:
        top = annual(data.get("salary_max") or data.get("salary_min"), data.get("salary_period"))
        if top is None:
            if not prefs.include_unlisted_salary:
                return "salary not stated"
        elif top < prefs.min_salary:
            return "salary"
    if prefs.min_label and label and LABEL_ORDER.index(label) < LABEL_ORDER.index(prefs.min_label):
        return "match label"
    return None
