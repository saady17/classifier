"""Reads a job post: required vs preferred skills, years of experience, degree and visa sponsorship."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from classifier.skills import matcher

REQ, PREF, DUTY, SKIP, BODY = "required", "preferred", "duties", "skip", "body"

_HEADINGS = [
    (PREF, r"preferred|nice[- ]to[- ]have|bonus|plus(es)?\b|desired|it would be great|extra credit|not required"),
    (REQ, r"requirements?|qualifications?|what you('ll)? (need|bring)|what we('re)? looking for|you (have|bring|are)|"
          r"must[- ]haves?|about you|who you are|skills|experience|your background|you might be|ideal candidate|competencies"),
    (DUTY, r"responsibilit|what you('ll)? do|the role|role overview|day[- ]to[- ]day|duties|in this role|you will|key tasks|impact"),
    (SKIP, r"benefits|perks|about (us|the company|the team)|who we are|why (join|work)|compensation|salary|pay range|"
           r"equal opportunity|eeo|diversity|accommodation|our (mission|values|culture)|life at|we offer|what we offer|"
           r"privacy|disclaimer|how to apply|application process|location"),
]
# Benefit lines without a heading ("medical, dental and vision", "401(k)") are not requirements.
_LINE_SKIP = re.compile(
    r"\b(401\(?k\)?|dental|vision insurance|health insurance|medical, dental|paid time off|\bpto\b|parental leave|"
    r"equity|stock options|rsus?|wellness|stipend|benefits|perks|reimbursement|holidays|retirement|commuter)\b|"
    # Pay transparency, equal-opportunity and agency boilerplate.
    r"\$\s?\d|salary|pay range|base pay|compensation range|total compensation|on-target earnings|\bote\b|"
    r"equal (employment )?opportunity|regardless of|gender identity|sexual orientation|veteran status|"
    r"protected (class|characteristic|veteran)|reasonable accommodation|e-?verify|recruitment agenc|recruiting agenc|"
    r"third[- ]party (recruit|agenc)|unsolicited resumes|background check|drug (screen|test)",
    re.I,
)
_LINE_PREF = re.compile(r"\b(preferred|a plus|is a plus|nice to have|bonus|ideally|desired|advantageous|beneficial)\b", re.I)

_YEARS = re.compile(
    r"(\d{1,2})\s*(?:\+|plus)?\s*(?:-|–|to)?\s*(\d{1,2})?\s*\+?\s*(?:years?|yrs?)\b(?:\s+of)?(?:[^.\n]{0,60}?)\b(experience|exp\b|working|in\b|as\b|of\b)",
    re.I,
)
_DEGREES = [
    (5, r"\b(ph\.?d|doctorate|doctoral)\b"),
    (4, r"\b(master'?s|m\.?s\.?c?\b|m\.?a\.?\b|mba|m\.eng)\b"),
    (3, r"\b(bachelor'?s|b\.?s\.?c?\b|b\.?a\.?\b|b\.?eng|undergraduate degree|4[- ]year degree|college degree|university degree)\b"),
    (2, r"\b(associate'?s degree|associate degree|2[- ]year degree)\b"),
    (1, r"\b(high school|ged|secondary school)\b"),
]
_EQUIV = re.compile(r"or equivalent|or related experience|equivalent (practical |work )?experience|in lieu of", re.I)
_NO_SPONSOR = re.compile(
    r"(unable|not able|will not|won't|cannot|can't|do(es)? not|are not able|not in a position) to (provide |offer )?(visa )?sponsor|"
    r"(no|not offering|without|does not offer|do not offer|cannot offer|unable to offer) (\w+ ){0,2}(visa |immigration )?sponsorship|"
    r"sponsorship (is )?not (available|offered|provided)|"
    r"must (be|have) (legally )?(authorized|eligible|authorization) to work[^.]{0,80}without[^.]{0,30}sponsor|"
    r"(u\.?s\.? citizens?(hip)? (is )?(required|only))|must be a u\.?s\.? citizen|green card holders? only",
    re.I,
)
_SPONSOR = re.compile(
    r"(visa|h-?1b|immigration) sponsorship (is )?(available|offered|provided)|we (will|can|do) sponsor|"
    r"(open|willing) to sponsor|sponsorship available|will provide (visa )?sponsorship|offers? (visa )?sponsorship",
    re.I,
)
_CLEARANCE = re.compile(r"\b(security clearance|secret clearance|ts/sci|top secret|public trust clearance|active clearance)\b", re.I)


@dataclass
class JobProfile:
    required: dict[str, int] = field(default_factory=dict)
    preferred: dict[str, int] = field(default_factory=dict)
    duties: dict[str, int] = field(default_factory=dict)
    min_years: float | None = None
    preferred_years: float | None = None
    degree: int = 0  # 0 none stated, 1 high school .. 5 PhD
    degree_required: bool = False
    degree_equivalent_ok: bool = False
    no_sponsorship: bool = False
    sponsors: bool = False
    clearance: bool = False
    structured: bool = False  # True when requirement sections were found


def _heading_kind(line: str) -> str | None:
    text = line.strip().strip("#*:–-—").strip()
    if not text or len(text) > 70 or text.startswith("- "):
        return None
    looks_heading = line.rstrip().endswith(":") or len(text.split()) <= 7 and not text.endswith(".")
    if not looks_heading:
        return None
    low = text.lower()
    for kind, pattern in _HEADINGS:
        if re.search(pattern, low):
            return kind
    return None


def sections(text: str) -> list[tuple[str, str]]:
    """(kind, line) for every line, kind taken from the nearest heading above."""
    kind, out = BODY, []
    for line in text.splitlines():
        heading = _heading_kind(line)
        if heading:
            kind = heading
            continue
        out.append((kind, line))
    return out


def _years(lines: list[str]) -> float | None:
    found = []
    for line in lines:
        for m in _YEARS.finditer(line):
            low = int(m.group(1))
            if 0 < low <= 20:
                found.append(low)
    return float(max(found)) if found else None


def _degree(text: str) -> int:
    low = text.lower()
    for level, pattern in _DEGREES:
        if re.search(pattern, low):
            return level
    return 0


def parse_job(description: str, company: str = "") -> JobProfile:
    m = matcher()
    company_key = re.sub(r"[^a-z0-9]", "", company.lower())
    lines = sections(description)
    profile = JobProfile(structured=any(kind in (REQ, PREF) for kind, _ in lines))
    req_lines, pref_lines = [], []
    for kind, line in lines:
        if kind == SKIP or not line.strip() or _LINE_SKIP.search(line):
            continue
        found = m.find(line)
        if company_key:
            # The employer's own product is not a requirement ("LinkedIn" at LinkedIn).
            found = {k: v for k, v in found.items() if re.sub(r"[^a-z0-9]", "", k.lower()) not in (company_key, company_key + "software")}
        if kind == PREF or (kind in (REQ, BODY, DUTY) and _LINE_PREF.search(line)):
            bucket, store = pref_lines, profile.preferred
        elif kind == DUTY:
            bucket, store = req_lines, profile.duties
        else:
            bucket, store = req_lines, profile.required
        bucket.append(line)
        for name, count in found.items():
            store[name] = store.get(name, 0) + count
    # A skill both required and preferred counts as required; duty mentions add weight to required.
    for name in list(profile.preferred):
        if name in profile.required:
            del profile.preferred[name]
    profile.min_years = _years(req_lines)
    profile.preferred_years = _years(pref_lines)
    req_text = "\n".join(req_lines)
    profile.degree = _degree(req_text) or _degree("\n".join(pref_lines))
    profile.degree_required = bool(_degree(req_text))
    profile.degree_equivalent_ok = bool(_EQUIV.search(description))
    whole = description
    profile.no_sponsorship = bool(_NO_SPONSOR.search(whole))
    profile.sponsors = bool(_SPONSOR.search(whole)) and not profile.no_sponsorship
    profile.clearance = bool(_CLEARANCE.search(whole))
    return profile
