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
    (5, r"\b(ph\.?\s?d|doctorate|doctoral|doctor of|d\.?n\.?p|pharm\.?d|m\.?d\.?|j\.?d\.?|psy\.?d|ed\.?d)\b"),
    (4, r"\b(masters?|master'?s|master of|m\.?s\.?c?|m\.?a\.?|mba|m\.?eng|msn|mph|msw|m\.?ed|llm)\b"),
    (3, r"\b(bachelors?|bachelor'?s|bachelor of|b\.?s\.?c?|b\.?a\.?|b\.?eng|bsn|b\.?comm?|bba|b\.?tech|b\.?e\.?|"
        r"undergraduate degree|4[- ]year degree|college degree|university degree)\b"),
    (2, r"\b(associate'?s degree|associate degree|associate of|a\.?a\.?s?|adn|2[- ]year degree|diploma in|hnd)\b"),
    (1, r"\b(high school|ged|secondary school|a[- ]levels?)\b"),
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
    if line.lstrip().startswith(("-", "•", "*", "·", "–")):
        return None  # a bullet is content, even when short ("- MBA preferred")
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


_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
                 "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15}
_NUMBER_WORD = re.compile(r"\b(" + "|".join(_NUMBER_WORDS) + r")\b(?=\s*(\(\d+\)\s*)?\+?\s*(-|to)?\s*\w*\s*(years?|yrs))", re.I)


def _years(lines: list[str]) -> float | None:
    found = []
    # "five years", "Three (3) years" -> digits, so one pattern handles both.
    lines = [_NUMBER_WORD.sub(lambda m: str(_NUMBER_WORDS[m.group(1).lower()]), re.sub(r"\b(\w+) \((\d+)\)", r"\2", line)) for line in lines]
    for line in lines:
        for m in _YEARS.finditer(line):
            low = int(m.group(1))
            if 0 < low <= 20:
                found.append(low)
    return float(max(found)) if found else None


def _degree(text: str) -> int:
    """Minimum degree a job asks for: "BS or MS" means a bachelor's is enough."""
    low = text.lower()
    found = [level for level, pattern in _DEGREES if re.search(pattern, low)]
    return min(found) if found else 0


def _names_employer(skill: str, company: str) -> bool:
    """'LinkedIn' at LinkedIn, 'Epic' at Epic Games, 'Datadog' at Datadog."""
    sk = re.sub(r"[^a-z0-9 ]", "", skill.lower()).split()
    co = re.sub(r"[^a-z0-9 ]", " ", company.lower()).split()
    if not sk or not co:
        return False
    joined = "".join(co)
    return "".join(sk) in (joined, joined + "software") or (len(sk) == 1 and sk[0] == co[0])


def parse_job(description: str, company: str = "") -> JobProfile:
    m = matcher()
    company_key = re.sub(r"[^a-z0-9]", "", company.lower())
    lines = sections(description)
    profile = JobProfile(structured=any(kind in (REQ, PREF) for kind, _ in lines))
    req_lines, pref_lines = [], []
    clauses = [(kind, c) for kind, line in lines for c in re.split(r";\s+|(?<=[a-z0-9)])\.\s+(?=[A-Z])", line)]
    for kind, line in clauses:
        if kind == SKIP or not line.strip() or _LINE_SKIP.search(line):
            continue
        found = m.find(line)
        if company_key:
            # The employer's own product is not a requirement ("LinkedIn" at LinkedIn).
            found = {k: v for k, v in found.items() if not _names_employer(k, company)}
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
