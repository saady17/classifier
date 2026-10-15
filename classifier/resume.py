"""Reads a resume: text, sections, work history (titles, dates, total years), degree, location, skills."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from datetime import date

from classifier.jobparse import _DEGREES
from classifier.locations import parse_location
from classifier.skills import matcher
from classifier.text import clean_text
from classifier.titles import index as title_index
from classifier.titles import level as title_level

SECTION_PATTERNS = [
    ("experience", r"(work |professional |relevant |employment |career )?(experience|history)|employment|work history|positions held|career"),
    ("education", r"education|academic|qualifications and education|degrees?"),
    ("skills", r"(technical |core |key |professional )?(skills|competencies|expertise|proficiencies|technologies|tools)|tech stack"),
    ("certifications", r"certifications?|licen[cs]es?|credentials|certificates?"),
    ("summary", r"summary|profile|objective|about( me)?|professional summary|career summary"),
    ("projects", r"projects?|portfolio"),
    ("other", r"volunteer(ing)?|awards?|honou?rs|publications|interests|languages|activities|references|leadership"),
]
_MONTHS = "jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec"
_MONTH_NUM = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
_DATE = rf"(?:(?:{_MONTHS})[a-z]*\.?\s*'?\d{{2,4}}|\d{{1,2}}\s*/\s*\d{{2,4}}|(?:spring|summer|fall|autumn|winter)\s+\d{{4}}|\d{{4}})"
_RANGE = re.compile(rf"({_DATE})\s*(?:-|–|—|to|until|through|–)\s*(present|current|now|today|ongoing|{_DATE})", re.I)


@dataclass
class Role:
    title: str
    start: date
    end: date
    occupation: str | None
    level: int


@dataclass
class ResumeProfile:
    text: str
    skills: dict[str, float] = field(default_factory=dict)  # skill -> evidence weight
    roles: list[Role] = field(default_factory=list)
    years: float = 0.0
    current_title: str | None = None
    occupation: str | None = None
    occupations: dict[str, float] = field(default_factory=dict)  # SOC -> weight (recent roles weigh more)
    level: int = 2
    degree: int = 0
    location_country: str | None = None
    location_region: str | None = None
    location_city: str | None = None


def extract_text(data: bytes, filename: str) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        import pdfplumber

        with pdfplumber.open(io.BytesIO(data)) as pdf:
            pages = [page.extract_text(x_tolerance=1.5, y_tolerance=3) or "" for page in pdf.pages[:6]]
        return clean_text("\n".join(pages))
    if name.endswith(".docx"):
        import docx

        doc = docx.Document(io.BytesIO(data))
        parts = [p.text for p in doc.paragraphs]
        for table in doc.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text for cell in row.cells))
        return clean_text("\n".join(parts))
    return clean_text(data.decode("utf-8", errors="ignore"))


def _section_of(line: str) -> str | None:
    text = line.strip().strip(":•|-–").strip()
    if not text or len(text) > 40 or len(text.split()) > 5:
        return None
    low = text.lower()
    for name, pattern in SECTION_PATTERNS:
        if re.fullmatch(rf"({pattern})s?", low):
            return name
    return None


def split_sections(text: str) -> list[tuple[str, str]]:
    current, out = "header", []
    for line in text.splitlines():
        sec = _section_of(line)
        if sec:
            current = sec
            continue
        out.append((current, line))
    return out


def _parse_date(text: str, is_end: bool) -> date | None:
    t = text.lower().strip().replace("'", "")
    today = date.today()
    if t in ("present", "current", "now", "today", "ongoing"):
        return today
    if m := re.match(rf"({_MONTHS})[a-z]*\.?\s*(\d{{2,4}})", t):
        y = int(m.group(2))
        y = y + 2000 if y < 50 else (y + 1900 if y < 100 else y)
        return date(y, _MONTH_NUM[m.group(1)[:3]], 1)
    if m := re.match(r"(\d{1,2})\s*/\s*(\d{2,4})", t):
        mo, y = int(m.group(1)), int(m.group(2))
        y = y + 2000 if y < 50 else (y + 1900 if y < 100 else y)
        return date(y, min(max(mo, 1), 12), 1)
    if m := re.match(r"(spring|summer|fall|autumn|winter)\s+(\d{4})", t):
        return date(int(m.group(2)), {"spring": 3, "summer": 6, "fall": 9, "autumn": 9, "winter": 12}[m.group(1)], 1)
    if m := re.match(r"(\d{4})", t):
        return date(int(m.group(1)), 12 if is_end else 1, 1)
    return None


def _merged_years(spans: list[tuple[date, date]]) -> float:
    total, last_end = 0, None
    for start, end in sorted(spans):
        if last_end and start <= last_end:
            if end > last_end:
                total += (end - last_end).days
                last_end = end
            continue
        total += (end - start).days
        last_end = end
    return round(total / 365.25, 1)


_SPLIT = re.compile(r"\s*(?:\||•|●|▪|·|,|\s[-–—/]\s|\sat\s|@|\t)\s*")
# A job title names a role. Without one of these words (or an exact known title) a line is a
# company, a location, a project or a sentence, not a title.
ROLE_WORDS = re.compile(
    r"\b(engineer|developer|programmer|architect|analyst|scientist|researcher|manager|director|head|lead|"
    r"officer|executive|president|vp|founder|co-founder|owner|partner|principal|consultant|advisor|adviser|"
    r"specialist|coordinator|associate|assistant|administrator|admin|representative|rep|agent|clerk|"
    r"technician|technologist|tech|designer|artist|writer|editor|producer|strategist|marketer|recruiter|"
    r"accountant|auditor|controller|bookkeeper|nurse|rn|lpn|np|physician|doctor|therapist|pharmacist|"
    r"counselor|teacher|tutor|instructor|professor|lecturer|educator|ta|intern|internship|trainee|apprentice|"
    r"fellow|volunteer|operator|driver|cashier|server|barista|cook|chef|supervisor|foreman|worker|handler|"
    r"associate|attorney|lawyer|paralegal|underwriter|broker|banker|trader|planner|buyer|scheduler|"
    r"receptionist|secretary|generalist|investigator|inspector|mechanic|electrician|welder|carpenter|"
    r"plumber|caregiver|aide|assistant|steward|ambassador|organizer|owner|member of technical staff|sde|swe|"
    r"devops|sre|qa|pm|ux|ui)s?\b",
    re.I,
)
_SIDE_JOB = re.compile(r"\b(student|part[- ]time|campus|seasonal|volunteer|server|barista|cashier|crew member|tutor)\b", re.I)
_NOT_TITLE = re.compile(r"^(present|current|now|today|ongoing|remote|hybrid|onsite|full[- ]time|part[- ]time|contract)$", re.I)


_GLUED = re.compile(rf"(?<=[a-z)\]])(?=(?:{_MONTHS})[a-z]*\.?\s*\d{{2,4}})|(?<=[A-Za-z)])(?=\d{{4}}\s*[-–—])", re.I)


def _unglue(line: str) -> str:
    """PDF text sometimes loses the space before the dates: "Researcher)Jan 2022", "Streamlit2025–Present"."""
    return _GLUED.sub(" ", line)


def _title_like(part: str) -> bool:
    words = part.split()
    if not 1 <= len(words) <= 7 or len(part) > 70 or _NOT_TITLE.match(part) or ":" in part:
        return False
    if re.search(r"\d{4}|%|\$", part) or part[:1].islower():
        return False  # dates, metrics, or a sentence fragment
    capitalized = sum(1 for w in words if w[:1].isupper() or not w[:1].isalpha())
    if capitalized < 0.6 * len(words):
        return False  # titles are written in Title Case or CAPS
    return bool(ROLE_WORDS.search(part))


def _best_title(candidates: list[str]) -> tuple[str | None, str | None, float]:
    ix = title_index()
    best = (None, None, 0.0)
    for cand in candidates:
        cand = re.sub(r"^[\s●•▪*\-–]+", "", cand)
        cand = _RANGE.sub(" | ", _unglue(cand))  # drop the dates before splitting on dashes
        for part in _SPLIT.split(cand):
            part = re.sub(r"\s*\([^)]*\)?", "", part).strip(" ()|")  # "(Graduate RA)", "(Volunteer"
            if not part or not _title_like(part):
                continue
            code, conf = ix.occupation(part)
            if conf > best[2]:
                best = (part, code, conf)
    return best


def parse_resume(text: str) -> ResumeProfile:
    profile = ResumeProfile(text=text)
    lines = split_sections(text)
    plain = [line for _, line in lines]

    # Work history: every date range in the experience section (or anywhere, if no section was found).
    in_exp = [i for i, (sec, _) in enumerate(lines) if sec == "experience"]
    scan = in_exp or [i for i, (sec, _) in enumerate(lines) if sec not in ("education",)]
    for i in scan:
        m = _RANGE.search(_unglue(plain[i]))
        if not m:
            continue
        start, end = _parse_date(m.group(1), False), _parse_date(m.group(2), True)
        if not start or not end or end < start or start.year < 1960 or (end - start).days > 365 * 45:
            continue
        nearby = [plain[i]] + [plain[j] for j in (i - 1, i + 1, i - 2, i + 2) if 0 <= j < len(plain)]
        title, code, conf = _best_title(nearby)
        if title and conf >= 0.7:
            profile.roles.append(Role(title, start, end, code, title_level(title)))
    profile.roles.sort(key=lambda r: (r.end, r.start), reverse=True)
    # Internships count a little toward experience; overlapping roles are not double-counted.
    full = [(r.start, r.end) for r in profile.roles if r.level > 0]
    profile.years = _merged_years(full) + 0.25 * _merged_years([(r.start, r.end) for r in profile.roles if r.level == 0])
    if profile.roles:
        for rank, role in enumerate(profile.roles[:6]):
            if role.occupation:
                weight = 1.0 / (1 + rank) * max(0.3, min(1.0, (role.end - role.start).days / 730))
                if _SIDE_JOB.search(role.title):
                    weight *= 0.35  # a student or part-time job alongside a career says less about the field
                profile.occupations[role.occupation] = profile.occupations.get(role.occupation, 0) + weight
        # The main occupation is the strongest across the history, not just the newest role.
        main = max(profile.occupations, key=profile.occupations.get) if profile.occupations else None
        current = next((r for r in profile.roles if r.occupation == main), profile.roles[0])
        profile.current_title, profile.occupation, profile.level = current.title, current.occupation, current.level
        if profile.level == 2 and profile.years >= 7:
            profile.level = 3  # long experience without a level word in the title
        if profile.years < 1.5 and profile.level >= 2:
            profile.level = 1
    else:
        # No dated roles: fall back to a title-like line in the header.
        title, code, conf = _best_title([line for sec, line in lines[:12] if sec in ("header", "summary")])
        if title and conf >= 0.85:
            profile.current_title, profile.occupation = title, code
            profile.occupations = {code: 1.0}

    # Highest degree, from the education section when there is one.
    edu = "\n".join(line for sec, line in lines if sec == "education") or text
    low = edu.lower()
    profile.degree = next((lvl for lvl, pattern in _DEGREES if re.search(pattern, low)), 0)

    # Location: first line near the top that names a US/UK/Canadian place.
    for _, line in lines[:10]:
        for part in re.split(r"\s*(?:[|•·]|\s[-–—]\s)\s*", line):
            if "@" in part or re.search(r"\d{3}.\d{3}", part):
                continue
            loc = parse_location(part)
            if loc.country:
                profile.location_country, profile.location_region, profile.location_city = loc.country, loc.region, loc.city
                break
        if profile.location_country:
            break

    # Skills: listed skills count once; skills used in work history count more (evidence), recent roles most.
    m = matcher()
    weights = {"skills": 1.0, "certifications": 1.2, "experience": 1.3, "projects": 1.0, "summary": 0.8, "header": 0.6, "education": 0.6, "other": 0.5}
    for sec, line in lines:
        for name, count in m.find(line).items():
            w = weights.get(sec, 0.7) * min(count, 2)
            profile.skills[name] = profile.skills.get(name, 0.0) + w
    return profile
