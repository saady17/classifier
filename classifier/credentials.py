"""State licenses and degree subjects, read the same way from job posts and resumes."""

from __future__ import annotations

import re

from classifier.locations import US_STATES

_NAME_TO_CODE = {name: code for code, name in US_STATES.items()}
_STATE_NAMES = sorted(_NAME_TO_CODE, key=len, reverse=True)
_STATE = r"(?:" + "|".join(re.escape(n) for n in _STATE_NAMES) + r"|[A-Z]{2})"
_PROFESSION = r"(?:RN|NP|PA|LPN|LVN|APRN|FNP|LCSW|LMSW|LMHC|LPC|LMFT|ASW|APCC|MH|nursing|nurse|registered nurse|nurse practitioner|educator'?s?|teaching|teacher|clinical|counseling|counselor|social work|mental health|pharmacy|pharmacist|bar|medical|massage|professional)"
# "licensed in NY, IN, MD, DC, or KY" / "license in the state of Virginia"
_LICENSED_IN = re.compile(rf"\b(?:licensed|licensure|license|licenses)\s+(?:in|for)\s+(?:the\s+state\s+of\s+)?({_STATE}(?:\s*(?:,|or|and|/)\s*(?:or\s+)?{_STATE})*)", re.I)
# "Active NY Nurse Practitioner license", "valid Tennessee teaching license", "Texas RN license", "CA MH Licensure"
_STATE_LICENSE = re.compile(rf"\b({_STATE})\s+(?:state\s+)?(?:(?:{_PROFESSION}|or|and|/)\s+){{0,5}}(?:license|licensure|licensed)\b", re.I)
# Resume style: "Registered Nurse, Texas (Compact)", "RN – California"
_RESUME_LICENSE = re.compile(rf"\b(?:RN|LPN|LVN|NP|APRN|registered nurse|licen[cs]ed?|license)\b[^\n]{{0,40}}?\b({_STATE})\b", re.I)
COMPACT = re.compile(r"\b(compact|multi-?state|nlc|nurse licensure compact)\b", re.I)
OBTAINABLE = re.compile(r"(ability|able|willing|eligible|eligibility)\s+to\s+(obtain|get|apply)|or (have )?submitted application|within \d+ (days|months) of hire|or proof of enrollment", re.I)


def _codes(text: str) -> set[str]:
    out = set()
    for part in re.split(r"\s*(?:,|\bor\b|\band\b|/)\s*", text):
        part = part.strip()
        if not part:
            continue
        if part.isupper() and part in US_STATES:
            out.add(part)
        elif part.lower() in _NAME_TO_CODE:
            out.add(_NAME_TO_CODE[part.lower()])
    return out


def license_states(text: str, resume: bool = False) -> set[str]:
    states: set[str] = set()
    for m in _LICENSED_IN.finditer(text):
        states |= _codes(m.group(1))
    for m in _STATE_LICENSE.finditer(text):
        states |= _codes(m.group(1))
    if resume:
        for m in _RESUME_LICENSE.finditer(text):
            states |= _codes(m.group(1))
    return states


# Degree subjects grouped into families; "or related field" accepts the same group.
SUBJECTS = {
    "computing": r"computer science|computer engineering|software engineering|information technology|information systems|informatics|\bcs\b|data science|computing",
    "engineering": r"mechanical|electrical|civil|chemical|industrial|aerospace|biomedical|manufacturing|materials|systems engineering|engineering",
    "math": r"mathematics|\bmath\b|statistics|physics|applied math|quantitative|operations research|actuarial",
    "business": r"business|finance|accounting|economics|commerce|management|\bmba\b|marketing|supply chain",
    "health": r"nursing|\bbsn\b|\bmsn\b|public health|pharmacy|health|medicine|kinesiology|nutrition",
    "life_science": r"biology|biochemistry|chemistry|life science|biotechnology|neuroscience|microbiology|molecular",
    "social": r"psychology|sociology|social work|counseling|political science|anthropology",
    "education": r"education|teaching",
    "law": r"\blaw\b|juris|legal studies",
    "arts": r"design|fine arts|art\b|graphic|communications|journalism|english|media",
}
GROUP = {"computing": "stem", "engineering": "stem", "math": "stem", "life_science": "stem", "health": "health",
         "business": "business", "social": "social", "education": "social", "law": "law", "arts": "arts"}
_DEGREE_IN = re.compile(r"(?:degree|bachelor'?s?|master'?s?|b\.?s\.?c?|m\.?s\.?c?|b\.?a\.?|m\.?a\.?|ph\.?d|bsn|msn|mba)\b[^.\n;]{0,60}?\b(?:in|of)\b\s+([^.\n;]{3,120})", re.I)


_DEGREE_NAMES = {"bsn": "health", "msn": "health", "dnp": "health", "pharmd": "health", "mba": "business", "bba": "business",
                 "jd": "law", "llm": "law", "b.comm": "business", "bcom": "business", "m.ed": "education", "msw": "social"}
_DEGREE_WORD = re.compile(r"\b(degree|bachelor|master|b\.?\s?s\.?c?|m\.?\s?s\.?c?|b\.?\s?a\.?|m\.?\s?a\.?|ph\.?\s?d|b\.?\s?tech|m\.?\s?tech|b\.?\s?e\.?|m\.?\s?eng|mba|bsn|msn|associate)\b", re.I)


def subjects(text: str, resume: bool = False) -> set[str]:
    """Degree subject families. Job posts: "degree in X"; resumes: any education line naming a degree."""
    found = set()
    for m in _DEGREE_IN.finditer(text):
        tail = m.group(1).lower()
        found |= {name for name, pattern in SUBJECTS.items() if re.search(pattern, tail)}
    for word, family in _DEGREE_NAMES.items():
        if re.search(rf"\b{re.escape(word)}\b", text, re.I):
            found.add(family)
    if resume:
        for line in text.splitlines():
            if _DEGREE_WORD.search(line):
                low = line.lower()
                found |= {name for name, pattern in SUBJECTS.items() if re.search(pattern, low)}
    return found
