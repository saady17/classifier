"""Job titles -> standard occupation (O*NET-SOC) and seniority level."""

from __future__ import annotations

import json
import re
from functools import lru_cache

from rapidfuzz import fuzz, process

from classifier.skilltext import fold

# 0 intern, 1 entry, 2 mid, 3 senior, 4 lead/staff/principal, 5 manager-of-managers / director, 6 VP+, 7 C-level
LEVELS = [
    (7, r"\b(chief [a-z]+ officer|ceo|cto|cfo|coo|cmo|cio|ciso|founder|co-founder)\b"),
    (6, r"\b(vp|vice president|svp|evp|head of)\b"),
    (5, r"\b(director|senior manager|sr\.? manager)\b"),
    (4, r"\b(principal|lead|architect|distinguished|fellow)\b|\bstaff (\w+ )?(engineer|developer|scientist|designer|researcher|analyst)\b"),
    (3, r"\b(senior|sr\.?|iii|level 3|l3|manager)\b"),
    (0, r"\b(intern|internship|co-?op|apprentice|trainee)\b"),
    (1, r"\b(junior|jr\.?|entry[- ]level|graduate|new grad|associate|assistant|i|level 1|l1)\b"),
    (2, r"\b(ii|level 2|l2|mid[- ]level)\b"),
]
_STRIP = re.compile(
    r"\b(senior|sr|jr|junior|lead|principal|staff|intern|internship|entry level|entry-level|mid level|"
    r"mid-level|new grad|graduate|i|ii|iii|iv|v|1|2|3|4|remote|hybrid|contract|temporary|temp|part time|"
    r"part-time|full time|full-time|per diem|prn)\b\.?"
)
GENERIC_LEVEL = 2


def level(title: str) -> int:
    t = fold(title)
    for value, pattern in LEVELS:
        if re.search(pattern, t):
            return value
    return GENERIC_LEVEL


def base_title(title: str) -> str:
    """'Sr. Software Engineer II (Backend) - Remote, US' -> 'software engineer'."""
    t = fold(title)
    t = re.sub(r"\(.*?\)|\[.*?\]", " ", t)
    t = re.split(r"\s[-–|]\s|,| @ | at ", t)[0]  # team, location or company after a separator
    t = _STRIP.sub(" ", t)
    t = re.sub(r"[^a-z0-9+#&/ ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


_EXEC = re.compile(r"^(head|vp|vice president|svp|evp|director|chief)( of)? (.+)$")


def _curated_rules(path: str = "data/taxonomy/curated_titles.txt") -> list[tuple[re.Pattern, str]]:
    rules = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#"):
            pattern, code = (p.strip() for p in line.rsplit("|", 1))
            rules.append((re.compile(pattern), code))
    return rules


class TitleIndex:
    def __init__(self):
        self.titles: dict[str, str] = json.load(open("data/taxonomy/titles.json", encoding="utf-8"))
        self.occupations: dict[str, dict] = json.load(open("data/taxonomy/occupations.json", encoding="utf-8"))
        self.keys = list(self.titles)
        self.rules = _curated_rules()

    def _lookup(self, b: str) -> tuple[str | None, float]:
        for pattern, code in self.rules:
            if pattern.search(b):
                return code, 1.0
        if b in self.titles:
            return self.titles[b], 1.0
        hit = process.extractOne(b, self.keys, scorer=fuzz.token_sort_ratio, score_cutoff=82)
        if hit:
            return self.titles[hit[0]], hit[1] / 100
        hit = process.extractOne(b, self.keys, scorer=fuzz.token_set_ratio, score_cutoff=90)
        if hit:
            return self.titles[hit[0]], 0.7 * hit[1] / 100
        return None, 0.0

    @lru_cache(maxsize=200_000)
    def occupation(self, title: str) -> tuple[str | None, float]:
        """(SOC code, confidence 0-1)."""
        b = base_title(title)
        candidates = [b] if b else []
        # "Head of Marketing", "VP, Finance" -> "marketing manager", "finance manager"
        folded = re.sub(r"[^a-z0-9&/ ]", " ", fold(title))
        folded = re.sub(r"\s+", " ", folded).strip()
        if m := _EXEC.match(folded):
            candidates.insert(0, f"{m.group(3)} manager")
        # "SDR - Sales Development Representative": also try the part after a dash.
        candidates += [base_title(part) for part in re.split(r"\s[-–|]\s", fold(title))[1:] if base_title(part)]
        for candidate in candidates:
            code, conf = self._lookup(candidate)
            if code:
                return code, conf
        return None, 0.0

    def relatedness(self, a: str | None, b: str | None) -> float:
        """How close two occupations are, 0-1."""
        if not a or not b:
            return 0.0
        if a == b:
            return 1.0
        if a[:7] == b[:7]:  # same detailed occupation, different O*NET specialty
            return 0.85
        tier = self.occupations.get(a, {}).get("related", {}).get(b) or self.occupations.get(b, {}).get("related", {}).get(a)
        if tier:
            return {"Primary-Short": 0.75, "Primary-Long": 0.6, "Supplemental": 0.45}.get(tier, 0.45)
        if a[:5] == b[:5]:  # same minor group, e.g. 15-12 computer occupations
            return 0.4
        if a[:2] == b[:2]:  # same major group
            return 0.2
        return 0.0

    def name(self, code: str | None) -> str | None:
        return self.occupations.get(code or "", {}).get("title")


@lru_cache(maxsize=1)
def index() -> TitleIndex:
    return TitleIndex()
