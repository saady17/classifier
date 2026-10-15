"""Finds skills from the dictionary in any text (resume or job post)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache

from classifier.skilltext import TOKEN_RE, fold, raw_tokens

MAX_NGRAM = 6
# Skills whose plain name is an everyday word in job posts ("ensure compliance", "networking events");
# only their specific aliases count.
NAME_NOT_AN_ALIAS = {"Compliance", "Networking", "Communications", "Communication", "Collaboration", "Insurance", "Lean"}
_SENTENCE_END = re.compile(r"(^|[.!?:;•\n]\s*|\n\s*-\s*)$")


@dataclass(frozen=True)
class Skill:
    name: str
    category: str
    source: str
    occupations: tuple[str, ...]


class SkillMatcher:
    def __init__(self, path: str = "data/taxonomy/skills.json"):
        self.skills: dict[str, Skill] = {}
        self.folded: dict[tuple[str, ...], str] = {}  # case-insensitive aliases
        self.exact: dict[tuple[str, ...], str] = {}  # case-sensitive aliases (and their ALL-CAPS form)
        # O*NET product names are proper nouns: their capitalized words must be capitalized in the text
        # too, so "REST Assured" matches but "rest assured" does not.
        self.proper: dict[tuple[str, ...], tuple[str, tuple[bool, ...]]] = {}
        for item in json.load(open(path, encoding="utf-8")):
            name = item["name"]
            self.skills[name] = Skill(name, item["category"], item["source"], tuple(item.get("occupations", [])))
            aliases = list(item["aliases"])
            # Very short names ("C") are never matched loosely; their aliases spell them out.
            if name not in NAME_NOT_AN_ALIAS and len(name) > 2 and not any(a.lstrip("=") == name for a in aliases):
                # The canonical name is an alias too, unless it is a case-sensitive word like "Go".
                aliases.append(("=" + name) if any(a.startswith("=") and a[1:].lower() == name.lower() for a in aliases) else name)
            for alias in aliases:
                if alias.startswith("="):
                    toks = tuple(raw_tokens(alias[1:]))
                    if toks:
                        self.exact.setdefault(toks, name)
                        self.exact.setdefault(tuple(t.upper() for t in toks), name)
                elif item["source"] == "onet":
                    raw = raw_tokens(alias)
                    if raw:
                        caps = tuple(t[:1].isupper() for t in raw)
                        self.proper.setdefault(tuple(fold(t) for t in raw), (name, caps))
                else:
                    toks = tuple(fold(t) for t in raw_tokens(alias))
                    if toks:
                        self.folded.setdefault(toks, name)

    def find(self, text: str) -> dict[str, int]:
        """Canonical skill name -> number of mentions. Longest match wins at each position."""
        spans = [(m.group(), m.start()) for m in TOKEN_RE.finditer(text)]
        toks = [t for t, _ in spans]
        low = [fold(t) for t in toks]
        found: dict[str, int] = {}
        i = 0
        while i < len(toks):
            hit, size = None, 0
            for n in range(min(MAX_NGRAM, len(toks) - i), 0, -1):
                name = self.folded.get(tuple(low[i : i + n]))
                if not name and (exact := self.exact.get(tuple(toks[i : i + n]))):
                    # "Go above and beyond": a capitalized word starting a sentence is not the skill.
                    word = toks[i]
                    at_start = n == 1 and word[:1].isupper() and not word.isupper() and _SENTENCE_END.search(text[max(0, spans[i][1] - 4) : spans[i][1]])
                    followed_by_word = i + 1 < len(toks) and toks[i + 1].isalpha() and toks[i + 1].islower() and text[spans[i][1] + len(word) : spans[i + 1][1]].strip() == ""
                    if not (at_start and followed_by_word):
                        name = exact
                if not name and (proper := self.proper.get(tuple(low[i : i + n]))):
                    if all(t[:1].isupper() for t, cap in zip(toks[i : i + n], proper[1]) if cap):
                        name = proper[0]
                if name:
                    hit, size = name, n
                    break
            if hit:
                found[hit] = found.get(hit, 0) + 1
                i += size
            else:
                i += 1
        return found


@lru_cache(maxsize=1)
def matcher() -> SkillMatcher:
    return SkillMatcher()
