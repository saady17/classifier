"""Finds skills from the dictionary in any text (resume or job post)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

from classifier.skilltext import fold, raw_tokens

MAX_NGRAM = 6


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
            if not any(a.lstrip("=") == name for a in aliases):
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
        toks = raw_tokens(text)
        low = [fold(t) for t in toks]
        found: dict[str, int] = {}
        i = 0
        while i < len(toks):
            hit, size = None, 0
            for n in range(min(MAX_NGRAM, len(toks) - i), 0, -1):
                name = self.folded.get(tuple(low[i : i + n])) or self.exact.get(tuple(toks[i : i + n]))
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
