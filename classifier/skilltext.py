"""Tokenizer shared by the skills dictionary and the matcher. Keeps C++, C#, .NET, CI/CD, A/B, FP&A,
ICD-10 and similar names whole."""

from __future__ import annotations

import re
import unicodedata

_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9+#.&/'-]*[A-Za-z0-9+#]|[A-Za-z0-9][+#]*|\.[A-Za-z]+")


def raw_tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text)
    return _TOKEN.findall(text)


def fold(token: str) -> str:
    token = unicodedata.normalize("NFKD", token)
    return "".join(ch for ch in token if not unicodedata.combining(ch)).lower().replace("’", "'")


def key(phrase: str) -> tuple[str, ...]:
    return tuple(fold(t) for t in raw_tokens(phrase))
