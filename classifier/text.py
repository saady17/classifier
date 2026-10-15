"""Plain-text helpers shared by the job and resume readers."""

from __future__ import annotations

import html
import re
import unicodedata

from selectolax.parser import HTMLParser

_BLOCK_TAGS = ("p", "div", "li", "br", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "ul", "ol", "section")


def html_to_text(markup: str | None) -> str:
    """Readable text from job-post HTML, one block per line, list items kept as "- " lines."""
    if not markup:
        return ""
    # Greenhouse returns HTML that is itself HTML-escaped.
    if "&lt;" in markup and "<" not in markup:
        markup = html.unescape(markup)
    tree = HTMLParser(markup)
    for node in tree.css("script, style, noscript"):
        node.decompose()
    for node in tree.css("li"):
        node.insert_before("\n- ")
    for tag in _BLOCK_TAGS:
        for node in tree.css(tag):
            node.insert_after("\n")
    text = tree.body.text(separator="") if tree.body else tree.text(separator="")
    return clean_text(text)


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace(" ", " ").replace("​", "")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    out: list[str] = []
    for line in lines:
        if line in ("", "-"):
            if out and out[-1] != "":
                out.append("")
            continue
        out.append(line)
    return "\n".join(out).strip()


def normalize(text: str) -> str:
    """Lowercase ASCII-ish text for matching; keeps + # . / so C++, C#, .NET and CI/CD survive."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    text = re.sub(r"[^a-z0-9+#./\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


_TOKEN = re.compile(r"[a-z0-9][a-z0-9+#.]*[a-z0-9+#]|[a-z0-9]")
STOPWORDS = frozenset(
    """a an the and or of to in on for with at by from as is are be been being this that these those
    it its we our you your they their will can may must should would could has have had do does
    not no yes all any each such than then there here who whom which what when where why how
    about into over under within across per via etc""".split()
)


def tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(normalize(text)) if t not in STOPWORDS]
