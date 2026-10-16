"""Rank jobs for a resume file from the command line:  python -m classifier.cli path/to/resume.pdf [--sponsor] [--remote]"""

from __future__ import annotations

import sys

from classifier.ranker import Preferences, rank
from classifier.resume import extract_text, parse_resume
from classifier.titles import index


def main(argv: list[str]) -> None:
    path = argv[0]
    text = extract_text(open(path, "rb").read(), path)
    resume = parse_resume(text)
    prefs = Preferences(needs_sponsorship="--sponsor" in argv, remote="remote_only" if "--remote" in argv else "any",
                        city=resume.location_city, region=resume.location_region,
                        countries=(resume.location_country,) if resume.location_country else ("US", "GB", "CA"))
    print(f"Resume: {resume.current_title} ({index().name(resume.occupation)}), level {resume.level}, {resume.years:g} yrs, "
          f"degree {resume.degree}, {resume.location_city or '?'}, {resume.location_region or '?'}")
    for i, m in enumerate(rank(resume, prefs, limit=int(next((a.split('=')[1] for a in argv if a.startswith('--top=')), 15))), 1):
        print(f"{i:2}. {m.score:5.1f} {m.label:15} {m.title} — {m.company} [{m.location[:40]}] {m.remote}")
        print(f"      matched: {', '.join(m.matched_required[:7])}" + (f" | missing: {', '.join(m.missing_required[:5])}" if m.missing_required else ""))
        f = m.factors
        print("      " + " ".join(f"{k[:6]}={v:.2f}" for k, v in f.items() if v is not None) + (f" | {'; '.join(m.notes)}" if m.notes else ""))


if __name__ == "__main__":
    main(sys.argv[1:])
