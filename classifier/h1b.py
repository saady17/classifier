"""H-1B sponsorship from US government data (Department of Labor LCA disclosure files).

Every employer that sponsors an H-1B worker must first get a Labor Condition Application (LCA)
certified; DOL publishes all of them with employer, job title, SOC occupation code and worksite.
This module loads certified H-1B LCAs into the job store and answers "has this employer sponsored
H-1B workers recently, and for which occupations?". Used as a HARD filter for people who need
sponsorship.

    python -m classifier.h1b data/h1b/LCA_Disclosure_Data_FY2025_Q4.xlsx data/h1b/LCA_Disclosure_Data_FY2026_Q3.xlsx

The files come from https://www.dol.gov/agencies/eta/foreign-labor/performance (the site blocks
server downloads, so they are downloaded in a browser and uploaded)."""

from __future__ import annotations

import csv
import json
import re
import sqlite3
import sys
from collections import defaultdict
from datetime import date, datetime
from functools import lru_cache

from rapidfuzz import fuzz, process

from classifier import store
from classifier.text import normalize

SCHEMA = """
CREATE TABLE IF NOT EXISTS h1b_employers (
  name_key TEXT PRIMARY KEY,   -- normalized legal name
  display TEXT NOT NULL,
  filings INTEGER NOT NULL,    -- certified H-1B LCAs (positions) in the loaded files
  occupations TEXT NOT NULL,   -- JSON {soc_code: positions}
  titles TEXT NOT NULL,        -- JSON {job title: positions}, top 30
  last_decision TEXT
);
"""
_SUFFIX = re.compile(
    r"\b(incorporated|inc|llc|l\.l\.c|ltd|limited|corp|corporation|co|company|lp|llp|plc|pllc|pc|p\.c|na|n\.a|"
    r"group|holdings|the|usa|us|america|international|technologies|technology|tech|labs|lab|software|systems|services|solutions)\b"
)


def name_key(name: str) -> str:
    """'Stripe, Inc.' -> 'stripe'; 'Included Health, Inc.' -> 'includedhealth'."""
    t = normalize(name).replace("&", " and ")
    t = re.sub(r"\bd/?b/?a\b.*$", "", t)  # "X LLC dba Y" keeps the legal name
    t = _SUFFIX.sub(" ", re.sub(r"[^a-z0-9& ]", " ", t))
    return re.sub(r"[^a-z0-9]", "", t)


def _rows(path: str):
    if path.lower().endswith(".csv"):
        with open(path, newline="", encoding="utf-8-sig") as f:
            yield from csv.DictReader(f)
        return
    from openpyxl import load_workbook

    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb.worksheets[0]
    it = ws.iter_rows(values_only=True)
    header = [str(h or "").strip().upper() for h in next(it)]
    for values in it:
        yield dict(zip(header, values))


def load(paths: list[str]) -> None:
    agg: dict[str, dict] = {}
    seen = kept = 0
    for path in paths:
        for row in _rows(path):
            seen += 1
            r = {str(k).upper(): v for k, v in row.items()}
            status = str(r.get("CASE_STATUS") or "").lower()
            visa = str(r.get("VISA_CLASS") or "").upper()
            if not status.startswith("certified") or "withdrawn" in status or visa != "H-1B":
                continue
            name = str(r.get("EMPLOYER_NAME") or "").strip()
            key = name_key(name)
            if not key:
                continue
            kept += 1
            positions = int(float(r.get("TOTAL_WORKER_POSITIONS") or 1) or 1)
            soc = str(r.get("SOC_CODE") or "").strip()[:7]
            title = str(r.get("JOB_TITLE") or "").strip().title()[:80]
            decided = r.get("DECISION_DATE")
            decided = decided.date().isoformat() if isinstance(decided, datetime) else str(decided or "")[:10]
            e = agg.setdefault(key, {"display": name, "filings": 0, "occ": defaultdict(int), "titles": defaultdict(int), "last": ""})
            e["filings"] += positions
            if soc:
                e["occ"][soc] += positions
            if title:
                e["titles"][title] += positions
            e["last"] = max(e["last"], decided)
        print(f"  {path}: {seen} rows read so far, {kept} certified H-1B", flush=True)
    db = store.connect()
    db.executescript(SCHEMA)
    db.execute("DELETE FROM h1b_employers")
    db.executemany(
        "INSERT INTO h1b_employers VALUES (?,?,?,?,?,?)",
        [(k, e["display"], e["filings"], json.dumps(dict(e["occ"])),
          json.dumps(dict(sorted(e["titles"].items(), key=lambda kv: -kv[1])[:30])), e["last"]) for k, e in agg.items()],
    )
    db.commit()
    print(f"h1b: {len(agg)} sponsoring employers from {kept} certified H-1B LCAs")


class Sponsors:
    def __init__(self, db: sqlite3.Connection):
        db.executescript(SCHEMA)
        self.rows = {r["name_key"]: r for r in db.execute("SELECT * FROM h1b_employers")}
        self.keys = list(self.rows)
        try:
            import yaml

            self.aliases = {name_key(k): name_key(v) for k, v in (yaml.safe_load(open("data/h1b_aliases.yaml")) or {}).items()}
        except OSError:
            self.aliases = {}

    @property
    def loaded(self) -> bool:
        return bool(self.rows)

    @lru_cache(maxsize=50_000)
    def lookup(self, company: str) -> sqlite3.Row | None:
        key = name_key(company)
        key = self.aliases.get(key, key)
        if not key:
            return None
        if key in self.rows:
            return self.rows[key]
        if len(key) >= 5:
            # "includedhealth" vs "includedhealthinc..." or "stripepayments": the longest sponsor key
            # that starts with the company key, or a near-identical spelling.
            hit = process.extractOne(key, self.keys, scorer=fuzz.ratio, score_cutoff=93)
            if hit:
                return self.rows[hit[0]]
            prefixed = [k for k in self.keys if k.startswith(key) and len(k) - len(key) <= 6]
            if prefixed:
                return max((self.rows[k] for k in prefixed), key=lambda r: r["filings"])
        return None

    def check(self, company: str, occupation: str | None) -> tuple[bool, str]:
        """(sponsors, note). sponsors is False when the employer has no certified H-1B LCA in the data."""
        row = self.lookup(company)
        if not row:
            return False, "no H-1B sponsorship record (US Department of Labor data)"
        occ = json.loads(row["occupations"])
        code = (occupation or "")[:7]  # "15-1252.00" -> "15-1252", the SOC detail level DOL uses
        same = sum(v for k, v in occ.items() if code and k[:7] == code)
        family = sum(v for k, v in occ.items() if code and k[:5] == code[:5])
        if same:
            return True, f"sponsored {same} H-1B {'position' if same == 1 else 'positions'} in this occupation (DOL data)"
        if family:
            return True, f"sponsored {family} H-1B positions in related occupations (DOL data)"
        return True, f"sponsored {row['filings']} H-1B positions (DOL data), none in this occupation"


@lru_cache(maxsize=1)
def sponsors() -> Sponsors:
    return Sponsors(store.connect())


def staleness() -> str | None:
    """Most recent decision date loaded, to warn when the data is old."""
    try:
        return store.connect().execute("SELECT max(last_decision) FROM h1b_employers").fetchone()[0]
    except sqlite3.Error:
        return None


if __name__ == "__main__":
    load(sys.argv[1:])
    print("latest decision in data:", staleness(), "| today:", date.today())
