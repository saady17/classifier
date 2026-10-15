"""Builds data/taxonomy/{skills,titles,occupations}.json from the curated list, O*NET 31.0 and the job
corpus (to spot product names that are also everyday words). Run after an O*NET update:

    python -m classifier.build_taxonomy
"""

from __future__ import annotations

import collections
import csv
import glob
import json
import re
import sqlite3

from classifier.skilltext import fold, key, raw_tokens

ONET = glob.glob("data/taxonomy/raw/onet/*/")[0]
OUT = "data/taxonomy"


def read(name):
    with open(ONET + name, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_curated():
    skills = []
    category = None
    for line in open(f"{OUT}/curated_skills.txt", encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, aliases, category = [p.strip() for p in line.split("|")]
        al = [a.strip() for a in aliases.split(";") if a.strip()]
        skills.append({"name": name, "category": category, "aliases": al, "source": "curated", "occupations": []})
    return skills


# O*NET entries that are company names more than skills.
TOO_VAGUE = {"Google", "Apple", "Microsoft", "Amazon", "Oracle", "IBM", "Meta"}

GENERIC_SOFTWARE = re.compile(r"\bsoftware\b|\bsystems?\b$|\bprograms?\b$|\bapplications?\b$|\bplatform\b$")


def onet_aliases(name: str, vendors: set[str]) -> list[str]:
    words = name.split()
    out = {name}
    # "Structured query language SQL" -> "SQL" and "Structured query language"
    if len(words) >= 2 and re.fullmatch(r"[A-Z][A-Z0-9+#./-]{1,7}", words[-1]):
        out.add(words[-1])
        out.add(" ".join(words[:-1]))
    rest = words[1:] if words[0] != "The" else words[1:]
    if words[0] == "The" and len(words) > 2:
        rest = words[2:]
        lead = words[1]
    else:
        lead = words[0]
    if lead in vendors and len(rest) >= 1:
        product = " ".join(rest)
        if len(product) >= 3 and not GENERIC_SOFTWARE.search(product.lower()):
            out.add(product)
    return sorted(out)


def common_word(token: str, lower: collections.Counter, upper: collections.Counter) -> bool:
    """Seen often in job posts in any case ("Navigator", "Workspace"): too risky as a bare alias."""
    return lower[token] + upper[token] >= 30


def build_skills(corpus_lower: collections.Counter, corpus_upper: collections.Counter):
    curated = load_curated()
    alias_owner = {}
    for s in curated:
        for a in s["aliases"] + [s["name"]]:
            alias_owner.setdefault(key(a.lstrip("=")), s["name"])

    rows = read("Software Skills.txt")
    by_example = collections.defaultdict(set)
    hot = set()
    for r in rows:
        by_example[r["Workplace Example"]].add(r["O*NET-SOC Code"])
        if r["Hot Technology"] == "Y" or r["In Demand"] == "Y":
            hot.add(r["Workplace Example"])
    firsts = collections.Counter(n.split()[0] for n in by_example if len(n.split()) > 1)
    firsts.update(n.split()[1] for n in by_example if n.startswith("The ") and len(n.split()) > 2)
    vendors = {w for w, c in firsts.items() if c >= 3 and w[:1].isupper() and w not in ("The", "Web", "Data", "Microsoft Office")}

    by_name = {s["name"]: s for s in curated}
    added = 0
    for example, socs in by_example.items():
        if example in TOO_VAGUE:
            continue
        words = example.split()
        # Category names ("Email software", "Web browser software") are not skills.
        if words[-1].lower() == "software" and all(w.islower() for w in words[1:]):
            continue
        aliases = onet_aliases(example, vendors)
        owner = next((alias_owner[key(a)] for a in aliases if key(a) in alias_owner), None)
        if owner:
            by_name[owner]["occupations"] = sorted(set(by_name[owner]["occupations"]) | socs)
            continue
        final = []
        for a in aliases:
            k = key(a)
            # Short auto acronyms collide across fields (ADL is a product and "activities of daily
            # living"); those need the full product name. Curated skills handle the important ones.
            if not k or (len(k) == 1 and len(k[0]) <= 4):
                continue
            if len(k) == 1 and a != example and common_word(k[0], corpus_lower, corpus_upper):
                continue  # bare product word left after removing the vendor, and it is common
            if len(k) == 1:
                low, up = corpus_lower[k[0]], corpus_upper[k[0]]
                # An everyday word in job posts ("access", "teams", "database"), even capitalized in
                # headings: only the full product name ("Microsoft Access") is kept.
                if low > 20 and low > 0.05 * (low + up):
                    continue
            elif all(corpus_lower[t] > 5 for t in k):
                # Every word is everyday English ("REST Assured" vs "rest assured"): exact case only.
                a = "=" + a
            final.append(a)
        if not final:
            continue
        skill = {"name": example, "category": "software", "aliases": final, "source": "onet",
                 "occupations": sorted(socs), "hot": example in hot}
        by_name[example] = skill
        for a in final:
            alias_owner.setdefault(key(a.lstrip("=")), example)
        added += 1
    skills = list(by_name.values())
    print(f"skills: {len(curated)} curated + {added} from O*NET = {len(skills)}; vendors detected: {len(vendors)}")
    return skills


def corpus_case_counts(limit=6000):
    lower, upper = collections.Counter(), collections.Counter()
    try:
        db = sqlite3.connect("data/jobs.db")
        for (text,) in db.execute("SELECT description FROM jobs ORDER BY random() LIMIT ?", (limit,)):
            for t in raw_tokens(text):
                if t.islower():
                    lower[t] += 1
                elif t[0].isupper():
                    upper[fold(t)] += 1
    except sqlite3.Error:
        pass
    return lower, upper


SENIORITY_WORDS = r"senior|sr|jr|junior|lead|principal|staff|intern|internship|entry level|entry-level|mid level|mid-level|i|ii|iii|iv|v|1|2|3|4"


def norm_title(title: str) -> str:
    t = fold(title)
    t = re.sub(r"\(.*?\)|\[.*?\]", " ", t)
    t = re.sub(rf"\b({SENIORITY_WORDS})\b\.?", " ", t)
    t = re.sub(r"[^a-z0-9+#&/ ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def build_titles():
    occupations = {r["O*NET-SOC Code"]: {"title": r["Title"]} for r in read("Occupation Data.txt")}
    for r in read("Job Zones.txt"):
        occupations.setdefault(r["O*NET-SOC Code"], {})["zone"] = int(r["Job Zone"])
    for r in read("Related Occupations.txt"):
        occ = occupations.setdefault(r["O*NET-SOC Code"], {})
        occ.setdefault("related", {})[r["Related O*NET-SOC Code"]] = r["Relatedness Tier"]
    votes = collections.defaultdict(collections.Counter)
    for code, occ in occupations.items():
        if "title" in occ:
            votes[norm_title(occ["title"].rstrip("s"))][code] += 5
            votes[norm_title(occ["title"])][code] += 5
    for r in read("Sample of Reported Titles.txt"):
        votes[norm_title(r["Reported Job Title"])][r["O*NET-SOC Code"]] += 3
    for r in read("Job Titles.txt"):
        votes[norm_title(r["Job Title"])][r["O*NET-SOC Code"]] += 1
        if r.get("Short Title") and r["Short Title"] != "n/a":
            votes[norm_title(r["Short Title"])][r["O*NET-SOC Code"]] += 1
    titles = {t: c.most_common(1)[0][0] for t, c in votes.items() if t}
    print(f"titles: {len(titles)} normalized titles -> {len(occupations)} occupations")
    return titles, occupations


def main():
    lower, upper = corpus_case_counts()
    skills = build_skills(lower, upper)
    titles, occupations = build_titles()
    json.dump(skills, open(f"{OUT}/skills.json", "w"), ensure_ascii=False, separators=(",", ":"))
    json.dump(titles, open(f"{OUT}/titles.json", "w"), ensure_ascii=False, separators=(",", ":"))
    json.dump(occupations, open(f"{OUT}/occupations.json", "w"), ensure_ascii=False, separators=(",", ":"))


if __name__ == "__main__":
    main()
