"""Ranks stored jobs against a resume. No AI: dictionary matching, rules and classic statistics.

Each job gets separate factor scores (0-1); a weighted sum gives the 0-100 match. The weights start
hand-set (WEIGHTS) and are later learned from human ratings (classifier.tune)."""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from dataclasses import dataclass, field
from functools import lru_cache

from rapidfuzz import fuzz

from classifier import filters, h1b, store
from classifier.credentials import GROUP
from classifier.resume import ResumeProfile
from classifier.skills import matcher
from classifier.titles import base_title
from classifier.titles import index as title_index

CANDIDATES_FTS = 800
CANDIDATES_OCC = 1500
SOFT_WEIGHT = 0.25  # communication, leadership... matter, but say little about fit
CATEGORY_WEIGHT = {"soft": SOFT_WEIGHT, "spoken_language": 0.6}

HAND_WEIGHTS = {
    "required_skills": 0.28,
    "preferred_skills": 0.05,
    "skill_overlap": 0.09,
    "title": 0.24,
    "seniority": 0.10,
    "keywords": 0.03,
    "content": 0.08,
    "degree": 0.04,
    "location": 0.05,
    "visa": 0.02,
}


# The judges were told to ignore location and were not asked about visa, so ratings say nothing
# about these; they keep their hand-set weights. Posting age is not scored: Jobzhub applies as soon
# as a job is fetched, so every job is fresh.
NOT_LEARNED = ("location", "visa")


def _load_weights() -> dict[str, float]:
    """Weights learned from ratings (classifier.tune) when available, otherwise the hand-set ones."""
    try:
        learned = json.load(open(os.environ.get("CLASSIFIER_WEIGHTS", "data/weights.json")))
    except (OSError, ValueError):
        return dict(HAND_WEIGHTS)
    return {k: (v if k in NOT_LEARNED else float(learned.get(k, v))) for k, v in HAND_WEIGHTS.items()}


WEIGHTS = _load_weights()


@dataclass
class Preferences:
    countries: tuple[str, ...] = ("US", "GB", "CA")
    remote: str = "any"  # any | remote_only | no_remote
    needs_sponsorship: bool = False
    has_clearance: bool = False
    city: str | None = None
    region: str | None = None
    willing_to_relocate: bool = True
    # citizen | permanent_resident | needs_sponsorship | unknown. Never guessed from the resume.
    work_authorization: str = "unknown"
    # Recruiter-style hard filters (classifier.filters). Empty means "any".
    employment_types: tuple[str, ...] = ()  # full_time, part_time, contract, temporary, internship
    work_modes: tuple[str, ...] = ()  # remote, hybrid, onsite
    locations: tuple[str, ...] = ()  # "Austin, TX", "New York", "Ontario", "UK"
    excluded_employers: tuple[str, ...] = ()
    only_employers: tuple[str, ...] = ()
    title_include: tuple[str, ...] = ()
    title_exclude: tuple[str, ...] = ()
    seniority: tuple[str, ...] = ()  # intern, entry, mid, senior, lead, director, executive, or "auto"
    min_salary: float | None = None  # annual, in the job's currency
    include_unlisted_salary: bool = True
    min_label: str | None = None  # "Excellent match" | "Good match" | "Possible match"


@dataclass
class Match:
    job_id: str
    title: str
    company: str
    url: str
    location: str
    remote: str
    posted_at: str | None
    score: float
    factors: dict[str, float]
    matched_required: list[str] = field(default_factory=list)
    missing_required: list[str] = field(default_factory=list)
    matched_preferred: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    features: dict[str, float] = field(default_factory=dict, repr=False)  # inputs to the learned model
    label: str = ""  # Excellent match | Good match | Possible match | Stretch (see classifier.calibrate)
    employment_type: str = "full_time"
    salary: str | None = None  # e.g. "USD 120,000-150,000 per year", when the post states it


def _skill_weights(db: sqlite3.Connection) -> dict[str, float]:
    m = matcher()
    weights = {}
    for skill, idf in db.execute("SELECT skill, idf FROM skill_stats"):
        cat = m.skills[skill].category if skill in m.skills else ""
        weights[skill] = idf * CATEGORY_WEIGHT.get(cat, 1.0)
    return weights


def _query_terms(resume: ResumeProfile) -> str:
    """FTS5 query: the resume's strongest skills plus its recent titles."""
    terms = [s for s, _ in sorted(resume.skills.items(), key=lambda x: -x[1])[:15]]
    terms += [r.title for r in resume.roles[:3]]
    if resume.current_title:
        terms.append(resume.current_title)
    words = []
    for t in terms:
        clean = re.sub(r"[^A-Za-z0-9 ]", " ", t).strip()
        if clean:
            words.append('"' + clean + '"')
    return " OR ".join(dict.fromkeys(words))


def _has(resume: ResumeProfile, skill: str) -> float:
    w = resume.skills.get(skill, 0.0)
    return 1.0 if w >= 1.0 else (0.7 if w > 0 else 0.0)


PRIOR_COVERAGE, PRIOR_WEIGHT = 0.3, 2.5  # a job listing one or two skills is weak evidence either way


def _coverage(skills: dict[str, int], resume: ResumeProfile, weights: dict[str, float]) -> tuple[float, list[str], list[str]]:
    if not skills:
        return math.nan, [], []
    total, got = PRIOR_WEIGHT, PRIOR_WEIGHT * PRIOR_COVERAGE
    hit, miss = [], []
    for s in skills:
        w = weights.get(s, 1.0)
        total += w
        h = _has(resume, s)
        got += w * h
        (hit if h else miss).append(s)
    def by_weight(skill: str) -> float:
        return -weights.get(skill, 1.0)

    soft = {n for n, sk in matcher().skills.items() if sk.category == "soft"}
    return got / total, sorted(hit, key=by_weight), sorted((s for s in miss if s not in soft), key=by_weight)


def _overlap(resume: ResumeProfile, job_skills: set[str], weights: dict[str, float]) -> float:
    """Share of the resume's strongest skills that this job uses at all (relevance from the candidate's side)."""
    top = sorted(resume.skills.items(), key=lambda x: -x[1] * weights.get(x[0], 1.0))[:12]
    if not top:
        return math.nan
    total = sum(weights.get(s, 1.0) * min(w, 3) for s, w in top)
    got = sum(weights.get(s, 1.0) * min(w, 3) for s, w in top if s in job_skills)
    return got / total if total else math.nan


def _title_fit(resume: ResumeProfile, job_title: str, job_occ: str | None) -> float:
    ix = title_index()
    occ = 0.0
    if job_occ and resume.occupations:
        norm = sum(resume.occupations.values())
        occ = max(ix.relatedness(job_occ, code) * (0.6 + 0.4 * w / norm) for code, w in resume.occupations.items())
    words = 0.0
    jb = base_title(job_title)
    titles = [r.title for r in resume.roles[:4]] or ([resume.current_title] if resume.current_title else [])
    for i, title in enumerate(titles):
        words = max(words, _title_words(jb, base_title(title)) * (1.0 - 0.1 * i))
    return max(occ, 0.9 * words)


_TITLE_FILLER = {"and", "of", "the", "for", "&", "to", "in", "with", "a"}


def _title_words(a: str, b: str) -> float:
    """Similarity of two titles that requires shared words ("account executive" vs "registered nurse" = 0)."""
    wa, wb = set(a.split()) - _TITLE_FILLER, set(b.split()) - _TITLE_FILLER
    if not wa or not wb or not wa & wb:
        return 0.0
    return fuzz.token_sort_ratio(a, b) / 100 * 0.5 + 0.5 * len(wa & wb) / len(wa | wb)


def _seniority(resume: ResumeProfile, job_level: int, min_years: float | None) -> tuple[float, list[str]]:
    notes = []
    diff = job_level - resume.level
    level_score = {0: 1.0, 1: 0.75, 2: 0.35}.get(diff, 0.1) if diff >= 0 else {-1: 0.8, -2: 0.5}.get(diff, 0.3)
    years_score = 1.0
    if min_years:
        gap = min_years - resume.years
        if gap > 0:
            years_score = math.exp(-gap / 2.5)
            notes.append(f"asks for {min_years:g}+ years; you have about {resume.years:g}")
        elif resume.years > min_years + 10 and job_level <= 2:
            years_score = 0.8
    return 0.5 * level_score + 0.5 * years_score, notes


def credential_fit(resume: ResumeProfile, job: dict, notes: list[str]) -> float:
    """Multiplier for a state license the job needs and a degree subject it asks for."""
    factor = 1.0
    states = set(job.get("license_states") or [])
    if states and (resume.license_states or any(s in resume.skills for s in ("Registered Nurse", "Licensed Practical Nurse", "Nurse Practitioner", "Teaching license", "Social work"))):
        if resume.license_states & states:
            pass
        elif job.get("license_compact_ok") and resume.license_compact:
            pass
        elif job.get("license_obtainable"):
            factor *= 0.85
            notes.append(f"needs a {'/'.join(sorted(states))} license (can be obtained after hiring)")
        else:
            factor *= 0.55
            notes.append(f"requires a {'/'.join(sorted(states))} license")
    wanted = set(job.get("degree_subjects") or [])
    if wanted and job.get("degree_required") and resume.degree_subjects and not (wanted & resume.degree_subjects):
        same_group = {GROUP[w] for w in wanted} & {GROUP[h] for h in resume.degree_subjects}
        if job.get("related_field_ok") and same_group:
            factor *= 0.92
        else:
            factor *= 0.75
            notes.append("asks for a degree in " + ", ".join(sorted(wanted)).replace("_", " "))
    return factor


def _salary_text(job: dict) -> str | None:
    lo, hi = job.get("salary_min"), job.get("salary_max")
    if not lo and not hi:
        return None
    rng = f"{lo:,.0f}-{hi:,.0f}" if lo and hi and lo != hi else f"{(lo or hi):,.0f}"
    return f"{job.get('salary_currency') or ''} {rng} per {job.get('salary_period') or 'year'}".strip()


ENTRY_JOB_LEVEL = 1
INTERN_TITLE = re.compile(r"\b(intern|internship|co-?op|summer analyst|summer associate)\b", re.I)


def level_fit(resume: ResumeProfile, job_level: int, min_years: float | None) -> float:
    """Multiplier for how well the job's level and years fit the person: below 1 for a clear gap, a
    small boost above 1 for an entry-level job that suits an early-career candidate."""
    factor = 1.0
    years = resume.years
    if min_years:
        gap = min_years - years
        if gap > 1.0:
            factor *= max(0.3, math.exp(-(gap - 1.0) / 2.5))
    up = job_level - resume.level
    if up >= 3:
        factor *= 0.45
    elif up == 2:
        factor *= 0.65
    early = years < 2.5 and resume.level <= 2
    if early and job_level <= ENTRY_JOB_LEVEL and (not min_years or min_years <= years + 1):
        factor *= 1.12  # an entry-level role is exactly what an early-career candidate should see first
    if years >= 8 and job_level <= ENTRY_JOB_LEVEL:
        factor *= 0.7  # very experienced people rarely want intern or entry-level jobs
    return factor


@lru_cache(maxsize=1)
def _label_thresholds() -> list[tuple[float, str]]:
    """Score thresholds measured by classifier.calibrate (data/labels.json); strictest first."""
    try:
        cal = json.load(open(os.environ.get("CLASSIFIER_LABELS", "data/labels.json")))
    except (OSError, ValueError):
        cal = {}
    names = (("excellent", "Excellent match"), ("good", "Good match"), ("possible", "Possible match"))
    return [(float(cal[k]["min_score"]), label) for k, label in names if cal.get(k)]


def match_label(score: float) -> str:
    for threshold, label in _label_thresholds():
        if score >= threshold:
            return label
    return "Stretch"


@lru_cache(maxsize=1)
def _model():
    """The learned ranking model (classifier.model), unless missing or CLASSIFIER_MODEL=off."""
    if os.environ.get("CLASSIFIER_MODEL", "on") == "off":
        return None
    import pickle

    try:
        with open(os.environ.get("CLASSIFIER_MODEL_PATH", "data/model.pkl"), "rb") as f:
            return pickle.load(f)
    except OSError:
        return None


@lru_cache(maxsize=1)
def _tfidf():
    import pickle

    try:
        with open(os.environ.get("CLASSIFIER_TFIDF", "data/tfidf.pkl"), "rb") as f:
            return pickle.load(f)
    except OSError:
        return None


def requirement_coverage(resume_text: str, jobs: dict[str, list[str]]) -> dict[str, dict[str, float]]:
    """Line-level matching (TF-IDF, no AI): for each requirement line of a job, its best match among
    the resume's lines; plus whether the requirements' most distinctive terms appear in the resume."""
    vec = _tfidf()
    if vec is None:
        return {}
    import numpy as np

    lines = [ln for ln in (x.strip(" -•*·●▪\t") for x in resume_text.splitlines()) if len(ln.split()) >= 3]
    if not lines:
        return {}
    R = vec.transform(lines + [resume_text])
    resume_lines, resume_all = R[:-1], R[-1]
    resume_terms = set(resume_all.indices)
    flat, owner = [], []
    for jid, reqs in jobs.items():
        for ln in reqs:
            flat.append(ln)
            owner.append(jid)
    out: dict[str, dict[str, float]] = {}
    if not flat:
        return out
    J = vec.transform(flat)
    best = (J @ resume_lines.T).max(axis=1).toarray().ravel()
    per: dict[str, list[float]] = {}
    terms: dict[str, list[int]] = {}
    for i, jid in enumerate(owner):
        per.setdefault(jid, []).append(float(best[i]))
        row = J.getrow(i)
        if row.nnz:
            top = row.indices[np.argsort(-row.data)[:3]]  # the line's most distinctive terms
            terms.setdefault(jid, []).extend(int(t) for t in top)
    for jid, sims in per.items():
        uniq = list(dict.fromkeys(terms.get(jid, [])))
        out[jid] = {
            "req_lines": float(len(sims)),
            "req_cov_mean": sum(sims) / len(sims),
            "req_cov_min": min(sims),
            "req_cov_share20": sum(s >= 0.20 for s in sims) / len(sims),
            "req_cov_share35": sum(s >= 0.35 for s in sims) / len(sims),
            "req_terms_found": (sum(t in resume_terms for t in uniq) / len(uniq)) if uniq else math.nan,
        }
    return out


def content_similarity(resume_text: str, docs: dict[str, str]) -> dict[str, float]:
    """Cosine similarity of whole-text TF-IDF vectors, scaled so the closest candidate is 1."""
    vec = _tfidf()
    if vec is None or not docs:
        return {}
    ids = list(docs)
    r = vec.transform([resume_text])
    sims = (vec.transform([docs[i] for i in ids]) @ r.T).toarray().ravel()
    top = float(sims.max()) or 1.0
    return {i: float(s) / top for i, s in zip(ids, sims)}


def _location(job_row, prefs: Preferences) -> tuple[float, bool]:
    """(score, allowed)."""
    locations = json.loads(job_row["locations"])
    countries = {loc["country"] for loc in locations if loc.get("country")}
    remote = job_row["remote"]
    if prefs.remote == "remote_only" and remote != "remote":
        return 0.0, False
    if prefs.remote == "no_remote" and remote == "remote":
        return 0.0, False
    if not countries & set(prefs.countries):
        return 0.0, False
    if remote == "remote":
        return 1.0, True
    if prefs.region and any(loc.get("region") == prefs.region for loc in locations):
        same_city = prefs.city and any((loc.get("city") or "").lower() == prefs.city.lower() for loc in locations)
        return (1.0 if same_city else 0.85), True
    home = prefs.countries[0] if len(prefs.countries) == 1 else None
    if home and home not in countries:
        return 0.25, True
    return (0.55 if prefs.willing_to_relocate else 0.15), True


def _combine(f: dict[str, float]) -> float:
    total = weight = 0.0
    for name, w in WEIGHTS.items():
        v = f.get(name)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue  # factor unknown for this job: its weight is spread over the rest
        total += w * v
        weight += w
    return 100 * total / weight if weight else 0.0


def bm25_scores(resume: ResumeProfile, db: sqlite3.Connection, limit: int = CANDIDATES_FTS, only: list[str] | None = None) -> dict[str, float]:
    """Plain keyword relevance (higher is better). Also the baseline the ranker must beat."""
    query = _query_terms(resume)
    if not query:
        return {}
    extra, args = "", [query]
    if only:
        extra = f" AND j.id IN ({','.join('?' * len(only))})"
        args += only
    rows = db.execute(
        "SELECT j.id, bm25(jobs_fts, 4.0, 1.0, 1.0) AS s FROM jobs_fts JOIN jobs j ON j.rowid = jobs_fts.rowid "
        f"WHERE jobs_fts MATCH ?{extra} ORDER BY s LIMIT ?",
        (*args, limit),
    )
    return {row["id"]: -row["s"] for row in rows}


def rank(
    resume: ResumeProfile,
    prefs: Preferences | None = None,
    limit: int = 50,
    db: sqlite3.Connection | None = None,
    only: list[str] | None = None,
    group: bool = True,
    use_model: bool = True,
) -> list[Match]:
    """Top matches. `only` scores exactly these job ids (evaluation pools) instead of searching."""
    prefs = prefs or Preferences()
    db = db or store.connect()
    weights = _skill_weights(db)

    # 1. Candidates: keyword search over the resume's skills and titles, plus same/related occupations.
    candidates: dict[str, float] = {}
    if only:
        candidates = {jid: 0.0 for jid in only}
        candidates.update(bm25_scores(resume, db, limit=len(only), only=only))
    query = None if only else _query_terms(resume)
    if query:
        for row in db.execute(
            "SELECT j.id, bm25(jobs_fts, 4.0, 1.0, 1.0) AS s FROM jobs_fts JOIN jobs j ON j.rowid = jobs_fts.rowid "
            "WHERE jobs_fts MATCH ? AND j.duplicate_of IS NULL ORDER BY s LIMIT ?",
            (query, CANDIDATES_FTS),
        ):
            candidates[row["id"]] = -row["s"]
    ix = title_index()
    related: set[str] = set()
    if not only:  # a fixed evaluation pool is scored as-is
        related = set(resume.occupations)
        for code in list(resume.occupations):
            related |= {c for c, tier in ix.occupations.get(code, {}).get("related", {}).items() if tier.startswith("Primary")}
    if related:
        marks = ",".join("?" * len(related))
        for (jid,) in db.execute(
            f"SELECT f.id FROM job_features f JOIN jobs j ON j.id = f.id WHERE j.duplicate_of IS NULL AND f.occupation IN ({marks}) LIMIT ?",
            (*related, CANDIDATES_OCC),
        ):
            candidates.setdefault(jid, 0.0)
    if not candidates:
        return []
    top_bm25 = max(candidates.values()) or 1.0

    # 2. Score every candidate.
    ids = list(candidates)
    results: list[Match] = []
    all_rows = []
    for chunk in range(0, len(ids), 900):
        part = ids[chunk : chunk + 900]
        marks = ",".join("?" * len(part))
        all_rows += db.execute(
            f"SELECT j.*, f.occupation, f.level AS job_level, f.data FROM jobs j JOIN job_features f ON f.id = j.id WHERE j.id IN ({marks})",
            part,
        ).fetchall()
    content = content_similarity(resume.text, {r["id"]: f"{r['title']}\n{r['description']}" for r in all_rows})
    coverage = requirement_coverage(resume.text, {r["id"]: json.loads(r["data"]).get("requirement_lines") or [] for r in all_rows})
    for rows in (all_rows,):
        for row in rows:
            loc_score, allowed = _location(row, prefs)
            if not allowed:
                continue
            job = json.loads(row["data"])
            notes = []
            if prefs.needs_sponsorship:
                if job["no_sponsorship"]:
                    continue
                # Hard check against US Department of Labor H-1B data: no record, no match.
                sp = h1b.sponsors()
                if sp.loaded:
                    ok, note = sp.check(row["company"], row["occupation"])
                    if not ok:
                        continue
                    notes.append(note)
            if job["clearance"] and not prefs.has_clearance:
                continue
            if not job.get("english", True):
                continue  # Jobzhub's markets are English-speaking; a German-language post is not a match
            if filters.check(row, job, prefs, resume) is not None:
                continue
            req, hit, miss = _coverage(job["required"], resume, weights)
            pref, phit, _ = _coverage(job["preferred"], resume, weights)
            all_skills = set(job["required"]) | set(job["preferred"]) | set(job["duties"])
            seniority, s_notes = _seniority(resume, row["job_level"], job["min_years"])
            notes += s_notes
            degree = 1.0
            if job["degree"] and resume.degree < job["degree"]:
                degree = 0.6 if (job["degree_equivalent_ok"] or not job["degree_required"]) else 0.2
                if job["degree_required"]:
                    notes.append("asks for a degree above yours" + (" (or equivalent experience)" if job["degree_equivalent_ok"] else ""))
            visa = math.nan
            if prefs.needs_sponsorship:
                visa = 1.0 if job["sponsors"] else 0.6
                if job["sponsors"]:
                    notes.append("mentions visa sponsorship")
            factors = {
                "required_skills": req,
                "preferred_skills": pref,
                "skill_overlap": _overlap(resume, all_skills, weights),
                "title": _title_fit(resume, row["title"], row["occupation"]),
                "seniority": seniority,
                "keywords": candidates[row["id"]] / top_bm25 if top_bm25 else 0.0,
                "content": content.get(row["id"], math.nan),
                "degree": degree,
                "location": loc_score,
                "visa": visa,
            }
            score = _combine(factors)
            # Level and experience are the commonest reason a match is wrong (judges' #1 reason), so a
            # clear gap scales the whole score rather than one factor.
            fit = level_fit(resume, row["job_level"], job["min_years"])
            if fit < 1.0:
                score *= fit
            elif fit > 1.0:
                score = min(100.0, score * fit)
            # Internships are for current students; graduates are usually ineligible (judges flagged this often).
            if INTERN_TITLE.search(row["title"]) and not resume.is_student:
                score *= 0.4
                notes.append("internship for current students")
            # ITAR / export-control and citizen-only jobs.
            if job.get("us_persons_only"):
                if prefs.needs_sponsorship or prefs.work_authorization == "needs_sponsorship":
                    continue
                if prefs.work_authorization == "unknown":
                    notes.append("US citizens or permanent residents only (ITAR/export control)")
            score *= credential_fit(resume, job, notes)
            missing_lang = [lang for lang in job.get("languages") or [] if lang not in resume.languages]
            if missing_lang:
                score *= 0.5
                notes.append("requires fluent " + ", ".join(lang.title() for lang in missing_lang))
            # A required license or certification the resume lacks (NP, CPA, CDL...) is close to a dealbreaker.
            licenses = {n for n, sk in matcher().skills.items() if sk.category in ("license", "certification")}
            missing_licenses = [s for s in miss if s in licenses]
            if missing_licenses:
                score *= 0.75 if len(missing_licenses) == 1 else 0.6
                notes.append("requires " + ", ".join(missing_licenses[:3]))
            # A job in an unrelated field is not a match however many skills happen to overlap.
            overlap = factors["skill_overlap"]
            if factors["title"] < 0.3 and (math.isnan(overlap) or overlap < 0.25):
                score *= 0.55
            elif factors["title"] < 0.3:
                score *= 0.8
            soft = {n for n, sk in matcher().skills.items() if sk.category == "soft"}
            req_hard = [k for k in job["required"] if k not in soft]
            pref_hard = [k for k in job["preferred"] if k not in soft]
            states = set(job.get("license_states") or [])
            wanted_subjects = set(job.get("degree_subjects") or [])
            feats = {
                **{f"f_{k}": (math.nan if v is None else v) for k, v in factors.items()},
                "rule_score": score,
                "years": resume.years,
                "min_years": job["min_years"] if job["min_years"] is not None else math.nan,
                "years_gap": (job["min_years"] - resume.years) if job["min_years"] is not None else math.nan,
                "pref_years_gap": (job["preferred_years"] - resume.years) if job.get("preferred_years") is not None else math.nan,
                "level_resume": resume.level,
                "level_job": row["job_level"],
                "level_gap": row["job_level"] - resume.level,
                "n_required": len(req_hard),
                "n_req_matched": sum(1 for k in req_hard if _has(resume, k)),
                "n_req_missing": sum(1 for k in req_hard if not _has(resume, k)),
                "n_preferred": len(pref_hard),
                "n_pref_matched": sum(1 for k in pref_hard if _has(resume, k)),
                "missing_licenses": len(missing_licenses),
                "license_mismatch": float(bool(states) and not (resume.license_states & states)),
                "degree_job": job["degree"],
                "degree_resume": resume.degree,
                "degree_gap": job["degree"] - resume.degree if job["degree"] else math.nan,
                "degree_required": float(job["degree_required"]),
                "degree_equiv_ok": float(job["degree_equivalent_ok"]),
                "subject_mismatch": float(bool(wanted_subjects) and bool(resume.degree_subjects) and not (wanted_subjects & resume.degree_subjects)),
                "languages_missing": len(missing_lang),
                "intern_job": float(bool(INTERN_TITLE.search(row["title"]))),
                "is_student": float(resume.is_student),
                "us_persons_only": float(bool(job.get("us_persons_only"))),
                "same_occupation": float(bool(row["occupation"]) and row["occupation"] == resume.occupation),
                "occupation_weight": resume.occupations.get(row["occupation"] or "", 0.0) / (sum(resume.occupations.values()) or 1.0),
                "title_words": max((_title_words(base_title(row["title"]), base_title(r.title)) for r in resume.roles[:4]), default=0.0),
                "structured": float(job["structured"]),
                "n_resume_skills": len(resume.skills),
                "remote": {"remote": 2.0, "hybrid": 1.0}.get(row["remote"], 0.0),
                **coverage.get(row["id"], {}),
            }
            model = _model() if use_model else None
            if model is not None:
                score = float(min(3.0, max(0.0, model.predict([[feats.get(k, math.nan) for k in model.feature_names]])[0])) * 100 / 3)
            locs = json.loads(row["locations"])
            results.append(
                Match(
                    job_id=row["id"], title=row["title"], company=row["company"], url=row["url"],
                    location="; ".join(loc["raw"] for loc in locs[:3]), remote=row["remote"], posted_at=row["posted_at"],
                    score=round(score, 1), factors={k: (None if isinstance(v, float) and math.isnan(v) else round(v, 3)) for k, v in factors.items()},
                    matched_required=hit[:10], missing_required=miss[:8], matched_preferred=phit[:6], notes=notes,
                    features=feats,
                    employment_type=job.get("employment_type") or "full_time",
                    salary=_salary_text(job),
                )
            )
    results.sort(key=lambda m: -m.score)
    for m in results:
        m.label = match_label(m.score)
    if prefs.min_label:
        results = [m for m in results if filters.LABEL_ORDER.index(m.label) >= filters.LABEL_ORDER.index(prefs.min_label)]
    if not group:
        return results[:limit]
    # The same role posted in several cities: keep the best-scoring posting, note the others.
    grouped: dict[tuple[str, str], Match] = {}
    for m in results:
        key = (m.company.lower(), base_title(m.title))
        if key in grouped:
            first = grouped[key]
            if len(first.notes) < 6 and m.location and m.location not in first.location:
                first.notes.append(f"also in {m.location}")
            continue
        grouped[key] = m
    return list(grouped.values())[:limit]
