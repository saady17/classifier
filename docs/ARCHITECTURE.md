# Architecture

```
official feeds ──ingest──▶ jobs.db (SQLite + FTS5) ──features──▶ job_features, skill_stats, tfidf.pkl
                                                                      │
resume (PDF/DOCX/TXT) ──resume.py──▶ ResumeProfile ──┐                │
preferences + filters ───────────────────────────────┼──▶ ranker.rank ◀┘──▶ matches (score, label, why)
                                                      │        │
                                     model.pkl (GBDT) ┘        └─ hard filters → factors → model → label
```

Nothing in this path calls an AI model. LLMs are used only offline, as judges that grade results
(see [EVALUATION.md](EVALUATION.md)).

## Modules (`classifier/`)

| Module | Role |
|---|---|
| `sources/*.py` | One fetcher per official feed → common `Job` model (`models.py`) |
| `ingest.py` | Fetch every account in `data/sources.yaml`, keep US/UK/CA, upsert, delete closed jobs, link duplicates |
| `discover.py` | Check candidate company IDs against each API (`data/candidates*.yaml`) |
| `store.py` | SQLite schema, FTS5 index (BM25), de-duplication |
| `locations.py` | Free-text locations → country / state-province / city |
| `skilltext.py`, `skills.py` | Tokenizer and skill matcher (aliases, exact-case words, proper nouns, sentence-start rules) |
| `build_taxonomy.py` | Builds `data/taxonomy/*.json` from curated lists + O*NET 31.0 + the job corpus |
| `titles.py` | Job title → O*NET-SOC occupation, seniority level, occupation relatedness |
| `jobparse.py` | Job post → required/preferred/duty skills, years, degree, visa, clearance, ITAR, languages, licenses, requirement lines |
| `credentials.py` | US state licenses and degree subject families (job and resume side) |
| `resume.py` | Resume → text, sections, dated roles, years, level, occupation(s), degree, subjects, licenses, languages, location, skills |
| `features.py` | Reads every new/changed job once (versioned by `READER_VERSION`), skill IDF, TF-IDF model |
| `filters.py` | Recruiter-style hard filters, employment type and salary detection |
| `h1b.py` | DOL LCA disclosure data → sponsoring employers by occupation (hard filter for sponsorship) |
| `ranker.py` | Candidates → filters → factor scores → learned model → label |
| `model.py`, `modelwrap.py` | Gradient-boosted trees trained on ratings (train resumes only) |
| `calibrate.py` | Chooses label thresholds so each label meets its precision target |
| `api.py`, `cli.py` | HTTP API and command line |
| `evaluation.py`, `judge.py`, `topk.py`, `quickeval.py`, `tune.py` | Evaluation tooling (offline) |
| `drive_fetch.py` | Downloads evaluation resumes from a Drive folder shared with the service account |

## How a match is scored

1. **Candidates.** Keyword search (FTS5 BM25) over the resume's top skills and titles, plus every job in
   the same or a closely related O*NET occupation (~1,500 jobs).
2. **Hard filters** (job removed, never just ranked low): country, work mode, employment type,
   locations, excluded/allowed employers, title keywords, seniority, salary, non-English posts,
   security clearance, and for people who need sponsorship: "no sponsorship" posts, ITAR/US-persons-only
   jobs and employers with no H-1B record in DOL data.
3. **Factors** (each 0–1): required and preferred skill coverage (rare skills weigh more), skill overlap,
   title/occupation fit, seniority and years, keyword relevance, whole-text TF-IDF similarity, degree,
   location, visa.
4. **Rule score**: weighted sum of factors (weights learned by `tune.py`) × multipliers for clear gaps
   (years/level, internships for non-students, missing licenses, wrong-state licenses, degree subject,
   missing required languages, unrelated field).
5. **Learned model**: `HistGradientBoostingRegressor` predicts the judges' 0–3 rating from ~47 numbers
   (the rule score, every factor, years and level gaps, required skills matched/missing, license,
   degree, language and requirement-line coverage...). Score = prediction × 100 / 3.
6. **Label** from calibrated thresholds in `data/labels.json` (Good match ≥ 82, Possible match ≥ 41,
   Stretch below; Excellent match appears only once a calibration measures ≥ 92% precision).
7. **Explanation**: matched/missing required skills, matched preferred skills, and notes such as
   "asks for 5+ years; you have about 3", "requires a NY license", "US citizens or permanent residents
   only (ITAR)", "sponsored 120 H-1B positions in this occupation (DOL data)".

## Data files

| File | Tracked | What |
|---|---|---|
| `data/sources.yaml` | yes | Employer accounts per feed (checked by hand) |
| `data/taxonomy/curated_skills.txt`, `curated_titles.txt` | yes | Hand-written skills/aliases and modern title rules |
| `data/taxonomy/*.json` | yes | Built dictionary (skills, titles, occupations) |
| `data/weights.json`, `data/model.pkl`, `data/labels.json`, `data/tfidf.pkl` | yes | Learned weights, model, label thresholds, TF-IDF vocabulary (no personal data) |
| `data/jobs.db` | no | Jobs, features, H-1B employers (rebuilt by ingest/features) |
| `eval/resumes*/`, `eval/private/` | no | Real resumes, pools, ratings: personal data, server only |
