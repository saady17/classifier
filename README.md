# classifier

Ranks jobs from official public job feeds against a resume, **without AI models**: no LLMs, no neural
networks, no AI APIs. Matching uses text rules, a skills and job-title dictionary (O*NET, ESCO) and
classic statistics (BM25, and simple models that learn feature weights from human ratings).

Markets: US, UK and Canada. Planned to merge into app.jobzhub.com.

## Job sources (official feeds only, no scraping)

| Source | Feed | Key needed |
|---|---|---|
| Greenhouse | Job Board API | no |
| Lever | Postings API | no |
| Ashby | Public job posting API | no |
| SmartRecruiters | Posting API | no |
| Workable | Careers widget API | no |
| Remotive | Remote jobs API (link back to Remotive) | no |
| USAJOBS | Search API | `USAJOBS_API_KEY`, `USAJOBS_EMAIL` |
| Adzuna | Search API (snippets only) | `ADZUNA_APP_ID`, `ADZUNA_APP_KEY` |

Companies are listed in `data/sources.yaml`. To add some, put candidate IDs in `data/candidates.yaml`,
run `python -m classifier.discover`, and check new matches really are the intended company.

## Run

```sh
docker build -t classifier-dev .
docker run --rm -v $PWD:/app -w /app classifier-dev python -m classifier.ingest   # fetch + store
```

Jobs are stored in `data/jobs.db` (SQLite with an FTS5 index), refreshed per account: closed jobs are
removed and duplicates across sources are linked.

## Pipeline

```sh
python -m classifier.ingest            # fetch jobs from data/sources.yaml (daily)
python -m classifier.features          # read each new/changed job once; skill rarity table
python -m classifier.build_taxonomy    # only after editing data/taxonomy/curated_*.txt or an O*NET update
python -m classifier.cli resume.pdf    # rank from the command line (--sponsor, --remote, --top=20)
python -m pytest -q                    # regression tests
```

## How ranking works

1. **Resume** (`resume.py`): text from PDF/DOCX, sections, dated roles mapped to O*NET occupations,
   merged years of experience, seniority, highest degree, location, skills weighted by where they appear.
2. **Job posts** (`jobparse.py`, precomputed by `features.py`): required vs preferred skills, years,
   minimum degree, visa sponsorship, clearance; benefits/EEO/pay boilerplate ignored.
3. **Skills dictionary** (`data/taxonomy/`): 352 hand-curated skills with aliases across many fields plus
   about 7,000 O*NET products, with rules against everyday-word false matches.
4. **Ranking** (`ranker.py`): candidates from keyword search (SQLite FTS5 BM25) plus same/related
   occupations; ten factor scores combined into 0–100 with an explanation; hard filters for country,
   remote preference, no-sponsorship and clearance.

## API

`uvicorn classifier.api:app` — set `CLASSIFIER_API_KEY` and send it as `X-API-Key`.

- `POST /v1/rank` — multipart: `file` (PDF/DOCX/TXT), optional `needs_sponsorship`, `remote`
  (`any`/`remote_only`/`no_remote`), `countries` (`US,CA`), `limit`.
- `POST /v1/rank-text` — JSON: `{"resume_text": "...", "preferences": {...}, "limit": 30}`.
- `GET /health` — job count and last fetch.

Each match has `score` (0–100), `factors`, `matched_required`, `missing_required`, `matched_preferred`
and `notes` (e.g. "asks for 5+ years; you have about 3").

## Evaluation (is it actually good?)

1. Put resumes in `eval/resumes/` (git-ignored; personal data never leaves the server).
2. `python -m classifier.evaluation pools` — for each resume, 25 jobs: the ranker's top 12, keyword
   search's top 8, 5 random same-country jobs, shuffled so raters can't tell which system chose them.
   A fixed third of resumes is the **test** set and never used for tuning.
3. Raters open `/eval` (set `RATER_PASSWORD`; the user name is the rater's name) and score each job
   0–3 (keys 0–3 work).
4. `python -m classifier.tune` — learns factor weights from the **train** resumes (`data/weights.json`).
5. `python -m classifier.evaluation report` — NDCG@10 and P@10 for the ranker vs keyword search vs
   random, on train and test. The test numbers are the honest ones.

## Data

O*NET® 31.0 Database by the U.S. Department of Labor, Employment and Training Administration
(USDOL/ETA), used under the CC BY 4.0 license. O*NET® is a trademark of USDOL/ETA.
Real resumes used for evaluation stay on the server (`eval/resumes/`, git-ignored) and are never committed.
