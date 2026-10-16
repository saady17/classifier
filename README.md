# classifier

Ranks jobs from official public job feeds against a resume, **without AI models at run time**: no
LLMs, no neural networks, no AI APIs. It reads the resume, applies the hard filters a recruiter or job
seeker would set (employment type, remote/hybrid/on-site, location, excluded employers, seniority,
salary, visa...), then scores every remaining job with rules, a skills and job-title dictionary
(O*NET, hand-curated) and classic statistics (BM25, TF-IDF and gradient-boosted decision trees
learned from ratings). Each match gets a 0–100 score, a calibrated label and a plain explanation.

Markets: US, UK and Canada. Built to be merged into app.jobzhub.com behind an HTTP API.

## Where it stands (held-out resumes, never used for tuning)

| | Result |
|---|---|
| "Good match" label is right (a good fit per a 3-judge majority) | **90.3%** of jobs scoring 82+ |
| "Possible match" label is at least a reasonable fit | 90.2% of jobs scoring 41+ |
| Top-5 jobs that are good fits (searching all ~29k jobs) | 62% (test set), 40% (fresh batch) |
| NDCG@10 vs keyword search (25-job pools) | 0.746 vs 0.455 (test), 0.705 vs 0.364 (fresh batch) |

Details, method and caveats: [docs/EVALUATION.md](docs/EVALUATION.md).

## Quick start

```sh
docker build -t classifier-dev .
alias run='docker run --rm -m 5g --env-file .env -v $PWD:/app -w /app classifier-dev'

run python -m classifier.ingest          # fetch jobs from data/sources.yaml (nightly by cron)
run python -m classifier.features        # read new/changed jobs once; skill rarity; TF-IDF
run python -m classifier.cli resume.pdf --top=20 [--sponsor] [--remote]
run uvicorn classifier.api:app --host 0.0.0.0 --port 8000
run python -m pytest -q                  # 57 regression tests
```

## Documentation

| Doc | What's in it |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Pipeline, modules, data, how a match is scored |
| [docs/API.md](docs/API.md) | Endpoints, every filter, labels, response format |
| [docs/EVALUATION.md](docs/EVALUATION.md) | AI judges as exam markers, splits, metrics, results history, known limits |
| [docs/OPERATIONS.md](docs/OPERATIONS.md) | Refresh, H-1B data, keys, memory, retraining, adding employers |
| [docs/LLM_GUIDE.md](docs/LLM_GUIDE.md) | Recommendations for using LLMs to build classifiers like this one |
| [AGENTS.md](AGENTS.md) | Rules for coding agents working in this repo |

## Job sources (official feeds only, no scraping)

| Source | Feed | Key needed |
|---|---|---|
| Greenhouse (230 employers) | Job Board API | no |
| Lever (33) | Postings API | no |
| Ashby (102) | Public job posting API | no |
| SmartRecruiters (24) | Posting API | no |
| Workable | Careers widget API | no |
| Remotive | Remote jobs API (link back to Remotive) | no |
| USAJOBS | Search API | `USAJOBS_API_KEY`, `USAJOBS_EMAIL` (not yet set) |
| Adzuna | Search API (snippets only) | `ADZUNA_APP_ID`, `ADZUNA_APP_KEY` (not yet set) |

LinkedIn, Indeed, Glassdoor and similar sites are deliberately **not** scraped.

## Data and privacy

O*NET® 31.0 Database by the U.S. Department of Labor, Employment and Training Administration
(USDOL/ETA), used under the CC BY 4.0 license. O*NET® is a trademark of USDOL/ETA. H-1B sponsorship
uses the Department of Labor's public LCA disclosure data.

Real resumes, rating pools and ratings (`eval/resumes*/`, `eval/private/`) are personal data: they stay
on the server, are git-ignored and are never committed. Resumes are redacted (name, email, phone, links,
street address) before any text is sent to an AI judge during evaluation.
