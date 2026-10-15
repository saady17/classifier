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

## Data

O*NET® 31.0 Database by the U.S. Department of Labor, Employment and Training Administration
(USDOL/ETA), used under the CC BY 4.0 license. O*NET® is a trademark of USDOL/ETA.
Real resumes used for evaluation stay on the server (`eval/resumes/`, git-ignored) and are never committed.
