# API

`uvicorn classifier.api:app --host 0.0.0.0 --port 8000`. Set `CLASSIFIER_API_KEY` and send it as
`X-API-Key` on every `/v1/*` request.

## Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| `POST` | `/v1/rank` | multipart: `file` (PDF/DOCX/TXT, ≤ 5 MB), `preferences` (JSON string, fields below), `limit` | resume summary + matches |
| `POST` | `/v1/rank-text` | JSON: `{"resume_text": "...", "preferences": {...}, "limit": 30}` | same |
| `GET` | `/v1/filters` | – | allowed values for each filter (for building a search form) |
| `GET` | `/health` | – | job count, last fetch |

## Preferences

All fields are optional. Empty lists mean "any". Unknown values are rejected with `422`.

### Facts about the person

| Field | Default | Meaning |
|---|---|---|
| `countries` | resume's country, else `["US","GB","CA"]` | Countries to search |
| `needs_sponsorship` | `false` | Removes "no sponsorship" posts, ITAR/US-persons-only jobs, and employers with **no certified H-1B filing in DOL data** (when loaded) |
| `work_authorization` | `"unknown"` | `citizen`, `permanent_resident`, `needs_sponsorship` or `unknown`. Never guessed from the resume. `unknown` only adds a note on ITAR jobs |
| `has_clearance` | `false` | Jobs requiring a security clearance are removed unless true |
| `willing_to_relocate` | `true` | Affects the location factor for on-site jobs elsewhere |

### Recruiter-style hard filters (a job that fails one is not shown)

| Field | Values | Notes |
|---|---|---|
| `employment_types` | `full_time`, `part_time`, `contract`, `temporary`, `internship` | From the feed's field, else the title, else the post's opening lines. Default when unstated: `full_time` |
| `work_modes` | `remote`, `hybrid`, `onsite` | `remote` (legacy: `any`/`remote_only`/`no_remote`) still works |
| `locations` | `["Austin, TX", "New York", "Ontario", "UK"]` | City, state/province or country. **Remote jobs in the same country also match** |
| `excluded_employers` | names | e.g. the current employer. Matched on normalized legal names ("Acme Corp" = "ACME CORPORATION") |
| `only_employers` | names | Allow-list |
| `title_include` | keywords | Title must contain at least one |
| `title_exclude` | keywords | Title must contain none |
| `seniority` | `intern`, `entry`, `mid`, `senior`, `lead`, `director`, `executive`, or `auto` | `auto` = the resume's level ± 1 |
| `min_salary` | number | Annual, in the job's currency; hourly ×2080, monthly ×12. Compared with the **top** of the posted range ("up to") |
| `include_unlisted_salary` | `true` | Keep jobs that state no salary when `min_salary` is set (~35% of jobs) |
| `min_label` | `Possible match`, `Good match`, `Excellent match` | Show only jobs with at least this label |

## Response

```json
{
  "resume": {"current_title": "Senior Software Engineer", "occupation": "Software Developers", "level": 3,
             "years": 8.3, "degree": 3, "location": "Seattle, WA, US", "top_skills": ["AWS", "Go", "..."]},
  "matches": [{
    "job_id": "greenhouse:twilio:1234", "title": "Software Engineer, Platform Engineering", "company": "Twilio",
    "url": "https://...", "location": "Remote - US", "remote": "remote", "posted_at": "2026-10-14T...",
    "score": 90.5, "label": "Good match", "employment_type": "full_time", "salary": "USD 116,960-146,200 per year",
    "factors": {"required_skills": 0.91, "title": 1.0, "seniority": 1.0, "...": "..."},
    "matched_required": ["Go", "Kubernetes"], "missing_required": ["Terraform"], "matched_preferred": [],
    "notes": ["asks for 5+ years; you have about 4"]
  }]
}
```

## Labels

| Label | Score | Measured precision (held-out resumes, 3-judge majority) |
|---|---|---|
| Excellent match | not yet issued | target ≥ 92% good fits, not reached with enough jobs yet |
| Good match | ≥ 82 | 90.3% are good fits |
| Possible match | ≥ 41 | 90.2% are at least a reasonable fit |
| Stretch | < 41 | – |

Thresholds live in `data/labels.json` and come from `python -m classifier.calibrate`. Re-run it after
every retrain, and check them on the next fresh batch of resumes.

## Example

```sh
curl -s -X POST http://localhost:8000/v1/rank -H "X-API-Key: $KEY" \
  -F file=@resume.pdf -F limit=20 \
  -F 'preferences={"work_modes":["remote","hybrid"],"employment_types":["full_time"],
                   "excluded_employers":["Current Employer Inc"],"seniority":["auto"],
                   "min_salary":120000,"min_label":"Possible match"}'
```
