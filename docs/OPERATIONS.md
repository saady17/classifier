# Operations

## Runtime

Everything runs in the `classifier-dev` Docker image (`docker build -t classifier-dev .`). The server
has 8 GB of RAM: give containers `-m 5g` and keep `FEATURE_WORKERS=1` (two workers ran out of memory).

## Nightly refresh

`deploy/classifier-refresh.cron` (installed as `/etc/cron.d/classifier-refresh`; **currently not
installed**, the classifier is stopped) runs at 05:00 UTC:

```sh
python -m classifier.ingest && python -m classifier.features
```

Ingest fetches every account in `data/sources.yaml`, keeps US/UK/Canada jobs, removes closed jobs and
links duplicates (~29k unique jobs, ~6 minutes). Features reads only new or changed jobs, then rebuilds
skill rarity and the TF-IDF vocabulary (~1–4 minutes). Log: `data/refresh.log`.

When the job reader's logic changes, bump `READER_VERSION` in `features.py` so every job is re-read.

## Keys (`/root/classifier/.env`, chmod 600, never committed)

| Variable | Used by | Notes |
|---|---|---|
| `CLASSIFIER_API_KEY` | API | Required header `X-API-Key` |
| `USAJOBS_API_KEY`, `USAJOBS_EMAIL` | ingest | Free: developer.usajobs.gov/apirequest. Not yet set |
| `ADZUNA_APP_ID`, `ADZUNA_APP_KEY` | ingest | Free: developer.adzuna.com. Not yet set |
| `GEMINI_API_KEY`, `OPENROUTER_API_KEY` | evaluation only | AI judges. The ranker never uses them |
| `GOOGLE_SA_EMAIL`, `GOOGLE_SA_PRIVATE_KEY` | `drive_fetch` | Downloading evaluation resumes from Drive |

The OpenRouter key was once pasted into a chat; rotate it on openrouter.ai.

## H-1B sponsorship data

`dol.gov` and `uscis.gov` block server downloads (403), so download in a browser:

1. https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/LCA_Disclosure_Data_FY2025_Q4.xlsx
2. https://www.dol.gov//media/LCA_Disclosure_Data_FY2026_Q3.xlsx (newer quarters as they appear)

Put them in `data/h1b/` and run `python -m classifier.h1b data/h1b/*.xlsx`. Until loaded, sponsorship
falls back to the post's own wording only. Add name mismatches to `data/h1b_aliases.yaml`. Refresh each
quarter.

## Adding employers

Put candidate IDs in a `data/candidates*.yaml`, run `CANDIDATES=... SOURCES_OUT=... python -m classifier.discover`,
**check each match is the intended company** (short names like "gong", "neon", "blackstone" matched
other companies), then merge into `data/sources.yaml`. Official feeds only.

## Retraining

1. Rate new resumes (see [EVALUATION.md](EVALUATION.md)).
2. `python -m classifier.topk build train` and judge the new pairs with all three judges.
3. `python -m classifier.model` (4-fold CV printed; compare with the previous model before keeping it).
4. `python -m classifier.quickeval test` / `new`, `python -m classifier.topk report`.
5. `python -m classifier.calibrate`, then commit `data/model.pkl`, `data/labels.json`.

## Rating page

`/eval` in the API (enabled by `RATER_PASSWORD`) shows a resume and a job side by side for 0–3 ratings,
the models' ratings after each rating, and `/eval/review` with agreement scores. It is **not running**
now; when needed run it bound to `127.0.0.1` and reach it through an SSH tunnel
(`ssh -L 8899:localhost:8899 root@<server>`), since it shows real resumes.
