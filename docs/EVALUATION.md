# Evaluation

The ranker uses no AI. To know whether it is any good, its results are graded offline by AI judges
acting like exam markers, plus a small human check.

## Data

- **309 real resumes** (git-ignored, server only): 275 in the first batch, 34 in a fresh batch.
- **Splits are fixed and never mixed:** `train` (164 resumes) for building and tuning, `test` (111,
  locked by a hash of the file name) and `new` (34, a later batch) for measuring only.
- **Pools:** for each resume, 25 jobs to rate, shuffled so raters can't tell who picked what: the
  ranker's top 12, keyword search's top 8, and 5 random same-country jobs.
- **Real top 10 (`topk.py`):** the ranker's actual top 10 when searching all jobs, the product's view.

## Judges

- Three independent LLMs rate each resume–job pair 0–3 with the same rubric human raters use
  (`judge.py`): Gemini (`gemini-3.7-flash`), DeepSeek (`deepseek/deepseek-v4.1-flash`) and GLM
  (`z-ai/glm-5.3-flash`) via OpenRouter, the latter restricted to providers that do not collect data.
- Resumes are redacted (name, email, phone, links, street address) before sending.
- Ratings are blind (no judge sees another's answer). The label is the **median of three** (majority);
  three-way splits (2%) go to GLM as referee, which reads all three reasonings.
- A human check: 16–22 ratings by the product owner. The final AI label was within 1 point of the
  human 88% of the time.

Rubric: 3 great fit (likely shortlisted), 2 good fit (worth applying, at most one gap), 1 weak fit
(stretch or neighbouring field), 0 not a fit (different field, wrong level, missing required
license/degree). Location, salary and company prestige are ignored by the judges.

## Agreement (how much to trust the labels)

| | Exact | Within 1 point |
|---|---|---|
| Gemini vs DeepSeek | 72% | 98% |
| Gemini vs GLM | 71% | 99% |
| DeepSeek vs GLM | 77% | 99% |
| All three identical | 61% | – |
| Two judges disagree on good vs not good (top-5 jobs) | 14% | – |

The last row is the measurement noise floor: with a two-judge average as the answer key, even a perfect
ranker would score roughly 86–93% "good" at the top. The three-judge majority reduces this.

## Metrics

- **NDCG@10** on pools: are the best of the 25 jobs at the top? The ideal order is computed over the
  jobs being ranked only.
- **Top-5 good** on the real top 10: share of the first 5 jobs shown that the judges rate 2 or 3.
- **Label precision** (`calibrate.py`): share of jobs carrying a label that truly deserve it.

## Results history (test split unless noted)

| Change | NDCG@10 test | Top-5 good (real search) |
|---|---|---|
| First rules (keyword search: 0.455) | 0.557 | 47% |
| Seniority/years as multipliers, internships, years in words | 0.645 (train) | |
| English filter, TF-IDF content factor, occupation from skills, learned weights | 0.702 | |
| State licenses, degree subjects, ITAR, languages, bare years | 0.704 | |
| Gradient-boosted model on ~40 features | 0.742 | 59% |
| Second training round, train on all rated jobs | 0.746 | 62% |
| Requirement-line coverage, majority labels | (CV 0.694, top-5 66%) | label precision below |

Fresh `new` batch: NDCG 0.705 (keyword search 0.364), top-5 good 40%.

**Label precision on held-out resumes (3-judge majority):** Good match (score ≥ 82) 90.3%;
Possible match (≥ 41) 90.2% at least reasonable. Excellent match (target 92%) not yet reached.

## Known limits

1. **Inventory.** Feeds are mostly tech companies. Top-5 good by field: data 75%, software 62%,
   HR/ops 51%, marketing 50%, finance 36%. 34 of 145 held-out resumes had fewer than 5 known good jobs.
   USAJOBS and Adzuna keys would add non-tech jobs.
2. **Domain depth.** Remaining misses are specific requirements ("10+ years SAP architecture" vs an SAP
   end user, gaming industry background, two-sided marketplace experience).
3. **Label thresholds were chosen on the held-out set**; confirm them on the next fresh batch.
4. **Some fixes were chosen after looking at the `new` batch's failures** (general rules, but that
   batch is no longer pristine).
5. AI judges are not people. Keep a human check on every new batch.

## Commands

```sh
python -m classifier.evaluation pools                  # pools for eval/resumes (RESUMES_DIR, POOL_SPLIT)
python -m classifier.judge --only=gemini|deepseek|glm  # blind ratings (EVAL_POOLS for another pools file)
python -m classifier.judge crosscheck                  # majority / referee verdicts
python -m classifier.judge agreement                   # judges vs each other and humans
python -m classifier.quickeval [train|test|new]        # NDCG@10, P@10 (train for iterating)
python -m classifier.topk build [splits]; ...report    # real top-10 and top-5 good
python -m classifier.model                             # 4-fold CV on train, then fit data/model.pkl
python -m classifier.calibrate                         # label thresholds -> data/labels.json
```
