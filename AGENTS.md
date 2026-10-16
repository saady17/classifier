# Rules for coding agents in this repo

1. **No AI in the ranking path.** `ranker.py`, `filters.py`, `resume.py`, `jobparse.py` and everything
   they import must not call LLMs, neural networks or AI APIs. Classic statistics (BM25, TF-IDF,
   logistic regression, gradient-boosted trees) are allowed. LLMs are allowed only in evaluation
   (`judge.py`).
2. **Official job feeds only.** Never scrape LinkedIn, Indeed, Glassdoor or any site whose terms forbid it.
3. **Personal data never leaves the server and is never committed:** `eval/resumes*/`, `eval/private/`,
   `.env`. Redact resumes before sending them to a judge.
4. **Do not infer citizenship or nationality** from resumes; work authorization is a stated preference.
5. **Evaluation discipline:** iterate with `python -m classifier.quickeval train`. Look at `test`/`new`
   only for milestone checks. Report baselines from the same run. See docs/EVALUATION.md.
6. **Every new matching rule:** check it on thousands of real job posts for false positives, add
   regression tests (true and false cases) to `tests/`, bump `READER_VERSION` in `features.py` if the job
   reader changed.
7. **Before committing:** `ruff check classifier --select E,F,W,I --ignore E501` and `python -m pytest -q`
   must both pass (check the exit code; do not pipe pytest into `tail` in a `&&` chain).
8. Run in Docker with `-m 5g`; the server has 8 GB of RAM.
9. After retraining: re-run `python -m classifier.calibrate` and commit `data/model.pkl` and `data/labels.json` together.
