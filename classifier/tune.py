"""Learns the ranker's factor weights from human ratings on the TRAIN resumes only.

    python -m classifier.tune          # writes data/weights.json; the ranker uses it when present

Pairwise logistic regression: for two jobs in the same resume's pool where raters preferred one, the
factor differences should predict which. Weights are kept non-negative (every factor can only help)
and scaled to sum to 1, so scores stay on 0-100."""

from __future__ import annotations

import json
import math

import numpy as np
from sklearn.linear_model import LogisticRegression

from classifier import store
from classifier.evaluation import _load_pools, averaged_ratings, ndcg_at
from classifier.ranker import HAND_WEIGHTS as WEIGHTS
from classifier.ranker import Preferences, rank
from classifier.resume import parse_resume

OUT = "data/weights.json"
FACTORS = list(WEIGHTS)


def vectors(split: str):
    pools, ratings = _load_pools(), averaged_ratings()
    db = store.connect()
    groups = []
    for rid, pool in pools.items():
        rel = ratings.get(rid, {})
        if pool["split"] != split or len(rel) < 0.8 * len(pool["order"]):
            continue
        live = [j for j in pool["order"] if j in rel and db.execute("SELECT 1 FROM jobs WHERE id=?", (j,)).fetchone()]
        matches = rank(parse_resume(pool["text"]), Preferences(), limit=len(live), db=db, only=live, group=False)
        rows = [([0.5 if m.factors.get(f) is None else m.factors[f] for f in FACTORS], rel[m.job_id], m.job_id) for m in matches]
        groups.append((rid, rows))
    return groups


def main() -> None:
    train = vectors("train")
    X, y = [], []
    for _, rows in train:
        for i, (fi, ri, _) in enumerate(rows):
            for fj, rj, _ in rows[i + 1 :]:
                if abs(ri - rj) < 0.5:
                    continue
                d = np.array(fi) - np.array(fj)
                X += [d, -d]
                y += [int(ri > rj), int(rj > ri)]
    if len(set(y)) < 2:
        print("Not enough rated training resumes yet.")
        return
    model = LogisticRegression(fit_intercept=False, C=1.0, max_iter=2000).fit(np.array(X), np.array(y))
    raw = np.clip(model.coef_[0], 0, None)
    if raw.sum() == 0:
        print("Learned weights were all zero; keeping the hand-set weights.")
        return
    learned = {f: round(float(w / raw.sum()), 4) for f, w in zip(FACTORS, raw)}
    json.dump(learned, open(OUT, "w"), indent=2)
    print(f"trained on {len(train)} resumes, {len(y) // 2} preference pairs")
    for f in FACTORS:
        print(f"  {f:18} hand {WEIGHTS[f]:.2f}  learned {learned[f]:.3f}")

    # Quick check on the train set itself; the honest number comes from `evaluation report` on test.
    def score(rows, w):
        return sorted(rows, key=lambda r: -sum(wf * v for wf, v in zip(w, r[0])))

    for name, w in (("hand", [WEIGHTS[f] for f in FACTORS]), ("learned", [learned[f] for f in FACTORS])):
        vals = [ndcg_at([r[2] for r in score(rows, w)], {r[2]: r[1] for r in rows}) for _, rows in train]
        vals = [v for v in vals if not math.isnan(v)]
        print(f"  train NDCG@10 with {name} weights: {sum(vals) / len(vals):.3f}")


if __name__ == "__main__":
    main()
