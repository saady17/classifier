"""Learned ranking model: gradient-boosted decision trees (classic statistics, no neural networks)
that predict the judges' 0-3 rating from ~35 numbers describing a resume-job pair. Trained on the
TRAIN resumes only (their 25-job pools plus the ranker's real top 10); test and new stay held out.

    python -m classifier.model          # cross-validate on train, then fit and save data/model.pkl"""

from __future__ import annotations

import json
import math
import os
import pickle
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from classifier import store
from classifier.evaluation import PRIVATE, _load_pools, averaged_ratings, ndcg_at
from classifier.ranker import Preferences, rank
from classifier.resume import parse_resume

OUT = "data/model.pkl"
PARAMS = dict(learning_rate=0.05, max_iter=350, max_leaf_nodes=24, min_samples_leaf=40, l2_regularization=1.0, random_state=7)
_POOLS: dict = {}
_LABELS: dict = {}


def _rows(rid: str):
    pool = _POOLS[rid]
    rel = _LABELS.get(rid, {})
    ids = [j for j in dict.fromkeys(pool["ids"]) if j in rel]
    if len(ids) < 10:
        return rid, []
    matches = rank(parse_resume(pool["text"]), Preferences(), limit=len(ids), db=store.connect(), only=ids, group=False, use_model=False)
    return rid, [(m.features, rel[m.job_id], m.job_id) for m in matches]


def collect(split: str) -> dict[str, list]:
    """Feature rows per resume: its pool plus its live top 10 (when built)."""
    global _POOLS, _LABELS
    pools = _load_pools()
    live_path = f"{PRIVATE}/live.json"
    live = json.load(open(live_path)) if os.path.exists(live_path) else {}
    _POOLS = {rid: {"text": p["text"], "ids": p["order"] + live.get(rid, {}).get("order", [])} for rid, p in pools.items() if p["split"] == split}
    _LABELS = averaged_ratings()
    with ProcessPoolExecutor(max_workers=2) as ex:
        return {rid: rows for rid, rows in ex.map(_rows, list(_POOLS)) if rows}


def _matrix(groups: dict[str, list], names: list[str]):
    X, y = [], []
    for rows in groups.values():
        for feats, label, _ in rows:
            X.append([feats.get(k, math.nan) for k in names])
            y.append(label)
    return np.array(X, dtype=float), np.array(y, dtype=float)


def _ndcg(model, groups, names) -> float:
    vals = []
    for rows in groups.values():
        preds = model.predict(np.array([[f.get(k, math.nan) for k in names] for f, _, _ in rows], dtype=float))
        ranked = [jid for _, (_, _, jid) in sorted(zip(preds, rows), key=lambda t: -t[0])]
        v = ndcg_at(ranked, {jid: lab for _, lab, jid in rows})
        if not math.isnan(v):
            vals.append(v)
    return sum(vals) / len(vals)


def main() -> None:
    groups = collect("train")
    names = sorted({k for rows in groups.values() for f, _, _ in rows for k in f})
    rids = sorted(groups)
    random.Random(3).shuffle(rids)
    folds = [rids[i::4] for i in range(4)]
    scores, base = [], []
    for k in range(4):
        hold = set(folds[k])
        tr = {r: groups[r] for r in rids if r not in hold}
        te = {r: groups[r] for r in hold}
        X, y = _matrix(tr, names)
        m = HistGradientBoostingRegressor(**PARAMS).fit(X, y)
        scores.append(_ndcg(m, te, names))
        # the rule-based score on the same held-out fold, for comparison
        bv = [ndcg_at([j for f, _, j in sorted(rows, key=lambda t: -t[0]["rule_score"])], {j: lab for _, lab, j in rows}) for rows in te.values()]
        bv = [v for v in bv if not math.isnan(v)]
        base.append(sum(bv) / len(bv))
    print(f"4-fold CV on {len(groups)} train resumes ({sum(len(v) for v in groups.values())} rated pairs): "
          f"model NDCG@10 {np.mean(scores):.3f} vs rule score {np.mean(base):.3f}")
    X, y = _matrix(groups, names)
    model = HistGradientBoostingRegressor(**PARAMS).fit(X, y)
    model.feature_names = names
    with open(OUT, "wb") as f:
        pickle.dump(model, f)
    print(f"saved {OUT} ({len(names)} features)")


if __name__ == "__main__":
    main()
