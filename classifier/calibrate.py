"""Chooses the score thresholds behind the ranker's match labels from rated data, so each label
means what it says. Writes data/labels.json.

    python -m classifier.calibrate [splits]     # default: test,new (held-out resumes)

Excellent match: >= 92% of these jobs are a good fit (3-judge majority rates them 2 or 3).
Good match:      >= 90% of Good + Excellent jobs are a good fit.
Possible match:  >= 90% are at least a reasonable fit (rated 1 or more).
Stretch:         everything else. Re-run after retraining and check on the next fresh batch."""

from __future__ import annotations

import json
import sys

from classifier import store
from classifier.evaluation import PRIVATE, _load_pools, averaged_ratings
from classifier.ranker import Preferences, rank
from classifier.resume import parse_resume

OUT = "data/labels.json"
TARGETS = {"excellent": (2.0, 0.92), "good": (2.0, 0.90), "possible": (1.0, 0.90)}
MIN_JOBS = 60  # do not trust a threshold measured on fewer jobs


def scored_pairs(splits: set[str]) -> list[tuple[float, float]]:
    pools = _load_pools()
    live = json.load(open(f"{PRIVATE}/live.json"))
    labels = averaged_ratings("ai")
    db = store.connect()
    out = []
    for rid, pool in pools.items():
        if pool["split"] not in splits:
            continue
        rel = labels.get(rid, {})
        ids = [j for j in dict.fromkeys(pool["order"] + live.get(rid, {}).get("order", [])) if j in rel]
        for m in rank(parse_resume(pool["text"]), Preferences(), limit=len(ids), db=db, only=ids, group=False):
            out.append((m.score, rel[m.job_id]))
    return out


def main(splits: set[str]) -> None:
    pairs = scored_pairs(splits)
    result = {}
    for name, (good_at, target) in TARGETS.items():
        chosen = None
        for t in range(40, 100):
            sel = [y for s, y in pairs if s >= t]
            if len(sel) < MIN_JOBS:
                break
            precision = sum(y >= good_at for y in sel) / len(sel)
            if precision >= target:
                chosen = {"min_score": t, "precision": round(precision, 3), "jobs": len(sel)}
                break
        result[name] = chosen
        print(f"{name:9}: {chosen or 'target not reached'}")
    result["measured_on"] = sorted(splits)
    result["pairs"] = len(pairs)
    json.dump(result, open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main(set((sys.argv[1] if len(sys.argv) > 1 else "test,new").split(",")))
