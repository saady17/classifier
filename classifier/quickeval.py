"""Fast iteration metric: ranker NDCG@10 / P@10 on the TRAIN resumes only (the test set stays locked).

    python -m classifier.quickeval            # train
    python -m classifier.quickeval test       # only for the final check"""

from __future__ import annotations

import math
import sys
from concurrent.futures import ProcessPoolExecutor

from classifier import store
from classifier.evaluation import _load_pools, averaged_ratings, ndcg_at, precision_at
from classifier.ranker import Preferences, rank
from classifier.resume import parse_resume

_POOLS: dict = {}
_LABELS: dict = {}


def _one(rid: str) -> tuple[float, float, float] | None:
    pool, rel = _POOLS[rid], _LABELS.get(rid, {})
    live = [j for j in pool["order"] if j in rel]
    if len(live) < 20:
        return None
    db = store.connect()
    ranked = [m.job_id for m in rank(parse_resume(pool["text"]), Preferences(), limit=len(live), db=db, only=live, group=False)]
    top5_bad = sum(1 for j in ranked[:5] if rel[j] < 1) / 5
    return ndcg_at(ranked, rel), precision_at(ranked, rel), top5_bad


def main(split: str = "train") -> None:
    global _POOLS, _LABELS
    _POOLS, _LABELS = _load_pools(), averaged_ratings()
    ids = [rid for rid, p in _POOLS.items() if p["split"] == split]
    with ProcessPoolExecutor(max_workers=2) as pool:
        results = [r for r in pool.map(_one, ids) if r and not math.isnan(r[0])]
    n = len(results)
    print(f"{split}: {n} resumes  NDCG@10 {sum(r[0] for r in results) / n:.3f}  P@10 {sum(r[1] for r in results) / n:.3f}  "
          f"top-5 picks rated 0: {sum(r[2] for r in results) / n:.0%}")


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["train"]))
