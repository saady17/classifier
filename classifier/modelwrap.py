"""Model wrappers kept in their own module so pickled models load from anywhere."""

from __future__ import annotations


class GoodFitModel:
    """Predicts P(good fit) and reports it on the 0-3 scale the ranker expects."""

    def __init__(self, clf, names):
        self.clf, self.feature_names = clf, names

    def predict(self, X):
        return 3.0 * self.clf.predict_proba(X)[:, 1]
