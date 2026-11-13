"""Tabular baselines B0-B2 (CLAUDE.md "Baselines"). Each model maps the feature row at a window's
end to P(event within H) for every horizon H. Sequence baselines (B3-B9) come later.

B0  threshold rule: score = 60-min mean of RI* (same for all H). No learning: the "skill vs B0"
    reference every learned model must beat.
B1  logistic regression, one per horizon, robust-scaled inputs, class-balanced.
B2  gradient-boosted trees (sklearn HistGradientBoosting = LightGBM-style histogram GBDT; used
    because LightGBM needs a system libomp not present on the M4), one per horizon.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer


class B0Threshold:
    name = "B0_threshold"

    def fit(self, X, y_by_h, seed=0):
        return self

    def predict(self, X: pd.DataFrame, horizons) -> dict:
        s = X["ri_star_mean60"].fillna(1.0).to_numpy()   # missing RI* -> "no restriction"
        return {h: s for h in horizons}


class PerHorizon:
    """Fit one binary classifier per horizon on windows whose label is known for that horizon."""

    def __init__(self, kind: str):
        self.kind, self.name, self.models = kind, {"lr": "B1_logreg", "gbdt": "B2_gbdt"}[kind], {}

    def _make(self, seed):
        if self.kind == "lr":
            # WHY RobustScaler: pressures are in Pa (1e6-1e7) next to 0/1 masks; median/IQR is not
            # dragged by spikes. Fitted inside the pipeline -> on training wells only.
            return make_pipeline(SimpleImputer(strategy="median"), RobustScaler(),
                                 LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1))
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                              l2_regularization=1.0, class_weight="balanced",
                                              random_state=seed)

    def fit(self, X: pd.DataFrame, y_by_h: dict, seed=0):
        for h, y in y_by_h.items():
            ok = y.notna().to_numpy()
            self.models[h] = self._make(seed).fit(X[ok], y[ok].astype(int))
        return self

    def predict(self, X, horizons) -> dict:
        return {h: self.models[h].predict_proba(X)[:, 1] for h in horizons}


def make(name: str):
    return {"B0": B0Threshold, "B1": lambda: PerHorizon("lr"), "B2": lambda: PerHorizon("gbdt")}[name]()
