"""Read Optuna results written by experiments/e1_tune.py (kept separate to avoid circular imports)."""

from __future__ import annotations

import json
from pathlib import Path

TUNING = Path(__file__).resolve().parents[2] / "results" / "tuning"


def tuned_params(name: str, test_well: str) -> dict:
    """Best params of a model for one fold. Raises if the study is missing: a silent fallback to the
    defaults would label an untuned run '-tuned' (B7-PIPS are tuned on Kaggle and copied in later)."""
    path = TUNING / f"{name}.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    if test_well not in res:
        raise KeyError(f"no Optuna result for {name} / {test_well} in {path} - run e1_tune.py first")
    return res[test_well]["params"]
