"""Synthetic checks for src/data/preprocess.py — no 3W download needed."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import preprocess as pp  # noqa: E402


def _instance(tmp_path, minutes=600, onset=120, rise=0.10, label=7):
    """1 Hz instance: flat P-MON-CKP, then linear rise of `rise` after `onset` minutes."""
    t = pd.date_range("2020-01-01", periods=minutes * 60, freq="1s")
    k = np.arange(len(t)) / 60.0
    p = 1e6 * (1 + np.clip(k - onset, 0, None) / (minutes - onset) * rise) + np.random.default_rng(0).normal(0, 50, len(t))
    df = pd.DataFrame({"P-MON-CKP": p, "T-JUS-CKP": 50.0 + np.sin(k)}, index=t)
    df["class"] = np.where(k < 60, np.nan, np.where(k < onset, 0, 107))
    d = tmp_path / str(label)
    d.mkdir()
    f = d / "WELL-00099_20200101000000.parquet"
    df.to_parquet(f)
    return f


def test_event_time_and_tau(tmp_path):
    df, meta = pp.process_instance(_instance(tmp_path))
    assert meta["well"] == "WELL-00099" and meta["event"]
    # +5 % of a +10 % linear rise over 480 min is reached ~240 min after onset (+ smoothing lag).
    lag = (meta["t_event"] - meta["t0"]).total_seconds() / 60
    assert 230 <= lag <= 300, lag
    assert df["tau_min"].iloc[0] > df["tau_min"].iloc[100] > 0
    assert df.loc[df.index >= meta["t_event"], "tau_min"].isna().all()


def test_no_event_without_rise(tmp_path):
    _, meta = pp.process_instance(_instance(tmp_path, rise=0.0))
    assert not meta["event"] and meta["t_censor"] == meta["t_end"]


def test_frozen_and_missing_masks():
    x = pd.Series([1.0, 2.0] + [3.0] * 40 + [4.0, np.nan], index=pd.date_range("2020", periods=44, freq="1min"))
    fr = pp.frozen_mask(x)
    assert fr.iloc[3:42].all() and not fr.iloc[:2].any() and not fr.iloc[-1]
