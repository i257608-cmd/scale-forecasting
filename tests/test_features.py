import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import build as fb  # noqa: E402


def test_horizon_bins():
    tau = np.array([30, 61, 179, 600, 2000, 5000, np.nan])  # minutes
    assert fb.horizon_bin(tau).tolist() == [0, 1, 1, 3, 5, 6, -1]


def test_windows_stop_before_event_and_are_causal():
    t = pd.date_range("2020", periods=600, freq="1min")
    m = pd.Series(dict(file="f", label=7, source="real", well="W", event=True,
                       t_event=t[300], t_censor=t[300]))
    w = fb.windows(m, t)
    assert w.t_end.max() < t[300] and w.end_pos.min() == fb.BASELINE_MIN
    assert (w.tau_min > 0).all() and np.allclose(w.tau_min, w.censor_min)


def test_slope_uses_only_past():
    t = pd.date_range("2020", periods=800, freq="1min")
    df = pd.DataFrame({c: np.arange(800.0) for c in fb.DYNAMIC}, index=t).assign(**{"class": 0.0, "tau_min": np.nan})
    f1 = fb.features(df)
    df.iloc[500:] = df.iloc[500:] * 10          # changing the future ...
    f2 = fb.features(df)
    pd.testing.assert_frame_equal(f1.iloc[:500], f2.iloc[:500])  # ... must not change the past
