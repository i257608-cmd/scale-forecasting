import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval import metrics as M  # noqa: E402


def _w():
    t = pd.date_range("2020", periods=10, freq="1h")
    ev = pd.DataFrame(dict(file="ev", label=7, event=True, t_end=t, tau_min=np.arange(10, 0, -1) * 60.0,
                           censor_min=np.arange(10, 0, -1) * 60.0))
    neg = pd.DataFrame(dict(file="neg", label=0, event=False, t_end=t, tau_min=np.nan,
                            censor_min=np.arange(10, 0, -1) * 60.0))
    return pd.concat([ev, neg], ignore_index=True)


def test_horizon_target_unknown_for_short_censored():
    y = M.horizon_target(_w(), 3)
    assert y.iloc[:10].tolist() == [0] * 7 + [1] * 3          # event within 3 h at last 3 windows
    assert y.iloc[10:18].eq(0).all() and y.iloc[18:].isna().all()  # <3 h observed left -> unknown


def test_event_metrics_hit_lead_and_false_alarm_episodes():
    w = _w()
    s = np.r_[[0] * 6 + [1] * 4, [0, 1, 1, 0, 1, 0, 0, 0, 0, 0]]
    r = M.event_metrics(w, s, 0.5)
    assert r["hits"] == 1 and r["lead_med_h"] == 4 and r["false_alarms"] == 2
