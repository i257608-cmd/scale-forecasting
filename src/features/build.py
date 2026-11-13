"""Per-minute features + forecast windows (CLAUDE.md "Proposed model" §2, §4).

Reads data/processed/<label>/<file>.parquet (from src/data/preprocess.py) and writes
    data/features/<label>/<file>.parquet   per-minute feature matrix (float32)
    data/features/windows.parquet          one row per forecast window (index + survival target)

A *window* = "standing at minute t, what do we know and when will the +5 % event happen?"
  * window end every STRIDE_MIN minutes, starting after the 60-min RI* baseline hour
  * windows at/after the event time t_E are dropped (CLAUDE.md §2: forecast, not detection)
  * target: tau_min = t_E - t (event) or censor_min = t_censor - t (no event seen)
  * bin index k over HORIZON_EDGES_H for the discrete-time survival head

Robust scaling is NOT done here on purpose: it must be fitted per fold on training wells only
(repo rule "never tune on test wells"), so it lives in the training code.

Run:  python src/features/build.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "features"

STRIDE_MIN = 10                                   # 10-min stride: hours-days process, cheap on M4
BASELINE_MIN = 60                                 # = preprocess.BASELINE_MIN (RI* undefined before)
ROLL_MIN = (10, 60, 360)                          # CLAUDE.md §4 rolling windows
DYNAMIC = ["ri_star", "P-MON-CKP", "T-JUS-CKP", "P-TPT"]   # physics-relevant channels get rolling stats
# Horizon bins (h) from EDA τ distribution (median 12 h, max 147 h): last bin open-ended.
HORIZON_EDGES_H = [1, 3, 6, 12, 24, 72]


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw channels + masks + rolling mean / std / slope. All causal (only past minutes used)."""
    f = df.drop(columns=["class", "tau_min"]).astype("float32")
    for c in DYNAMIC:
        x = df[c]
        for w in ROLL_MIN:
            r = x.rolling(w, min_periods=max(2, w // 4))
            f[f"{c}_mean{w}"] = r.mean()
            f[f"{c}_std{w}"] = r.std()
            # slope = change of the w-min mean over w minutes, per hour -> scale-free trend signal.
            # WHY: scaling is a slow monotonic rise; slugging/hydrate oscillate or jump.
            f[f"{c}_slope{w}"] = (r.mean() - r.mean().shift(w)) * (60 / w)
    return f.astype("float32")


def horizon_bin(tau_min: np.ndarray) -> np.ndarray:
    """Bin index 0..K-1 for event times (K = len(edges)+1). NaN (censored) -> -1."""
    k = np.searchsorted(np.array(HORIZON_EDGES_H) * 60, tau_min, side="right")
    return np.where(np.isnan(tau_min), -1, k)


def windows(meta: pd.Series, t: pd.DatetimeIndex) -> pd.DataFrame:
    end_pos = np.arange(BASELINE_MIN, len(t), STRIDE_MIN)
    te = t[end_pos]
    if meta.event:
        keep = te < meta.t_event
        end_pos, te = end_pos[keep], te[keep]
        tau = (meta.t_event - te).total_seconds().to_numpy() / 60
    else:
        tau = np.full(len(te), np.nan)
    censor = (meta.t_censor - te).total_seconds().to_numpy() / 60
    return pd.DataFrame(dict(file=meta.file, label=meta.label, source=meta.source, well=meta.well,
                             end_pos=end_pos, t_end=te, event=meta.event,
                             tau_min=tau, censor_min=censor, k=horizon_bin(tau)))


def build() -> pd.DataFrame:
    idx = pd.read_csv(PROC / "instances.csv", parse_dates=["t_start", "t_end", "t0", "t_event", "t_censor"])
    wins = []
    for i, m in idx.iterrows():
        df = pd.read_parquet(PROC / str(m.label) / m.file)
        dest = OUT / str(m.label) / m.file
        dest.parent.mkdir(parents=True, exist_ok=True)
        features(df).to_parquet(dest)
        wins.append(windows(m, df.index))
        if i % 300 == 0:
            print(f"{i}/{len(idx)}", flush=True)
    w = pd.concat(wins, ignore_index=True)
    w.to_parquet(OUT / "windows.parquet")
    return w


if __name__ == "__main__":
    w = build()
    print(f"{len(w):,} windows")
    print(w[w.label == 7].groupby(["source", "k"]).size().unstack(0).rename(
        index=lambda k: "censored" if k == -1 else f"bin{k}"))
