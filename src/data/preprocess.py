"""Preprocessing + survival labels for 3W v2 instances (CLAUDE.md "Proposed model" §1, §2, §4).

One instance (Parquet, 1 Hz) -> one 1-min DataFrame with
    value channels | mask channels (m_*) | ri_star | class | minute-level labels
plus one row of instance-level metadata (well, source, onset t0, event time t_E, censoring).

Run as a script to build data/processed/ for every instance:
    python src/data/preprocess.py            # all 2,228 instances
    python src/data/preprocess.py --quick    # class 0 + 7 real only (fast smoke run)

WHY each choice is made is commented next to it. Decisions trace to notebooks/01_EDA.ipynb.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "data" / "3w" / "dataset"
OUT = ROOT / "data" / "processed"

# WHY these 9: rule "drop vars missing in >50 % of real instances" applied in EDA
# (results/tables/channel_missingness_real.csv). P-JUS-CKP / ABER-CKP / T-MON-CKP fail it.
CHANNELS = ["P-MON-CKP", "T-JUS-CKP", "P-TPT", "T-TPT", "P-ANULAR", "P-PDG",
            "P-JUS-CKGL", "T-PDG", "ESTADO-SDV-P"]
SCALING, SCALING_TRANSIENT = 7, 107
RI_THRESHOLD = 1.05            # approved 2026-11-14: event = scaling grown 5 %
FROZEN_MIN = 30                # >30 min constant = frozen sensor (3W v2 leaves these untreated)
FFILL_MIN = 10                 # short gaps are comms dropouts -> safe to carry forward
BASELINE_MIN = 60              # causal RI* baseline: first hour (always before onset, see below)
RI_SMOOTH_MIN = 60             # 1 h rolling median for the *label* -> one spike can't trigger an event


def source_and_well(path: Path) -> tuple[str, str | None]:
    """Filename encodes source; only real files carry a well ID (needed for grouped splits)."""
    m = re.match(r"(WELL-\d+)_", path.stem)
    return ("real", m.group(1)) if m else (path.stem.split("_")[0].lower(), None)


def frozen_mask(x: pd.Series, min_len: int = FROZEN_MIN) -> pd.Series:
    """True where the value has not changed for >= min_len consecutive minutes.
    WHY: a stuck transmitter reads a plausible constant; the model must be told it is not real."""
    same = x.diff().eq(0)
    run_id = (~same).cumsum()                       # new run starts whenever value changes
    run_len = same.groupby(run_id).transform("size")  # length of each constant run
    return (same & (run_len >= min_len - 1)) & x.notna()


def to_minutes(raw: pd.DataFrame, source: str = "real", sample_s: int = 1) -> pd.DataFrame:
    """1 Hz -> 1-min mean + mask channels + short-gap forward fill.
    WHY 1 min: scaling evolves over hours-days; 1 Hz only adds noise and 60x compute.
    sample_s > 1 (E3 resampling test only): simulate a historian that logs one point every sample_s
    seconds; minutes without a point are linearly interpolated between logged points (what a 1-min
    pipeline fed by a 2-5 min historian would see), so they are NOT flagged as missing."""
    vals = raw.reindex(columns=CHANNELS).astype("float64")
    if sample_s > 1:
        vals = vals.iloc[::sample_s]
    if source == "drawn":
        # WHY: hand-drawn pressures are in bar (P-MON-CKP median 45 vs 3.8e6 Pa real; P-PDG 253 vs
        # 2.3e7). Unconverted, models learned "low pressure = scaling" and inverted on real wells
        # (B3 AUROC 0.10 on WELL-00024, found 2026-11-14). Temperatures already match (°C).
        p = [c for c in CHANNELS if c.startswith("P-")]
        vals[p] = vals[p] * 1e5
    # WHY: 3W has sentinel/garbage readings (e.g. P-PDG = 1.18e42 Pa in WELL-00006). Nothing real in
    # these channels exceeds 1e9 (Pa: 10,000 bar; °C; 0/0.5/1 states), so mark as missing
    # BEFORE averaging -> one bad second cannot poison a whole minute.
    vals = vals.where(vals.abs() < 1e9).resample("1min").mean()
    if sample_s > 60:
        vals = vals.interpolate("time", limit=sample_s // 60, limit_area="inside")
    out = pd.DataFrame(index=vals.index)
    for c in CHANNELS:
        x = vals[c]
        missing = x.isna()
        frozen = frozen_mask(x)
        x = x.mask(frozen)                          # frozen readings are treated as missing values
        # WHY mask computed BEFORE ffill: the model must know a value was imputed.
        out[f"m_{c}"] = (missing | frozen).astype("float32")
        out[c] = x.ffill(limit=FFILL_MIN)
    # Labels: max of the 1 Hz class in each minute (NaN stays NaN = unlabelled).
    # WHY max: a minute containing any event second counts as event -> onset is never placed late,
    # and it is vectorised (a per-minute mode() would take hours over 2,228 files).
    out["class"] = raw["class"].astype("float64").resample("1min").max()
    return out


def ri_star(p_mon: pd.Series, baseline: float) -> pd.Series:
    """RI* = P-MON-CKP / baseline. Upstream-of-choke pressure rises as scale narrows the throat."""
    return p_mon / baseline if np.isfinite(baseline) and baseline > 0 else p_mon * np.nan


def process_instance(path: Path) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_parquet(path)
    label = int(path.parent.name)
    source, well = source_and_well(path)
    df = to_minutes(raw, source)
    t = df.index
    cls = df["class"]

    # Onset t0 = first 107 or 7 minute (EDA: 31/36 real instances are labelled 107 only).
    is_scaling = cls.isin([SCALING, SCALING_TRANSIENT]).to_numpy()
    t0 = t[is_scaling.argmax()] if is_scaling.any() else pd.NaT

    p = df["P-MON-CKP"]
    # FEATURE baseline must be causal (no knowledge of t0 at deployment):
    # median of the first hour. WHY safe: every 3W instance starts with >= 1 h unlabelled+normal
    # data (EDA: min pre-onset span 1.5 h), so the first hour never contains scaling.
    df["ri_star"] = ri_star(p, p.iloc[:BASELINE_MIN].median()).astype("float32")

    # LABEL baseline may use expert info (t0): median of all pre-onset minutes (approved definition).
    # Labels are allowed to look at the future; features are not.
    t_event = pd.NaT
    if label == SCALING and pd.notna(t0):
        ri_lab = ri_star(p, p[t < t0].median()).rolling(RI_SMOOTH_MIN, min_periods=10).median()
        hit = ri_lab[(t >= t0) & (ri_lab >= RI_THRESHOLD)]
        t_event = hit.index[0] if len(hit) else pd.NaT
    # WHY RI* crossings in other classes are NOT events: EDA shows hydrate (8), slugging (3)
    # etc. also cross 1.05; the expert label says they are not scaling -> negatives / censored.

    t_end = t[-1]
    # Minutes until event (survival target) for every minute before t_E; NaN after t_E or if censored.
    if pd.notna(t_event):
        df["tau_min"] = np.where(t < t_event, (t_event - t).total_seconds() / 60, np.nan).astype("float32")
    else:
        df["tau_min"] = np.float32(np.nan)
    meta = dict(file=path.name, label=label, source=source, well=well,
                t_start=t[0], t_end=t_end, t0=t0, t_event=t_event,
                event=bool(pd.notna(t_event)),
                # Censoring time: last minute observed without event (normal + non-crossing scaling).
                t_censor=t_event if pd.notna(t_event) else t_end,
                n_min=len(df), p_mon_missing=float(df["m_P-MON-CKP"].mean()))
    return df, meta


def build(paths: list[Path], out: Path = OUT) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    metas = []
    for i, f in enumerate(paths):
        df, meta = process_instance(f)
        dest = out / f.parent.name / f.name
        dest.parent.mkdir(exist_ok=True)
        df.to_parquet(dest)
        metas.append(meta)
        if i % 200 == 0:
            print(f"{i}/{len(paths)} {f.name}", flush=True)
    idx = pd.DataFrame(metas)
    idx.to_csv(out / "instances.csv", index=False)
    return idx


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="real class 0 + 7 only")
    a = ap.parse_args()
    paths = sorted(DS.glob("*/*.parquet"))
    if a.quick:
        paths = [p for p in paths if p.parent.name in ("0", "7") and p.stem.startswith("WELL")]
    idx = build(paths)
    print(idx.groupby(["label", "source"]).agg(n=("file", "size"), events=("event", "sum")))
