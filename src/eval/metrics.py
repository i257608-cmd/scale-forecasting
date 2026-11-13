"""Evaluation for scaling-severity forecasts (CLAUDE.md "Metrics" + "Alarm").

Two levels, because a forecaster can rank windows well and still be useless in a control room:

1. Window level, per horizon H: y = 1 if the +5 % event happens within H.
   Censored windows whose remaining observed time < H have unknown y -> excluded (not called 0).
   Metrics: AUPRC (primary; positives are rare) and AUROC.

2. Event level (alarm rule agreed 2026-11-14):
   * alarm = score >= thr at a window; consecutive windows above thr are ONE alarm episode
   * event instance: hit if the first alarm comes before t_E; lead time = t_E - first alarm
   * false alarm = an episode in an instance that has no event and is not class 7
     (class-7 instances that never reach +5 % are scaling wells: alarms there are reported
      separately as "early/unconfirmed", not as false)
   * thr = lowest score with <= 1 false alarm per well-month, chosen on OUT-OF-FOLD predictions over
     the training wells (inner 5-fold grouped by well). WHY not the single calibration well: with
     1 event it gave thr = 0.9999 that never transferred (recall 0 on every test well, 2026-11-14).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HORIZONS_H = [1, 3, 6, 12, 24, 72]
ALARM_H = 24                      # alarm horizon k*: "event within 24 h"
MAX_FA_PER_DAY = 1 / 30           # <= 1 false alarm per well-month


def horizon_target(w: pd.DataFrame, h: float) -> pd.Series:
    """1 / 0 / NaN(unknown) for 'event within h hours'."""
    hm = h * 60
    y = pd.Series(np.nan, index=w.index)
    y[w.event & (w.tau_min <= hm)] = 1.0
    y[(w.event & (w.tau_min > hm)) | (~w.event & (w.censor_min >= hm))] = 0.0
    return y


def window_metrics(w: pd.DataFrame, scores: dict[float, np.ndarray]) -> dict:
    out = {}
    for h, s in scores.items():
        y = horizon_target(w, h)
        ok = y.notna().to_numpy()
        yy, ss = y[ok].to_numpy(), s[ok]
        if 0 < yy.sum() < len(yy):
            out[f"auprc_{h}h"] = average_precision_score(yy, ss)
            out[f"auroc_{h}h"] = roc_auc_score(yy, ss)
            out[f"prev_{h}h"] = yy.mean()   # AUPRC of a random scorer = prevalence
    return out


def _episodes(above: np.ndarray) -> int:
    return int(above[0] + (np.diff(above.astype(int)) == 1).sum()) if len(above) else 0


def event_metrics(w: pd.DataFrame, s: np.ndarray, thr: float) -> dict:
    """w must be sorted by (file, t_end). s aligned with w."""
    w = w.assign(s=s, above=s >= thr)
    hits, leads, n_ev, fa, days, early = 0, [], 0, 0, 0.0, 0
    for _, g in w.groupby("file", sort=False):
        a = g.above.to_numpy()
        if g.event.iloc[0]:
            n_ev += 1
            if a.any():
                hits += 1
                leads.append(g.tau_min.to_numpy()[a.argmax()] / 60)
        elif g.label.iloc[0] == 7:
            early += int(a.any())
        else:
            fa += _episodes(a)
            days += (g.t_end.iloc[-1] - g.t_end.iloc[0]).total_seconds() / 86400 + 1 / 144
    return dict(thr=thr, events=n_ev, hits=hits, recall=hits / n_ev if n_ev else np.nan,
                lead_med_h=float(np.median(leads)) if leads else np.nan,
                lead_q1_h=float(np.percentile(leads, 25)) if leads else np.nan,
                lead_q3_h=float(np.percentile(leads, 75)) if leads else np.nan,
                false_alarms=fa, fa_per_day=fa / days if days else np.nan, neg_days=days,
                alarms_in_unconfirmed_scaling=early)


def choose_threshold(w: pd.DataFrame, s: np.ndarray) -> float:
    """Lowest threshold meeting the false-alarm budget on calibration wells (max recall under budget)."""
    # Grid finer near the top: alarm thresholds live in the upper tail of the score distribution.
    qs = np.r_[np.linspace(0.5, 0.99, 50), 1 - np.logspace(-2, -5, 60), 1.0]
    for thr in np.unique(np.quantile(s, qs)):
        if event_metrics(w, s, thr)["fa_per_day"] <= MAX_FA_PER_DAY:
            return float(thr)
    return float(np.max(s)) + 1e-9   # budget unreachable -> never alarm (reported honestly)


FA_BUDGETS = (1 / 30, 0.1, 1.0)


def hits_at_fa(w: pd.DataFrame, s: np.ndarray, budgets=FA_BUDGETS) -> dict:
    """Threshold-free operating curve: max hits with FA/day <= budget, sweeping thresholds on the
    set itself. Like AUPRC it does NOT pick a deployable threshold - report it as "recall @ FA rate",
    never as the alarm system's performance (that is event_metrics with a train-chosen thr)."""
    qs = np.r_[np.linspace(0.0, 0.99, 100), 1 - np.logspace(-2, -5, 60), 1.0]
    curve = [event_metrics(w, s, thr) for thr in np.unique(np.quantile(s, qs))]
    out = {}
    for b in budgets:
        ok = [c for c in curve if c["fa_per_day"] <= b]
        best = max(ok, key=lambda c: (c["hits"], -c["thr"])) if ok else None
        out[f"hits@fa{b:.3g}"] = best["hits"] if best else 0
        out[f"lead@fa{b:.3g}"] = best["lead_med_h"] if best else np.nan
    return out
