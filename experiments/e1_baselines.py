"""E1 (part 1): tabular baselines B0-B2 on the 6 leave-one-well-out folds.

    python experiments/e1_baselines.py            # all folds, B0 B1 B2
    python experiments/e1_baselines.py --quick    # 2 folds (wells 22, 24)

Per fold: fit on train -> alarm threshold from out-of-fold scores on train wells -> score test wells once.
Writes results/tables/e1_baselines_folds.csv, e1_baselines_summary.csv and results/runs/<ts>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.splits import make_folds, quick, v1v2_fold  # noqa: E402
from src.eval import metrics as M  # noqa: E402
from src.models.baselines import make  # noqa: E402
from src.features.si import with_si  # noqa: E402
from src.models.tuned import tuned_params  # noqa: E402

FEAT = ROOT / "data" / "features"
TAB = FEAT / "tabular.parquet"


def tabular() -> pd.DataFrame:
    """Feature row at every window end (cached: 2,228 file reads take ~30 s)."""
    win = FEAT / "windows.parquet"
    # Rebuild if features were rebuilt after the cache was written (stale cache = silent wrong results).
    if TAB.exists() and TAB.stat().st_mtime > win.stat().st_mtime:
        return pd.read_parquet(TAB)
    w = pd.read_parquet(win)
    parts = []
    for (label, f), g in w.groupby(["label", "file"], sort=False):
        X = pd.read_parquet(FEAT / str(label) / f).iloc[g.end_pos.to_numpy()].reset_index(drop=True)
        parts.append(pd.concat([g.reset_index(drop=True), X], axis=1))
    t = pd.concat(parts, ignore_index=True).sort_values(["file", "t_end"], ignore_index=True)
    t.to_parquet(TAB)
    return t


def build_model(name: str, test_well: str):
    """'B2' -> fixed defaults; 'B2-tuned' -> Optuna params of this fold (e1_tune.py); '-SI' = extra features."""
    base = name.removesuffix("-tuned").removesuffix("-SI")
    return make(base, **(tuned_params(base, test_well) if name.endswith("-tuned") else {}))


def oof_scores(name, train, feat_cols, seed, n_splits=5, test_well=""):
    """Out-of-fold ALARM_H scores for real training windows, grouped by well.
    Synthetic windows always stay in the inner training part (never scored)."""
    real = train[train.source == "real"]
    synth = train[train.source != "real"]
    oof = np.zeros(len(real))
    for _, va in GroupKFold(n_splits=n_splits).split(real, groups=real.well):
        tr = pd.concat([real.drop(real.index[va]), synth])
        m = build_model(name, test_well).fit(tr[feat_cols], {M.ALARM_H: M.horizon_target(tr, M.ALARM_H)}, seed=seed)
        oof[va] = m.predict(real.iloc[va][feat_cols], [M.ALARM_H])[M.ALARM_H]
    return real, oof


def main(models, folds_mode, seed=0, split="lowo"):
    t = tabular()
    if any(n.endswith("-SI") for n in models):   # chemistry prior (src/features/si.py) as extra features
        assert all(n.endswith("-SI") for n in models), "run -SI variants separately (feature set differs)"
        t = with_si(t)
    meta_cols = ["file", "label", "source", "well", "end_pos", "t_end", "event", "tau_min", "censor_min", "k"]
    feat_cols = [c for c in t.columns if c not in meta_cols]
    inst = pd.read_csv(ROOT / "data" / "processed" / "instances.csv")
    folds = v1v2_fold(inst) if split == "v1v2" else make_folds(inst)   # E4 v1 -> v2 or LOWO (E1)
    folds = quick(folds) if folds_mode == "quick" else folds
    pred_dir = ROOT / "results" / ("preds_v1v2" if split == "v1v2" else "preds")
    rows = []
    for f in folds:
        part = {s: t[t.file.isin(f[s])] for s in ("train", "calib", "test")}
        y_tr = {h: M.horizon_target(part["train"], h) for h in M.HORIZONS_H}
        for name in models:
            t0 = time.time()
            m = build_model(name, f["test_well"]).fit(part["train"][feat_cols], y_tr, seed=seed)
            sc = {s: m.predict(part[s][feat_cols], M.HORIZONS_H) for s in ("calib", "test")}
            # Alarm threshold from out-of-fold scores on TRAINING wells (never test wells).
            tr_real, oof = oof_scores(name, part["train"], feat_cols, seed, test_well=f["test_well"])
            thr = M.choose_threshold(tr_real, oof)
            pred_dir.mkdir(parents=True, exist_ok=True)
            np.savez(pred_dir / f"{name}_{f['test_well']}_s{seed}.npz", oof_thr=thr,
                     **{f"{s}_{h}": v for s in sc for h, v in sc[s].items()})
            r = dict(model=m.name + name[2:], test_well=f["test_well"], seed=seed, secs=round(time.time() - t0, 1),
                     **M.window_metrics(part["test"], sc["test"]),
                     **M.event_metrics(part["test"], sc["test"][M.ALARM_H], thr),
                     **M.hits_at_fa(part["test"], sc["test"][M.ALARM_H]))
            rows.append(r)
            print(f"{r['model']:>13} {r['test_well']}: AUPRC24={r.get('auprc_24h', np.nan):.3f} "
                  f"recall={r['recall']:.2f} lead={r['lead_med_h']:.1f}h FA/day={r['fa_per_day']:.3f}", flush=True)
    res = pd.DataFrame(rows)
    out = ROOT / "results" / "tables"
    tag = ("" if split == "lowo" else f"_{split}") + ("" if models == ["B0", "B1", "B2"] else "_" + "_".join(models))
    res.to_csv(out / f"e1_baselines{tag}_folds.csv", index=False)
    # Event metrics pooled over folds (sum hits/events/FAs) are more stable than mean of per-fold ratios.
    summ = res.groupby("model").agg(
        auprc_24h=("auprc_24h", "mean"), auroc_24h=("auroc_24h", "mean"), auprc_6h=("auprc_6h", "mean"),
        hits=("hits", "sum"), events=("events", "sum"), false_alarms=("false_alarms", "sum"),
        neg_days=("neg_days", "sum"), lead_med_h=("lead_med_h", "median"))
    summ["recall"] = summ.hits / summ.events
    for b in M.FA_BUDGETS:   # pooled recall @ FA budget (threshold-free operating curve)
        summ[f"recall@fa{b:.3g}"] = res.groupby("model")[f"hits@fa{b:.3g}"].sum() / summ.events
    summ["fa_per_day"] = summ.false_alarms / summ.neg_days
    summ.round(4).to_csv(out / f"e1_baselines{tag}_summary.csv")
    runs = ROOT / "results" / "runs"
    runs.mkdir(exist_ok=True)
    (runs / f"e1_baselines{tag}_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(json.dumps(
        dict(models=models, folds=folds_mode, seed=seed, horizons_h=M.HORIZONS_H, alarm_h=M.ALARM_H,
             max_fa_per_day=M.MAX_FA_PER_DAY, n_features=len(feat_cols),
             summary=summ.reset_index().to_dict("records")), default=str, indent=1))
    print(summ.round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--models", nargs="+", default=["B0", "B1", "B2"])
    ap.add_argument("--split", choices=["lowo", "v1v2"], default="lowo")
    a = ap.parse_args()
    main(a.models, "quick" if a.quick else "full", split=a.split)
