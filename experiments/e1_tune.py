"""E1 hyper-parameter tuning with an EQUAL Optuna budget for every model (CLAUDE.md "Experiments" E1).

    python experiments/e1_tune.py --models B1 B2 B3 B4 B5 B6 B7 B8 B8i B9 PIPS     # all folds
    python experiments/e1_tune.py --models B2 --trials 4 --quick                   # smoke test
-> results/tuning/<model>.json  {test_well: {best params, best value, n_trials}}
Then:  e1_baselines.py / e1_sequence.py / b9_lstm_ae.py --tuned  re-train with these params
(predictions saved as "<model>-tuned", so the fixed-hyper-parameter results stay untouched).

Rules (no test-well leakage, CLAUDE.md "Rules"):
  * tuning is NESTED: a separate study per LOWO fold, scored only on that fold's CALIBRATION wells
    (the same wells used for early stopping). Test wells are never seen.
  * every model gets the same budget: N trials (TPE sampler, seed 0) + median pruning per epoch.
  * objective = the model's own validation loss on calibration windows (survival NLL for PI-PatchSurv,
    per-horizon BCE for B1-B8, reconstruction MSE on normal windows for B9). WHY loss, not AUPRC:
    calibration AUPRC was flat (~0.05) during early-stopping experiments (one small calibration well,
    1-3 events) -> no signal to tune on; the loss uses every calibration window.
  * sequence models tune with 4 epochs (best epoch was 0-1 in 28/30 PI-PatchSurv runs, so longer
    training only costs time); the final re-training keeps 12 epochs + early stopping.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.e1_baselines import tabular  # noqa: E402
from src.data.splits import make_folds, quick  # noqa: E402
from src.eval import metrics as M  # noqa: E402
from src.models.baselines import make  # noqa: E402
from src.train.trainer import FoldData, Trainer  # noqa: E402

from src.models.tuned import TUNING as OUT  # noqa: E402
META = ["file", "label", "source", "well", "end_pos", "t_end", "event", "tau_min", "censor_min", "k"]


def space(trial, name):
    if name == "B1":
        return dict(C=trial.suggest_float("C", 1e-3, 10, log=True))
    if name == "B2":
        return dict(learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 8, 63, log=True),
                    l2_regularization=trial.suggest_float("l2_regularization", 1e-3, 10, log=True),
                    max_iter=trial.suggest_int("max_iter", 100, 600, step=50))
    p = dict(lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
             wd=trial.suggest_float("wd", 1e-6, 1e-2, log=True),
             bs=trial.suggest_categorical("bs", [128, 256, 512]))
    if name.startswith("PIPS"):
        p["lam_mono"] = trial.suggest_float("lam_mono", 0.0, 0.5)
    return p


def objective_tab(trial, name, part, feat):
    hp = space(trial, name)
    y_tr = {h: M.horizon_target(part["train"], h) for h in M.HORIZONS_H}
    m = make(name, **hp).fit(part["train"][feat], y_tr)
    sc = m.predict(part["calib"][feat], M.HORIZONS_H)
    losses = []
    for h in M.HORIZONS_H:   # mean BCE over horizons, calibration windows with known label
        y = M.horizon_target(part["calib"], h).to_numpy()
        ok = ~np.isnan(y)
        if ok.any():
            p = np.clip(sc[h][ok], 1e-6, 1 - 1e-6)
            losses.append(-np.mean(y[ok] * np.log(p) + (1 - y[ok]) * np.log(1 - p)))
    return float(np.mean(losses))


def objective_seq(trial, name, data, part):
    hp = space(trial, name)

    def report(ep, vl):
        trial.report(vl, ep)
        if trial.should_prune():
            raise optuna.TrialPruned()
    if name == "B9":
        from experiments.b9_lstm_ae import fit
        normal = {s: part[s][part[s].label == 0] for s in ("train", "calib")}
        _, log = fit(data, normal["train"], normal["calib"], 0, epochs=4, on_epoch=report, **hp)
        return min(l["val"] for l in log)
    tr = Trainer(name, data, seed=0, epochs=4, **hp).fit(part["train"], part["calib"], on_epoch=report)
    return min(l["val"] for l in tr.log)


def main(models, n_trials, quick_mode, max_hours=None):
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    t = tabular()
    feat = [c for c in t.columns if c not in META]
    folds = make_folds(pd.read_csv(ROOT / "data" / "processed" / "instances.csv"))
    folds = quick(folds) if quick_mode else folds
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for name in models:
        path = OUT / f"{name}{'_quick' if quick_mode else ''}.json"
        res = json.loads(path.read_text()) if path.exists() else {}
        for f in folds:
            if f["test_well"] in res:          # resumable: one fold = one saved study
                continue
            # session limit (Kaggle 12 h): stop before a fold that may not finish; re-run resumes
            if max_hours and time.time() - t_start > max_hours * 3600:
                print(f"time budget {max_hours} h used, stopping before {name} {f['test_well']}", flush=True)
                return
            t0 = time.time()
            part = {s: t[t.file.isin(f[s])].reset_index(drop=True) for s in ("train", "calib")}
            study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0),
                                        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1))
            if name in ("B1", "B2"):
                study.optimize(lambda tr: objective_tab(tr, name, part, feat), n_trials=n_trials)
            else:
                data = FoldData(t, f["train"])
                study.optimize(lambda tr: objective_seq(tr, name, data, part), n_trials=n_trials)
            res[f["test_well"]] = dict(params=study.best_params, value=study.best_value, n_trials=n_trials,
                                       n_pruned=sum(x.state == optuna.trial.TrialState.PRUNED for x in study.trials),
                                       secs=round(time.time() - t0))
            path.write_text(json.dumps(res, indent=1))
            print(f"{name} {f['test_well']}: best {study.best_value:.4f} {study.best_params} "
                  f"({res[f['test_well']]['secs']}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8i", "B9", "PIPS"])
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--max-hours", type=float, default=None)
    a = ap.parse_args()
    main(a.models, a.trials, a.quick, a.max_hours)
