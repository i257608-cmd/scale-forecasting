"""E1 (part 2) + E2: sequence baselines B3-B6, PI-PatchSurv and its ablations on the LOWO folds.

    python experiments/e1_sequence.py --quick                       # 2 folds, 1 seed, all models
    python experiments/e1_sequence.py --models PIPS --seeds 0 1 2 3 4
    python experiments/e1_sequence.py --models PIPS PIPS-phys PIPS-mono PIPS-surv PIPS-patch  # E2

Primary comparison = threshold-free metrics (AUPRC/AUROC per horizon, recall @ FA budget), the
same columns as e1_baselines.py, so B0-B2 and B3+ sit in one table. Deployed-threshold alarm
metrics for all models come in the calibration step (threshold transfer is the open problem).
Every run: checkpoint -> results/checkpoints/, predictions -> results/preds/, log -> results/runs/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.e1_baselines import tabular  # noqa: E402
from src.data.splits import make_folds, quick, v1v2_fold  # noqa: E402
from src.eval import metrics as M  # noqa: E402
from src.features.si import with_si  # noqa: E402
from src.models.tuned import tuned_params  # noqa: E402
from src.train.trainer import DEVICE, FoldData, Trainer  # noqa: E402


def main(models, seeds, quick_mode, tag, lr=1e-3, split="lowo", resume=False):
    t = tabular()
    if any(n.endswith("-SI") for n in models):   # chemistry prior (src/features/si.py) -> extra phi columns
        assert all(n.endswith("-SI") for n in models), "run -SI variants separately (phi width differs)"
        t = with_si(t)
    inst = pd.read_csv(ROOT / "data" / "processed" / "instances.csv")
    real = set(inst.file[inst.source == "real"])
    folds = v1v2_fold(inst) if split == "v1v2" else make_folds(inst)   # E4 v1 -> v2 or LOWO (E1)
    preds = "preds_v1v2" if split == "v1v2" else "preds"
    folds = quick(folds) if quick_mode else folds
    for d in ("checkpoints", preds, "runs", "tables"):
        (ROOT / "results" / d).mkdir(exist_ok=True)
    rows = []
    for f in folds:
        data = FoldData(t, f["train"])          # scaling fitted on this fold's train files only
        part = {s: t[t.file.isin(f[s])].reset_index(drop=True) for s in ("train", "calib", "test")}
        for name in models:
            # E2 "-synthetic": train on real instances only (simulated + hand-drawn removed).
            wtr = part["train"][part["train"].file.isin(real)] if "-nosyn" in name else part["train"]
            for seed in seeds:
                t0 = time.time()
                out = ROOT / "results" / preds / f"{name}_{f['test_well']}_s{seed}.npz"
                # --resume (after a crash): skip runs whose predictions are newer than the trainer code
                # (older files predate the label/file key fix and must be re-run)
                if resume and out.exists() and out.stat().st_mtime > (ROOT / "src/train/trainer.py").stat().st_mtime:
                    continue
                # "-tuned": Optuna params of this fold (e1_tune.py); "-SI" / "-tuned" suffixes are not architectures
                arch = name.removesuffix("-tuned").removesuffix("-SI")     # "PIPS-<ablation>" stays as is
                hp = dict(lr=lr) | (tuned_params(arch, f["test_well"]) if name.endswith("-tuned") else {})
                tr = Trainer(arch, data, seed=seed, **hp).fit(wtr, part["calib"])
                sc = {s: tr.predict(part[s]) for s in ("calib", "test")}
                key = f"{name}_{f['test_well']}_s{seed}"
                torch.save(tr.m.state_dict(), ROOT / "results" / "checkpoints" / f"{key}.pt")
                np.savez(ROOT / "results" / preds / f"{key}.npz",
                         **{f"{s}_{h}": v for s in sc for h, v in sc[s].items()})
                r = dict(model=name, test_well=f["test_well"], seed=seed, secs=round(time.time() - t0),
                         params=sum(p.numel() for p in tr.m.parameters()),
                         best_epoch=int(np.argmin([l["val"] for l in tr.log])),
                         gate=getattr(tr.m.body, "last_gate", None),
                         **M.window_metrics(part["test"], sc["test"]),
                         **M.hits_at_fa(part["test"], sc["test"][M.ALARM_H]))
                r["events"] = int(part["test"][part["test"].event].file.nunique())
                rows.append(r)
                (ROOT / "results" / "runs" / f"{key}_{tag}_log.json").write_text(json.dumps(tr.log))
                print(f"{name:>10} {f['test_well']} s{seed}: AUPRC24={r.get('auprc_24h', np.nan):.3f} "
                      f"AUROC24={r.get('auroc_24h', np.nan):.3f} hits@1FA/mo={r['hits@fa0.0333']}/{r['events']} "
                      f"({r['secs']}s, {r['params']:,} params, {DEVICE})", flush=True)
    if not rows:
        return
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "results" / "tables" / f"e1_sequence_{tag}_folds.csv", index=False)
    # Mean over seeds first, then pool over folds (hits summed, AUPRC averaged).
    per_fold = res.groupby(["model", "test_well"]).mean(numeric_only=True).reset_index()
    summ = per_fold.groupby("model").agg(auprc_24h=("auprc_24h", "mean"), auroc_24h=("auroc_24h", "mean"),
                                         auprc_6h=("auprc_6h", "mean"), events=("events", "sum"),
                                         params=("params", "first"), secs=("secs", "mean"))
    for b in M.FA_BUDGETS:
        summ[f"recall@fa{b:.3g}"] = per_fold.groupby("model")[f"hits@fa{b:.3g}"].sum() / summ.events
    summ.round(4).to_csv(ROOT / "results" / "tables" / f"e1_sequence_{tag}_summary.csv")
    (ROOT / "results" / "runs" / f"e1_sequence_{tag}_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(
        json.dumps(dict(models=models, seeds=seeds, quick=quick_mode, device=DEVICE,
                        summary=summ.reset_index().to_dict("records")), default=str, indent=1))
    print(summ.round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--models", nargs="+", default=["B3", "B4", "B5", "B6", "PIPS"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--tag", default="main")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--split", choices=["lowo", "v1v2"], default="lowo")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    main(a.models, a.seeds, a.quick, a.tag + ("_quick" if a.quick else ""), a.lr, a.split, a.resume)
