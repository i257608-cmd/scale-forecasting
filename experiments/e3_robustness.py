"""E3 robustness: corrupt the TEST inputs of already-trained sequence models and re-score.

    python experiments/e3_robustness.py --models PIPS B3

Inference only (checkpoints from e1_sequence.py; seeds of one model are averaged = ensemble).
Corruptions act on the sequence input x (scaled values + masks):
  noise p %   Gaussian noise with std = p % of each channel's typical level (|train median|)
  dropout n   n random sensors per window read nothing: value -> 0 (= median), mask -> 1
  frozen      one random sensor per window stuck for the last 60 min at its value 60 min ago,
              mask NOT set (a stuck transmitter the frozen-detector has not caught yet: worst case)
Limitation (reported): the physics vector phi (rolling stats) is left clean. The resampling test
(30 s - 5 min) lives in experiments/e3_resampling.py (rebuilds the features per rate).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.e1_baselines import tabular  # noqa: E402
from src.data.splits import make_folds  # noqa: E402
from src.eval import metrics as M  # noqa: E402
from src.train.trainer import DEVICE, VALUE_COLS, FoldData, Trainer  # noqa: E402

NV = len(VALUE_COLS)                 # 9 sensors + RI*; RI* is index NV-1, P-MON-CKP index 0
CONDITIONS = [("clean", 0), ("noise", 0.01), ("noise", 0.03), ("noise", 0.05),
              ("dropout", 1), ("dropout", 2), ("dropout", 3), ("frozen", 60)]


def corrupt(x: np.ndarray, kind: str, level, scale: np.ndarray, rng) -> np.ndarray:
    x = x.copy()
    B, L, _ = x.shape
    if kind == "noise":
        x[:, :, :NV] += rng.normal(0, 1, (B, L, NV)).astype(np.float32) * (level * scale)
    elif kind == "dropout":
        for i in range(B):
            for c in rng.choice(NV - 1, level, replace=False):
                cols = [c, NV - 1] if c == 0 else [c]          # losing P-MON-CKP also loses RI*
                x[i, :, cols] = 0.0
                x[i, :, [NV + c]] = 1.0
    elif kind == "frozen":
        for i in range(B):
            c = rng.integers(NV - 1)
            cols = [c, NV - 1] if c == 0 else [c]
            for cc in cols:
                x[i, L - level:, cc] = x[i, L - level - 1, cc]
    return x


def main(models, quick_mode):
    t = tabular()
    folds = make_folds(pd.read_csv(ROOT / "data" / "processed" / "instances.csv"))
    if quick_mode:
        folds = [f for f in folds if f["test_well"] == "WELL-00024"]
    rows = []
    for f in folds:
        data = FoldData(t, f["train"])
        scale = (data.med.abs() / data.iqr).clip(upper=50).to_numpy(np.float32)   # p % of level, in scaled units
        te = t[t.file.isin(f["test"])].reset_index(drop=True)
        for name in models:
            cks = sorted((ROOT / "results" / "checkpoints").glob(f"{name}_{f['test_well']}_s*.pt"))
            if not cks:
                continue
            trs = []
            for ck in cks:
                tr = Trainer(name, data)
                tr.m.load_state_dict(torch.load(ck, map_location=DEVICE))
                trs.append(tr)
            for kind, level in CONDITIONS:
                rng = np.random.default_rng(0)
                orig = data.x
                data.x = lambda files, ends, _o=orig: corrupt(_o(files, ends), kind, level, scale, rng)
                try:
                    s = np.mean([tr.predict(te)[M.ALARM_H] for tr in trs], 0)
                finally:
                    data.x = orig
                r = dict(model=name, test_well=f["test_well"], condition=f"{kind} {level}",
                         **M.window_metrics(te, {M.ALARM_H: s}), **M.hits_at_fa(te, s, budgets=(1 / 30,)))
                rows.append(r)
                print(f"{name} {f['test_well']} {kind} {level}: AUPRC24={r.get('auprc_24h', np.nan):.3f}", flush=True)
    res = pd.DataFrame(rows)
    out = ROOT / "results" / "tables"
    res.to_csv(out / "e3_robustness_folds.csv", index=False)
    summ = res.groupby(["model", "condition"], sort=False).agg(auprc_24h=("auprc_24h", "mean"),
                                                                auroc_24h=("auroc_24h", "mean"),
                                                                hits=("hits@fa0.0333", "sum")).reset_index()
    summ.round(4).to_csv(out / "e3_robustness.csv", index=False)
    print(summ.round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["PIPS", "B3"])
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    main(a.models, a.quick)
