"""B9: LSTM-autoencoder anomaly DETECTOR (detection reference, Aranha et al. 2024 SPE J / Lopes 2025).

    python experiments/b9_lstm_ae.py                 # 6 LOWO folds
    python experiments/b9_lstm_ae.py --split v1v2    # E4 v1 -> v2

WHY it is different from B1-B8: it is unsupervised. It never sees a scaling label; it learns to
reconstruct NORMAL operation (class-0 windows of the training split) and flags windows it cannot
reconstruct. This is how most published 3W detectors work, so it shows what a detector gives when
asked to forecast.
Score: reconstruction error of the window -> percentile among calibration-split normal windows
(0..1). The same score is written for every horizon (a detector has no notion of "within H hours"),
so per-horizon AUPRC/recall are comparable but Brier / ECE / conformal are NOT meaningful for B9.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.e1_baselines import tabular  # noqa: E402
from src.data.splits import make_folds, quick, v1v2_fold  # noqa: E402
from src.eval import metrics as M  # noqa: E402
from src.train.trainer import DEVICE, VALUE_COLS, FoldData, ikey  # noqa: E402
from src.models.tuned import tuned_params  # noqa: E402

NV, POOL, H = len(VALUE_COLS), 4, 64


class LSTMAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.LSTM(NV, H, batch_first=True)
        self.dec = nn.LSTM(H, H, batch_first=True)
        self.out = nn.Linear(H, NV)

    def forward(self, v):                       # v: (B, T, NV) pooled values
        _, (h, _) = self.enc(v)
        z = h[-1][:, None].expand(-1, v.shape[1], -1)   # bottleneck repeated over time
        return self.out(self.dec(z)[0])


def prep(x: np.ndarray):
    """(B, 360, C) -> pooled values (B, 90, NV) and weights 1 = observed (RI* follows P-MON-CKP mask)."""
    v = torch.from_numpy(x[..., :NV]).to(DEVICE)
    m = torch.from_numpy(x[..., NV:]).to(DEVICE)
    obs = 1 - torch.cat([m, m[..., :1]], -1)
    pool = lambda a: nn.functional.avg_pool1d(a.transpose(1, 2), POOL).transpose(1, 2)  # noqa: E731
    return pool(v), pool(obs)


def err(model, data, w, bs=1024):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(w), bs):
            ww = w.iloc[i:i + bs]
            v, o = prep(data.x(ikey(ww), ww.end_pos.to_numpy()))
            out.append((((model(v) - v) ** 2 * o).sum((1, 2)) / o.sum((1, 2)).clamp(min=1)).cpu().numpy())
    return np.concatenate(out)


def fit(data, wtr, wca, seed, epochs=12, steps=150, bs=256, lr=1e-3, wd=1e-4, on_epoch=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = LSTMAE().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    va = wca.iloc[rng.choice(len(wca), min(len(wca), 8000), replace=False)]
    best, state, log = np.inf, None, []
    for ep in range(epochs):
        model.train()
        for _ in range(steps):
            ww = wtr.iloc[rng.integers(0, len(wtr), bs)]
            v, o = prep(data.x(ikey(ww), ww.end_pos.to_numpy()))
            loss = ((model(v) - v) ** 2 * o).sum() / o.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        vl = float(err(model, data, va).mean())          # early stopping: calib NORMAL windows only
        log.append(dict(epoch=ep, val=vl))
        if vl < best:
            best, state = vl, copy.deepcopy(model.state_dict())
        if on_epoch:
            on_epoch(ep, vl)
    model.load_state_dict(state)
    return model, log


def main(split, quick_mode, seed=0, tuned=False):
    t = tabular()
    inst = pd.read_csv(ROOT / "data" / "processed" / "instances.csv")
    folds = v1v2_fold(inst) if split == "v1v2" else make_folds(inst)
    folds = quick(folds) if quick_mode else folds
    pred_dir = ROOT / "results" / ("preds_v1v2" if split == "v1v2" else "preds")
    rows = []
    for f in folds:
        t0 = time.time()
        data = FoldData(t, f["train"])
        part = {s: t[t.file.isin(f[s])].reset_index(drop=True) for s in ("train", "calib", "test")}
        normal = {s: part[s][part[s].label == 0] for s in ("train", "calib")}
        hp = tuned_params("B9", f["test_well"]) if tuned else {}     # Optuna params (e1_tune.py)
        model, log = fit(data, normal["train"], normal["calib"], seed, **hp)
        ref = np.sort(err(model, data, normal["calib"]))        # calibration-normal error distribution
        sc = {s: np.searchsorted(ref, err(model, data, part[s])) / len(ref) for s in ("calib", "test")}
        key = f"B9{'-tuned' if tuned else ''}_{f['test_well']}_s{seed}"
        torch.save(model.state_dict(), ROOT / "results" / "checkpoints" / f"{key}.pt")
        np.savez(pred_dir / f"{key}.npz", **{f"{s}_{h}": sc[s] for s in sc for h in M.HORIZONS_H})
        te = {h: sc["test"] for h in M.HORIZONS_H}
        r = dict(model="B9-tuned" if tuned else "B9", test_well=f["test_well"], seed=seed, secs=round(time.time() - t0),
                 params=sum(p.numel() for p in model.parameters()),
                 best_epoch=int(np.argmin([l["val"] for l in log])),
                 **M.window_metrics(part["test"], te), **M.hits_at_fa(part["test"], sc["test"]))
        rows.append(r)
        print(f"B9 {f['test_well']}: AUPRC24={r.get('auprc_24h', np.nan):.3f} AUROC24={r.get('auroc_24h', np.nan):.3f} "
              f"({r['secs']}s, best epoch {r['best_epoch']})", flush=True)
    tag = ("" if split == "lowo" else f"_{split}") + ("_tuned" if tuned else "")
    pd.DataFrame(rows).to_csv(ROOT / "results" / "tables" / f"e1_b9{tag}{'_quick' if quick_mode else ''}_folds.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["lowo", "v1v2"], default="lowo")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tuned", action="store_true")
    a = ap.parse_args()
    main(a.split, a.quick, tuned=a.tuned)
