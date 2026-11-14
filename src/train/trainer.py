"""Training loop for sequence models (one fold, one seed).

Data flow per fold:
  1. robust scaling (median / IQR) fitted on TRAINING files only, applied to all splits
  2. a window = last L minutes ending at end_pos (left-padded with "missing" for early windows)
  3. loss: "bce" -> per-horizon BCE with unknown labels masked;
           "surv" -> censor-aware discrete-time NLL + λ·L_mono
  4. early stopping on CALIBRATION wells' loss (never test wells)
  5. predict test windows -> {horizon_h: P(event within h)} for src/eval/metrics.py
"""

from __future__ import annotations

import copy
import os
import time

import numpy as np
import pandas as pd
import torch

from src.data.preprocess import CHANNELS
from src.eval import metrics as M
from src.features.build import OUT as FEAT_DIR, HORIZON_EDGES_H, STRIDE_MIN, BASELINE_MIN
from src.models.seq import build

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

VALUE_COLS = CHANNELS + ["ri_star"]
SEQ_COLS = VALUE_COLS + [f"m_{c}" for c in CHANNELS]     # masks as extra channels (CLAUDE.md §5)
L = 360
K = len(HORIZON_EDGES_H) + 1
EDGES_MIN = np.array(HORIZON_EDGES_H) * 60
assert M.HORIZONS_H == HORIZON_EDGES_H, "metric horizons must equal survival bin edges"


def phys_cols(tab_cols) -> list[str]:
    # Physics features: rolling level/spread/trend of RI*, P-MON-CKP, T-JUS-CKP, P-TPT at window end.
    return [c for c in tab_cols if any(s in c for s in ("_mean", "_std", "_slope"))]


class FoldData:
    """Holds every instance's per-minute array (scaled with TRAIN statistics) and slices windows."""

    def __init__(self, tab: pd.DataFrame, train_files: list[str]):
        self.pcols = phys_cols(tab.columns)
        files = tab[["file", "label"]].drop_duplicates()
        raw = {f: pd.read_parquet(FEAT_DIR / str(l) / f, columns=SEQ_COLS) for f, l in files.itertuples(index=False)}
        tr = pd.concat([raw[f][VALUE_COLS] for f in train_files if f in raw])
        med, iqr = tr.median(), (tr.quantile(0.75) - tr.quantile(0.25)).replace(0, 1).fillna(1)
        self.arr = {}
        for f, d in raw.items():
            v = ((d[VALUE_COLS] - med) / iqr).clip(-20, 20).fillna(0)  # clip: one bad sensor can't blow up a batch
            self.arr[f] = np.concatenate([v.to_numpy(np.float32), d[SEQ_COLS[len(VALUE_COLS):]].to_numpy(np.float32)], 1)
        ptr = tab[tab.file.isin(train_files)][self.pcols]
        self.pmed, self.piqr = ptr.median(), (ptr.quantile(0.75) - ptr.quantile(0.25)).replace(0, 1).fillna(1)
        self.c_in, self.f_phys = len(SEQ_COLS), len(self.pcols)

    def phi(self, w: pd.DataFrame) -> np.ndarray:
        return ((w[self.pcols] - self.pmed) / self.piqr).clip(-20, 20).fillna(0).to_numpy(np.float32)

    def x(self, files, ends) -> np.ndarray:
        out = np.zeros((len(files), L, self.c_in), np.float32)
        out[:, :, len(VALUE_COLS):] = 1.0                       # padding = "missing"
        for i, (f, e) in enumerate(zip(files, ends)):
            seg = self.arr[f][max(0, e - L + 1): e + 1]
            out[i, L - len(seg):] = seg
        return out


def targets(w: pd.DataFrame):
    """bce: (N, K-1) with NaN = unknown. surv: event flag, bin index (event bin or censor bin)."""
    ybce = np.stack([M.horizon_target(w, h).to_numpy() for h in HORIZON_EDGES_H], 1).astype(np.float32)
    ev = w.event.to_numpy()
    t = np.where(ev, w.tau_min.to_numpy(), w.censor_min.to_numpy())
    b = np.searchsorted(EDGES_MIN, t, side="right")
    return ybce, ev.astype(np.float32), b.astype(np.int64)


def surv_nll(logits, ev, b, w_ev):
    """Discrete-time NLL. Event in bin b: survived bins < b, hazard fires in b.
    Censored in bin b: only known to survive bins < b (conservative: bin b itself not credited)."""
    logh = torch.nn.functional.logsigmoid(logits)
    log1mh = torch.nn.functional.logsigmoid(-logits)
    k = torch.arange(logits.shape[1], device=logits.device)[None]
    before = (k < b[:, None]).float()
    ll = (log1mh * before).sum(1) + ev * logh.gather(1, b[:, None]).squeeze(1)
    wt = torch.where(ev > 0, w_ev, torch.ones_like(ev))        # class weighting: events are rare
    return -(wt * ll).sum() / wt.sum()


def bce_loss(logits, y, pos_w):
    ok = ~torch.isnan(y)
    l = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.nan_to_num(y), pos_weight=pos_w, reduction="none")
    return (l * ok).sum() / ok.sum().clamp(min=1)


class Trainer:
    def __init__(self, name, data: FoldData, seed=0, lam_mono=0.1, epochs=12, steps=150, bs=256, lr=1e-3):
        self.name, self.d, self.seed = name, data, seed
        self.lam = 0.0 if name in ("PIPS-mono",) else lam_mono
        self.epochs, self.steps, self.bs, self.lr = epochs, steps, bs, lr
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.m = build(name, data.c_in, data.f_phys, K, L).to(DEVICE)
        self.kind = self.m.head.kind

    def _batch(self, w, idx, prev=False):
        ww = w.iloc[idx]
        ends = ww.end_pos.to_numpy() - (STRIDE_MIN if prev else 0)
        x = torch.from_numpy(self.d.x(ww.file.to_numpy(), np.maximum(ends, 0))).to(DEVICE)
        return x, torch.from_numpy(self.d.phi(ww)).to(DEVICE)

    def _loss(self, w, T, idx, train=True):
        x, phi = self._batch(w, idx)
        logits = self.m(x, phi)
        ybce, ev, b = (torch.from_numpy(a[idx]).to(DEVICE) for a in T)
        if self.kind == "bce":
            return bce_loss(logits, ybce, self.pos_w)
        loss = surv_nll(logits, ev, b, self.w_ev)
        if train and self.lam > 0:
            # L_mono: on scaling instances, P(event within H) must not drop between consecutive windows.
            # WHY only class 7: there the restriction only grows (no choke data to detect interventions).
            sel = (w.label.to_numpy()[idx] == 7) & (w.end_pos.to_numpy()[idx] - STRIDE_MIN >= BASELINE_MIN)
            if sel.any():
                xp, pp = self._batch(w, idx[sel], prev=True)
                Fp = self.m.head.cdf(self.m(xp, pp))
                Fc = self.m.head.cdf(logits[torch.from_numpy(sel).to(DEVICE)])
                loss = loss + self.lam * torch.relu(Fp - Fc).mean()
        return loss

    def fit(self, wtr: pd.DataFrame, wca: pd.DataFrame):
        Ttr, Tca = targets(wtr), targets(wca)
        yb = Ttr[0]
        pos = np.nansum(yb, 0)
        neg = np.sum(~np.isnan(yb), 0) - pos
        self.pos_w = torch.tensor(np.clip(neg / np.maximum(pos, 1), 1, 100), dtype=torch.float32, device=DEVICE)
        self.w_ev = torch.tensor(float(np.clip((1 - Ttr[1].mean()) / max(Ttr[1].mean(), 1e-3), 1, 100)), device=DEVICE)
        opt = torch.optim.AdamW(self.m.parameters(), lr=self.lr, weight_decay=1e-4)
        rng = np.random.default_rng(self.seed)
        ca_idx = rng.choice(len(wca), min(len(wca), 8000), replace=False)   # fixed val subset (speed)
        best, best_state, self.log = np.inf, None, []
        for ep in range(self.epochs):
            self.m.train()
            t0, tl = time.time(), 0.0
            for _ in range(self.steps):
                idx = rng.integers(0, len(wtr), self.bs)
                loss = self._loss(wtr, Ttr, idx)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.m.parameters(), 1.0)
                opt.step()
                tl += loss.item()
            vl = self.evaluate_loss(wca, Tca, ca_idx)
            self.log.append(dict(epoch=ep, train=tl / self.steps, val=vl, secs=time.time() - t0))
            if vl < best:
                best, best_state = vl, copy.deepcopy(self.m.state_dict())
        self.m.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def evaluate_loss(self, w, T, idx):
        self.m.eval()
        return float(np.mean([self._loss(w, T, idx[i:i + 1024], train=False).item() for i in range(0, len(idx), 1024)]))

    @torch.no_grad()
    def predict(self, w: pd.DataFrame) -> dict:
        self.m.eval()
        out = []
        for i in range(0, len(w), 1024):
            x, phi = self._batch(w, np.arange(i, min(i + 1024, len(w))))
            out.append(self.m.head.cdf(self.m(x, phi)).cpu().numpy())
        F = np.concatenate(out)
        return {h: F[:, j] for j, h in enumerate(HORIZON_EDGES_H)}
