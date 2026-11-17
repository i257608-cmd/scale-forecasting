"""Training loop for sequence models (one fold, one seed).

Data flow per fold:
  1. robust scaling (median / IQR) fitted on TRAINING files only, applied to all splits
  2. a window = last L minutes ending at end_pos (left-padded with "missing" for early windows)
  3. loss: "bce" -> per-horizon BCE with unknown labels masked;
           "surv" -> censor-aware discrete-time NLL + λ·L_mono
  4. early stopping on CALIBRATION wells' loss (never test wells). Calibration AUPRC@24h is logged
     too; it was tried as the stopping criterion (2026-11-14) but stayed flat at ~0.05 (one small
     calibration well, 1-3 events) -> no signal, so the loss rule is kept for ALL sequence models.
     Finding: calib loss is lowest after epoch 1 on every fold -> models overfit the training
     wells quickly (cross-well shift), so short training generalises best.
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


def ikey(w: pd.DataFrame) -> np.ndarray:
    """Instance key "label/file". WHY not the file name alone: simulated and hand-drawn file names repeat
    across class folders (885 of 2,228 instances, e.g. DRAWN_00001 is both a class-1 and a class-7
    drawing). Keying by file alone (bug until 2026-11-15) fed ~half of the synthetic windows the
    signal of another class; real file names are unique, so test / calibration were not affected."""
    return (w.label.astype(int).astype(str) + "/" + w.file).to_numpy()


class FoldData:
    """Holds every instance's per-minute array (scaled with TRAIN statistics) and slices windows."""

    def __init__(self, tab: pd.DataFrame, train_files: list[str]):
        self.pcols = phys_cols(tab.columns)
        files = tab[["file", "label"]].drop_duplicates()
        raw = {f"{l}/{f}": pd.read_parquet(FEAT_DIR / str(l) / f, columns=SEQ_COLS) for f, l in files.itertuples(index=False)}
        train = set(train_files)
        tr = pd.concat([d[VALUE_COLS] for k, d in raw.items() if k.split("/", 1)[1] in train])
        med, iqr = tr.median(), (tr.quantile(0.75) - tr.quantile(0.25)).replace(0, 1).fillna(1)
        self.med, self.iqr = med, iqr   # kept for E3 (noise relative to the signal level)
        self.arr = {f: self.scale(d) for f, d in raw.items()}
        ptr = tab[tab.file.isin(train_files)][self.pcols]
        self.pmed, self.piqr = ptr.median(), (ptr.quantile(0.75) - ptr.quantile(0.25)).replace(0, 1).fillna(1)
        self.c_in, self.f_phys = len(SEQ_COLS), len(self.pcols)

    def scale(self, d: pd.DataFrame) -> np.ndarray:
        """Per-minute features -> (minutes, C) with TRAIN statistics (E3 re-scales resampled test files)."""
        v = ((d[VALUE_COLS] - self.med) / self.iqr).clip(-20, 20).fillna(0)  # clip: one bad sensor can't blow up a batch
        return np.concatenate([v.to_numpy(np.float32), d[SEQ_COLS[len(VALUE_COLS):]].to_numpy(np.float32)], 1)

    def phi(self, w: pd.DataFrame) -> np.ndarray:
        return ((w[self.pcols] - self.pmed) / self.piqr).clip(-20, 20).fillna(0).to_numpy(np.float32)

    def x(self, keys, ends) -> np.ndarray:
        """keys = ikey(windows) ("label/file"), ends = window end positions (minutes)."""
        out = np.zeros((len(keys), L, self.c_in), np.float32)
        out[:, :, len(VALUE_COLS):] = 1.0                       # padding = "missing"
        for i, (k, e) in enumerate(zip(keys, ends)):
            seg = self.arr[k][max(0, e - L + 1): e + 1]
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
    def __init__(self, name, data: FoldData, seed=0, lam_mono=0.1, epochs=12, steps=150, bs=256, lr=1e-3, wd=1e-4):
        self.name, self.d, self.seed = name, data, seed
        self.lam = 0.0 if name in ("PIPS-mono",) else lam_mono
        self.epochs, self.steps, self.bs, self.lr, self.wd = epochs, steps, bs, lr, wd
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.m = build(name, data.c_in, data.f_phys, K, L).to(DEVICE)
        self.kind = self.m.head.kind
        # WHY: "-patching" (patch 1) has 360 tokens per channel and runs out of MPS memory at bs 256.
        # Gradient accumulation over micro-batches keeps the effective batch (and the experiment) unchanged.
        self.micro, self.ebs = (32, 64) if name == "PIPS-nopatch" else (bs, 1024)

    def _batch(self, w, idx, prev=False):
        ww = w.iloc[idx]
        ends = ww.end_pos.to_numpy() - (STRIDE_MIN if prev else 0)
        x = torch.from_numpy(self.d.x(ikey(ww), np.maximum(ends, 0))).to(DEVICE)
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

    def fit(self, wtr: pd.DataFrame, wca: pd.DataFrame, on_epoch=None):
        """on_epoch(epoch, calib_loss): optional hook (Optuna pruning in experiments/e1_tune.py)."""
        Ttr, Tca = targets(wtr), targets(wca)
        yb = Ttr[0]
        pos = np.nansum(yb, 0)
        neg = np.sum(~np.isnan(yb), 0) - pos
        self.pos_w = torch.tensor(np.clip(neg / np.maximum(pos, 1), 1, 100), dtype=torch.float32, device=DEVICE)
        self.w_ev = torch.tensor(float(np.clip((1 - Ttr[1].mean()) / max(Ttr[1].mean(), 1e-3), 1, 100)), device=DEVICE)
        opt = torch.optim.AdamW(self.m.parameters(), lr=self.lr, weight_decay=self.wd)
        rng = np.random.default_rng(self.seed)
        ca_idx = rng.choice(len(wca), min(len(wca), 8000), replace=False)   # fixed val subset (speed)
        best, best_state, self.log = np.inf, None, []
        for ep in range(self.epochs):
            self.m.train()
            t0, tl = time.time(), 0.0
            for _ in range(self.steps):
                idx = rng.integers(0, len(wtr), self.bs)
                opt.zero_grad()
                loss = 0.0
                for j in range(0, self.bs, self.micro):     # one pass when micro == bs
                    part = self._loss(wtr, Ttr, idx[j:j + self.micro]) * (len(idx[j:j + self.micro]) / self.bs)
                    part.backward()
                    loss += part.detach()
                torch.nn.utils.clip_grad_norm_(self.m.parameters(), 1.0)
                opt.step()
                tl += loss.item()
            vl = self.evaluate_loss(wca, Tca, ca_idx)
            va = M.window_metrics(wca, {M.ALARM_H: self.predict(wca)[M.ALARM_H]})   # all calib windows: positives are few
            vauprc = va.get(f"auprc_{M.ALARM_H}h", np.nan)
            self.log.append(dict(epoch=ep, train=tl / self.steps, val=vl, val_auprc=vauprc, secs=time.time() - t0))
            if vl < best:
                best, best_state = vl, copy.deepcopy(self.m.state_dict())
            if on_epoch:
                on_epoch(ep, vl)
        self.m.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def evaluate_loss(self, w, T, idx):
        self.m.eval()
        return float(np.mean([self._loss(w, T, idx[i:i + self.ebs], train=False).item() for i in range(0, len(idx), self.ebs)]))

    @torch.no_grad()
    def predict(self, w: pd.DataFrame) -> dict:
        self.m.eval()
        out = []
        for i in range(0, len(w), self.ebs):
            x, phi = self._batch(w, np.arange(i, min(i + self.ebs, len(w))))
            out.append(self.m.head.cdf(self.m(x, phi)).cpu().numpy())
        F = np.concatenate(out)
        return {h: F[:, j] for j, h in enumerate(HORIZON_EDGES_H)}
