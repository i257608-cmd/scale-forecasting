"""E3 (part 2) resampling robustness: what if the historian logs every 30 s, 1, 2 or 5 min instead of 1 Hz?

    python experiments/e3_resampling.py --models PIPS B3      # ~15 min build per rate (cached) + inference

For every REAL instance (all are test instances in some fold) the 1 Hz data are point-sampled every
r seconds (src/data/preprocess.to_minutes(sample_s=r)), re-aggregated to the 1-min grid, and the
causal features (RI*, rolling stats) are recomputed -> data/features_rs/<r>/. Labels, event times
and window positions stay the ORIGINAL ones (the truth does not change with the logging rate).
Models are NOT retrained (trained on 1 Hz -> 1-min means); scaling statistics stay the training
ones. Seeds of a model are averaged (ensemble), as in e3_robustness.py.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.e1_baselines import tabular  # noqa: E402
from src.data.preprocess import BASELINE_MIN, DS, OUT as PROC, ri_star, to_minutes  # noqa: E402
from src.data.splits import make_folds  # noqa: E402
from src.eval import metrics as M  # noqa: E402
from src.features.build import features  # noqa: E402
from src.train.trainer import DEVICE, SEQ_COLS, FoldData, Trainer, ikey  # noqa: E402

RATES_S = [30, 60, 120, 300]
RS = ROOT / "data" / "features_rs"


def one(args):
    """Resampled per-minute features of one real instance, on the original minute grid."""
    r, label, f, ends = args
    orig = pd.read_parquet(PROC / str(label) / f, columns=["class"])
    df = to_minutes(pd.read_parquet(DS / str(label) / f), "real", sample_s=r).reindex(orig.index)
    mcols = [c for c in df.columns if c.startswith("m_")]
    df[mcols] = df[mcols].fillna(1.0)                     # minutes lost at the tail = missing
    p = df["P-MON-CKP"]
    df["ri_star"] = ri_star(p, p.iloc[:BASELINE_MIN].median()).astype("float32")   # same causal baseline
    df["tau_min"] = np.nan
    feat = features(df)
    dest = RS / str(r) / str(label) / f
    dest.parent.mkdir(parents=True, exist_ok=True)
    feat[SEQ_COLS].to_parquet(dest)
    return feat.drop(columns=SEQ_COLS).iloc[ends].reset_index(drop=True).assign(file=f, end_pos=ends)


def build_rate(r: int, t: pd.DataFrame) -> pd.DataFrame:
    out = RS / str(r) / "tabular.parquet"
    if out.exists():
        return pd.read_parquet(out)
    jobs = [(r, int(g.label.iloc[0]), f, g.end_pos.to_numpy()) for f, g in t[t.source == "real"].groupby("file")]
    with ProcessPoolExecutor(6) as ex:
        parts = list(ex.map(one, jobs, chunksize=8))
    tab = pd.concat(parts, ignore_index=True)
    tab.to_parquet(out)
    return tab


def main(models):
    t = tabular()
    meta = ["file", "label", "source", "well", "end_pos", "t_end", "event", "tau_min", "censor_min", "k"]
    folds = make_folds(pd.read_csv(ROOT / "data" / "processed" / "instances.csv"))
    rows = []
    for r in RATES_S:
        rs = build_rate(r, t)
        print(f"rate {r}s: {len(rs):,} windows", flush=True)
        for f in folds:
            data = FoldData(t, f["train"])
            te = t[t.file.isin(f["test"])].reset_index(drop=True)
            te_rs = te[meta].merge(rs, on=["file", "end_pos"], how="left", validate="one_to_one")
            clean_arr = {k: data.arr[k] for k in np.unique(ikey(te))}      # test files are real -> unique names
            for k in clean_arr:
                data.arr[k] = data.scale(pd.read_parquet(RS / str(r) / k))
            for name in models:
                cks = sorted((ROOT / "results" / "checkpoints").glob(f"{name}_{f['test_well']}_s*.pt"))
                trs = []
                for ck in cks:
                    tr = Trainer(name, data)
                    tr.m.load_state_dict(torch.load(ck, map_location=DEVICE))
                    trs.append(tr)
                if not trs:
                    continue
                s = np.mean([tr.predict(te_rs)[M.ALARM_H] for tr in trs], 0)
                rows.append(dict(model=name, test_well=f["test_well"], rate_s=r,
                                 **M.window_metrics(te, {M.ALARM_H: s}), **M.hits_at_fa(te, s, budgets=(1 / 30,))))
                print(f"  {name} {f['test_well']} {r}s: AUPRC24={rows[-1].get('auprc_24h', np.nan):.3f}", flush=True)
            data.arr.update(clean_arr)
    res = pd.DataFrame(rows)
    out = ROOT / "results" / "tables"
    res.to_csv(out / "e3_resampling_folds.csv", index=False)
    summ = res.groupby(["model", "rate_s"]).agg(auprc_24h=("auprc_24h", "mean"), auroc_24h=("auroc_24h", "mean"),
                                                hits=("hits@fa0.0333", "sum")).reset_index()
    summ.round(4).to_csv(out / "e3_resampling.csv", index=False)
    print(summ.round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["PIPS", "B3"])
    main(ap.parse_args().models)
