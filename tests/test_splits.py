"""Leakage checks for src/data/splits.py on a toy index (no data download needed)."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.splits import make_folds  # noqa: E402


def _toy():
    rows = []
    # wells 1-4 have scaling (well 1: most events, well 4: none), wells 5-10 do not
    n_events = {1: 3, 2: 1, 3: 1, 4: 0}
    for w in range(1, 11):
        rows.append(dict(file=f"WELL-{w:05d}_a", source="real", well=f"WELL-{w:05d}", label=0, event=False))
        for j in range(max(n_events.get(w, 0), 1) if w in n_events else 0):
            rows.append(dict(file=f"WELL-{w:05d}_b{j}", source="real", well=f"WELL-{w:05d}", label=7,
                             event=j < n_events[w]))
    rows += [dict(file="SIMULATED_00001", source="simulated", well=None, label=6, event=False),
             dict(file="DRAWN_00001", source="drawn", well=None, label=7, event=True)]
    return pd.DataFrame(rows)


def test_calibration_well_has_events_and_never_the_dominant_well():
    idx = _toy()
    for f in make_folds(idx):
        calib = idx[idx.file.isin(f["calib"])]
        assert calib.event.any()                      # alarm threshold needs positives
        assert "WELL-00001" not in set(calib.well)    # dominant well stays in train


def test_no_well_leak_and_synthetic_train_only():
    idx = _toy()
    folds = make_folds(idx)
    assert len(folds) == 4
    well_of = idx.set_index("file").well
    for f in folds:
        sets = {s: set(well_of[f[s]].dropna()) for s in ("train", "calib", "test")}
        assert not (sets["train"] & sets["test"]) and not (sets["train"] & sets["calib"]) and not (sets["calib"] & sets["test"])
        assert {"SIMULATED_00001", "DRAWN_00001"} <= set(f["train"])
        assert not any(x.startswith(("SIMULATED", "DRAWN")) for x in f["calib"] + f["test"])
        assert len(f["train"]) + len(f["calib"]) + len(f["test"]) == len(idx)  # nothing dropped
