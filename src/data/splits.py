"""Well-grouped leave-one-well-out (LOWO) folds (CLAUDE.md "Proposed model" §3 + repo rules).

Fold k:
  test        = every REAL instance of scaling well k  + one group of non-scaling wells
  calibration = every REAL instance of the next scaling well (conformal + alarm threshold τ)
                + another group of non-scaling wells
  train       = all remaining REAL instances + ALL simulated + ALL hand-drawn

WHY each rule:
  * grouped by well -> no window of a test well ever leaks into training (G3: unseen-well eval).
  * non-scaling wells also rotate into test/calibration -> false alarms per well-day are measured
    on wells the model never saw, and τ is tuned on calibration wells, never on test.
  * simulated / hand-drawn never leave train (they have no well and are not real evidence).
  * well 23 has no +5 % event (all censored) -> it is still a test fold: it measures false alarms
    on a well that scales slowly, which is an honest, useful negative case.
"""

from __future__ import annotations

import pandas as pd


def make_folds(idx: pd.DataFrame, scaling_label: int = 7) -> list[dict[str, list[str]]]:
    """idx = data/processed/instances.csv. Returns list of {train, calib, test: [file, ...]}."""
    real = idx[idx.source == "real"]
    synth = idx[idx.source != "real"].file.tolist()
    scaling_wells = sorted(real[real.label == scaling_label].well.unique())
    other_wells = sorted(set(real.well) - set(scaling_wells))
    k = len(scaling_wells)
    # Deterministic round-robin groups of non-scaling wells (sorted -> reproducible, no seed needed).
    groups = [other_wells[i::k] for i in range(k)]

    # Calibration-well candidates. WHY: calibration needs >= 1 event (alarm threshold, conformal),
    # so event-free well 23 is excluded; the well with most events (24: 17/24) is excluded too,
    # otherwise one fold's training set would lose >70 % of all real events.
    ev = real[real.label == scaling_label].groupby("well").event.sum()
    calib_pool = [w for w in scaling_wells if ev[w] > 0 and w != ev.idxmax()]

    folds = []
    for i, w in enumerate(scaling_wells):
        test_wells = {w, *groups[i]}
        # next pool well after w in sorted order (cyclic), never the test well itself
        cw = next(c for c in sorted(calib_pool, key=lambda c: (c <= w, c)) if c != w)
        calib_wells = {cw, *groups[(i + 1) % k]}
        files = lambda wells: real[real.well.isin(wells)].file.tolist()  # noqa: E731
        folds.append(dict(
            test_well=w,
            test=files(test_wells),
            calib=files(calib_wells),
            train=files(set(real.well) - test_wells - calib_wells) + synth,
        ))
    return folds


def quick(folds: list[dict]) -> list[dict]:
    """--quick mode (CLAUDE.md hardware note): 2 folds, the two scaling wells with most events."""
    return [f for f in folds if f["test_well"] in ("WELL-00022", "WELL-00024")]


if __name__ == "__main__":
    from pathlib import Path
    idx = pd.read_csv(Path(__file__).resolve().parents[2] / "data" / "processed" / "instances.csv")
    for f in make_folds(idx):
        sub = lambda s: idx[idx.file.isin(f[s])]  # noqa: E731
        print(f"test {f['test_well']}: " + " | ".join(
            f"{s} n={len(f[s])} wells={sub(s).well.nunique()} scaling_events={int(sub(s).event.sum())}"
            for s in ("train", "calib", "test")))
