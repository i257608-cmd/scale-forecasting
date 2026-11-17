import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.seq import build  # noqa: E402
from src.train.trainer import surv_nll  # noqa: E402


def test_all_models_shapes_params_and_monotone_survival_cdf():
    x, phi = torch.randn(4, 360, 19), torch.randn(4, 36)
    for name in ["B3", "B4", "B5", "B6", "B7", "B8", "B8i", "PIPS", "PIPS-phys", "PIPS-surv", "PIPS-patch"]:
        m = build(name, 19, 36, 7)
        F = m.head.cdf(m(x, phi))
        assert F.shape == (4, 6), name
        assert sum(p.numel() for p in m.parameters()) < 1_000_000, name
        if m.head.kind == "surv":
            assert (F[:, 1:] >= F[:, :-1] - 1e-6).all()     # P(within 24h) >= P(within 6h)


def test_surv_nll_prefers_correct_bin():
    ev, b, w = torch.tensor([1.0]), torch.tensor([2]), torch.tensor(1.0)
    good = torch.tensor([[-5.0, -5.0, 5.0, 0, 0, 0, 0]])   # hazard fires in bin 2
    bad = torch.tensor([[5.0, -5.0, -5.0, 0, 0, 0, 0]])    # fires in bin 0
    assert surv_nll(good, ev, b, w) < surv_nll(bad, ev, b, w)
    # censored in bin 2: high hazards before bin 2 are penalised, after are not
    assert surv_nll(bad, torch.tensor([0.0]), b, w) > surv_nll(good, torch.tensor([0.0]), b, w)


def test_folddata_keeps_same_named_instances_of_different_classes_apart(tmp_path, monkeypatch):
    # 3W reuses synthetic file names across class folders (DRAWN_00001 is class 1 AND class 7).
    import numpy as np
    import pandas as pd
    import src.train.trainer as T
    monkeypatch.setattr(T, "FEAT_DIR", tmp_path)
    for label, level in ((1, 1.0), (7, 5.0)):
        d = pd.DataFrame({c: np.full(400, level, np.float32) for c in T.VALUE_COLS})
        d[[c for c in T.SEQ_COLS if c.startswith("m_")]] = 0.0
        (tmp_path / str(label)).mkdir()
        d.to_parquet(tmp_path / str(label) / "DRAWN_00001.parquet")
    tab = pd.DataFrame(dict(file=["DRAWN_00001.parquet"] * 2, label=[1, 7], end_pos=[399, 399]))
    data = T.FoldData(tab, ["DRAWN_00001.parquet"])
    x = data.x(T.ikey(tab), tab.end_pos.to_numpy())
    assert x[0, -1, 0] != x[1, -1, 0]          # each window reads its own class's signal
