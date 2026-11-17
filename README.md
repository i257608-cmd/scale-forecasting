# scale-forecasting

Early forecasting of mineral scale build-up in production wells (Petrobras 3W v2.0) with
PI-PatchSurv, a physics-informed patch Transformer with a discrete-time survival head.

MS AI research project, FAST-NUCES — Phase 3: implementation and experimental evaluation.

## Setup (macOS, Apple Silicon)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## Layout

| Path | Content |
|---|---|
| `src/data/` | 3W download (`download_3w.py`), 1 Hz → 1 min cleaning, masks, RI*, labels (`preprocess.py`), well-grouped LOWO and v1 → v2 splits (`splits.py`) |
| `src/features/` | rolling physics features and survival windows (`build.py`); PHREEQC saturation-index prior from USGS brines (`si.py`) |
| `src/models/` | tabular baselines B0–B2 (`baselines.py`), sequence baselines B3–B8i and PI-PatchSurv (`seq.py`), tuned-parameter lookup (`tuned.py`) |
| `src/train/` | training loop, survival loss, L_mono, early stopping (`trainer.py`) |
| `src/eval/` | window, event, false-alarm and threshold metrics (`metrics.py`) |
| `experiments/` | `e1_baselines.py` (B0–B2), `e1_sequence.py` (B3–B8i, PI-PatchSurv, E2 ablations), `b9_lstm_ae.py` (B9 detector), `e1_tune.py` (Optuna), `e3_robustness.py`, `e3_resampling.py`, `analyze.py` (all final tables, statistics and figures) |
| `scripts/` | `phase2_queue.sh`: the run order used for the full Phase 2 methodology |
| `notebooks/` | `01_EDA.ipynb` |
| `results/` | `tables/`, `figures/`, `runs/` (logs), `preds/` (saved predictions), `tuning/` (Optuna studies) |
| `docs/` | `Phase3_Report.md`, `Step_by_Step_Guide.md`, `Approach_and_Best_Practices.md` |

Data (`data/`) is not committed. See `CLAUDE.md` for full project context.
