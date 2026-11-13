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

| Folder | Content |
|---|---|
| `src/data` | 3W download, loading, cleaning, masks, onset labels, well-grouped splits |
| `src/features` | restriction index, ΔT, SI lookup, rolling statistics |
| `src/models` | baselines B0–B9 and PI-PatchSurv |
| `src/train` | training loop, tuning, ensemble, conformal calibration |
| `src/eval` | event, probabilistic and calibration metrics |
| `experiments` | E1–E5 run scripts |
| `notebooks` | EDA and result figures |
| `results` | metrics tables and figures |
| `docs` | Phase 3 report, step-by-step guide, approach and best practices |

Data (`data/`) is not committed. See `CLAUDE.md` for full project context.
