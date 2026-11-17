# Step-by-Step Guide — reproducing every Phase 3 result

Audience: anyone (group members, instructor, examiner) who wants to re-run the project from an empty
folder on a laptop. Every command is run from the repository root. Times are for a MacBook Air M4
(no CUDA); the code also runs on CUDA or CPU.

---

## 0. Environment (5 min)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTORCH_ENABLE_MPS_FALLBACK=1      # Apple-Silicon only: CPU fallback for missing MPS ops
pytest -q tests                           # 14 tests, ~3 s — must pass before anything else
```

LightGBM needs the system `libomp` library on macOS. It is not required: baseline B2 uses
scikit-learn's HistGradientBoosting instead (same algorithm family).

## 1. Download Petrobras 3W v2.0.0 (≈10 min, ≈5 GB)

```bash
python src/data/download_3w.py
```

Sparse, shallow git checkout of the `dataset/` folder of github.com/petrobras/3W at the pinned commit
`826fec4` → `data/3w/dataset/<label>/*.parquet`. The script checks the version (2.0.0) and the counts
1,119 real / 1,089 simulated / 20 hand-drawn, and stops if they differ.

## 2. Exploratory data analysis (optional, ≈5 min)

Open `notebooks/01_EDA.ipynb` and run all cells (class-7 instances, wells, durations, pre-onset time,
missingness; justifies the target revision). Outputs: `results/tables/class7_*.csv`,
`results/figures/eda_*.png`.

## 3. Preprocessing and labels (≈15 min)

```bash
python src/data/preprocess.py
python src/data/splits.py                 # prints the 6 leave-one-well-out folds and the v1 -> v2 fold
```

1 Hz → 1-minute means; frozen / missing → mask channels; gaps ≤ 10 min forward-filled; |x| > 1e9 →
missing; hand-drawn pressures bar → Pa; RI*, onset t0 and event time t_E. Output:
`data/processed/<label>/*.parquet`, `data/processed/instances.csv`.

## 4. Features, survival windows and the chemistry prior (≈20 min)

```bash
python src/features/build.py              # rolling features + 109,547 windows (data/features/)
python src/features/si.py                 # ~10 min: PHREEQC SI grid (USGS PWGD v3.0 brines) + SI per window
```

## 5. E1 — tabular baselines B0–B2 (≈40 min)

```bash
python experiments/e1_baselines.py                     # B0 B1 B2, all 6 folds
python experiments/e1_baselines.py --models B2-SI      # + chemistry prior
python experiments/e1_baselines.py --quick             # 2-fold smoke test
```

## 6. E1/E2 — sequence models and PI-PatchSurv (≈1.5 h per model-seed over 6 folds)

```bash
python experiments/e1_sequence.py --models B3 B4 B5 B6 B7 B8 B8i --tag main      # sequence baselines
python experiments/b9_lstm_ae.py                                                  # B9 detector
python experiments/e1_sequence.py --models PIPS --seeds 0 1 2 3 4 --tag main_pips # 5-seed ensemble
python experiments/e1_sequence.py --models PIPS-phys PIPS-mono PIPS-surv PIPS-patch PIPS-nosyn --tag e2
python experiments/e1_sequence.py --models PIPS-nopatch --tag e2_nopatch          # "-patching" (slow)
python experiments/e1_sequence.py --models PIPS-SI --seeds 0 1 2 3 4 --tag si     # + chemistry prior
python experiments/e1_sequence.py --quick --models PIPS                           # 2-fold smoke test
```

Each run writes a checkpoint (`results/checkpoints/`), predictions (`results/preds/`), the per-epoch
training log (`results/runs/*_log.json`) and fold tables (`results/tables/`). `--resume` skips folds
whose predictions already exist.

## 7. E1 — Optuna tuning with an equal budget (≈30 h in total)

```bash
python experiments/e1_tune.py --models B1 B2 B3 B4 --trials 20          # locally (M4)
# B5-B9 and PIPS were tuned on Kaggle (2 x T4): scripts/kaggle/tune.py + kernel-metadata.json
python experiments/e1_baselines.py --models B1-tuned B2-tuned
python experiments/b9_lstm_ae.py --tuned
python experiments/e1_sequence.py --models B3-tuned B4-tuned B5-tuned B6-tuned B7-tuned B8-tuned B8i-tuned --tag tuned
python experiments/e1_sequence.py --models PIPS-tuned --seeds 0 1 2 3 4 --tag tuned_pips
```

Tuning is nested per fold and scored only on calibration wells (`results/tuning/<model>.json`; Kaggle
logs in `results/tuning/kaggle_logs/`).

## 8. E3 — robustness and resampling (≈1 h, inference only)

```bash
python experiments/e3_robustness.py --models PIPS B3     # noise, sensor dropout, frozen sensor
python experiments/e3_resampling.py --models PIPS B3     # 30 s / 1 / 2 / 5 min (features rebuilt per rate)
```

## 9. E4 — v1 → v2 generalisation (≈2 h)

```bash
python experiments/e1_sequence.py --split v1v2 --models B3 B4 B5 B6 B7 B8 B8i --tag v1v2
python experiments/e1_sequence.py --split v1v2 --models PIPS --seeds 0 1 2 3 4 --tag v1v2_pips
python experiments/b9_lstm_ae.py --split v1v2
```

## 10. Final analysis (≈1.5 h, no training)

```bash
python experiments/analyze.py              # LOWO: final_*.csv + all figures
python experiments/analyze.py --split v1v2 # E4: e4_v1v2_*.csv
```

Reads every saved prediction and writes `results/tables/final_*.csv`, `e4_v1v2_*.csv` and
`results/figures/*.png`. These two commands regenerate every number and figure in
`docs/Phase3_Report.md`. The whole chain of steps 5–10 is scripted in `scripts/phase2_queue.sh`
(and `scripts/finish_tuned.sh` for the part that waited on Kaggle).

## 11. Where to find what

| Question | File |
|---|---|
| Main comparison (AUPRC/AUROC per horizon, C-index, Brier, IBS, ECE, skill vs B0, recall at budgets) | `results/tables/final_main.csv` |
| Deployed alarms (recall, FA/day, event precision, lead time) | `results/tables/final_alarm.csv` |
| False alarms by confounder class (E4) | `results/tables/final_fa_by_class.csv` |
| Uncertainty intervals (PICP, width, ΔT) | `results/tables/final_conformal.csv` |
| Significance tests (bootstrap, Wilcoxon, Holm, Friedman, Nemenyi) | `results/tables/final_stats.csv`, `final_mean_ranks.csv` |
| Robustness / resampling (E3) | `results/tables/e3_robustness.csv`, `e3_resampling.csv` |
| v1 → v2 (E4) | `results/tables/e4_v1v2_*.csv` |
| Tuned hyper-parameters | `results/tuning/*.json` |
| Figures | `results/figures/` |
| Full report | `docs/Phase3_Report.md` |

## Rules that the code enforces (do not break them when extending)

1. Splits are grouped by well. A window of a test well never enters training.
2. Thresholds and hyper-parameters are chosen on training (out-of-fold) or calibration wells — never on
   test wells.
3. Simulated and hand-drawn instances are used for training only.
4. Instances are identified by `label/file`, never by file name alone (3W repeats synthetic names
   across class folders).
5. Seeds are fixed; every run is logged under `results/`.
