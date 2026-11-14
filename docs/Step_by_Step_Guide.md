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
pytest -q tests                           # 12 tests, ~3 s — must pass before anything else
```

LightGBM needs the system `libomp` library on macOS. It is not required: baseline B2 uses
scikit-learn's HistGradientBoosting instead (same algorithm family).

## 1. Download Petrobras 3W v2.0.0 (≈10 min, ≈5 GB)

```bash
python src/data/download_3w.py
```

*What it does:* sparse, shallow git checkout of only the `dataset/` folder of
github.com/petrobras/3W at the pinned commit `826fec4` → `data/3w/dataset/<label>/*.parquet`.
*Why pinned:* the dataset is versioned; pinning the commit makes every number in the report
reproducible. The script checks the version (2.0.0) and the counts 1,119 real / 1,089 simulated /
20 hand-drawn instances, and stops if they differ.

## 2. Exploratory data analysis (optional, ≈5 min)

Open `notebooks/01_EDA.ipynb` and run all cells. It lists every class-7 ("Scaling in PCK")
instance with its well, duration and pre-onset normal time, shows sensor missingness, and
justifies the target revision (see `docs/Approach_and_Best_Practices.md` §2). Outputs go to
`results/tables/class7_*.csv` and `results/figures/eda_*.png`.

## 3. Preprocessing and labels (≈15 min)

```bash
python src/data/preprocess.py
```

For each of the 2,228 instances: 1 Hz → 1-minute means; frozen (>30 min constant) and missing
values → mask channels; gaps ≤10 min forward-filled; |x| > 1e9 treated as missing; hand-drawn
pressures converted bar → Pa; RI* computed; onset t0 and event time t_E (RI* ≥ 1.05) found.
Output: `data/processed/<label>/*.parquet` and `data/processed/instances.csv`.

Check the folds: `python src/data/splits.py` prints 6 leave-one-well-out folds (one per scaling well).

## 4. Features and survival windows (≈10 min)

```bash
python src/features/build.py
```

Rolling mean / std / slope over 10, 60 and 360 minutes for RI*, P-MON-CKP, T-JUS-CKP and P-TPT,
plus one row per 10-minute window with its time-to-event τ, censoring time and horizon bin.
Output: `data/features/<label>/*.parquet`, `data/features/windows.parquet` (109,547 windows).

## 5. E1 — tabular baselines B0–B2 (≈40 min)

```bash
python experiments/e1_baselines.py            # all 6 folds
python experiments/e1_baselines.py --quick    # 2 folds (wells 22 and 24), smoke test
```

## 6. E1/E2 — sequence models and PI-PatchSurv (≈1.5 h per seed for all 5 models)

```bash
python experiments/e1_sequence.py --models B3 B4 B5 B6 --tag main            # sequence baselines
python experiments/e1_sequence.py --models PIPS --seeds 0 1 2 3 4 --tag main # 5-seed deep ensemble
python experiments/e1_sequence.py --models PIPS-phys PIPS-mono PIPS-surv PIPS-patch --tag e2  # ablations
python experiments/e1_sequence.py --quick --models PIPS                      # 2-fold smoke test
```

Each run writes a checkpoint (`results/checkpoints/`), its predictions (`results/preds/`), the
per-epoch training log (`results/runs/*_log.json`) and the fold tables (`results/tables/`).

## 7. E3 — robustness (≈20 min, inference only)

```bash
python experiments/e3_robustness.py --models PIPS B3
```

## 8. Final analysis: deployed alarms, calibration, conformal, statistics, figures (≈10 min)

```bash
python experiments/analyze.py
```

Reads every saved prediction (no training) and writes `results/tables/final_*.csv` and
`results/figures/*.png`. This one command regenerates every number and figure in
`docs/Phase3_Report.md`.

## 9. Where to find what

| Question | File |
|---|---|
| Main comparison table | `results/tables/final_main.csv` |
| Deployed alarm performance (recall, FA/day, lead time) | `results/tables/final_alarm.csv` |
| False alarms by confounder class (E4) | `results/tables/final_fa_by_class.csv` |
| Uncertainty intervals (PICP, width) | `results/tables/final_conformal.csv` |
| Significance tests | `results/tables/final_stats.csv`, `final_mean_ranks.csv` |
| Robustness (E3) | `results/tables/e3_robustness.csv` |
| Figures | `results/figures/` |
| Full report | `docs/Phase3_Report.md` |

## Rules that the code enforces (do not break them when extending)

1. Splits are grouped by well. A window of a test well never enters training.
2. Thresholds are chosen on training (out-of-fold) or calibration wells — never on test wells.
3. Simulated and hand-drawn instances are used for training only.
4. Seeds are fixed; every run is logged under `results/`.
