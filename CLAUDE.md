# CLAUDE.md — project context (read this first)

This file hands the project over from the planning chat (Claude Cowork) to Claude Code in VS Code.
Everything below was decided in Phases 0–2. Do not redesign without checking with the student.

## Who / what
- Course: Advanced Artificial Intelligence (MS AI), FAST-NUCES Islamabad. Instructor: Sir Atif Saeed.
- Group: Muhammad Haseeb (25I-7608), Muavia Shakeel (25I-76021).
- GitHub: https://github.com/i257608-cmd — repo `scale-forecasting` (private).
- Current task: **Phase 3 — Research Implementation and Experimental Evaluation** (scaling only).
- Liquid loading and water breakthrough are OUT of the MS scope (separate Mari/PetroGear industry track).

## Topic
Physics-informed deep learning for **early forecasting of mineral scale build-up** in oil and gas
production wells from multivariate sensor time series.

## Dataset
- **Petrobras 3W v2.0.0** (Vargas et al., Sci. Data 13:949, 2026, doi 10.1038/s41597-026-07225-z), CC BY 4.0.
  GitHub: https://github.com/petrobras/3W . Parquet files, 1 Hz, 27 variables, 2,228 instances
  (1,119 real / 1,089 simulated / 20 hand-drawn), 42 real wells.
- Target: **class 7 "Scaling in PCK"** (production choke): 36 real + 10 hand-drawn, 0 simulated.
  Class 0 = normal; classes 1–6, 8, 9 = negatives/confounders
  (class 6 "Quick restriction in PCK" is the main look-alike).
- **EDA facts (notebooks/01_EDA.ipynb, 3W commit 826fec4) — supersede earlier assumptions:**
  - Class 7 DOES have a transient label **107** (TRANSIENT_OFFSET = 100). 31/36 real instances go
    0 → 107 and never reach steady 7. First hour of every instance is unlabelled (NaN).
  - Real class-7 instances come from only **6 wells**: 22 (8), 23 (4), 24 (19), 21 (2), 6 (2), 1 (1).
    Only wells 1 and 6 have class-0 files; 40 distinct real wells appear in filenames.
  - Pre-onset labelled-normal time: median 6.5 h, max 28 h → forecasting the *start* of scaling is
    limited to hours.
  - **P-JUS-CKP and T-MON-CKP are missing in 100 %** of real class-7 instances, ABER-CKP in 78 %
    → ΔP across choke and ΔT across choke cannot be computed. P-MON-CKP is always present and rises
    after onset in 89 % of instances (median +6.6 %); T-JUS-CKP falls in 56 % (Joule–Thomson).
- Frozen/missing values are deliberately left untreated in v2 → handle with masks.
- Optional chemistry prior: USGS National Produced Waters Geochemical DB v2.3 (doi 10.5066/F7J964W8)
  + PHREEQC (phreeqpython) → SI(T, P) lookup, used only as a relative driver.

## Proposed model: PI-PatchSurv (Phase 2)
1. Preprocess: resample 1 Hz → 1 min mean; frozen (>30 min constant while open) / missing → mask
   channels; forward-fill gaps ≤ 10 min; drop vars missing in >50 % of real instances;
   robust scaling fitted on training folds only.
2. Labels (**revised after EDA, approved by student 2026-11-14**): forecast *severity*, not start.
   Onset t0 = first timestamp labelled 107 or 7. Event time t_E = first time RI* ≥ **1.05**
   (scaling has grown 5 %). For windows ending at t < t_E, τ = t_E − t binned into K horizon bins
   spanning ~1 h to ~5 days (well 24 crosses in hours, wells 22/23 in days); instances that never
   cross and normal series are right-censored; windows after t_E dropped.
3. Splits: grouped by well (leave-one-well-out over wells with real scaling), time order kept inside
   wells; simulated + hand-drawn instances in TRAINING ONLY; one well group held out for calibration.
4. Features (**revised**): restriction proxy **RI* = P-MON-CKP / median(P-MON-CKP before t0)** of
   the same instance (self-referenced, because P-JUS-CKP/ABER-CKP are missing and 4/6 scaling
   wells have no normal files; for normal instances use a trailing median of the instance's
   first hours); T-JUS-CKP (JT-cooling signal); P-TPT; SI trajectory; rolling mean/slope/std
   (10, 60, 360 min). Original RI = ΔP_PCK / ΔP_ref(a) is NOT computable on 3W v2 class 7.
   Limitation to report: RI* does not control for choke-opening changes → check false alarms on class 6.
5. Model: patch Transformer encoder (P = 16, stride 8, channel-independent, masks as channels)
   → z_s; physics MLP (2 × 64, GELU, dropout) on φ_t → z_p; gated fusion
   g = σ(W_g[z_s; z_p]), z = g⊙z_s + (1−g)⊙z_p; discrete-time survival head h_k = σ(w_kᵀz + b_k),
   F(k) = 1 − Π_{j≤k}(1 − h_j).
6. Loss: class-weighted censor-aware survival NLL + λ·L_mono (penalise drops in F between
   consecutive windows of the same well when no intervention/choke change).
7. Uncertainty: deep ensemble (5 seeds) + split-conformal calibration on calibration wells.
8. Alarm: F̂(k*) > τ, τ chosen on validation for ≤ 1 false alarm per well-month.

## Baselines (B0–B9)
B0 threshold rule on RI · B1 logistic regression · B2 XGBoost/LightGBM · B3 LSTM · B4 GRU ·
B5 CNN-LSTM · B6 TCN · B7 Transformer-LSTM (Lai 2026) · B8 PatchTST / iTransformer ·
B9 LSTM-autoencoder detector (detection reference, Lopes 2025 / Aranha 2025).

## Experiments
- E1 baseline comparison (same folds, equal Optuna budget)
- E2 ablation: −physics branch, −L_mono, −survival head (per-horizon classifiers), −synthetic data,
  −patching, −ensemble/conformal
- E3 robustness: Gaussian noise 1–5 %, sensor dropout 1–3, injected frozen segments, resampling 30 s–5 min
- E4 generalisation: LOWO; train on 21 v1 wells → test on 21 new v2 wells; false alarms on class 6
- E5 horizon sensitivity
- Stats: Friedman–Nemenyi, Wilcoxon signed-rank, Holm correction, bootstrap 95 % CI

## Metrics (no accuracy / R² as primary)
Event recall & precision, false alarms per well-day, lead time (median, IQR), ΔT; AUPRC & AUROC per
horizon; C-index; integrated Brier; ECE + reliability plot; PICP & interval width; skill vs B0.

## Known risk + fallback — RESOLVED 2026-11-14
EDA showed little pre-onset data (median 6.5 h) and missing ΔP sensors → fallback adopted:
survival target = time until RI* (P-MON-CKP proxy) crosses +5 %. See Proposed model §2 and §4.
Open risk: RI* behaviour on class 0 / class 6 (false alarms) not yet checked.

## Hardware
MacBook Air M4, no CUDA. Use torch `mps` device, float32 only, `PYTORCH_ENABLE_MPS_FALLBACK=1`.
Keep models small (<1 M params). Provide `--quick` (2 folds, 1 seed) and full modes. Checkpoint every
run. Heavy tuning may move to Kaggle/Colab (code must also run on CUDA/CPU).

## Phase 3 deliverables (instructor brief, all 7 required)
1. Implementation + documented changes from class feedback (TODO: student to paste feedback below)
2. Dataset, preprocessing, feature engineering, setup, splits, implementation details
3. Baseline evaluation with all metrics
4. Graphs (PR curves, reliability, lead-time histograms, ablation bars, horizon curves, example forecasts)
5. Comparison with published work (Lopes 2025, Aranha 2025, Lai 2026 …) + fairness limitations
6. Ablation studies
7. Results analysis & discussion vs gaps G1–G4
Plus (student request): `docs/Step_by_Step_Guide.md`, `docs/Approach_and_Best_Practices.md`,
heavily commented code explaining WHY.

## Research gaps (Phase 1)
G1 no timed forecasting of scaling (max lead 6 d in literature) · G2 chemistry + sensors never fused ·
G3 no unseen-well evaluation · G4 rare events, private data, no uncertainty.

## Rules for working in this repo
- Never split windows randomly; always group by well.
- Never tune or choose thresholds on test wells.
- Hand-drawn / simulated instances never enter validation or test.
- Fixed seeds; log every run (config + metrics) to results/.
- Report honest negative results.

## Progress log
- **Step 1 (done 2026-11-14):** `src/data/download_3w.py`, `notebooks/01_EDA.ipynb` → target revised (see above).
- **Step 2 (done 2026-11-14):** `src/data/preprocess.py` → `data/processed/<label>/*.parquet` +
  `instances.csv`; `src/data/splits.py` (6 LOWO folds, one per scaling well); tests in `tests/`.
  - Channels kept (≤50 % missing in real instances): P-MON-CKP, T-JUS-CKP, P-TPT, T-TPT, P-ANULAR,
    P-PDG, P-JUS-CKGL, T-PDG, ESTADO-SDV-P (+ mask channel each).
  - Feature RI* uses a causal baseline = median of the instance's first 60 min (always pre-onset);
    label RI* uses the pre-t0 median. RI* ≈ 1.008 just before onset (median) → causal baseline OK.
  - Events (+5 %): 24/36 real (well 24: 17, 22: 3, 6: 2, 1: 1, 21: 1, 23: 0), 7/10 hand-drawn.
    τ available before event: median 12 h, max 147 h. Proposed horizon bins (h): 0–1, 1–3, 3–6,
    6–12, 12–24, 24–72, >72.
  - Calibration well = next scaling well with ≥1 event, never well 24 (keeps 17 events in train).
  - False-alarm check (`results/tables/ri_star_false_alarm_check.csv`): raw RI* ≥ 1.05 on normal
    class 0 = 0.014 / day (1/308 instances) but 78 % of slugging (3), 86 % of hydrate-in-production-line (8),
    40 % of class 9 instances also cross → a threshold rule (B0) will false-alarm on these; the
    model must use shape (slow monotonic rise, T-JUS-CKP drop). Real class 6: only 3 usable files,
    1 crosses but within minutes (scaling: median 7 h).
  - Values with |x| > 1e9 are treated as missing (garbage P-PDG = 1.18e42 Pa in WELL-00006).
- **Step 3 (done 2026-11-14):** `src/features/build.py` → `data/features/<label>/*.parquet`
  (channels, masks, RI*, rolling mean/std/slope 10/60/360 min on RI*, P-MON-CKP, T-JUS-CKP, P-TPT)
  + `data/features/windows.parquet`: 109,547 windows, stride 10 min, start after the 60-min baseline,
  dropped at/after t_E; columns tau_min, censor_min, k (bin, −1 = censored). Real class-7 windows per
  bin 0..6: 142 / 278 / 412 / 617 / 490 / 887 / 856, censored 5,032. Robust scaling deliberately
  left to training code (fit per fold on train wells). SI (chemistry) feature not built yet.
- **Step 4 (done 2026-11-14): E1 part 1, baselines B0–B2** (`src/eval/metrics.py`,
  `src/models/baselines.py`, `experiments/e1_baselines.py`; results in
  `results/tables/e1_baselines_*.csv`, run log in `results/runs/`).
  - Alarm rule (approved): hit = first alarm before RI* reaches +5 %; alarm horizon 24 h;
    budget ≤ 1 false alarm per well-month; FA counted only on non-class-7 instances.
  - Threshold chosen on OUT-OF-FOLD scores over training wells (inner 5-fold by well). The single
    calibration well (1 event) gave thr 0.9999 → recall 0; calibration wells kept for conformal.
  - B2 is HistGradientBoosting (sklearn), not LightGBM: LightGBM needs system libomp (absent).
  - Pooled over 6 folds (24 events, 248 negative well-days):

    | model | AUPRC@24h | AUROC@24h | recall (deployed thr) | FA/day | recall @ ≤1 FA/month (threshold-free) |
    |---|---|---|---|---|---|
    | B0 RI* threshold | 0.53 | 0.83 | 0.00 | 0.032 | 0.00 (0.25 @ 1 FA/day) |
    | B1 logistic | 0.50 | 0.74 | 0.04 | 0.012 | 0.04 |
    | B2 GBDT | 0.67 | 0.90 | 0.13 | 0.476 | 0.67 |

  - Key finding: ranking is good (B2 recall 0.67 at budget if the threshold were right) but the
    threshold does not transfer across wells (score scales differ per well) → per-well
    calibration / conformal is the main gap PI-PatchSurv must close. B2 fails on slow-scaling
    well 22 (AUPRC 0.15; training dominated by fast well 24) where plain B0 ranks well (0.69).
- **Next:** sequence baselines B3–B6 + PI-PatchSurv (torch, MPS), then per-well calibration.

## Class feedback (from 3-minute presentation) — TODO
that was good. 

## Phase 3 due date — TODO
Due Nov 15, 11:59 PM