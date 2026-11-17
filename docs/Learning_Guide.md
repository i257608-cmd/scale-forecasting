# Learning Guide — machine learning through this project

**For:** Muhammad Haseeb and Muavia Shakeel. Use it to understand every concept, file and decision in
`scale-forecasting`, and to explain the code to the instructor.

**How to read it:**
- Part 1 is the big picture.
- Part 2 teaches the ML concepts, each tied to the place in this project where it is used.
- Part 3 walks through every code file in pipeline order.
- Part 4 covers the tools.
- Part 5 is the project diary: what happened, what changed, what did not work and what was modified.
- Part 6 covers how to read the results.
- Part 7 is viva questions with short answers.
- Part 8 has exercises.

The companion documents are:
- `docs/Step_by_Step_Guide.md`: the exact commands to reproduce the results.
- `docs/Approach_and_Best_Practices.md`: why the method looks the way it does.
- `docs/Phase3_Report.md`: the results report.
- `CLAUDE.md`: the spec and progress log.

---

## Part 1 — The big picture

### 1.1 The problem

In oil and gas wells, minerals such as calcite (CaCO₃) and barite (BaSO₄) come out of the produced
water and stick to the equipment. This is **scale**. On the production choke (PCK), which is a valve
on the wellhead, scale narrows the opening:
- The flow is restricted.
- The pressure upstream of the choke rises.
- Production is lost.
- Eventually someone must stop the well and clean it.

If we can **forecast** scale build-up hours or days before it becomes serious, the operator can plan
an inhibitor squeeze or a cleaning job instead of reacting to it.

### 1.2 Detection vs prediction vs forecasting

| Task | Question it answers | Example in this project |
|---|---|---|
| **Detection** | "Is something abnormal *now*?" | B9 LSTM-autoencoder: flags windows that do not look normal |
| **Classification** | "Which fault is this?" | Most 3W papers: label a window as class 0–9 |
| **Forecasting (what we do)** | "*How long until* the problem reaches a given severity?" | PI-PatchSurv: P(scaling reaches +5 % within 1, 3, 6, 12, 24, 72 h) |

Most published 3W work is detection or classification. Research gap **G1** is that nobody forecasts
scaling with a time horizon. This is why the project uses **survival analysis** (Part 2.6) rather
than a plain classifier.

### 1.3 The pipeline in one picture

```
Petrobras 3W v2.0.0 (Parquet, 1 Hz, 27 sensors, 2,228 instances)
        │  src/data/download_3w.py      download a pinned version of the data
        ▼
notebooks/01_EDA.ipynb                  look at the data -> decide target, channels, rules
        │
        ▼  src/data/preprocess.py       1 Hz -> 1 min, masks, RI*, onset t0, event time t_E
data/processed/<class>/<file>.parquet + instances.csv
        │
        ▼  src/data/splits.py           leave-one-well-out folds (train / calibration / test)
        ▼  src/features/build.py        rolling features + one "window" every 10 min + survival target
        ▼  src/features/si.py           (optional) chemistry prior: saturation index from PHREEQC
data/features/...  windows.parquet (109,547 windows)
        │
        ├── src/models/baselines.py     B0 rule, B1 logistic regression, B2 gradient boosting
        ├── src/models/seq.py           B3 LSTM … B8i iTransformer, PI-PatchSurv
        └── experiments/b9_lstm_ae.py   B9 autoencoder detector
        │
        ▼  src/train/trainer.py         training loop (scaling, loss, early stopping, prediction)
        ▼  experiments/e1_*.py          run every model on every fold -> results/preds/*.npz
        ▼  experiments/e1_tune.py       Optuna hyper-parameter search (Mac + Kaggle GPUs)
        ▼  experiments/e3_*.py          robustness: noise, sensor dropout, frozen sensors, sample rate
        ▼  src/eval/metrics.py          AUPRC, AUROC, alarms, lead time, false alarms
        ▼  experiments/analyze.py       final tables, statistics, figures -> results/
        ▼
docs/Phase3_Report.md
```

---

## Part 2 — ML concepts, explained with this project

### 2.1 Time series, sampling and resampling

A **time series** is a sequence of measurements in time order. The 3W data record 27 sensors once
per second (1 Hz).

We **resample** to 1-minute means (`to_minutes` in `preprocess.py`):
- Scaling grows over hours to days, so second-by-second detail is mostly noise.
- 1-minute data is 60× smaller, so training is 60× cheaper.

**Concept to remember:** choose the time resolution from the speed of the physical process, not
from the speed of the sensor.

### 2.2 Missing data, frozen sensors and masks

Real sensors fail in two ways:
1. They report nothing (**missing**, NaN).
2. They get stuck on a plausible value (**frozen**).

3W v2 deliberately leaves both untreated. Our handling:
- **Frozen detection:** a value that has not changed for ≥ 30 min is treated as missing (`frozen_mask`).
- **Forward fill** for short gaps (≤ 10 min): a short comms dropout is safe to bridge with the last value.
- **Mask channels** `m_<sensor>`: 1 = "this value was missing or imputed", 0 = "real reading". The
  mask is computed **before** filling, so the model always knows which values were invented.
- **Garbage values:** |x| > 1e9 becomes missing. One file had P-PDG = 1.18e42 Pa, which is impossible.

**Concept:** never let the model believe an imputed value is a real one. Masks are the standard
trick to tell it.

### 2.3 Feature engineering and physics-informed features

A **feature** is a number the model sees. We build three kinds.
- **Raw channels** (9 sensors that survive the "≤ 50 % missing" rule).
- **The physics proxy RI\*** = P-MON-CKP / (median of P-MON-CKP in the instance's first hour).
  - Scale narrows the choke, so the pressure upstream of it rises. RI* = 1.05 means "+5 % above this
    well's own normal".
  - It is **self-referenced** (divided by the well's own baseline). Each well has a different
    normal pressure, so absolute values do not transfer between wells.
- **Rolling statistics** of RI*, P-MON-CKP, T-JUS-CKP and P-TPT over 10, 60 and 360 min: mean, std
  and slope.
  - Scaling is a *slow, monotonic rise*. Slugging or hydrates *oscillate or jump*.
  - Slope and std let even a simple model separate these shapes.
- **Chemistry prior (SI):** from real brine analyses (USGS) and the PHREEQC geochemistry program we
  compute how supersaturated calcite and barite would be at the measured temperature and pressure.
  This is research gap **G2**: chemistry and sensors had never been fused.

**Causality rule:** every feature at minute *t* may only use data up to *t*. Rolling windows look
backwards. The RI* baseline uses the first hour, which is always before scaling starts. If a feature
used the future, results would look great and fail in a real control room. That is called
**leakage** (2.5).

### 2.4 Labels, windows and the prediction target

- **Onset t0:** the first minute the expert label says "scaling" (class 7, or its transient form 107).
- **Event time t_E:** the first minute after t0 where the smoothed RI* ≥ 1.05. This is the event we
  forecast.
- **Window:** "standing at minute *t*, with the last 6 h (360 min) of data, how long until t_E?"
  - A new window ends every 10 min (stride).
  - Windows at or after t_E are dropped. Once the event has happened, it is detection, not forecasting.
- **τ (tau)** = t_E − t, the time remaining until the event.
- **Horizon bins:** 0–1, 1–3, 3–6, 6–12, 12–24, 24–72 and > 72 h. The model outputs a probability
  for each.

Labels may use future knowledge (t0, t_E are decided afterwards by an expert). **Features may not.**
That asymmetry is normal and correct.

### 2.5 Train / validation (calibration) / test, cross-validation and leakage

- **Training set:** the model learns from it.
- **Validation set** (here called **calibration**): used to choose things (when to stop training,
  hyper-parameters, alarm threshold, conformal intervals).
- **Test set:** touched once, at the very end, to report performance.

**Leakage** means information from the test set reaching training or choices. Common forms, and how
this repo blocks them:

| Leak | Why it is wrong | Guard in this repo |
|---|---|---|
| Random split of windows | Neighbouring windows of the same well are almost identical, so the "test" is memorised | **Grouped by well** (`splits.py`) |
| Scaling statistics from all data | Test means and variances leak in | Scaler fitted on **training files only** (`FoldData`) |
| Threshold or hyper-parameters tuned on test | Optimistic numbers | Chosen on **calibration** wells or out-of-fold training wells |
| Synthetic data in test | Not real evidence | Simulated and hand-drawn data are used **only in training** |

**Leave-one-well-out (LOWO) cross-validation:**
- Only 6 real wells have scaling, so we make 6 folds.
- In each fold, one scaling well is the test well and another is the calibration well.
- Every result is "performance on a well the model has never seen". This is gap **G3**.

### 2.6 Survival analysis and censoring

Classification asks "yes or no?". **Survival analysis** asks "*when*?". It also handles a special
kind of missing label called **censoring**:
- Some scaling instances end before RI* reaches +5 %.
- Normal instances never have the event.
- For both, we only know "no event *until* the recording ends". That is right-censored.
- Calling them "no event ever" (0) would be wrong. Throwing them away would waste data.

**Discrete-time survival model:**
- For each bin *k* the model predicts a **hazard** h_k = P(event in bin k | no event before k).
- The probability of the event within bin *k* is F(k) = 1 − Π_{j≤k}(1 − h_j).
- F can only grow with *k*: "within 24 h" is never less likely than "within 6 h". This is
  **monotone by construction**.
- **Loss (negative log-likelihood):**
  - An event in bin b contributes log(1 − h) for every bin before b, plus log h_b.
  - A censored window only contributes "survived the bins before b" (`surv_nll` in `trainer.py`).

**Concept:** the loss function encodes what we know and what we do not. Censor-aware loss = honest
use of partial labels.

### 2.7 Class imbalance

Events are rare: only 24 real events across all wells. Most windows are negative. Without care, a
model learns "always say no" and still looks accurate. This is why **accuracy is never used here**.

The fixes used:
- **Class weights** (`class_weight="balanced"` in sklearn; `pos_weight` / `w_ev` in PyTorch): mistakes
  on rare positives cost more.
- **Synthetic data** in training (simulated and hand-drawn instances) adds more scaling examples. The
  E2 ablation "−synthetic" measures whether this helps.
- **Metrics that respect rarity:** AUPRC and recall at a fixed false-alarm rate (2.11).

### 2.8 The models, from simple to complex

Always start with simple baselines. A complex model is only worth it if it beats them.

| ID | Model | Idea in one line | File |
|---|---|---|---|
| B0 | Threshold rule | Score = 60-min mean of RI*. No learning: the "skill" reference | `baselines.py` |
| B1 | Logistic regression | Weighted sum of features → sigmoid. Linear | `baselines.py` |
| B2 | Gradient-boosted trees | Many small decision trees, each fixing the previous ones' errors | `baselines.py` |
| B3 | LSTM | Recurrent network with a memory cell. Reads the 360-min sequence step by step | `seq.py` |
| B4 | GRU | A simpler LSTM (fewer gates) | `seq.py` |
| B5 | CNN-LSTM | Convolutions find local shapes and shorten the sequence, then an LSTM models the trend | `seq.py` |
| B6 | TCN | Causal dilated convolutions. Sees 511 min of history without recurrence | `seq.py` |
| B7 | Transformer-LSTM | Self-attention for global context, then an LSTM (Lai 2026 style) | `seq.py` |
| B8 | PatchTST | Cut each sensor's series into patches, Transformer over patches | `seq.py` |
| B8i | iTransformer | One token per *sensor*; attention learns cross-sensor relations | `seq.py` |
| B9 | LSTM autoencoder | Unsupervised. Learns to reconstruct normal data; a large error means anomaly | `b9_lstm_ae.py` |
| **PIPS** | **PI-PatchSurv** (ours) | PatchTST sequence branch + physics MLP branch, gated fusion, survival head | `seq.py` |

**Key neural-network ideas used:**
- **MLP / Linear layer:** y = Wx + b. Stacked with non-linearities (GELU), it can learn curves.
- **Recurrent networks (LSTM/GRU):** keep a hidden state that is updated each time step. Good for
  order; slow on long sequences.
- **Convolution:** a small filter slides over time and detects local patterns.
  - **Dilation** skips steps, so the filter sees further back.
  - **Causal padding** (padding on the left only) means minute *t* never sees minute *t+1*.
- **Self-attention / Transformer:** every token looks at every other token and decides what is
  relevant. Positional embeddings tell it the order.
- **Patching** (PatchTST): instead of 360 one-minute tokens, use 16-minute patches with stride 8, giving
  44 tokens. Each token carries local shape, and attention becomes much cheaper.
- **Channel independence:** each sensor is encoded separately by the same shared encoder. This needs
  fewer parameters and helps with 24 events.
- **Gated fusion:**
  - The gate is g = σ(W[z_s; z_p]) and the output is z = g·z_s + (1 − g)·z_p.
  - The model learns per sample how much to trust the sensor-sequence branch versus the physics branch.
  - The average gate (≈ 0.5) tells us how much the physics branch is actually used.
- **Dropout, weight decay, gradient clipping:** regularisers that stop over-fitting and exploding
  updates.

### 2.9 Training: loss, optimiser, epochs, early stopping, over-fitting

- **Loss:** the number training minimises. Here it is BCE (binary cross-entropy) per horizon for
  baselines, and survival NLL plus a monotonicity penalty λ·L_mono for PIPS.
- **Optimiser AdamW:** a gradient-descent variant with adaptive step sizes and weight decay.
  - **Learning rate (lr):** the step size.
  - **Batch size (bs):** how many windows are used per step.
- **Epoch:** here, 150 random batches. We train 12 epochs.
- **Over-fitting:** the training loss keeps falling while the loss on unseen wells rises.
- **Early stopping:** after every epoch we measure the loss on the **calibration** wells and keep the
  best weights.
  - Finding: the best epoch was 0–1 in most runs. The models over-fit the training wells very fast,
    because every well behaves differently (cross-well shift).
  - Lesson: with tiny data, short training and strong regularisation win.

### 2.10 Hyper-parameter tuning with Optuna

**Hyper-parameters** are settings chosen *before* training: lr, weight decay, batch size, tree depth,
and λ for L_mono.

**Optuna** searches them automatically:
- **TPE sampler:** it learns which regions of the search space gave good results and samples more
  there. It is smarter than grid or random search.
- **MedianPruner:** it stops a trial early if, after an epoch, its loss is worse than the median of
  previous trials. This saves time.
- **Nested tuning:** a separate study per fold, scored only on that fold's calibration wells. Test
  wells are never used.
- **Equal budget:** every model gets 20 trials per fold. Without that, a model with more tuning looks
  better unfairly.

### 2.11 Evaluation metrics and why each is used

Window-level metrics (each window gets a score and a 0/1 truth for "event within H"):

| Metric | What it measures | Note |
|---|---|---|
| **AUROC** | P(a random positive window scores above a random negative) | 0.5 = random. Can look high when positives are rare |
| **AUPRC** (primary) | Area under precision–recall; focuses on the positives | Random = prevalence (~0.1–0.3 here). Honest under imbalance |
| **C-index** | Do windows with less time remaining get higher risk? | Survival version of AUROC |
| **Brier score** | Mean squared error of the probability | Lower is better; mixes ranking and calibration |
| **ECE + reliability plot** | Does "70 %" really happen 70 % of the time? | Calibration of probabilities |

Event-level (alarm) metrics, which are what an operator cares about:
- **Alarm:** the score crosses a threshold. Consecutive windows above the threshold count as one alarm.
- **Hit:** the first alarm comes before t_E. **Recall** = hits / events.
- **False alarm (FA):** an alarm on a non-scaling instance. It is reported per well-day. The budget
  is ≤ 1 per well-month (1/30 per day).
- **Lead time:** how many hours before t_E the first alarm came.
- **Recall @ FA budget (threshold-free):** the best recall achievable at that FA rate if the threshold
  were perfect.
  - It shows the *ranking* quality.
  - The *deployed* recall uses a threshold chosen on calibration wells. The gap between the two is a
    key finding (Part 6).

Uncertainty:
- **Deep ensemble:** train 5 seeds and average them. This gives more stable predictions, and their
  spread is an uncertainty signal.
- **Split-conformal prediction:** use calibration wells to measure how wrong the predicted
  time-to-event is, then widen the intervals until 90 % of calibration cases are covered.
  - **PICP** = observed coverage. **Width** = the price paid for it.

Statistics (are the differences real or just luck with 6 folds?):
- **Bootstrap 95 % CI:** resample instances many times and see how much the metric moves.
- **Wilcoxon signed-rank:** a paired test over folds (model A vs model B on the same wells).
- **Friedman–Nemenyi:** do the model ranks differ across folds?
- **Holm correction:** many comparisons give many chances of false "significance", so p-values are
  corrected.

### 2.12 Ablation and robustness

- **Ablation (E2):** remove one component and measure the drop. This shows *which part* of the model
  actually helps. Variants:
  - −physics branch
  - −L_mono
  - −survival head (per-horizon BCE instead)
  - −patching
  - −synthetic data
  - −ensemble
  - +SI chemistry
- **Robustness (E3):** corrupt the test inputs and see how gracefully performance degrades:
  - Gaussian noise 1–5 %
  - 1–3 sensors dropped
  - frozen sensor
  - different logging rates (30 s – 5 min)
- **Generalisation (E4):** train only on the wells of the older 3W v1, test on the 24 wells new in
  v2. This is the hardest test of "unseen wells".

---

## Part 3 — Code walkthrough, file by file

Every file has a docstring at the top and `# WHY` comments. Read this section with the file open.

### 3.1 `src/data/download_3w.py` — get the data reproducibly

- **`download(commit)`** does a *sparse, shallow* git clone of only `dataset/` from
  github.com/petrobras/3W, pinned to commit `826fec4`.
  - Pinned so every number traces back to exact bytes, because Petrobras keeps updating `main`.
  - Sparse so we download about 1.7 GB, not the whole repo history.
- **`verify(ds)`** asserts that `dataset.ini` says version 2.0.0, then counts instances per class and
  source. They must match the paper: 1,119 real / 1,089 simulated / 20 hand-drawn.
- **Lesson:** data provenance is part of the experiment.

### 3.2 `notebooks/01_EDA.ipynb` — look before you model

Exploratory Data Analysis (EDA) changed the whole project:
- The choke-downstream pressure (P-JUS-CKP) is missing in 100 % of real scaling files, so the planned
  ΔP-across-choke feature was impossible.
- The median pre-onset time is only 6.5 h, so forecasting the *start* of scaling had almost no data.
- Only 6 wells have real scaling.

Result: the target was changed to "time until RI* reaches +5 %" (approved 2026-11-14).
**Lesson:** EDA is where the research question meets reality.

### 3.3 `src/data/preprocess.py` — clean data and make labels

| Function | What it does |
|---|---|
| `source_and_well(path)` | File name → source (real / simulated / drawn) and well ID (needed for grouped splits) |
| `frozen_mask(x)` | True where a value has been constant ≥ 30 min. Uses run-length counting with `diff`, `cumsum`, `groupby` |
| `to_minutes(raw, source, sample_s)` | 1 Hz → 1 min. Converts drawn pressures from bar to Pa (×1e5). Turns \|x\|>1e9 into NaN. Builds masks, forward-fills ≤ 10 min, takes the label as the minute max. `sample_s` is used only by the E3 resampling test |
| `ri_star(p, baseline)` | RI* = P-MON-CKP / baseline |
| `process_instance(path)` | Finds onset t0 and builds the causal feature RI* (first-hour baseline). Finds t_E using the *label* RI* (pre-t0 baseline, 1-h rolling median, so one spike cannot trigger an event). Computes `tau_min` and the metadata row |
| `build(paths)` | Loops over all 2,228 files → `data/processed/<class>/<file>.parquet` + `instances.csv` |

**Lesson in this file:** the *label* baseline may use expert knowledge (t0); the *feature* baseline
may not. Read the comments around `df["ri_star"]` vs `ri_lab`.

### 3.4 `src/data/splits.py` — who is train, calibration and test

- **`make_folds(idx)`**:
  - For each of the 6 scaling wells: test = that well plus a group of non-scaling wells (to measure
    false alarms).
  - Calibration = the next scaling well that has ≥ 1 event (never well 24, which holds 17 of 24
    events) plus another non-scaling group.
  - Train = everything else plus *all* synthetic data.
- **`v1v2_fold(idx)`** is the E4 split:
  - Train on wells present in 3W v1.1.1, mapped by time to v2 IDs (v1 has 18 real wells, not 21).
  - Test on the 24 wells new in v2.
- **`quick(folds)`** keeps only 2 folds, for fast smoke tests (`--quick`).

### 3.5 `src/features/build.py` — features and windows

- **`features(df)`** keeps the raw channels and masks, and adds rolling mean/std/slope for
  `DYNAMIC = [ri_star, P-MON-CKP, T-JUS-CKP, P-TPT]` over 10/60/360 min. Slope is the change of the
  rolling mean, per hour.
- **`horizon_bin(tau)`** maps minutes-to-event to bin 0–6, or −1 if censored. It uses
  `np.searchsorted` on the bin edges.
- **`windows(meta, t)`** creates one row per window end (every 10 min after the first hour). Rows at
  or after t_E are dropped. Each row gets `tau_min`, `censor_min` and `k`.
- **`build()`** writes the per-minute features and `windows.parquet` (109,547 windows).
- **Robust scaling is deliberately not done here.** It must be fitted per fold on training wells only.

### 3.6 `src/features/si.py` — chemistry prior

- **`brines()`:**
  - Loads 1,423 Gulf of Mexico offshore produced-water analyses (USGS PWGD v3.0).
  - Keeps the usable ones and fills missing ions with the median.
  - Samples 200 brines.
- **`si_grid()`:**
  - For each brine, temperature (0–150 °C) and pressure (1–401 bar), PHREEQC (Pitzer database, for
    salty brines) computes the saturation index (SI) of calcite and barite.
  - The median over brines gives a "typical brine" SI(T, P) table.
  - It takes about 10 min.
- **`lookup()`** interpolates that table at every minute's measured (T-JUS-CKP, P-MON-CKP).
- **`build()`** adds rolling stats of SI to every window → `data/features_si/tabular_si.parquet`.
  **`with_si()`** merges them for the `-SI` model variants.
- **Limitations, written in the docstring:** US brines, not Brazilian ones; no gas phase (CO₂
  degassing); fixed pH. So SI is a *relative driver*, not the true SI of the well.

### 3.7 `src/eval/metrics.py` — the scoreboard

| Function | Role |
|---|---|
| `horizon_target(w, h)` | Truth for "event within h hours": 1 / 0 / NaN. NaN when a censored window ended too early to know |
| `window_metrics(w, scores)` | AUPRC, AUROC and prevalence per horizon |
| `event_metrics(w, s, thr)` | Turns scores into alarms at threshold `thr`. Counts hits, lead times and false-alarm episodes per well-day |
| `choose_threshold(w, s)` | Lowest threshold meeting ≤ 1 FA per well-month on the given (non-test) data |
| `hits_at_fa(w, s)` | Threshold-free operating points: best hits at FA ≤ 1/30, 0.1 and 1 per day |

### 3.8 `src/models/baselines.py` — B0, B1, B2

- **`B0Threshold`:** the score is `ri_star_mean60`. `fit` does nothing.
- **`PerHorizon("lr")`:** a sklearn `Pipeline` of median imputer → `RobustScaler` → `LogisticRegression`,
  with `class_weight="balanced"`.
- **`PerHorizon("gbdt")`:** `HistGradientBoostingClassifier`. One model per horizon, trained only on
  windows whose label is known for that horizon.
- **`make(name, **params)`** is the factory. `params` come from Optuna for the `-tuned` runs.

### 3.9 `src/models/seq.py` — all neural networks

- Every model has a **body** that turns `x` (batch, 360 min, 19 channels: 10 values + 9 masks) and
  `phi` (physics features) into a 64-dimensional embedding `z`.
- A **head** turns `z` into probabilities:
  - `Head("bce")`: one sigmoid per horizon. Not forced to be monotone.
  - `Head("surv")`: hazards, then `cdf()` = 1 − cumprod(1 − h). Monotone.
- Bodies: `RNN` (B3/B4), `CNNLSTM` (B5), `TCN` (B6), `TransLSTM` (B7), `ITransformer` (B8i) and
  `PIPatchSurv` (B8 when `use_physics=False`, PIPS when it is on).
- Walk through `PIPatchSurv.forward` slowly. It is the core of the thesis:
  1. Split the values and masks. Stack each value with its mask, giving shape (B, L, C, 2).
  2. Cut patches with `unfold` (16-minute patches, stride 8), giving (B·C, n_patches, 32). Each sensor
     becomes its own sequence (channel independence).
  3. Embed and add the positional encoding, then run the Transformer encoder and average over patches.
     Add a learned channel embedding, then flatten to z_s.
  4. The physics MLP turns `phi` into z_p.
  5. The gate g = σ(W[z_s; z_p]) gives z = g·z_s + (1−g)·z_p. The mean gate is stored in `last_gate`.
- **`build(name, ...)`** maps names to architectures. Ablations are flags:
  - `PIPS-phys` has no physics branch.
  - `PIPS-patch` uses patch 8.
  - `PIPS-surv` uses the BCE head.
  - `PIPS-mono` sets λ=0 in the trainer.

### 3.10 `src/train/trainer.py` — the training loop

- **`FoldData`:**
  - Loads every instance's per-minute array.
  - Fits **robust scaling** (median / IQR) on training files only, clips to ±20, and fills NaN with 0
    (the median).
  - `x(keys, ends)` slices 360-minute windows. Early windows are left-padded, with masks = 1 meaning
    "missing".
- **`ikey(w)`** = `"label/file"`. This is the fix for the file-name collision bug (Part 5, item 9).
- **`targets(w)`** builds per-horizon 0/1/NaN targets (for BCE), plus the event flag and bin (for
  survival).
- **`surv_nll`** is the censor-aware survival loss with event weighting. **`bce_loss`** is BCE with
  unknown labels masked and `pos_weight` for imbalance.
- **`Trainer.fit`:**
  - AdamW; 12 epochs × 150 random batches; gradient clipping at 1.0.
  - After each epoch: calibration loss (and calibration AUPRC, logged only). The best state is kept.
  - The `on_epoch` hook lets Optuna prune trials.
- **`Trainer._loss`** adds **L_mono**. On scaling instances, the predicted F of the previous window
  (10 min earlier) must not be higher than now. Penalty: relu(F_prev − F_now).
- **`Trainer.predict`** returns `{horizon: P(event within horizon)}` for every window.
- **Device:** `mps` (Apple GPU) → `cuda` (NVIDIA, e.g. Kaggle) → `cpu`. The same code runs everywhere.

### 3.11 `src/models/tuned.py` — read Optuna results

`tuned_params(name, test_well)` returns the best hyper-parameters for one fold. It **raises an error**
if they are missing. Before 2026-11-16 it silently fell back to the defaults, which could have
labelled an untuned run "-tuned".

### 3.12 The experiment scripts (`experiments/`)

| Script | Experiment | What it does |
|---|---|---|
| `e1_baselines.py` | E1, E4 | `tabular()` builds or caches one feature row per window. Fits B0–B2 per fold. Out-of-fold threshold on training wells. Saves preds/tables/run log |
| `e1_sequence.py` | E1, E2, E4 | For each fold → `FoldData` → each model and seed → `Trainer.fit` → predictions, checkpoint, log. `--resume` skips finished runs; `--split v1v2` runs E4 |
| `b9_lstm_ae.py` | B9 | Trains an LSTM autoencoder on normal windows only. Score = percentile of the reconstruction error |
| `e1_tune.py` | E1 tuning | Nested Optuna studies; resumable per fold; `--max-hours` for the Kaggle time limit |
| `e3_robustness.py` | E3 | Loads checkpoints, corrupts test inputs (noise/dropout/frozen), re-scores. Inference only |
| `e3_resampling.py` | E3 | Rebuilds test features from 1 Hz data sampled every 30 s – 5 min, re-scores the same models |
| `analyze.py` | All | Reads every saved prediction → final tables (metrics, deployed alarms, conformal, stats) and figures. No training, about 1 min |

### 3.13 `scripts/` — running long jobs unattended

- **`phase2_queue.sh`:** three chains run in parallel on the Mac (sequence baselines / PIPS runs /
  tuning), then post-steps (robustness, tuned re-training, analysis). Every step is logged to
  `results/runs/queue.log`.
- **`resume_main.sh`:** after a crash, waits for chain A and re-runs the baselines with `--resume`.
- **`stop_local_tune_after_b4.sh`:** stops local tuning once B4 is done, because B5–B9 and PIPS are
  tuned on Kaggle.

### 3.14 `tests/` — small checks that catch big bugs

Run `pytest -q`.
- `test_preprocess.py`: event time and τ are correct; no event without a rise; frozen and missing masks.
- `test_features.py`: horizon bins; windows stop before the event; slopes use only past data
  (causality).
- `test_splits.py`:
  - no well appears in two splits, and synthetic data is in train only;
  - the calibration well has events and is never well 24;
  - the v1→v2 split tests only new wells.
- `test_metrics.py`: an unknown label for short censored windows; hits, lead time and false-alarm
  episodes.
- `test_seq.py`:
  - every model's output shape, parameter count < 1 M and monotone survival CDF;
  - the survival loss prefers the correct bin;
  - same-named files of different classes stay apart (the `ikey` regression test).

---

## Part 4 — Tools used and what each is for

| Tool | Used for | Where |
|---|---|---|
| **Python 3** + virtual env `.venv` | Everything; isolated package versions | — |
| **pandas** | Tables, time indexes, `resample`, `rolling`, `groupby` | Preprocessing, features, metrics |
| **NumPy** | Fast arrays, `searchsorted`, window slicing | Everywhere |
| **Parquet / pyarrow** | Compressed column storage of 1 Hz data | `data/` |
| **scikit-learn** | Logistic regression, gradient boosting, `GroupKFold`, AUROC/AUPRC | B1, B2, metrics |
| **PyTorch** | Neural networks, autograd, AdamW, devices (MPS/CUDA/CPU) | `seq.py`, `trainer.py` |
| **Optuna** | Hyper-parameter search (TPE + pruning) | `e1_tune.py` |
| **phreeqpython (PHREEQC)** | Geochemical saturation indices | `si.py` |
| **SciPy** | Grid interpolation, statistical tests | `si.py`, `analyze.py` |
| **matplotlib** | Figures | `analyze.py` |
| **pytest** | Unit tests | `tests/` |
| **Jupyter** | EDA notebook | `notebooks/` |
| **git + GitHub** | Version control, pinned data download | Repo |
| **zsh scripts, `nohup`, `caffeinate`** | Long unattended runs; stop the Mac sleeping | `scripts/` |
| **Kaggle (2× Tesla T4 GPU)** | Heavy Optuna tuning, ~10–30× faster than the M4 for these jobs | See 4.1 |

### 4.1 How Kaggle was used

1. **Account:** phone verification is needed for GPUs. The API token is stored in
   `~/.kaggle/access_token` with permissions 600.
   - **Never paste a token in chat or code.** The first token was exposed and had to be revoked.
2. **Private dataset** `scale-forecasting-tuning`: a zip of `src/`, `experiments/`, `data/features/`
   and `instances.csv`. It is uploaded once, and Kaggle mounts it read-only at `/kaggle/input/`.
3. **Script kernel** `scale-forecasting-tune` (`tune.py`):
   - copies the inputs to `/kaggle/working/sf`;
   - runs `pip install optuna`;
   - starts **two processes, one per GPU** (`CUDA_VISIBLE_DEVICES=0/1`);
   - copies `results/tuning/*.json` and the logs to `/kaggle/working`. Only files there survive.
4. **Push and run in the background:** `kaggle kernels push -p <dir> --accelerator NvidiaTeslaT4`.
   It keeps running with the browser closed.
5. **Check:** `kaggle kernels status <user>/<kernel>` returns QUEUED / RUNNING / COMPLETE / ERROR.
   Fetch the files with `kaggle kernels output ... -p <dir>`.
6. **Limits:**
   - 12 h per session, so `--max-hours 9.5` stops before a fold could be cut off. Re-running resumes
     from the saved JSON.
   - About 30 GPU-hours per week.
   - "Updated X ago" in the notebook list is when that version *started*, not proof that it is running.

---

## Part 5 — Project diary: what happened, what changed, what did not work

In date order. Each item: **what happened → why → what was changed.**

1. **Target redesigned after EDA (2026-11-14).**
   - What happened: the ΔP-across-choke feature was impossible (P-JUS-CKP is missing in 100 % of real
     scaling files), and there is very little time before onset (median 6.5 h).
   - Change: the target became the time until RI* reaches +5 % (scaling severity), with horizons from
     1 h to more than 72 h. Approved by the student.
2. **Garbage sensor values.**
   - What happened: P-PDG = 1.18e42 Pa in WELL-00006.
   - Change: any |x| > 1e9 is treated as missing before averaging.
3. **Hand-drawn data in the wrong units.**
   - What happened: the drawn pressures are in bar, the real ones in Pa. The models learned "low
     pressure = scaling" and inverted on real wells (B3 AUROC 0.10 on well 24).
   - Change: drawn pressures are multiplied by 1e5. All data was rebuilt and all models re-run.
4. **Alarm threshold from one calibration well failed.**
   - What happened: with 1 event the threshold became 0.9999, giving recall 0 everywhere.
   - Change: for B0–B2, the threshold comes from out-of-fold scores over the training wells.
   - Later change: for all models, a "calib-self" threshold on a self-referenced score (score minus
     the instance's own first-hour level), because score scales differ per well.
5. **LightGBM would not install.**
   - What happened: it needs the system library libomp, which is not on the Mac.
   - Change: sklearn's HistGradientBoosting (the same histogram-GBDT idea) is used as B2.
6. **PI-PatchSurv was too slow.**
   - Change: each mask is folded into its value's patch token (2P inputs). This halves the channel
     count, giving about 450 s per fold on the M4.
7. **Early stopping signal.**
   - What happened: the best epoch was 0 in 28 of 30 PIPS runs. Early stopping on calibration AUPRC
     was tried, but the AUPRC stayed flat at about 0.05 (too few calibration events).
   - Change: the loss rule was kept. Reported as a finding: there is fast over-fitting to training
     wells, because each well behaves differently.
8. **First full results (Step 6, 2026-11-15)** looked good for PIPS: the best AUROC, C-index and ECE,
   and 0 false alarms on normal and class-6 data.
9. **Critical bug found: file-name collision (2026-11-15).**
   - What happened: simulated and hand-drawn file names repeat across class folders (885 of 2,228;
     e.g. `DRAWN_00001` is both a class-1 and a class-7 drawing). `FoldData` stored arrays by file
     name only, so about half of the synthetic training windows got *another class's signal*.
   - Change: everything is keyed by `"label/file"` (`ikey`), with a regression test.
   - Real files are unique, so the test and calibration data were never wrong. But **every sequence
     model was retrained**.
   - Result: after the fix, plain baselines (LSTM B3, TCN B6) caught up with or passed PIPS. The
     earlier PIPS advantage was partly the bug. This is reported honestly as a negative result.
10. **Phase 2 completed without cuts** (student instruction: "complete phase 2 as it is"). Added:
    - B7 Transformer-LSTM, B8 PatchTST, B8i iTransformer, B9 LSTM-autoencoder;
    - the −synthetic ablation;
    - the SI chemistry prior;
    - E3 resampling;
    - E4 v1→v2;
    - equal-budget Optuna tuning.
11. **3W v1 has 18 real wells, not 21.**
    - What happened: CLAUDE.md said "21". Matching v1 files to v2 by time showed 18.
    - Change: the E4 split was corrected (train on 4 real scaling events, test on 20 events in new
      wells 22–24).
12. **USGS chemistry database v2.3 was blocked** by a bot check. Change: v3.0 was used via the USGS
    ArcGIS service, keeping Gulf of Mexico offshore brines as the closest public analogue.
13. **Mac crash (segfault, exit 139)** while 3 GPU jobs ran together. Change: added `--resume` and
    `resume_main.sh`, which finished the missing runs.
14. **Mac slowdown.**
    - What happened: swap was almost full (14.5/15 GB), with Chrome using about 5 GB. One PIPS-SI seed
      took 6,813 s instead of about 550 s.
    - Change: close Chrome during runs, keep the Mac plugged in, and use `caffeinate`.
15. **Optuna on the Mac was too slow.**
    - What happened: B4 took up to 3.5 h per fold, so the full tuning would have taken days.
    - Change: tuning moved to Kaggle.
    - Kaggle problems solved on the way:
      - the CLI needed the `six` package;
      - the first token leaked and was revoked;
      - without phone verification, the run used the CPU (about 600 s per trial);
      - after verification it got 2× T4; B7 ran 20 trials on a fold in 391 s.
    - Split: B1–B4 tuned on the Mac; B5–B9 and PIPS on Kaggle.
16. **Safety fix:** `tuned_params` now raises an error when tuning results are missing, instead of
    silently using the defaults.
17. **Tuned results (17 Nov).** Tuning helped the GRU (B4) and Transformer-LSTM (B7) on deployed
    alarms, hurt B3, B5, B6, B8, and did not help PIPS. Lesson: with one small calibration well as the
    tuning objective, tuning mostly adds variance.
18. **"−patching" ran out of memory.**
    - What happened: patch length 1 gives 360 tokens per channel; one attention call asked for 5 GB on
      top of 16.6 GB in use (MPS limit 20 GB).
    - Change: gradient accumulation (`Trainer.micro`): micro-batches of 32 add up to the same batch of
      256, so the experiment stays comparable. Other models are untouched.
19. **Missing Phase-2 metrics added** to `analyze.py`: integrated Brier score (all horizons), skill vs
    B0, event precision of deployed alarms, ΔT (error of the predicted time-to-event) and the Nemenyi
    critical difference.
20. **Literature check (17 Nov).** No survival / time-to-event work on 3W and no class-7 forecast with
    a lead time was found, so the framing is new. But a held-out-well benchmark for 3W was proposed in
    the official repo (issue #231, Oct 2026), so "unseen-well evaluation" is no longer unique: we
    align with it and cite it.

**What did not work (keep these, they are results too):**
- Early stopping on calibration AUPRC.
- Single-well thresholds.
- PIPS's advantage after the bug fix.
- The survival head for alarms. The per-horizon-heads ablation gave *better* recall at 1 FA/month
  (0.75 vs 0.50).
- SI prior: small, non-significant gains (pooled AUPRC 0.60 vs 0.54 for PIPS; B2 deployed 10/24 vs
  4/24). Generic Gulf-of-Mexico brines are not the wells' own water.
- Tuning: did not help PIPS (pooled AUPRC 0.54 → 0.56, per-fold 0.70 → 0.60).
- B9 detector as a forecaster (AUROC ≈ 0.37). This shows that detection ≠ forecasting (gap G1).

---

## Part 6 — Reading the results

Final numbers (17 Nov 2026, `results/tables/final_*.csv`). Full tables are in `docs/Phase3_Report.md`.

**Deployed alarms** (threshold fixed on calibration wells, calib-self strategy, 24 events):

| Model | Events caught | False alarms / well-day | Median warning |
|---|---|---|---|
| B4 GRU, tuned | **17** | 0.052 (≈ 1.6 / month) | 2.9 h |
| B7 Transformer-LSTM, tuned | 16 | **0.044** | 4.1 h |
| B4 GRU | 16 | 0.052 | 1.8 h |
| B8i iTransformer, tuned | 14 | 0.105 | 4.7 h |
| B7 Transformer-LSTM | 13 | 0.052 | 5.3 h |
| PIPS 5-seed ensemble | 9 | 0.190 | 4.2 h |

**Ranking (threshold-free):**

| Model | AUPRC@24h (mean of folds) | AUROC@24h | Pooled AUPRC@24h |
|---|---|---|---|
| B3 LSTM | **.811** | .945 | .454 |
| B7-tuned | .752 | .930 | .512 |
| PIPS ensemble | .697 | .780 | .538 |
| PIPS per-horizon heads (ablation) | .704 | .848 | **.819** |
| B8 PatchTST | .613 | .813 | .753 |
| B2 GBDT | .679 | .892 | .279 |
| B0 rule | .533 | .828 | .218 |

How to read it:
- **Two kinds of "good".** Ranking (AUPRC, AUROC) asks "are dangerous windows scored higher?";
  deployed alarms ask "with a threshold fixed in advance, how many events are caught and how many false
  alarms?". A model can be good at one and bad at the other (PIPS-surv: best pooled AUPRC, 1/24 deployed).
- **Mean-of-folds vs pooled AUPRC** differ because pooled mixes all wells: a model whose scores are on
  different scales per well looks worse pooled.
- **Significance:** only "better than B0, B1, B9" survives Holm correction; Friedman p = 0.06. With 24
  events, 17 from one well, no winner among the deep models can be claimed. Say this explicitly.
- **PIPS's real strengths:** no loss with 3 of 9 sensors missing (LSTM −30 %), conformal coverage 0.87
  of a 90 % target (most deep baselines 0.4–0.6), physics branch worth about 0.07 AUROC.
- **E4 v1→v2:** B8 PatchTST best (AUPRC .61); PIPS .28. Only 4 training events → generalising to new
  wells is the open problem.

The thesis story to tell, honestly:
1. The task is new: time-to-restriction (survival) forecasting of choke scaling on 3W.
2. Plain GRU/LSTM are strong once data bugs are fixed; our physics model is not the best ranker.
3. Ranking transfers across wells, raw thresholds do not; self-referencing the score helps a lot.
4. Physics-informed design buys robustness and calibrated uncertainty, not raw ranking.
5. Detectors do not forecast (B9), and more real events per well (field data) are the bottleneck.

---

## Part 7 — Likely viva questions, short answers

- **Why not accuracy?** Events are rare. Saying "no" always gives high accuracy. AUPRC and recall at
  a false-alarm budget measure what matters.
- **Why group by well?** Neighbouring windows of one well are almost the same. Random splits leak the
  test well into training and inflate the scores.
- **Why survival analysis?** We forecast *when*, and many series end without the event (censored).
  Survival loss uses those correctly. Classification would mislabel them.
- **Why RI\* and not ΔP across the choke?** P-JUS-CKP is missing in all real scaling files.
  Self-referencing to the first hour makes wells with different pressures comparable.
- **What is "physics-informed" here?** Three things:
  - physics-derived features (RI*, Joule–Thomson temperature drop, SI from PHREEQC) in a separate
    branch;
  - a monotonicity loss (scale only grows);
  - a survival head that makes the forecast monotone in horizon.
- **Does the physics branch matter?** Yes, modestly: removing it costs about 0.07 AUROC, 0.08 C-index
  and worsens calibration (ECE 0.23 → 0.29). It is not significant on pooled AUPRC.
- **Why did PIPS lose its lead?** A data bug had corrupted half the synthetic training windows. After
  fixing it, simple baselines improved more. We report it rather than hide it.
- **How do you avoid tuning on test?** Optuna, early stopping, thresholds and conformal intervals all
  use calibration wells or out-of-fold training wells only. Test wells are scored once.
- **Why Kaggle?** The M4 took hours per tuning fold. The T4 GPUs take minutes. The code is
  device-agnostic (`mps` / `cuda` / `cpu`).
- **Biggest limitation?** 24 real events from 6 wells, 17 of them from one well. Every conclusion has
  wide uncertainty.

---

## Part 8 — Exercises to learn by doing

1. Run `pytest -q`, then break `ikey` (return only the file name) and see which test fails.
2. Open `notebooks/01_EDA.ipynb` and plot P-MON-CKP and RI* for one class-7 file from well 24. Mark t0
   and t_E.
3. In a notebook, load `data/features/windows.parquet` and count windows per horizon bin for real
   class 7.
4. Run `python experiments/e1_baselines.py --quick` and read the printed AUPRC/recall per fold.
5. Change `RI_THRESHOLD` to 1.03 on a copy of the repo, rebuild the data, and see how the number of
   events changes. That is the trade-off behind the target choice.
6. Read `PIPatchSurv.forward` and write the tensor shape after every line for B=2, L=360, C=10.
7. Run `python experiments/e1_tune.py --models B2 --trials 4 --quick` and open `results/tuning/B2_quick.json`.
8. Explain the survival NLL for one event window in bin 3 and one censored window in bin 3, on paper.
