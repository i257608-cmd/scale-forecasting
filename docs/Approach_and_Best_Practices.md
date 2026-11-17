# Approach and Best Practices

Audience: readers who want to understand *why* the pipeline is built the way it is — the
reasoning behind each design decision, and the practices that keep the evaluation honest.
For the commands, see `Step_by_Step_Guide.md`; for the results, see `Phase3_Report.md`.

---

## 1. The problem in one paragraph

Mineral scale (e.g. CaCO₃, BaSO₄) precipitates inside the production choke (PCK) and narrows its
throat. It grows slowly, so an operator who sees it coming can plan a chemical squeeze or an acid
job instead of losing production. The task is: from the live sensor stream of a well, estimate
**how soon the restriction will reach a given severity**, with a calibrated probability and an
uncertainty interval, on a well the model has never seen.

## 2. Why the target was revised (EDA-driven, approved 2026-11-14)

The Phase 2 plan forecast the *start* of scaling with a restriction index RI = ΔP_PCK / ΔP_ref.
The EDA on 3W v2.0.0 showed both parts were not feasible:

| Finding | Consequence |
|---|---|
| P-JUS-CKP and T-MON-CKP missing in 100 % of real class-7 instances | ΔP across the choke cannot be computed → original RI impossible |
| Pre-onset normal data: median 6.5 h | forecasting the *start* has at most hours of lead |
| 4 of the 6 scaling wells have no class-0 files | no external normal reference for these wells |
| P-MON-CKP always present, rises after onset in 89 % of instances | usable restriction proxy |

So the target became **severity**: *time until the restriction has grown 5 %*, measured by the
self-referenced proxy

  RI* = P-MON-CKP / median(P-MON-CKP before onset)   (label)
  RI* = P-MON-CKP / median(first 60 min of the instance)   (feature, causal)

Scaling is gradual, so "time until +5 %" is a meaningful, physically interpretable forecast
(hours on fast well 24, days on wells 22/23).

*Why self-referenced:* every well has a different choke, flow rate and pressure regime. Dividing
by the well's own recent baseline removes that level, so a model can learn the *shape* of a rise
across wells — the same idea as normalising by a reference state in reservoir engineering.

## 3. Why survival analysis and not classification

Most windows never see an event within the observed time (the instance ends, or scaling never
reaches +5 %). A classifier must call these "negative", which is wrong — they are **censored**:
we only know the event did not happen *yet*. The discrete-time survival head

  h_k = σ(w_kᵀz + b_k),  F(k) = 1 − Π_{j≤k} (1 − h_j)

uses censored windows correctly (they only say "no event before bin k") and produces one
**monotone** curve P(event within 1, 3, 6, 12, 24, 72 h) — P(within 24 h) can never be smaller
than P(within 6 h), which independent per-horizon classifiers do not guarantee.

## 4. Why physics-informed

With only 24 real events, a purely data-driven model has little to learn from. The physics branch
gives the model features an engineer would look at, computed causally:

- RI* level, spread and slope over 10 min, 1 h and 6 h → "is the upstream pressure rising slowly
  and steadily?" (scale) vs "jumping" (slugging, quick restriction);
- T-JUS-CKP → Joule–Thomson cooling increases as the throat narrows;
- P-TPT → a rise in the whole system (not a choke restriction) moves P-TPT too.

A learned gate g = σ(W_g[z_s; z_p]) decides per window how much to trust the sequence branch vs
the physics branch. Removing the physics branch costs about 0.07 AUROC (E2), so it is used.

The monotonicity loss L_mono adds a second piece of physics: on a scaling well, without an
intervention, the restriction does not shrink, so the forecast probability should not drop
between consecutive windows.

## 5. Why patching, channel independence and masks

- **Patches (16 min, stride 8):** a 6-h window becomes 44 tokens instead of 360 time steps → an
  attention model fits on a laptop, and each token sees a local shape (a slope, a step).
- **Channel-independent encoder:** each sensor is encoded by the same Transformer. This shares
  parameters across sensors (fewer weights for 24 events) and copes with sensors missing in some
  wells.
- **Masks folded into tokens:** 3W v2 deliberately leaves frozen and missing values untreated. Each
  patch token carries the value *and* a "this was imputed" mask, so the model can learn to ignore
  imputed segments instead of trusting a flat line.

## 6. Evaluation practices that keep the results honest

| Practice | Why |
|---|---|
| Leave-one-well-out folds, grouped by well | windows of one well are strongly correlated; a random split leaks the test well into training and inflates every metric |
| Non-scaling wells rotate into test and calibration | false alarms are measured on wells the model never saw |
| Synthetic (simulated, hand-drawn) data in training only | they are not real evidence; they cannot vouch for real-well performance |
| Thresholds from training (out-of-fold) or calibration wells only | choosing a threshold on test wells is the most common hidden leak in alarm papers |
| Robust scaling fitted per fold on training files | scaler statistics are also "learned" from data |
| Censored windows with unknown labels excluded from window metrics | calling them negative would reward models that ignore late events |
| AUPRC, event recall, false alarms per well-day, lead time — not accuracy | positives are < 5 % of windows; accuracy of an "always normal" model is > 95 % |
| Threshold-free *and* deployed-threshold metrics both reported | ranking quality and alarm usefulness are different things (see §7) |
| Instance-level bootstrap, Holm correction | windows of one instance are not independent; many comparisons inflate false discoveries |
| Fixed seeds, every run logged with config, checkpoint and predictions | anyone can regenerate every figure with `experiments/analyze.py` |
| Negative results reported | e.g. our proposed PI-PatchSurv is not the best model; plain GRU/LSTM rank and deploy better; tuning did not help it |

## 7. The main lesson: ranking ≠ deployable alarm

All good models rank windows well *within* a well, but the absolute score scale differs between
wells, so a threshold chosen on one set of wells over- or under-fires on another. This is why the
report always shows two numbers: recall at a false-alarm budget (threshold-free, best case) and
recall/false alarms with a threshold fixed before seeing the test well (what an operator would get).
Self-referencing the score to the instance's own first hour (`calib-self`) is the cheap analogue
of RI* = P / own baseline and is evaluated as a remedy. In the final results it roughly doubles deployed
recall for several models (GRU 12 → 16 hits, Transformer-LSTM 5 → 13) at fewer false alarms.

## 8. Engineering practices

- Small models (< 1 M parameters, PI-PatchSurv ≈ 129 k): 24 events cannot support more.
- Apple-Silicon `mps` device in float32, with CPU fallback; same code runs on CUDA.
- `--quick` mode (2 folds, 1 seed) for every experiment → a smoke test in minutes before a long run.
- Unit tests for preprocessing, splits, features, metrics and model shapes / monotone survival CDF.
- Every block of code carries a "WHY" comment explaining the decision, not just the operation.
- Bugs found are documented where they were fixed (e.g. hand-drawn pressures in bar → ×1e5,
  sentinel values 1e42 → missing), so a reader can see what went wrong and why the fix is correct.

## 9. Lessons from the final runs

- **Key data by its full identity.** 3W repeats synthetic file names across class folders. Keying
  arrays by file name silently mixed classes for half of the synthetic training windows; every sequence
  result changed after the fix (`trainer.ikey`, regression test). Always key by `label/file`, and add a
  test that fails if two instances share a key.
- **Simple models can win with few events.** With 24 events, GRU/LSTM states integrate the slow
  restriction trend more directly than a patch Transformer. Benchmark simple models under the same
  protocol before claiming a complex one.
- **Tune on the same evidence you stop on.** Optuna was scored on calibration-well loss (AUPRC there
  is flat with 1–3 events). With one small calibration well, tuning mostly adds variance: it helped two
  models and hurt four.
- **Choose variants on calibration data, not test results.** The per-horizon-head variant looked best
  on the test wells; adopting it after seeing them would be test-set selection, so it is reported as a
  finding instead.
- **Keep experiments equal when compute forces a change.** The no-patching ablation did not fit in
  memory; gradient accumulation (micro-batches of 32, effective batch 256) kept it comparable.
