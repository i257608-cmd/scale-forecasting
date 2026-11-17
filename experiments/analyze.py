"""Final analysis over every saved prediction (results/preds/<model>_<well>_s<seed>.npz).

    python experiments/analyze.py

No training here: everything reads predictions, so the report can be regenerated in ~1 min.
Produces
  results/tables/final_main.csv        window metrics (AUPRC/AUROC per horizon, C-index, Brier, ECE)
  results/tables/final_alarm.csv       DEPLOYED alarm metrics for each threshold strategy
  results/tables/final_conformal.csv   time-to-event intervals: PICP / width (split conformal)
  results/tables/final_stats.csv       paired bootstrap + Wilcoxon vs PIPS, Holm-corrected; Friedman
  results/figures/*.png                PR curves, reliability, lead times, horizons, ablations, examples

Threshold strategies (never use test wells):
  calib        thr = lowest score with <= 1 FA / well-month on the CALIBRATION wells
  calib-self   same, on a self-referenced score s - median(s over the instance's first hour of windows).
               WHY: score scales differ per well (step 4 finding); subtracting the instance's own early
               level is the score-space analogue of RI* = P / own baseline. Causal: uses only past windows.
  oof          (B0-B2 only) thr from out-of-fold scores over training wells (e1_baselines.py)
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.e1_baselines import tabular  # noqa: E402
from src.data.splits import make_folds, v1v2_fold  # noqa: E402
from src.eval import metrics as M  # noqa: E402

PRED, TAB, FIG = ROOT / "results" / "preds", ROOT / "results" / "tables", ROOT / "results" / "figures"
OUT = "final"                 # table prefix; "e4_v1v2" for the v1 -> v2 split (set in __main__)
H = M.ALARM_H
SELF_REF_WINDOWS = 6          # first hour of windows (stride 10 min) after the 60-min RI* baseline
COVER = 0.9                   # conformal target coverage
RNG = np.random.default_rng(0)
MAIN = ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8i", "B9", "PIPS"]
TUNED = ["B0"] + [f"{m}-tuned" for m in MAIN[1:]]          # E1 with an equal Optuna budget (e1_tune.py)


def reference(m: str) -> str:
    """Which PIPS variant a model is compared with (like with like: seeds and tuning must match)."""
    if m.endswith("-single"):
        return "PIPS-single"
    if m in ("PIPS-tuned", "PIPS-SI"):
        return "PIPS"                      # 5-seed ensemble vs 5-seed ensemble
    if m.endswith("-tuned"):
        return "PIPS-tuned"
    if m.startswith("PIPS-"):
        return "PIPS-single"               # single-seed ablations (E2) vs seed 0
    return "PIPS"


# ---------------------------------------------------------------- loading
def load_preds():
    """{model: {well: {split_h: array}}}, averaging seeds (deep ensemble). Models with >1 seed also
    get '<model>-single' = seed 0 only (the '-ensemble' ablation)."""
    by = defaultdict(lambda: defaultdict(list))
    for f in sorted(PRED.glob("*.npz")):
        m = re.match(r"(.+)_(WELL-\d+|V2-NEW)_s(\d+)$", f.stem)
        by[m.group(1)][m.group(2)].append((int(m.group(3)), dict(np.load(f))))
    out = {}
    for model, wells in by.items():
        out[model] = {w: {k: np.mean([d[k] for _, d in runs], 0) for k in runs[0][1]} for w, runs in wells.items()}
        if any(len(r) > 1 for r in wells.values()):
            out[f"{model}-single"] = {w: min(runs, key=lambda r: r[0])[1] for w, runs in wells.items()}
    return out


def self_ref(w: pd.DataFrame, s: np.ndarray) -> np.ndarray:
    ref = pd.Series(s).groupby(w.file.to_numpy()).transform(lambda x: np.median(x.iloc[:SELF_REF_WINDOWS]))
    return s - ref.to_numpy()


# ---------------------------------------------------------------- metrics not in src/eval
def leads(w, s, thr):
    out = []
    for _, g in w.assign(a=s >= thr).groupby("file", sort=False):
        if g.event.iloc[0] and g.a.any():
            out.append(g.tau_min.to_numpy()[g.a.to_numpy().argmax()] / 60)
    return out


def fa_by_label(w, s, thr):
    """E4: false-alarm episodes and well-days per class label of non-scaling instances
    (class 6 'quick restriction in PCK' is the main look-alike)."""
    out = []
    for f, g in w.assign(a=s >= thr)[w.label != 7].groupby("file", sort=False):
        a = g.a.to_numpy()
        out.append(dict(label=int(g.label.iloc[0]), episodes=int(a[0] + (np.diff(a.astype(int)) == 1).sum()),
                        any_alarm=bool(a.any()),
                        days=(g.t_end.iloc[-1] - g.t_end.iloc[0]).total_seconds() / 86400 + 1 / 144))
    return out


def ece(y, p, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())


def c_index(w, risk, n=4000):
    """Harrell's C on a window subsample (O(n^2)); time = tau (event) or censor_min."""
    i = RNG.choice(len(w), min(n, len(w)), replace=False)
    t = np.where(w.event, w.tau_min, w.censor_min).astype(float)[i]
    e, r = w.event.to_numpy()[i], risk[i]
    comp = e[:, None] & (t[:, None] < t[None, :])
    conc = (r[:, None] > r[None, :]) + 0.5 * (r[:, None] == r[None, :])
    return float((conc * comp).sum() / comp.sum()) if comp.sum() else np.nan


def tau_hat(F: np.ndarray) -> np.ndarray:
    """Predicted median time-to-event (h): first horizon where F >= 0.5, log-interpolated; > 72 h -> 144."""
    hs = np.log(np.array(M.HORIZONS_H, float))
    F = np.maximum.accumulate(F, 1)            # bce heads may be non-monotone
    out = np.full(len(F), np.log(144.0))
    for i, row in enumerate(F):
        j = np.argmax(row >= 0.5)
        if row[j] >= 0.5:
            out[i] = hs[0] if j == 0 else np.interp(0.5, [row[j - 1], row[j]], [hs[j - 1], hs[j]])
    return out


def fmat(d, split):
    return np.stack([d[f"{split}_{h}"] for h in M.HORIZONS_H], 1)


# ---------------------------------------------------------------- main
def main(split="lowo"):
    FIG.mkdir(parents=True, exist_ok=True)
    t = tabular()
    inst = pd.read_csv(ROOT / "data" / "processed" / "instances.csv")
    folds = {f["test_well"]: f for f in (v1v2_fold(inst) if split == "v1v2" else make_folds(inst))}
    parts = {w: {s: t[t.file.isin(f[s])].reset_index(drop=True) for s in ("calib", "test")} for w, f in folds.items()}
    preds = load_preds()
    rows, alarm, conf, fal, pooled = [], [], [], [], defaultdict(lambda: dict(y=[], s=[], file=[], leads=defaultdict(list)))
    for model, wells in sorted(preds.items()):
        for well, d in wells.items():
            te, ca = parts[well]["test"], parts[well]["calib"]
            sc_te = {h: d[f"test_{h}"] for h in M.HORIZONS_H}
            r = dict(model=model, test_well=well, **M.window_metrics(te, sc_te),
                     **M.hits_at_fa(te, sc_te[H]))
            y = M.horizon_target(te, H).to_numpy()
            ok = ~np.isnan(y)
            p = np.clip(sc_te[H][ok], 0, 1)
            ibs = []   # integrated Brier: mean Brier over all forecast horizons (Phase 2 metric list)
            for h in M.HORIZONS_H:
                yh = M.horizon_target(te, h).to_numpy(); okh = ~np.isnan(yh)
                ibs.append(np.mean((np.clip(sc_te[h][okh], 0, 1) - yh[okh]) ** 2))
            r.update(brier_24h=float(np.mean((p - y[ok]) ** 2)), ibs=float(np.mean(ibs)), ece_24h=float(ece(y[ok], p)),
                     c_index=c_index(te, sc_te[H]), events=int(te[te.event].file.nunique()))
            rows.append(r)
            P = pooled[model]
            P["y"].append(y[ok]); P["s"].append(sc_te[H][ok]); P["file"].append(te.file.to_numpy()[ok])
            # deployed thresholds
            strat = {"calib": (d[f"calib_{H}"], sc_te[H]),
                     "calib-self": (self_ref(ca, d[f"calib_{H}"]), self_ref(te, sc_te[H]))}
            for name, (sc_ca, s_te) in strat.items():
                thr = M.choose_threshold(ca, sc_ca)
                alarm.append(dict(model=model, test_well=well, strategy=name, **M.event_metrics(te, s_te, thr)))
                P["leads"][name] += leads(te, s_te, thr)
                fal += [dict(model=model, strategy=name, **x) for x in fa_by_label(te, s_te, thr)]
            if "oof_thr" in d:
                thr = float(d["oof_thr"])
                alarm.append(dict(model=model, test_well=well, strategy="oof", **M.event_metrics(te, sc_te[H], thr)))
                P["leads"]["oof"] += leads(te, sc_te[H], thr)
            # split-conformal interval on log time-to-event, calibrated on calib-well event windows
            if model not in ("B0",):
                ce, tev = ca.event.to_numpy(), te.event.to_numpy()
                if ce.sum() and tev.sum():
                    res = np.abs(np.log(ca.tau_min.to_numpy()[ce] / 60) - tau_hat(fmat(d, "calib")[ce]))
                    q = np.quantile(res, min(1, np.ceil((len(res) + 1) * COVER) / len(res)))
                    th = tau_hat(fmat(d, "test")[tev])
                    lt = np.log(te.tau_min.to_numpy()[tev] / 60)
                    conf.append(dict(model=model, test_well=well, n=int(tev.sum()), q_log=q,
                                     delta_t_h=float(np.median(np.abs(np.exp(th) - np.exp(lt)))),   # ΔT = |τ̂ − τ|
                                     picp=float(np.mean(np.abs(lt - th) <= q)),
                                     width_factor=float(np.exp(2 * q))))   # interval = [τ̂/e^q, τ̂·e^q]
    res = pd.DataFrame(rows)
    res.to_csv(TAB / f"{OUT}_main_folds.csv", index=False)
    al = pd.DataFrame(alarm)
    al.to_csv(TAB / f"{OUT}_alarm_folds.csv", index=False)

    # ---- summaries (pooled event counts; mean of per-fold window metrics; pooled bootstrap CI)
    summ = res.groupby("model").agg(**{c: (c, "mean") for c in res.columns if c.startswith(("auprc", "auroc"))
                                       or c in ("brier_24h", "ibs", "ece_24h", "c_index")}, events=("events", "sum"))
    if "B0" in set(res.model):   # skill vs the physics threshold rule B0, same fold: 1 = perfect, 0 = no better than B0
        b0 = res[res.model == "B0"].set_index("test_well")
        sk = res.join(b0[["brier_24h", f"auprc_{H}h"]], on="test_well", rsuffix="_b0")
        sk["bss_vs_b0"] = 1 - sk.brier_24h / sk.brier_24h_b0
        sk["auprc_skill_vs_b0"] = (sk[f"auprc_{H}h"] - sk[f"auprc_{H}h_b0"]) / (1 - sk[f"auprc_{H}h_b0"])
        summ = summ.join(sk.groupby("model")[["bss_vs_b0", "auprc_skill_vs_b0"]].mean())
    for b in M.FA_BUDGETS:
        summ[f"recall@fa{b:.3g}"] = res.groupby("model")[f"hits@fa{b:.3g}"].sum() / summ.events
    ci = {}
    for model, P in pooled.items():
        y, s, f = (np.concatenate(P[k]) for k in ("y", "s", "file"))
        files = np.unique(f)
        grp = {u: np.flatnonzero(f == u) for u in files}
        boots = []
        for _ in range(500):   # resample whole instances (windows of one instance are correlated)
            ix = np.concatenate([grp[u] for u in RNG.choice(files, len(files))])
            if 0 < y[ix].sum() < len(ix):
                boots.append(average_precision_score(y[ix], s[ix]))
        ci[model] = (average_precision_score(y, s), *np.percentile(boots, [2.5, 97.5]))
    summ = summ.join(pd.DataFrame(ci, index=["pooled_auprc_24h", "pooled_lo", "pooled_hi"]).T)
    summ.round(4).to_csv(TAB / f"{OUT}_main.csv")

    als = al.groupby(["model", "strategy"]).agg(hits=("hits", "sum"), events=("events", "sum"),
                                                false_alarms=("false_alarms", "sum"), neg_days=("neg_days", "sum"),
                                                early=("alarms_in_unconfirmed_scaling", "sum")).reset_index()
    als["recall"] = als.hits / als.events
    als["fa_per_day"] = als.false_alarms / als.neg_days
    als["event_precision"] = als.hits / (als.hits + als.false_alarms).replace(0, np.nan)   # true / all alarm episodes
    als["lead_med_h"] = [np.median(pooled[m]["leads"][s]) if pooled[m]["leads"][s] else np.nan
                         for m, s in zip(als.model, als.strategy)]
    als["lead_iqr_h"] = [np.subtract(*np.percentile(pooled[m]["leads"][s], [75, 25])) if pooled[m]["leads"][s] else np.nan
                         for m, s in zip(als.model, als.strategy)]
    als.round(4).to_csv(TAB / f"{OUT}_alarm.csv", index=False)
    cf = pd.DataFrame(conf)
    if len(cf):
        cf.to_csv(TAB / f"{OUT}_conformal_folds.csv", index=False)
        cg = cf.assign(cov=cf.picp * cf.n).groupby("model").agg(n=("n", "sum"), cov=("cov", "sum"),
                                                                  width_factor=("width_factor", "median"),
                                                                  delta_t_med_h=("delta_t_h", "median"))
        cg["picp"] = cg["cov"] / cg.n
        cg.drop(columns="cov").round(3).to_csv(TAB / f"{OUT}_conformal.csv")

    fl = pd.DataFrame(fal)
    if len(fl):
        fl = fl[fl.model.isin(MAIN + TUNED + ["PIPS-SI"])].groupby(["model", "strategy", "label"]).agg(
            instances=("episodes", "size"), with_alarm=("any_alarm", "sum"), episodes=("episodes", "sum"),
            days=("days", "sum")).reset_index()
        fl["fa_per_day"] = fl.episodes / fl.days
        fl["frac_instances_alarmed"] = fl.with_alarm / fl.instances
        fl.round(4).to_csv(TAB / f"{OUT}_fa_by_class.csv", index=False)
    if split == "lowo":   # one v1 -> v2 fold: no per-fold statistics or LOWO figures
        stats_table(res, pooled)
        seed_table(parts)
        plots(res, als, pooled, preds, parts)
    print(summ.round(3).to_string())
    print(als.round(3).to_string())


def seed_table(parts):
    """mean +/- SD over random seeds (each seed scored on its own, not ensembled): window metrics
    averaged over folds, then deployed calib-self alarms pooled over folds. Models with >= 2 seeds only."""
    by = defaultdict(lambda: defaultdict(dict))
    for f in sorted(PRED.glob("*.npz")):
        m = re.match(r"(.+)_(WELL-\d+)_s(\d+)$", f.stem)
        if m:
            by[m.group(1)][int(m.group(3))][m.group(2)] = dict(np.load(f))
    rows = []
    for model, seeds in by.items():
        if len(seeds) < 2:
            continue
        for seed, wells in seeds.items():
            per, hits, fa, days = [], 0, 0, 0.0
            for well, d in wells.items():
                te, ca = parts[well]["test"], parts[well]["calib"]
                per.append(M.window_metrics(te, {H: d[f"test_{H}"]}))
                thr = M.choose_threshold(ca, self_ref(ca, d[f"calib_{H}"]))
                ev = M.event_metrics(te, self_ref(te, d[f"test_{H}"]), thr)
                hits, fa, days = hits + ev["hits"], fa + ev["false_alarms"], days + ev["neg_days"]
            pf = pd.DataFrame(per)
            rows.append(dict(model=model, seed=seed, folds=len(wells), auprc_24h=pf[f"auprc_{H}h"].mean(),
                             auroc_24h=pf[f"auroc_{H}h"].mean(), hits=hits, fa_per_day=fa / days))
    if not rows:
        return
    sd = pd.DataFrame(rows)
    sd.to_csv(TAB / f"{OUT}_seed_runs.csv", index=False)
    agg = sd.groupby("model").agg(seeds=("seed", "nunique"), **{f"{c}_{fn}": (c, fn) for c in
                                  ("auprc_24h", "auroc_24h", "hits", "fa_per_day") for fn in ("mean", "std")})
    agg.round(4).to_csv(TAB / f"{OUT}_seeds.csv")


def stats_table(res, pooled):
    """PIPS vs each other model: paired instance bootstrap on pooled AUPRC@24h + Wilcoxon over folds.
    Holm correction across comparisons. Friedman over models with all folds present."""
    ref = "PIPS" if "PIPS" in pooled else None
    if ref is None:
        return
    rows = []
    yref, sref, fref = (np.concatenate(pooled[ref][k]) for k in ("y", "s", "file"))
    files = np.unique(fref)
    grp = {u: np.flatnonzero(fref == u) for u in files}
    piv = res.pivot(index="test_well", columns="model", values=f"auprc_{H}h").dropna()
    for m in pooled:
        if m == ref:
            continue
        # WHY: ablations are single-seed runs -> compare to single-seed PIPS (seed 0), not the ensemble
        r = reference(m) if reference(m) in pooled and reference(m) != m else ref
        yref, sref = (np.concatenate(pooled[r][k]) for k in ("y", "s"))
        y, s, f = (np.concatenate(pooled[m][k]) for k in ("y", "s", "file"))
        if len(y) != len(yref) or not (f == fref).all():
            continue
        diffs = []
        for _ in range(1000):
            ix = np.concatenate([grp[u] for u in RNG.choice(files, len(files))])
            if 0 < yref[ix].sum() < len(ix):
                diffs.append(average_precision_score(yref[ix], sref[ix]) - average_precision_score(y[ix], s[ix]))
        diffs = np.array(diffs)
        p_boot = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
        p_w = stats.wilcoxon(piv[r], piv[m]).pvalue if m in piv and (piv[r] != piv[m]).any() else np.nan
        rows.append(dict(model=m, reference=r, delta_auprc=float(np.mean(diffs)), lo=float(np.percentile(diffs, 2.5)),
                         hi=float(np.percentile(diffs, 97.5)), p_boot=p_boot, p_wilcoxon_folds=p_w))
    st = pd.DataFrame(rows).sort_values("p_boot")
    n = len(st)
    st["p_boot_holm"] = np.minimum(1, np.maximum.accumulate(st.p_boot.to_numpy() * (n - np.arange(n))))
    ranks = {}
    for label, group in (("fixed", MAIN), ("tuned", TUNED)):
        main = [m for m in group if m in piv]
        if len(main) > 2:
            fr = stats.friedmanchisquare(*[piv[m] for m in main])
            st = pd.concat([st, pd.DataFrame([dict(model=f"Friedman {label} ({','.join(main)}) over {len(piv)} folds",
                                                   p_boot=fr.pvalue, delta_auprc=fr.statistic)])])
            ranks[f"mean_rank_auprc24_{label}"] = piv[main].rank(axis=1, ascending=False).mean()
            # Nemenyi post-hoc: two models differ if their mean ranks differ by more than CD (Demsar 2006)
            q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
                   11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391}
            k, nf = len(main), len(piv)
            cd = q05[min(k, 15)] * np.sqrt(k * (k + 1) / (6 * nf))
            mr = ranks[f"mean_rank_auprc24_{label}"]
            sig = [f"{a}>{b}" for a in main for b in main if mr[b] - mr[a] > cd]
            st = pd.concat([st, pd.DataFrame([dict(model=f"Nemenyi CD {label} (k={k}, N={nf})", delta_auprc=cd,
                                                   reference="; ".join(sig) or "no pair differs")])])
    if ranks:
        pd.DataFrame(ranks).round(2).to_csv(TAB / f"{OUT}_mean_ranks.csv")
    st.round(4).to_csv(TAB / f"{OUT}_stats.csv", index=False)


def save(fig, name):
    """Publication output: 300-dpi PNG for documents + vector PDF for the LaTeX paper."""
    fig.tight_layout(); fig.savefig(FIG / f"{name}.png", dpi=300); fig.savefig(FIG / f"{name}.pdf"); plt.close(fig)


LABEL = {"B0": "B0 RI* rule", "B1": "B1 logistic", "B2": "B2 GBDT", "B3": "B3 LSTM", "B4": "B4 GRU", "B5": "B5 CNN-LSTM",
         "B6": "B6 TCN", "B7": "B7 Transformer-LSTM", "B8": "B8 PatchTST", "B8i": "B8i iTransformer",
         "B9": "B9 LSTM-AE", "PIPS": "PI-PatchSurv (ours)"}


def plots(res, als, pooled, preds, parts):
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "legend.fontsize": 7.5, "axes.spines.top": False,
                         "axes.spines.right": False})
    lab = lambda m: LABEL.get(m.removesuffix("-tuned"), m) + (" (tuned)" if m.endswith("-tuned") else "")  # noqa: E731
    pr_ = lambda m: (np.concatenate(pooled[m]["y"]), np.concatenate(pooled[m]["s"]))  # noqa: E731
    # 1 PR curves at 24 h, pooled over folds: (a) rule / classical / detector, (b) deep forecasters
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, group, title in ((axes[0], ["B0", "B1", "B2", "B9", "PIPS"], "(a) rule, classical, detector"),
                             (axes[1], ["B3", "B4", "B5", "B6", "B7", "B8", "B8i", "PIPS"], "(b) deep forecasters")):
        for m in [x for x in group if x in pooled]:
            y, sc = pr_(m)
            pr, rc, _ = precision_recall_curve(y, sc)
            ax.plot(rc, pr, lw=2.2 if m == "PIPS" else 1, color="k" if m == "PIPS" else None,
                    label=f"{lab(m)} (AP {average_precision_score(y, sc):.2f})")
        ax.axhline(y.mean(), color="grey", ls=":", label=f"random ({y.mean():.2f})")
        ax.set(xlabel="recall", title=title); ax.legend(loc="upper right")
    axes[0].set_ylabel("precision")
    save(fig, "pr_curves_24h")
    # 2 reliability (calibration) at 24 h
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    for m in [x for x in ("B2", "B3", "B4", "B7", "PIPS") if x in pooled]:
        y, p = pr_(m); p = np.clip(p, 0, 1)
        b = np.minimum((p * 10).astype(int), 9)
        keep = [i for i in range(10) if (b == i).sum() > 20]
        ax.plot([p[b == i].mean() for i in keep], [y[b == i].mean() for i in keep], "o-", ms=3,
                lw=2 if m == "PIPS" else 1, color="k" if m == "PIPS" else None, label=f"{lab(m)} (ECE {ece(y, p):.2f})")
    ax.plot([0, 1], [0, 1], ":", color="grey")
    ax.set(xlabel="predicted P(event within 24 h)", ylabel="observed frequency"); ax.legend()
    save(fig, "reliability_24h")
    # 3 lead times of hits under the deployed calib-self threshold
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for m in [x for x in ("B4-tuned", "B7-tuned", "PIPS") if x in pooled]:
        L = pooled[m]["leads"]["calib-self"]
        if L:
            ax.hist(L, bins=np.logspace(-1, 2.3, 16), alpha=0.45, label=f"{lab(m)} (n = {len(L)}, median {np.median(L):.1f} h)")
    ax.set_xscale("log"); ax.set(xlabel="lead time before RI* reaches +5 % (h)", ylabel="events caught"); ax.legend()
    save(fig, "lead_time_hist")
    # 4 horizon sensitivity (E5)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for m in [x for x in ("B0", "B2", "B3", "B4", "B7", "B8", "PIPS") if x in pooled]:
        v = [res[res.model == m][f"auprc_{h}h"].mean() for h in M.HORIZONS_H]
        ax.plot(M.HORIZONS_H, v, "o-", ms=3, lw=2.2 if m == "PIPS" else 1, color="k" if m == "PIPS" else None, label=lab(m))
    ax.set_xscale("log"); ax.set_xticks(M.HORIZONS_H, [str(h) for h in M.HORIZONS_H])
    ax.set(xlabel="forecast horizon (h)", ylabel="AUPRC (mean over folds)"); ax.legend(ncol=2)
    save(fig, "horizon_auprc")
    # 5 ablations: mean +/- SD over seeds where several seeds exist (single-seed scoring, no ensembling)
    sf = TAB / f"{OUT}_seed_runs.csv"
    abl = [("PIPS", "full model"), ("PIPS-phys", "− physics"), ("PIPS-mono", "− L_mono"), ("PIPS-surv", "− survival\n(BCE heads)"),
           ("PIPS-nopatch", "− patching"), ("PIPS-patch", "patch 8"), ("PIPS-nosyn", "− synthetic"), ("PIPS-SI", "+ SI prior"),
           ("PIPS-tuned", "tuned")]
    sr = pd.read_csv(sf) if sf.exists() else pd.DataFrame(columns=["model", "auprc_24h", "hits"])
    rows = []
    for m, name in abl:
        r = sr[sr.model == m]
        if len(r):
            rows.append((name, r.auprc_24h.mean(), r.auprc_24h.std(ddof=1) if len(r) > 1 else 0, r.hits.mean(),
                         r.hits.std(ddof=1) if len(r) > 1 else 0, len(r)))
        elif m in pooled:      # single seed only: score it from the fold table
            g = res[res.model == m]
            hits = als[(als.model == m) & (als.strategy == "calib-self")].hits
            rows.append((name, g[f"auprc_{H}h"].mean(), 0, float(hits.iloc[0]) if len(hits) else np.nan, 0, 1))
    if len(rows) > 1:
        ab = pd.DataFrame(rows, columns=["name", "a", "a_sd", "h", "h_sd", "n"])
        fig, ax1 = plt.subplots(figsize=(8, 3.6))
        x = np.arange(len(ab))
        ax1.bar(x - 0.2, ab.a, 0.4, yerr=ab.a_sd, capsize=2, label="AUPRC@24h (left)", color="#4a73ab")
        ax2 = ax1.twinx()
        ax2.bar(x + 0.2, ab.h, 0.4, yerr=ab.h_sd, capsize=2, label="deployed hits / 24 (right)", color="#efb65c")
        ax1.set_xticks(x, [f"{a}\n(n={n})" for a, n in zip(ab.name, ab.n)], fontsize=7.5)
        ax1.set_ylabel("AUPRC@24h, mean over folds"); ax2.set_ylabel("events caught (calib-self)")
        ax2.spines["right"].set_visible(True)
        h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper right")
        save(fig, "ablation_bars")
    # 6 qualitative examples: one event instance on fast well 24 and one on slow well 22
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.2))
    for ax, well in zip(axes, ("WELL-00024", "WELL-00022")):
        te = parts[well]["test"]
        ev_files = te[te.event].file.unique()
        if not len(ev_files):
            continue
        f = ev_files[np.argmax([(te.file == x).sum() for x in ev_files])]
        sel = (te.file == f).to_numpy()
        g = te[sel]
        days = (g.t_end - g.t_end.iloc[0]).dt.total_seconds() / 86400
        ax.plot(days, (g.ri_star - 1) * 100, color="grey", lw=1, label="restriction RI* − 1 (%)")
        ax.axhline(5, color="grey", ls=":")
        ax2 = ax.twinx()
        for m, c in (("B4-tuned", "#d6455f"), ("PIPS", "k")):
            if m in preds and well in preds[m]:
                ax2.plot(days, preds[m][well][f"test_{H}"][sel], color=c, lw=1.4, label=f"{lab(m)}: P(event ≤ 24 h)")
        tE = ((g.t_end.iloc[-1] - g.t_end.iloc[0]).total_seconds() / 60 + float(g.tau_min.iloc[-1])) / 1440
        ax.axvline(tE, color="#c62828", ls="--", lw=1, label="t_E (+5 % reached)")
        ax.set_title(f"{well.replace('WELL-000', 'well ')} ({'fast' if well.endswith('24') else 'slow'} scaling)")
        ax.set_ylabel("RI* − 1 (%)"); ax2.set_ylim(0, 1); ax2.set_ylabel("risk"); ax2.spines["right"].set_visible(True)
        hd = ax.get_legend_handles_labels(), ax2.get_legend_handles_labels()
    axes[-1].set_xlabel("days since start of recording")
    fig.legend(hd[0][0] + hd[1][0], hd[0][1] + hd[1][1], loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(FIG / "example_forecasts.png", dpi=300)
    fig.savefig(FIG / "example_forecasts.pdf"); plt.close(fig)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["lowo", "v1v2"], default="lowo")
    a = ap.parse_args()
    if a.split == "v1v2":
        PRED, OUT = ROOT / "results" / "preds_v1v2", "e4_v1v2"
    main(a.split)
