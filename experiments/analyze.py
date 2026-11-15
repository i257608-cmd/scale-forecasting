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
from src.data.splits import make_folds  # noqa: E402
from src.eval import metrics as M  # noqa: E402

PRED, TAB, FIG = ROOT / "results" / "preds", ROOT / "results" / "tables", ROOT / "results" / "figures"
H = M.ALARM_H
SELF_REF_WINDOWS = 6          # first hour of windows (stride 10 min) after the 60-min RI* baseline
COVER = 0.9                   # conformal target coverage
RNG = np.random.default_rng(0)
MAIN = ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "PIPS"]


# ---------------------------------------------------------------- loading
def load_preds():
    """{model: {well: {split_h: array}}}, averaging seeds (deep ensemble). Models with >1 seed also
    get '<model>-single' = seed 0 only (the '-ensemble' ablation)."""
    by = defaultdict(lambda: defaultdict(list))
    for f in sorted(PRED.glob("*.npz")):
        m = re.match(r"(.+)_(WELL-\d+)_s(\d+)$", f.stem)
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
def main():
    FIG.mkdir(parents=True, exist_ok=True)
    t = tabular()
    folds = {f["test_well"]: f for f in make_folds(pd.read_csv(ROOT / "data" / "processed" / "instances.csv"))}
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
            r.update(brier_24h=float(np.mean((p - y[ok]) ** 2)), ece_24h=float(ece(y[ok], p)),
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
                                     picp=float(np.mean(np.abs(lt - th) <= q)),
                                     width_factor=float(np.exp(2 * q))))   # interval = [τ̂/e^q, τ̂·e^q]
    res = pd.DataFrame(rows)
    res.to_csv(TAB / "final_main_folds.csv", index=False)
    al = pd.DataFrame(alarm)
    al.to_csv(TAB / "final_alarm_folds.csv", index=False)

    # ---- summaries (pooled event counts; mean of per-fold window metrics; pooled bootstrap CI)
    summ = res.groupby("model").agg(**{c: (c, "mean") for c in res.columns if c.startswith(("auprc", "auroc"))
                                       or c in ("brier_24h", "ece_24h", "c_index")}, events=("events", "sum"))
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
    summ.round(4).to_csv(TAB / "final_main.csv")

    als = al.groupby(["model", "strategy"]).agg(hits=("hits", "sum"), events=("events", "sum"),
                                                false_alarms=("false_alarms", "sum"), neg_days=("neg_days", "sum"),
                                                early=("alarms_in_unconfirmed_scaling", "sum")).reset_index()
    als["recall"] = als.hits / als.events
    als["fa_per_day"] = als.false_alarms / als.neg_days
    als["lead_med_h"] = [np.median(pooled[m]["leads"][s]) if pooled[m]["leads"][s] else np.nan
                         for m, s in zip(als.model, als.strategy)]
    als["lead_iqr_h"] = [np.subtract(*np.percentile(pooled[m]["leads"][s], [75, 25])) if pooled[m]["leads"][s] else np.nan
                         for m, s in zip(als.model, als.strategy)]
    als.round(4).to_csv(TAB / "final_alarm.csv", index=False)
    cf = pd.DataFrame(conf)
    if len(cf):
        cf.to_csv(TAB / "final_conformal_folds.csv", index=False)
        cg = cf.assign(cov=cf.picp * cf.n).groupby("model").agg(n=("n", "sum"), cov=("cov", "sum"),
                                                                  width_factor=("width_factor", "median"))
        cg["picp"] = cg["cov"] / cg.n
        cg.drop(columns="cov").round(3).to_csv(TAB / "final_conformal.csv")

    fl = pd.DataFrame(fal)
    if len(fl):
        fl = fl[fl.model.isin(MAIN)].groupby(["model", "strategy", "label"]).agg(
            instances=("episodes", "size"), with_alarm=("any_alarm", "sum"), episodes=("episodes", "sum"),
            days=("days", "sum")).reset_index()
        fl["fa_per_day"] = fl.episodes / fl.days
        fl["frac_instances_alarmed"] = fl.with_alarm / fl.instances
        fl.round(4).to_csv(TAB / "final_fa_by_class.csv", index=False)
    stats_table(res, pooled)
    plots(res, als, pooled, preds, parts)
    print(summ.round(3).to_string())
    print(als.round(3).to_string())


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
        r = "PIPS-single" if m.startswith("PIPS-") and m != "PIPS-single" and "PIPS-single" in pooled else ref
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
    main = [m for m in MAIN if m in piv]
    if len(main) > 2:
        fr = stats.friedmanchisquare(*[piv[m] for m in main])
        st.attrs["friedman"] = fr
        st = pd.concat([st, pd.DataFrame([dict(model=f"Friedman({','.join(main)}) over {len(piv)} folds",
                                               p_boot=fr.pvalue, delta_auprc=fr.statistic)])])
        ranks = piv[main].rank(axis=1, ascending=False).mean()
        ranks.round(2).to_csv(TAB / "final_mean_ranks.csv", header=["mean_rank_auprc24"])
    st.round(4).to_csv(TAB / "final_stats.csv", index=False)


def plots(res, als, pooled, preds, parts):
    present = [m for m in MAIN if m in pooled]
    # 1 PR curves at 24 h, pooled over folds
    fig, ax = plt.subplots(figsize=(6, 5))
    for m in present:
        y, s = np.concatenate(pooled[m]["y"]), np.concatenate(pooled[m]["s"])
        pr, rc, _ = precision_recall_curve(y, s)
        ax.plot(rc, pr, lw=2.5 if m == "PIPS" else 1, color="k" if m == "PIPS" else None, label=f"{m} (AP {average_precision_score(y, s):.2f})")
    ax.axhline(y.mean(), color="grey", ls=":", label="random")
    ax.set(xlabel="recall", ylabel="precision", title="PR curves, event within 24 h (pooled test wells)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "pr_curves_24h.png", dpi=150); plt.close(fig)
    # 2 reliability
    fig, ax = plt.subplots(figsize=(5, 5))
    for m in [x for x in ("B2", "B3", "PIPS") if x in pooled]:
        y, p = np.concatenate(pooled[m]["y"]), np.clip(np.concatenate(pooled[m]["s"]), 0, 1)
        b = np.minimum((p * 10).astype(int), 9)
        xs = [p[b == i].mean() for i in range(10) if (b == i).sum() > 20]
        ys = [y[b == i].mean() for i in range(10) if (b == i).sum() > 20]
        ax.plot(xs, ys, "o-", label=f"{m} (ECE {ece(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "k:")
    ax.set(xlabel="predicted P(event ≤ 24 h)", ylabel="observed frequency", title="Reliability, 24 h")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "reliability_24h.png", dpi=150); plt.close(fig)
    # 3 lead-time histogram (calib-self strategy, deployable)
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in [x for x in ("B2", "PIPS") if x in pooled]:
        for s in ("calib-self",):
            L = pooled[m]["leads"][s]
            if L:
                ax.hist(L, bins=np.logspace(-1, 2.3, 15), alpha=0.5, label=f"{m} {s} (n={len(L)})")
    ax.set_xscale("log"); ax.set(xlabel="lead time before RI* +5 % (h)", ylabel="hits", title="Lead times of hits")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "lead_time_hist.png", dpi=150); plt.close(fig)
    # 4 horizon curves
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in present:
        v = [res[res.model == m][f"auprc_{h}h"].mean() for h in M.HORIZONS_H]
        ax.plot(M.HORIZONS_H, v, "o-", lw=2 if m == "PIPS" else 1, label=m)
    ax.set_xscale("log"); ax.set(xlabel="horizon (h)", ylabel="mean AUPRC over folds", title="Horizon sensitivity (E5)")
    ax.legend(fontsize=8, ncol=2); fig.tight_layout(); fig.savefig(FIG / "horizon_auprc.png", dpi=150); plt.close(fig)
    # 5 ablation bars
    abl = [m for m in ("PIPS", "PIPS-single", "PIPS-phys", "PIPS-mono", "PIPS-surv", "PIPS-patch") if m in pooled]
    if len(abl) > 1:
        g = res[res.model.isin(abl)].groupby("model")
        a = g[f"auprc_{H}h"].mean().reindex(abl)
        rec = (g["hits@fa0.0333"].sum() / g["events"].sum()).reindex(abl)
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(abl))
        ax.bar(x - 0.2, a, 0.4, label="AUPRC@24h"); ax.bar(x + 0.2, rec, 0.4, label="recall @ 1 FA/month")
        names = {"PIPS": "PIPS\n5-seed ens.", "PIPS-single": "PIPS seed 0\n(reference)", "PIPS-phys": "−physics",
                 "PIPS-mono": "−L_mono", "PIPS-surv": "−survival\n(BCE heads)", "PIPS-patch": "patch 8\nno overlap"}
        ax.set_xticks(x, [names[m] for m in abl], fontsize=8)
        ax.set(title="E2 ablations (single seed; compare with 'PIPS seed 0')"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(FIG / "ablation_bars.png", dpi=150); plt.close(fig)
    # 6 example forecasts: one event instance each in wells 24 and 22
    fig, axes = plt.subplots(2, 1, figsize=(9, 6))
    for ax, well in zip(axes, ("WELL-00024", "WELL-00022")):
        te = parts[well]["test"]
        ev_files = te[te.event].file.unique()
        if not len(ev_files):
            continue
        f = ev_files[np.argmax([(te.file == x).sum() for x in ev_files])]
        sel = (te.file == f).to_numpy()
        g = te[sel]
        ax.plot(g.t_end, g.ri_star, "k", label="RI* (feature)")
        ax.axhline(1.05, color="k", ls=":")
        ax2 = ax.twinx()
        for m in [x for x in ("B2", "PIPS") if x in preds and well in preds[x]]:
            ax2.plot(g.t_end, preds[m][well][f"test_{H}"][sel], label=f"{m} P(≤24 h)")
        tE = g.t_end.iloc[-1] + pd.Timedelta(minutes=float(g.tau_min.iloc[-1]))
        ax.axvline(tE, color="r", ls="--", label="t_E (+5 %)")
        ax.set_title(f"{well}: {f}", fontsize=9); ax2.set_ylim(0, 1)
        ax.legend(loc="upper left", fontsize=7); ax2.legend(loc="center left", fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "example_forecasts.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
