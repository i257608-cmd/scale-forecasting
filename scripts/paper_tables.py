"""Write the LaTeX result tables of paper/main.tex from results/tables/*.csv (no hand-typed numbers).

    python scripts/paper_tables.py          # -> paper/tables/*.tex
Run after `python experiments/analyze.py` (and `--split v1v2`).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T, OUT = ROOT / "results" / "tables", ROOT / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
m = pd.read_csv(T / "final_main.csv").set_index("model")
al = pd.read_csv(T / "final_alarm.csv")
cf = pd.read_csv(T / "final_conformal.csv").set_index("model")
st = pd.read_csv(T / "final_stats.csv")
sd = pd.read_csv(T / "final_seeds.csv").set_index("model") if (T / "final_seeds.csv").exists() else pd.DataFrame()
NAME = {"B0": "B0 $\\widetilde{\\mathrm{RI}}^*$ rule", "B1": "B1 Logistic regression", "B2": "B2 GBDT", "B3": "B3 LSTM",
        "B4": "B4 GRU", "B5": "B5 CNN-LSTM", "B6": "B6 TCN", "B7": "B7 Transformer-LSTM$^\\dagger$",
        "B8": "B8 PatchTST$^\\dagger$", "B8i": "B8i iTransformer", "B9": "B9 LSTM-AE detector$^\\dagger$",
        "PIPS": "PI-PatchSurv (proposed)"}
cs = al[al.strategy == "calib-self"].set_index("model")


def f(v, d=3):
    return "--" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def pm(model, col, d=3):
    """mean +/- SD over seeds if several seeds exist, else the single value from the main table."""
    if model in sd.index and sd.loc[model, "seeds"] > 1:
        return f"{sd.loc[model, col + '_mean']:.{d}f}$\\pm${sd.loc[model, col + '_std']:.{d}f}"
    return None


def write(name, text):
    (OUT / f"{name}.tex").write_text(text)
    print("wrote", name)


def main_table(models, label, caption):
    rows = []
    for k in models:
        if k not in m.index:
            continue
        r = m.loc[k]
        nm = NAME.get(k.removesuffix("-tuned"), k)
        a = pm(k, "auprc_24h") or f(r.auprc_24h)
        b = pm(k, "auroc_24h") or f(r.auroc_24h)
        hits = pm(k, "hits", 1) or (f"{int(cs.loc[k].hits)}" if k in cs.index else "--")
        nseed = int(sd.loc[k, "seeds"]) if k in sd.index else 1
        bold = "\\textbf{" if k == "PIPS" else "{"
        rows.append(f"{bold}{nm}}} & {nseed} & {a} & {b} & {f(r.c_index)} & {f(r.get('ibs'))} & {f(r.ece_24h)} & "
                    f"{f(r.get('bss_vs_b0'), 2)} & {f(r['recall@fa0.0333'], 2)} & {hits} & "
                    f"{f(cf.picp.get(k), 2) if k in cf.index else '--'} \\\\")
    write(label, f"""\\begin{{table*}}[t]
\\caption{{{caption}}}\\label{{tab:{label}}}
\\scriptsize\\setlength{{\\tabcolsep}}{{3pt}}
\\resizebox{{\\textwidth}}{{!}}{{\\begin{{tabular}}{{lcccccccccc}}
\\toprule
Model & Seeds & AUPRC@24h & AUROC@24h & C-index & IBS & ECE & BSS & Rec.@1/mo & Hits/24 & PICP \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}}}
\\footnotetext{{AUPRC/AUROC and hits: mean$\\pm$SD over single-seed runs where several seeds exist; other columns: ensemble of the available seeds. BSS: Brier skill score at 24~h relative to B0. Rec.@1/mo: threshold-free recall at one false alarm per well-month. Hits: events caught out of 24 with the deployed calib-self threshold. PICP: coverage of the 90\\,\\% conformal interval. $^\\dagger$Re-implementation of a base-paper architecture.}}
\\end{{table*}}
""")


FIXED = ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8i", "B9", "PIPS"]
main_table(FIXED, "main", "Forecast quality on unseen wells (six leave-one-well-out folds, 24 events), fixed hyper-parameters")
main_table([f"{x}-tuned" for x in FIXED[1:]], "tuned", "As Table~\\ref{tab:main}, with Optuna-tuned hyper-parameters (equal budget, tuned on calibration wells)")

# deployed alarms: one row per model, both threshold strategies side by side
rows = []
for k in ["B2", "B3", "B4", "B4-tuned", "B6", "B7", "B7-tuned", "B8", "B8i-tuned", "PIPS", "PIPS-tuned"]:
    r1 = al[(al.model == k) & (al.strategy == "calib")]
    r2 = al[(al.model == k) & (al.strategy == "calib-self")]
    if not (len(r1) and len(r2)):
        continue
    r1, r2 = r1.iloc[0], r2.iloc[0]
    lead = "--" if np.isnan(r2.lead_med_h) else f"{r2.lead_med_h:.1f} ({r2.lead_iqr_h:.1f})"
    nm = NAME.get(k.removesuffix("-tuned"), k).replace("$^\\dagger$", "").replace(" (proposed)", "") + (" (tuned)" if k.endswith("-tuned") else "")
    rows.append(f"{nm} & {int(r1.hits)} & {r1.fa_per_day:.3f} & {int(r2.hits)} & {r2.fa_per_day:.3f} & {f(r2.get('event_precision'), 2)} & {lead} \\\\")
write("alarms", f"""\\begin{{table*}}[t]
\\caption{{Deployed alarms with thresholds fixed on the calibration wells (ensemble of the available seeds; 24 events, about 248 negative well-days). FA: false alarms per negative well-day}}\\label{{tab:alarms}}
\\footnotesize
\\resizebox{{\\textwidth}}{{!}}{{\\begin{{tabular}}{{lcccccc}}
\\toprule
 & \\multicolumn{{2}}{{c}}{{Raw score (calib)}} & \\multicolumn{{4}}{{c}}{{Self-referenced score (calib-self)}} \\\\
\\cmidrule{{2-3}}\\cmidrule{{4-7}}
Model & Hits/24 & FA & Hits/24 & FA & Event precision & Lead, h: median (IQR) \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}}}
\\end{{table*}}
""")

# ablations
ABL = [("PIPS", "Full model (reference)"), ("PIPS-phys", "$-$ physics branch"), ("PIPS-mono", "$-$ $\\mathcal{L}_{\\mathrm{mono}}$"),
       ("PIPS-surv", "$-$ survival head (per-horizon BCE)"), ("PIPS-nopatch", "$-$ patching (patch length 1)"),
       ("PIPS-patch", "patch 8, no overlap"), ("PIPS-nosyn", "$-$ synthetic training data"),
       ("PIPS-SI", "$+$ PHREEQC SI prior"), ("PIPS-tuned", "Optuna-tuned")]
rows = []
for k, lab in ABL:
    if k not in m.index:
        rows.append(f"{lab} & \\multicolumn{{7}}{{c}}{{not available}} \\\\")
        continue
    r = m.loc[k]
    n = int(sd.loc[k, "seeds"]) if k in sd.index else 1
    a = pm(k, "auprc_24h") or f(r.auprc_24h)
    b = pm(k, "auroc_24h") or f(r.auroc_24h)
    hits = pm(k, "hits", 1) or (f"{int(cs.loc[k].hits)}" if k in cs.index else "--")
    s = st[st.model == k]
    d = "--" if not len(s) else f"{s.delta_auprc.iloc[0]:+.3f} [{s.lo.iloc[0]:+.2f}, {s.hi.iloc[0]:+.2f}], {s.p_boot_holm.iloc[0]:.2f}"
    rows.append(f"{lab} & {n} & {a} & {b} & {f(r.c_index)} & {f(r.ece_24h)} & {hits} & {d} \\\\")
write("ablation", f"""\\begin{{table*}}[t]
\\caption{{Ablation of PI-PatchSurv. $\\Delta$: reference minus variant on pooled AUPRC@24h (paired instance bootstrap, 95\\,\\% CI, Holm-corrected $p$)}}\\label{{tab:ablation}}
\\scriptsize\\setlength{{\\tabcolsep}}{{3pt}}
\\resizebox{{\\textwidth}}{{!}}{{\\begin{{tabular}}{{lccccccc}}
\\toprule
Variant & Seeds & AUPRC@24h & AUROC@24h & C-index & ECE & Hits/24 & $\\Delta$ pooled AUPRC [CI], $p$ \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}}}
\\end{{table*}}
""")

# robustness (E3) and resampling
rb = pd.read_csv(T / "e3_robustness.csv").pivot(index="condition", columns="model", values="auprc_24h")
rs = pd.read_csv(T / "e3_resampling.csv").pivot(index="rate_s", columns="model", values="auprc_24h")
order = [("clean 0", "clean"), ("noise 0.01", "Gaussian noise 1\\,\\%"), ("noise 0.03", "Gaussian noise 3\\,\\%"),
         ("noise 0.05", "Gaussian noise 5\\,\\%"), ("dropout 1", "1 sensor dropped"), ("dropout 2", "2 sensors dropped"),
         ("dropout 3", "3 sensors dropped"), ("frozen 60", "1 sensor frozen 60 min")]
rows = [f"{lab} & {rb.loc[c, 'PIPS']:.3f} & {rb.loc[c, 'B3']:.3f} \\\\" for c, lab in order if c in rb.index]
rows += [f"resampled to {int(r)} s & {rs.loc[r, 'PIPS']:.3f} & {rs.loc[r, 'B3']:.3f} \\\\" for r in rs.index]
write("robustness", f"""\\begin{{table}}[t]
\\caption{{Robustness: AUPRC@24h under corruptions of the test inputs}}\\label{{tab:robust}}
\\scriptsize
\\begin{{tabular*}}{{\\columnwidth}}{{@{{\\extracolsep\\fill}}lcc}}
\\toprule
Condition & PI-PatchSurv & B3 LSTM \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular*}}
\\end{{table}}
""")

# false alarms by class
fa = pd.read_csv(T / "final_fa_by_class.csv")
fa = fa[fa.strategy == "calib-self"]
cls = {0: "0 normal", 6: "6 quick restr.", 8: "8 hydrate (prod.)", 9: "9 hydrate (serv.)", 3: "3 slugging"}
models = ["B2", "B4-tuned", "B7-tuned", "PIPS"]
rows = []
for c, lab in cls.items():
    g = fa[fa.label == c].set_index("model")
    n = int(g.instances.iloc[0]) if len(g) else 0
    rows.append(f"{lab} & {n} & " + " & ".join(str(int(g.loc[mm, 'episodes'])) if mm in g.index else "--" for mm in models) + " \\\\")
other = fa[fa.label.isin([1, 2, 4, 5])].groupby("model").agg(instances=("instances", "sum"), episodes=("episodes", "sum"))
rows.append(f"1, 2, 4, 5 & {int(other.instances.iloc[0])} & " + " & ".join(str(int(other.loc[mm, 'episodes'])) if mm in other.index else "--" for mm in models) + " \\\\")
write("fa_class", f"""\\begin{{table}}[t]
\\caption{{False-alarm episodes per confounder class (calib-self; all real non-scaling instances tested once); $^\\ddagger$tuned}}\\label{{tab:faclass}}
\\scriptsize
\\begin{{tabular*}}{{\\columnwidth}}{{@{{\\extracolsep\\fill}}lccccc}}
\\toprule
Class & Inst. & B2 & B4$^\\ddagger$ & B7$^\\ddagger$ & Ours \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular*}}
\\end{{table}}
""")

# v1 -> v2
v = pd.read_csv(T / "e4_v1v2_main.csv").set_index("model")
va = pd.read_csv(T / "e4_v1v2_alarm.csv")
va = va[va.strategy == "calib-self"].set_index("model")
rows = []
for k in ["B0", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8i", "B9", "PIPS"]:
    if k in v.index:
        h = f"{int(va.loc[k].hits)} / {va.loc[k].fa_per_day:.3f}" if k in va.index else "--"
        rows.append(f"{k if k != 'PIPS' else 'PI-PatchSurv'} & {v.loc[k].auprc_24h:.3f} & {v.loc[k].auroc_24h:.3f} & {h} \\\\")
write("v1v2", f"""\\begin{{table}}[t]
\\caption{{Generalisation: trained on wells of 3W v1.1.1, tested on wells added in v2.0.0 (20 events)}}\\label{{tab:v1v2}}
\\scriptsize
\\begin{{tabular}}{{lccc}}
\\toprule
Model & AUPRC@24h & AUROC@24h & Hits/20 / FA \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")
