"""Print the markdown tables of docs/Phase3_Report.md from results/tables/final_*.csv.

    python scripts/report_tables.py > /tmp/tables.md
Run after `python experiments/analyze.py`; paste the sections into the report. No numbers are typed by hand.
"""
from pathlib import Path

import pandas as pd

T = Path(__file__).resolve().parents[1] / "results" / "tables"
m = pd.read_csv(T / "final_main.csv").set_index("model")
a = pd.read_csv(T / "final_alarm.csv")
cf = pd.read_csv(T / "final_conformal.csv").set_index("model")
st = pd.read_csv(T / "final_stats.csv")
NAMES = {"B0": "B0 RI* threshold", "B1": "B1 logistic", "B2": "B2 GBDT", "B3": "B3 LSTM", "B4": "B4 GRU",
         "B5": "B5 CNN-LSTM", "B6": "B6 TCN", "B7": "B7 Transformer-LSTM", "B8": "B8 PatchTST",
         "B8i": "B8i iTransformer", "B9": "B9 LSTM-AE detector", "PIPS": "**PI-PatchSurv** (5 seeds)"}
f2 = lambda v: "–" if pd.isna(v) else f"{v:.2f}"   # noqa: E731
f3 = lambda v: "–" if pd.isna(v) else f"{v:.3f}"   # noqa: E731


def main_table(models):
    out = ["| Model | AUPRC@24h | AUROC@24h | C-index | Brier@24h | IBS | ECE@24h | Brier skill vs B0 | Recall @ 1 FA/month | @ 1 FA/day | Pooled AUPRC [95 % CI] | PICP (90 %) |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in models:
        if k not in m.index:
            continue
        r = m.loc[k]
        name = NAMES.get(k.removesuffix("-tuned"), k) + (" (tuned)" if k.endswith("-tuned") else "")
        out.append(f"| {name} | {f3(r.auprc_24h)} | {f3(r.auroc_24h)} | {f3(r.c_index)} | {f3(r.brier_24h)} | "
                   f"{f3(r.get('ibs'))} | {f3(r.ece_24h)} | {f2(r.get('bss_vs_b0'))} | {f2(r['recall@fa0.0333'])} | "
                   f"{f2(r['recall@fa1'])} | {f2(r.pooled_auprc_24h)} [{f2(r.pooled_lo)}, {f2(r.pooled_hi)}] | "
                   f"{f2(cf.picp.get(k)) if k in cf.index else '–'} |")
    return "\n".join(out)


FIXED = ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8i", "B9", "PIPS"]
TUNED = ["B1-tuned", "B2-tuned", "B3-tuned", "B4-tuned", "B5-tuned", "B6-tuned", "B7-tuned", "B8-tuned",
         "B8i-tuned", "B9-tuned", "PIPS-tuned"]
print("## TABLE_FIXED\n" + main_table(FIXED))
print("\n## TABLE_TUNED\n" + main_table(TUNED))

al = a[a.strategy.isin(["calib", "calib-self"])].copy()
keep = ["B2", "B3", "B4", "B4-tuned", "B6", "B7", "B7-tuned", "B8", "B8i", "B8i-tuned", "PIPS", "PIPS-tuned"]
out = ["| Model | Strategy | Hits / 24 | False alarms / well-day | Event precision | Lead time median (IQR), h |",
       "|---|---|---|---|---|---|"]
for k in keep:
    for s in ("calib", "calib-self"):
        r = al[(al.model == k) & (al.strategy == s)]
        if len(r):
            r = r.iloc[0]
            lead = "–" if pd.isna(r.lead_med_h) else f"{r.lead_med_h:.1f} ({r.lead_iqr_h:.1f})"
            out.append(f"| {k} | {s} | {int(r.hits)} | {r.fa_per_day:.3f} | {f2(r.get('event_precision'))} | {lead} |")
print("\n## TABLE_ALARM\n" + "\n".join(out))

ABL = [("PIPS", "5-seed ensemble"), ("PIPS-single", "seed 0 (reference)"), ("PIPS-phys", "− physics branch"),
       ("PIPS-mono", "− L_mono"), ("PIPS-surv", "− survival head (per-horizon BCE)"),
       ("PIPS-nopatch", "− patching (patch 1)"), ("PIPS-patch", "patch 8, no overlap"),
       ("PIPS-nosyn", "− synthetic data (real only)"), ("PIPS-SI", "+ SI chemistry prior (5 seeds)"),
       ("PIPS-tuned", "Optuna-tuned (5 seeds)")]
calself = a[a.strategy == "calib-self"].set_index("model")
out = ["| Variant | AUPRC@24h | AUROC@24h | C-index | Pooled AUPRC | Recall @ 1 FA/month | Deployed hits / FA per day (calib-self) | Δ pooled AUPRC vs reference [95 % CI], Holm p |",
       "|---|---|---|---|---|---|---|---|"]
for k, label in ABL:
    if k not in m.index:
        out.append(f"| {label} | not available | | | | | | |")
        continue
    r, s = m.loc[k], st[st.model == k]
    d = "—" if not len(s) else f"{s.delta_auprc.iloc[0]:+.3f} [{s.lo.iloc[0]:+.2f}, {s.hi.iloc[0]:+.2f}] vs {s.reference.iloc[0]}, p = {s.p_boot_holm.iloc[0]:.2f}"
    dep = f"{int(calself.loc[k].hits)} / {calself.loc[k].fa_per_day:.3f}" if k in calself.index else "–"
    out.append(f"| {label} | {f3(r.auprc_24h)} | {f3(r.auroc_24h)} | {f3(r.c_index)} | {f2(r.pooled_auprc_24h)} | "
               f"{f2(r['recall@fa0.0333'])} | {dep} | {d} |")
print("\n## TABLE_ABLATION\n" + "\n".join(out))
print("\n## NEMENYI\n" + "\n".join(f"{r.model}: CD = {r.delta_auprc:.2f} ranks; {r.reference}"
                                    for r in st[st.model.str.startswith("Nemenyi")].itertuples()))
print("\n## DELTA_T (median |τ̂ − τ|, h)\n" + cf["delta_t_med_h"].round(1).to_string() if "delta_t_med_h" in cf else "")
