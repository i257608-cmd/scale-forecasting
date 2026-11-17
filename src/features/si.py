"""Chemistry prior: mineral saturation index SI(T, P) as a physics feature (CLAUDE.md §4 "SI
trajectory", research gap G2 "chemistry + sensors never fused").

    python src/features/si.py        # ~10 min: PHREEQC grid, then SI features for every window

3W has NO water chemistry, so the prior is built from an external brine population:
  * USGS National Produced Waters Geochemical Database v3.0 (Blondes et al. 2023,
    doi 10.5066/P9DSRCZJ; supersedes v2.3 named in Phase 2 - v2.3 is behind a ScienceBase bot
    check, v3.0 is served by USGS's public ArcGIS FeatureServer "Produced_Waters_V3"),
    filtered to Gulf of Mexico OFFSHORE samples (STATE OffShore-Louisiana / OffShore-Texas, 1,423):
    the closest public analogue of deep-water Brazilian (Campos / Santos) produced brines.
    -> data/usgs/pwgd_v3_gom_offshore.csv
  * PHREEQC (phreeqpython, Pitzer database: valid for brines up to ~6 mol/kg, GoM TDS reaches 300 g/L)
    computes SI of calcite (CaCO3) and barite (BaSO4) for each brine on a T x P grid; the grid value is
    the MEDIAN over brines = SI of a "typical offshore brine" at that temperature and pressure.
  * Feature = grid looked up at the sensor state (T-JUS-CKP, P-MON-CKP) every minute, plus rolling
    mean / std / slope over 10, 60, 360 min like the other physics features.

WHY it can still help without well chemistry: the composition is fixed, so SI(t) is a known,
non-linear function of the measured T and P: it encodes the thermodynamic DIRECTION of the scaling
drive (calcite: more supersaturated when hotter / lower pressure; barite: the opposite), which a
small network would otherwise have to learn from 24 events. It is a relative driver, not a true SI.
Limitations (reported): no CO2 degassing (no gas phase), pH fixed at the sample value, US basin
brines instead of Brazilian ones.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[2]
USGS = ROOT / "data" / "usgs" / "pwgd_v3_gom_offshore.csv"
GRID = ROOT / "data" / "usgs" / "si_grid.npz"
OUT = ROOT / "data" / "features_si" / "tabular_si.parquet"
GRID_T = np.arange(0.0, 152.5, 2.5)          # °C  (T-JUS-CKP range in 3W is ~5-120 °C)
GRID_P = np.arange(1.0, 411.0, 10.0)         # bar (P-MON-CKP range ~5-300 bar)
MINERALS = ("Calcite", "Barite")
IONS = ["Na", "K", "Ca", "Mg", "Ba", "Sr", "Cl", "SO4", "HCO3", "PH"]
N_BRINES = 200


def brines(seed: int = 0) -> pd.DataFrame:
    """Usable brines: Ca, Cl, HCO3 and pH (5-9) measured; ions not measured in a sample are filled
    with the offshore median (Ba / SO4 / Sr are often missing; barite then uses the typical value)."""
    d = pd.read_csv(USGS)
    d = d[d[["Ca", "Cl", "HCO3", "PH"]].notna().all(1) & d.PH.between(5, 9) & (d.Cl > 1000)]
    med = d[IONS].median()
    d = d[IONS].fillna(med)
    return d.sample(min(N_BRINES, len(d)), random_state=seed)


def si_grid() -> dict:
    from phreeqpython import PhreeqPython
    pp = PhreeqPython(database="pitzer.dat")
    B = brines()
    out = {m: np.full((len(B), len(GRID_T), len(GRID_P)), np.nan) for m in MINERALS}
    for b, r in enumerate(B.itertuples(index=False)):
        for i, T in enumerate(GRID_T):
            for j, P in enumerate(GRID_P):
                try:
                    s = pp.add_solution({"units": "mg/L", "temp": T, "pressure": P / 1.01325, "pH": r.PH,
                                         "Na": r.Na, "K": r.K, "Ca": r.Ca, "Mg": r.Mg, "Ba": r.Ba, "Sr": r.Sr,
                                         "Cl": f"{r.Cl} charge", "S(6)": r.SO4, "Alkalinity": f"{r.HCO3} as HCO3"})
                    for m in MINERALS:
                        out[m][b, i, j] = s.si(m)
                    s.forget()
                except Exception:   # PHREEQC non-convergence for an extreme brine -> leave NaN
                    pass
        if b % 25 == 0:
            print(f"brine {b}/{len(B)}", flush=True)
    grid = {m: np.nanmedian(v, 0) for m, v in out.items()}
    np.savez(GRID, T=GRID_T, P=GRID_P, n_brines=len(B), **grid,
             **{f"{m}_iqr": np.subtract(*np.nanpercentile(v, [75, 25], 0)) for m, v in out.items()})
    return grid


def lookup():
    g = np.load(GRID)
    f = {m: RegularGridInterpolator((g["T"], g["P"]), g[m]) for m in MINERALS}

    def si(T: np.ndarray, P_pa: np.ndarray) -> dict:
        pts = np.stack([np.clip(T, g["T"][0], g["T"][-1]), np.clip(P_pa / 1e5, g["P"][0], g["P"][-1])], -1)
        ok = ~np.isnan(pts).any(-1)
        res = {}
        for m in MINERALS:
            v = np.full(len(pts), np.nan)
            v[ok] = f[m](pts[ok])
            res[f"si_{m.lower()}"] = v
        return res
    return si


def build() -> pd.DataFrame:
    from src.features.build import ROLL_MIN, PROC
    si = lookup()
    w = pd.read_parquet(ROOT / "data" / "features" / "windows.parquet", columns=["file", "label", "end_pos"])
    parts = []
    for (label, f), g in w.groupby(["label", "file"], sort=False):
        df = pd.read_parquet(PROC / str(label) / f, columns=["T-JUS-CKP", "P-MON-CKP"])
        s = pd.DataFrame(si(df["T-JUS-CKP"].to_numpy(float), df["P-MON-CKP"].to_numpy(float)))
        feats = {}
        for c in s.columns:   # same causal rolling stats as src/features/build.py -> picked by phys_cols()
            for wmin in ROLL_MIN:
                r = s[c].rolling(wmin, min_periods=max(2, wmin // 4))
                feats[f"{c}_mean{wmin}"] = r.mean()
                feats[f"{c}_std{wmin}"] = r.std()
                feats[f"{c}_slope{wmin}"] = (r.mean() - r.mean().shift(wmin)) * (60 / wmin)
        X = pd.DataFrame(feats).iloc[g.end_pos.to_numpy()].reset_index(drop=True).astype("float32")
        parts.append(pd.concat([g[["file", "label", "end_pos"]].reset_index(drop=True), X], axis=1))
    t = pd.concat(parts, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    t.to_parquet(OUT)
    return t


def with_si(tab: pd.DataFrame) -> pd.DataFrame:
    """Add the SI columns to a tabular() frame (only for '-SI' model variants)."""
    # label in the key: synthetic file names repeat across class folders
    return tab.merge(pd.read_parquet(OUT), on=["file", "label", "end_pos"], how="left", validate="one_to_one")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    if not GRID.exists():
        si_grid()
    g = np.load(GRID)
    for m in MINERALS:
        print(m, "SI at (20 °C, 1 bar), (60 °C, 100 bar), (100 °C, 300 bar):",
              [round(float(RegularGridInterpolator((g["T"], g["P"]), g[m])([[a, b]])[0]), 2)
               for a, b in ((20, 1), (60, 101), (100, 301))])
    t = build()
    print(f"{len(t):,} windows with SI features -> {OUT}")
