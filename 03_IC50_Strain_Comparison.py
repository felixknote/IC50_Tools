#!/usr/bin/env python3
"""
IC50 Strain Comparison: MG1655 vs ACE-1
Author: Felix Knote

Design: 11 plates per strain, 22 ABs (2 per plate), one plate per day.
        3 technical row replicates per AB per plate. No biological replicates.
        → Error bars from within-plate technical SD (propagated for log2 FC).
        → No significance testing / stars.
"""

import os
import re
import warnings

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
BASE_DIR = r"L:\43-RVZ\AIMicroscopy\Mitarbeiter\3_Users\Felix\2026_04_24_AI4AB_IC50_Determination"

STRAINS = {
    "MG1655": {
        "dir": os.path.join(BASE_DIR, "MG1655"),
        "map": os.path.join(BASE_DIR, "MG1655", "AB Plate Map.csv"),
        "color": "#2166ac",
    },
    "ACE-1": {
        "dir": os.path.join(BASE_DIR, "ACE-1"),
        "map": os.path.join(BASE_DIR, "ACE-1", "AB Plate Map.csv"),
        "color": "#d6604d",
    },
}

OUTPUT_DIR = os.path.join(BASE_DIR, "Strain Comparison")
# ─────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit

# ─────────────────────────────────────────────────────────────
# Plate geometry
# ─────────────────────────────────────────────────────────────
N_DILUTIONS = 10
DILUTION_COLS = list(range(2, 12))
BLANK_WELLS = ["A1", "A12", "H1", "H12"]
GROWTH_CTRL_WELLS = [f"A{c}" for c in DILUTION_COLS]
AB1_ROWS = list("BCD")
AB2_ROWS = list("EFG")

# ─────────────────────────────────────────────────────────────
# Data loading & parsing
# ─────────────────────────────────────────────────────────────
_WELL_LINE = re.compile(r'^([A-Ha-h])(\d{1,2})\s*;\s*([0-9.,\-]+)')

def _to_float(val) -> float:
    if pd.isna(val):
        return np.nan
    s = str(val).strip().replace(" ", "")
    if s.count(",") == 1 and s.count(".") >= 1 and s.index(",") > s.rindex("."):
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan

def load_plate(filepath: str) -> pd.DataFrame:
    records = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = _WELL_LINE.match(line.strip())
            if m:
                records.append({
                    "Well": m.group(1).upper() + str(int(m.group(2))),
                    "OD": _to_float(m.group(3)),
                })
    if not records:
        raise ValueError(f"No well data found in: {filepath}")
    df = pd.DataFrame(records)
    if len(df) != 96:
        warnings.warn(f"{os.path.basename(filepath)}: expected 96 wells, got {len(df)}")
    return df

def load_plate_map(filepath: str) -> pd.DataFrame:
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        raw = None
        for sep in [",", ";", "\t"]:
            try:
                candidate = pd.read_csv(filepath, sep=sep)
                if candidate.shape[1] >= 3:
                    raw = candidate
                    break
            except Exception:
                continue
        if raw is None:
            raw = pd.read_csv(filepath)
    else:
        raw = pd.read_excel(filepath)

    raw.columns = [c.strip() for c in raw.columns]
    name_col, conc_col = raw.columns[1], raw.columns[2]

    records = []
    for i in range(0, len(raw), 2):
        ab1 = raw.iloc[i]
        ab2 = raw.iloc[i + 1] if (i + 1) < len(raw) else None
        records.append({
            "plate_id": i // 2 + 1,
            "antibiotic_1_name": str(ab1[name_col]).strip(),
            "antibiotic_1_max_conc": float(ab1[conc_col]),
            "antibiotic_2_name": str(ab2[name_col]).strip() if ab2 is not None else None,
            "antibiotic_2_max_conc": float(ab2[conc_col]) if ab2 is not None else None,
        })
    return pd.DataFrame(records)

def parse_plate_layout(plate_df: pd.DataFrame):
    def wells_od(wells):
        return plate_df.loc[plate_df["Well"].isin(wells), "OD"].values.astype(float)

    blank_od = float(np.nanmean(wells_od(BLANK_WELLS)))
    growth_od = wells_od(GROWTH_CTRL_WELLS)
    ab1_od = {row: wells_od([f"{row}{c}" for c in DILUTION_COLS]) for row in AB1_ROWS}
    ab2_od = {row: wells_od([f"{row}{c}" for c in DILUTION_COLS]) for row in AB2_ROWS}
    return blank_od, growth_od, ab1_od, ab2_od

def normalize_data(raw_od: np.ndarray, blank_od: float, growth_od: np.ndarray) -> np.ndarray:
    corrected = np.clip(raw_od - blank_od, 0, None)
    growth_corrected = np.clip(growth_od - blank_od, 0, None)
    norm_factor = float(np.nanmean(growth_corrected))
    if norm_factor == 0:
        warnings.warn("Growth control mean is zero – normalization skipped.")
        return corrected
    return corrected / norm_factor * 100.0

# ─────────────────────────────────────────────────────────────
# Dose-response fitting (4PL)
# ─────────────────────────────────────────────────────────────
def four_param_logistic(x, bottom, top, log_ic50, hill):
    return bottom + (top - bottom) / (1.0 + 10.0 ** ((log_ic50 - np.log10(x)) * hill))

def fit_dose_response(concentrations: np.ndarray, responses: np.ndarray):
    mask = ~np.isnan(responses) & (concentrations > 0)
    x, y = concentrations[mask], responses[mask]
    if len(x) < 4:
        raise RuntimeError("Fewer than 4 valid data points.")

    lx = np.log10(x)
    lb = [ 0.0,  0.0, lx.min() - 1.0, 0.1]
    ub = [200.0, 200.0, lx.max() + 1.0, 10.0]
    p0 = [
        np.clip(y.min(),       lb[0] + 1e-9, ub[0] - 1e-9),
        np.clip(y.max(),       lb[1] + 1e-9, ub[1] - 1e-9),
        np.clip(np.median(lx), lb[2] + 1e-9, ub[2] - 1e-9),
        np.clip(1.0,           lb[3] + 1e-9, ub[3] - 1e-9),
    ]

    popt, _ = curve_fit(
        four_param_logistic, x, y,
        p0=p0, bounds=(lb, ub), maxfev=50_000, method="trf",
    )

    y_pred = four_param_logistic(x, *popt)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return 10.0 ** popt[2], popt, r2

def calculate_ic50(rep_responses: dict, concentrations: np.ndarray):
    """
    Fits one 4PL curve per technical row replicate.
    Returns per-rep results, the mean IC50, and the within-plate technical SD.
    (n=3 technical replicates per plate; no biological replicates across days.)
    """
    results = []
    for rep, resp in rep_responses.items():
        try:
            ic50, popt, r2 = fit_dose_response(concentrations, resp)
            results.append({"rep": rep, "ic50": ic50, "popt": popt, "r2": r2, "responses": resp})
        except Exception as exc:
            warnings.warn(f"Fit failed for {rep}: {exc}")
            results.append({"rep": rep, "ic50": np.nan, "popt": None, "r2": np.nan, "responses": resp})

    ic50_vals = [r["ic50"] for r in results if not np.isnan(r["ic50"])]
    mean_ic50 = float(np.mean(ic50_vals)) if ic50_vals else np.nan
    # SD across the 3 technical row replicates (within-plate technical SD)
    tech_sd = float(np.std(ic50_vals, ddof=1)) if len(ic50_vals) > 1 else 0.0
    return results, mean_ic50, tech_sd

# ─────────────────────────────────────────────────────────────
# File discovery
# ─────────────────────────────────────────────────────────────
_STRAIN_FILE_RE = re.compile(
    r'\d{4}_\d{2}_\d{2}_FK(?:_IC50)?_([A-Za-z0-9][A-Za-z0-9-]*)_P\d+\.csv$',
    re.IGNORECASE,
)

def _extract_plate_id(filename: str) -> int:
    m = re.search(r'_[Pp](\d+)', filename)
    if m:
        return int(m.group(1))
    m = re.search(r'[Pp]late[_\s\-]?(\d+)', filename)
    if m:
        return int(m.group(1))
    m = re.search(r'(\d+)', filename)
    return int(m.group(1)) if m else 0

# ─────────────────────────────────────────────────────────────
# Process one strain
# ─────────────────────────────────────────────────────────────
def process_strain(strain_name: str, input_dir: str, plate_map_path: str) -> list:
    plate_map = load_plate_map(plate_map_path)
    map_basename = os.path.basename(plate_map_path)
    plate_files = sorted(
        f for f in os.listdir(input_dir)
        if not f.startswith("~") and f != map_basename
        and (
            (m := _STRAIN_FILE_RE.search(f)) is not None
            and m.group(1).lower() == strain_name.lower()
        )
    )

    results = []
    for fname in plate_files:
        fpath = os.path.join(input_dir, fname)
        plate_id = _extract_plate_id(fname)
        map_row = plate_map[plate_map["plate_id"] == plate_id]
        if map_row.empty:
            continue
        map_row = map_row.iloc[0]

        try:
            plate_df = load_plate(fpath)
        except Exception as exc:
            warnings.warn(f"[{strain_name}] Could not load {fname}: {exc}")
            continue

        blank_od, growth_od, ab1_od, ab2_od = parse_plate_layout(plate_df)

        antibiotics = []
        if map_row["antibiotic_1_name"]:
            antibiotics.append((map_row["antibiotic_1_name"], float(map_row["antibiotic_1_max_conc"]), ab1_od))
        if map_row["antibiotic_2_name"]:
            antibiotics.append((map_row["antibiotic_2_name"], float(map_row["antibiotic_2_max_conc"]), ab2_od))

        for ab_name, ab_max_conc, ab_od in antibiotics:
            conc = np.array([ab_max_conc / 2 ** i for i in range(N_DILUTIONS)])
            ab_norm = {f"Rep {i+1}": normalize_data(od, blank_od, growth_od)
                       for i, od in enumerate(ab_od.values())}

            rep_results, mean_ic50, tech_sd = calculate_ic50(ab_norm, conc)
            results.append({
                "ab_name": ab_name,
                "plate_id": plate_id,
                "mean_ic50": mean_ic50,
                "sd_ic50": tech_sd,   # within-plate technical SD from 3 row replicates
                "rep_ic50s": [r["ic50"] for r in rep_results],
                "concentrations": conc,
                "rep_results": rep_results,
            })

    return results

# ─────────────────────────────────────────────────────────────
# Comparison table
# ─────────────────────────────────────────────────────────────
def build_comparison_df(strain_results: dict) -> pd.DataFrame:
    """
    Single-plate design: n=1 per strain per AB (no biological replicates).
    Error bars come exclusively from within-plate technical SD (3 row replicates).
    Fold-change error is propagated from technical SDs.
    No Mann-Whitney / significance testing.
    """
    strain_names = list(strain_results.keys())
    s0, s1 = strain_names

    ab_data: dict = {}
    for strain, results in strain_results.items():
        for r in results:
            entry = ab_data.setdefault(r["ab_name"], {}).setdefault(
                strain, {"ic50": np.nan, "tech_sd": np.nan}
            )
            entry["ic50"] = r["mean_ic50"]
            entry["tech_sd"] = r["sd_ic50"]

    rows = []
    for ab_name in sorted(ab_data):
        row = {"Antibiotic": ab_name}
        for strain in strain_names:
            strain_entry = ab_data[ab_name].get(strain, {"ic50": np.nan, "tech_sd": np.nan})
            row[f"{strain}_IC50"] = strain_entry["ic50"]
            row[f"{strain}_SD"] = strain_entry["tech_sd"]
        rows.append(row)

    df = pd.DataFrame(rows)
    df["Fold_Change"] = df[f"{s1}_IC50"] / df[f"{s0}_IC50"]
    df["Log2_FC"] = np.log2(df["Fold_Change"])

    # Propagated tech-SD error on log2(FC): σ = (1/ln2) * sqrt((σ1/IC50_1)² + (σ0/IC50_0)²)
    log2_fc_err = []
    for _, row in df.iterrows():
        ic50_0, ic50_1 = row[f"{s0}_IC50"], row[f"{s1}_IC50"]
        tsd_0, tsd_1 = row[f"{s0}_SD"], row[f"{s1}_SD"]
        if all(not np.isnan(v) and v > 0 for v in [ic50_0, ic50_1, tsd_0, tsd_1]):
            err = (1.0 / np.log(2)) * np.sqrt((tsd_1 / ic50_1) ** 2 + (tsd_0 / ic50_0) ** 2)
        else:
            err = np.nan
        log2_fc_err.append(err)
    df["Log2_FC_tech_err"] = log2_fc_err

    # Aliases expected by main() and plot_fold_change()
    df["Signed_FC"] = df["Log2_FC"]
    df["Signed_FC_err"] = df["Log2_FC_tech_err"]

    return df

# ─────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────

def plot_grouped_bar(df: pd.DataFrame, strain_names: list, save_dir: str) -> str:
    """Grouped bar chart with IC50 ± within-plate technical SD. No significance stars."""
    s0, s1 = strain_names
    c0, c1 = STRAINS[s0]["color"], STRAINS[s1]["color"]
    ab_names = df["Antibiotic"].tolist()
    x = np.arange(len(ab_names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(14, len(ab_names) * 0.9), 7))

    ax.bar(x - w / 2, df[f"{s0}_IC50"], w,
           yerr=df[f"{s0}_SD"], capsize=4, color=c0, alpha=0.85, label=s0, zorder=3)
    ax.bar(x + w / 2, df[f"{s1}_IC50"], w,
           yerr=df[f"{s1}_SD"], capsize=4, color=c1, alpha=0.85, label=s1, zorder=3)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(ab_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("IC50 (µg/mL)", fontsize=12)
    ax.set_title(
        f"IC50 Comparison: {s0} vs {s1}\nError bars = ± technical SD (3 row replicates per plate)",
        fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, which="both", axis="y", ls="--", lw=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()

    path = os.path.join(save_dir, "01_IC50_grouped_bar.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_fold_change(df: pd.DataFrame, strain_names: list, save_dir: str) -> str:
    """Horizontal bar chart of log2 FC with propagated technical SD. No significance stars."""
    s0, s1 = strain_names
    sub = df.dropna(subset=["Signed_FC"]).sort_values("Signed_FC", ascending=True)
    colors = [STRAINS[s1]["color"] if v > 0 else STRAINS[s0]["color"] for v in sub["Log2_FC"]]
    y = np.arange(len(sub))

    xerr = sub["Signed_FC_err"].fillna(0).values
    err_label = "± propagated technical SD (3 row replicates per plate)"

    fig, ax = plt.subplots(figsize=(9, max(6, len(sub) * 0.42)))
    ax.barh(y, sub["Log2_FC"], xerr=xerr, color=colors, alpha=0.85, zorder=3,
            capsize=4, error_kw={"elinewidth": 1.2, "ecolor": "black", "alpha": 0.7})
    ax.axvline(1,  color="gray", lw=0.8, ls=":")   

    ax.set_yticks(y)
    ax.set_yticklabels(sub["Antibiotic"], fontsize=9)
    ax.set_xlabel(f"Fold-change  (IC50 {s1} / IC50 {s0})", fontsize=12)
    ax.set_title(
        f"Fold-Change in IC50: {s1} relative to {s0}\n{err_label}",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, axis="x", ls="--", lw=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(handles=[
        Line2D([0], [0], color=STRAINS[s1]["color"], lw=8, alpha=0.85,
               label=f"{s1} higher IC50 (more resistant)"),
        Line2D([0], [0], color=STRAINS[s0]["color"], lw=8, alpha=0.85,
               label=f"{s0} higher IC50 (more resistant)"),
    ], fontsize=9, loc="lower right")
    plt.tight_layout()

    path = os.path.join(save_dir, "02_fold_change.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


_REP_MARKERS = ["o", "s", "^"]

def plot_individual_overlays(strain_results: dict, strain_names: list, save_dir: str) -> str:
    """
    One figure per antibiotic: 3 technical row-replicate curves for each strain,
    mean 4PL fit, ±tech SD band. IC50 annotated as mean ± technical SD.
    """
    s0, s1 = strain_names
    c0, c1 = STRAINS[s0]["color"], STRAINS[s1]["color"]
    curves_dir = os.path.join(save_dir, "dose_response_curves")
    os.makedirs(curves_dir, exist_ok=True)

    lookup: dict = {}
    for s in strain_names:
        ab_dict: dict = {}
        for r in strain_results[s]:
            ab_dict.setdefault(r["ab_name"], []).append(r)
        lookup[s] = ab_dict

    all_abs = sorted(set(lookup[s0]) | set(lookup[s1]))

    for ab_name in all_abs:
        fig, ax = plt.subplots(figsize=(8, 6))

        for strain, color in [(s0, c0), (s1, c1)]:
            plate_list = lookup[strain].get(ab_name, [])
            if not plate_list:
                continue

            res = plate_list[0]  # single plate per AB
            conc = res["concentrations"]
            xfit = np.logspace(np.log10(conc.min()), np.log10(conc.max()), 300)

            rep_curves = []
            for i, rep in enumerate(res["rep_results"]):
                marker = _REP_MARKERS[i % len(_REP_MARKERS)]
                ax.scatter(conc, rep["responses"],
                           color=color, marker=marker, s=40, zorder=5,
                           alpha=0.75, label=f"{strain} {rep['rep']}")
                if rep["popt"] is not None:
                    yf = four_param_logistic(xfit, *rep["popt"])
                    ax.plot(xfit, yf, color=color, lw=1.2, alpha=0.55)
                    rep_curves.append(yf)

            if rep_curves:
                ymean = np.nanmean(rep_curves, axis=0)
                ystd  = np.nanstd(rep_curves, axis=0)
                ax.plot(xfit, ymean, color=color, lw=2.5,
                        ls="--" if strain == s0 else "-",
                        label=f"{strain} mean fit")
                ax.fill_between(xfit, ymean - ystd, ymean + ystd,
                                color=color, alpha=0.15, label=f"{strain} ±tech SD")

            mean_ic50 = res["mean_ic50"]
            tech_sd   = res["sd_ic50"]
            if not np.isnan(mean_ic50):
                sd_str = f" ± {tech_sd:.4g}" if not np.isnan(tech_sd) else ""
                ax.axvline(mean_ic50, color=color, ls=":", lw=2,
                           label=f"IC50 {strain} = {mean_ic50:.4g}{sd_str} µg/mL")

        ax.set_xscale("log")
        ax.set_xlabel("Concentration (µg/mL)", fontsize=12)
        ax.set_ylabel("Relative growth (%)", fontsize=12)
        ax.set_ylim(bottom=0)
        ax.set_title(ab_name, fontsize=14, fontweight="bold")
        ax.grid(True, which="both", ls="--", lw=0.6, alpha=0.6)
        ax.legend(fontsize=9)
        plt.tight_layout()

        fname = ab_name.replace(" ", "_").replace("/", "-") + ".png"
        fig.savefig(os.path.join(curves_dir, fname), dpi=300, bbox_inches="tight")
        plt.close(fig)

    return curves_dir

# ─────────────────────────────────────────────────────────────
# Excel export
# ─────────────────────────────────────────────────────────────
def save_comparison_excel(df: pd.DataFrame, strain_names: list, save_dir: str) -> str:
    s0, s1 = strain_names
    cols_in  = ["Antibiotic",
                f"{s0}_IC50", f"{s0}_SD",
                f"{s1}_IC50", f"{s1}_SD",
                "Fold_Change", "Log2_FC", "Log2_FC_tech_err"]
    cols_out = ["Antibiotic",
                f"{s0} IC50 (µg/mL)", f"{s0} tech SD (3 row reps)",
                f"{s1} IC50 (µg/mL)", f"{s1} tech SD (3 row reps)",
                f"Fold Change ({s1}/{s0})", "log2(Fold Change)",
                "log2(FC) propagated tech err"]
    export = df[[c for c in cols_in if c in df.columns]].copy()
    export.columns = cols_out[:len(export.columns)]

    path = os.path.join(save_dir, "IC50_comparison.xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        export.to_excel(writer, index=False, sheet_name="Comparison")
        ws = writer.sheets["Comparison"]
        for col in ws.columns:
            width = max(len(str(cell.value or "")) for cell in col) + 4
            ws.column_dimensions[col[0].column_letter].width = min(width, 45)
    return path

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    strain_names = list(STRAINS.keys())
    s0, s1 = strain_names

    print("=" * 65)
    print(" IC50 Strain Comparison Pipeline")
    print("=" * 65)
    print(f" Strains : {s0} vs {s1}")
    print(f" Output  : {OUTPUT_DIR}")
    print(f" Design  : 1 plate per AB per strain | error = within-plate tech SD")
    print()

    strain_results = {}
    for strain_name, cfg in STRAINS.items():
        print(f"[Processing {strain_name}]")
        results = process_strain(strain_name, cfg["dir"], cfg["map"])
        strain_results[strain_name] = results
        print(f" -> {len(results)} antibiotics processed")
        for r in results:
            ic50_str = f"{r['mean_ic50']:.4g}" if not np.isnan(r["mean_ic50"]) else "N/A"
            print(f"   {r['ab_name']:<35} IC50 = {ic50_str} µg/mL  tech SD = {r['sd_ic50']:.4g}")
        print()

    print("[Building comparison table]")
    df = build_comparison_df(strain_results)
    valid = df.dropna(subset=[f"{s0}_IC50", f"{s1}_IC50"])
    print(f" -> {len(valid)} antibiotics matched in both strains")

    fc_vals = valid["Fold_Change"].values
    signed_vals = valid["Signed_FC"].values
    print(f" Median fold-change        : {np.nanmedian(fc_vals):.2f}x")
    print(f" Max signed fold-change    : {np.nanmax(signed_vals):.2f}  "
          f"({valid.loc[valid['Signed_FC'].idxmax(), 'Antibiotic']})")
    print(f" Min signed fold-change    : {np.nanmin(signed_vals):.2f}  "
          f"({valid.loc[valid['Signed_FC'].idxmin(), 'Antibiotic']})")
    print()

    for label, fn, args in [
        ("[1/3] Grouped bar chart     ", plot_grouped_bar,        (df, strain_names, OUTPUT_DIR)),
        ("[2/3] Fold-change chart     ", plot_fold_change,        (df, strain_names, OUTPUT_DIR)),
        ("[3/3] Per-antibiotic curves ", plot_individual_overlays, (strain_results, strain_names, OUTPUT_DIR)),
    ]:
        print(label + "...")
        p = fn(*args)
        print(f"       Saved: {os.path.basename(p) if os.path.isfile(p) else p}")

    print()
    print("Exporting Excel...")
    p = save_comparison_excel(df, strain_names, OUTPUT_DIR)
    print(f" Saved: {os.path.basename(p)}")

    print()
    print("=" * 65)
    print(" Done.")
    print(f" Output folder: {OUTPUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()