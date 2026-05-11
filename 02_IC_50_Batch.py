#!/usr/bin/env python3
"""
IC50 Batch Processing Pipeline
Author: Felix Knote

Plate reader format (per-file):
  Vertical semicolon-delimited list:  A01;  0.118
  Header metadata lines are auto-skipped.

Plate layout (96-well, 8x12):
  Blanks:         A1, A12, H1, H12
  Growth control: A2-A11  (cells only, 100% growth reference)
  DMSO control:   H2-H11  (2-fold dilution, H2 = DMSO_MAX_PCT %)
  Antibiotic 1:   rows B, C, D  (3 replicates, columns 2-11)
  Antibiotic 2:   rows E, F, G  (3 replicates, columns 2-11)

Plate map CSV (auto-pairs antibiotics into plates):
  Index | Antibiotics | Highest concentration [ug/mL]
  Rows 1-2 -> Plate 1,  rows 3-4 -> Plate 2,  etc.
"""

import os
import re
import warnings

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
INPUT_DIR = r"L:\43-RVZ\AIMicroscopy\Mitarbeiter\3_Users\Felix\2026_04_24_AI4AB_IC50_Determination\ACE-1"
PLATE_MAP = r"L:\43-RVZ\AIMicroscopy\Mitarbeiter\3_Users\Felix\2026_04_24_AI4AB_IC50_Determination\ACE-1\AB Plate Map.csv"

OUTLIER_THRESHOLDS = {
    "fit_min_r2":      0.90,
    "fit_hill_range":  (0.3, 5.0),
    "fit_top_min":     0.70,
    "fit_bottom_max":  0.30,
    "blank_max_od":    0.05,
    "growth_min_od":   0.20,
    "growth_max_cv":   15.0,
}
# ─────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit

# ─────────────────────────────────────────────────────────────
# Plate geometry
# ─────────────────────────────────────────────────────────────
N_DILUTIONS   = 10
DILUTION_COLS = list(range(2, 12))

BLANK_WELLS       = ["A1", "A12", "H1", "H12"]
GROWTH_CTRL_WELLS = [f"A{c}" for c in DILUTION_COLS]
DMSO_CTRL_WELLS   = [f"H{c}" for c in DILUTION_COLS]
AB1_ROWS          = list("BCD")
AB2_ROWS          = list("EFG")

DMSO_MAX_PCT = 5.0
REP_PALETTE  = [matplotlib.cm.plasma(v) for v in (0.15, 0.55, 0.90)]  # type: ignore[attr-defined]


# ─────────────────────────────────────────────────────────────
# 1. Data loading
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
    """Parse vertical 'A01;  0.118' plate-reader format. Returns tidy (Well, OD) DataFrame."""
    records = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = _WELL_LINE.match(line.strip())
            if m:
                records.append({
                    "Well": m.group(1).upper() + str(int(m.group(2))),
                    "OD":   _to_float(m.group(3)),
                })
    if not records:
        raise ValueError(f"No well data found in: {filepath}")
    df = pd.DataFrame(records)
    if len(df) != 96:
        warnings.warn(f"{os.path.basename(filepath)}: expected 96 wells, got {len(df)}")
    return df


def load_plate_map(filepath: str) -> pd.DataFrame:
    """
    Load index-based antibiotic map and pair into plates.
    Input:  Index | Antibiotics | Highest concentration [ug/mL]
    Output: plate_id | antibiotic_1_name | antibiotic_1_max_conc | antibiotic_2_name | antibiotic_2_max_conc
    """
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

    raw.columns  = [c.strip() for c in raw.columns]
    name_col     = raw.columns[1]
    conc_col     = raw.columns[2]

    records = []
    for i in range(0, len(raw), 2):
        ab1 = raw.iloc[i]
        ab2 = raw.iloc[i + 1] if (i + 1) < len(raw) else None
        records.append({
            "plate_id":             i // 2 + 1,
            "antibiotic_1_name":    str(ab1[name_col]).strip(),
            "antibiotic_1_max_conc": float(ab1[conc_col]),
            "antibiotic_2_name":    str(ab2[name_col]).strip() if ab2 is not None else None,
            "antibiotic_2_max_conc": float(ab2[conc_col])      if ab2 is not None else None,
        })
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────
# 2. Plate layout parsing
# ─────────────────────────────────────────────────────────────
def parse_plate_layout(plate_df: pd.DataFrame):
    """Returns blank_od (scalar), growth_od, dmso_od, ab1_od dict, ab2_od dict."""
    def wells_od(wells):
        return plate_df.loc[plate_df["Well"].isin(wells), "OD"].values.astype(float)

    blank_od  = float(np.nanmean(wells_od(BLANK_WELLS)))
    growth_od = wells_od(GROWTH_CTRL_WELLS)
    dmso_od   = wells_od(DMSO_CTRL_WELLS)
    ab1_od    = {row: wells_od([f"{row}{c}" for c in DILUTION_COLS]) for row in AB1_ROWS}
    ab2_od    = {row: wells_od([f"{row}{c}" for c in DILUTION_COLS]) for row in AB2_ROWS}

    return blank_od, growth_od, dmso_od, ab1_od, ab2_od


# ─────────────────────────────────────────────────────────────
# 3. Normalization
# ─────────────────────────────────────────────────────────────
def normalize_data(raw_od: np.ndarray, blank_od: float, growth_od: np.ndarray) -> np.ndarray:
    """Background-subtract then express as % of mean growth control."""
    corrected        = np.clip(raw_od    - blank_od, 0, None)
    growth_corrected = np.clip(growth_od - blank_od, 0, None)
    norm_factor = float(np.nanmean(growth_corrected))
    if norm_factor == 0:
        warnings.warn("Growth control mean is zero – normalization skipped.")
        return corrected
    return corrected / norm_factor * 100.0


# ─────────────────────────────────────────────────────────────
# 4. Dose-response model (4PL)
# ─────────────────────────────────────────────────────────────
def four_param_logistic(x, bottom, top, log_ic50, hill):
    return bottom + (top - bottom) / (1.0 + 10.0 ** ((log_ic50 - np.log10(x)) * hill))


def fit_dose_response(concentrations: np.ndarray, responses: np.ndarray):
    """Fit 4PL. Returns (ic50, popt, r2). Raises RuntimeError if fit is not possible."""
    mask = ~np.isnan(responses) & (concentrations > 0)
    x, y = concentrations[mask], responses[mask]
    if len(x) < 4:
        raise RuntimeError("Fewer than 4 valid data points.")

    lx = np.log10(x)
    lb = [  0.0,   0.0, lx.min() - 1.0,  0.1]
    ub = [200.0, 200.0, lx.max() + 1.0, 10.0]
    p0 = [
        np.clip(y.min(),       lb[0] + 1e-9, ub[0] - 1e-9),
        np.clip(y.max(),       lb[1] + 1e-9, ub[1] - 1e-9),
        np.clip(np.median(lx), lb[2] + 1e-9, ub[2] - 1e-9),
        np.clip( 1.0,          lb[3] + 1e-9, ub[3] - 1e-9),
    ]

    popt, pcov = curve_fit(  # type: ignore[misc]
        four_param_logistic, x, y,
        p0=p0, bounds=(lb, ub), maxfev=50_000, method="trf",
    )

    y_pred = four_param_logistic(x, *popt)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    return 10.0 ** popt[2], popt, r2, pcov


# ─────────────────────────────────────────────────────────────
# 5. IC50 across replicates
# ─────────────────────────────────────────────────────────────
def calculate_ic50(rep_responses: dict, concentrations: np.ndarray):
    """
    rep_responses: {"Rep 1": array(10,), "Rep 2": ..., "Rep 3": ...}
    Returns (rep_results list, mean_ic50, sd_ic50).
    Fits failing OUTLIER_THRESHOLDS are excluded from the mean and flagged.
    """
    results = []
    for rep, resp in rep_responses.items():
        try:
            ic50, popt, r2, pcov = fit_dose_response(concentrations, resp)
            bottom, top, log_ic50, hill = popt
            log_ic50_se = float(np.sqrt(pcov[2, 2])) if np.isfinite(pcov[2, 2]) else np.nan
            ic50_ci_lo  = 10.0 ** (log_ic50 - 1.96 * log_ic50_se) if not np.isnan(log_ic50_se) else np.nan
            ic50_ci_hi  = 10.0 ** (log_ic50 + 1.96 * log_ic50_se) if not np.isnan(log_ic50_se) else np.nan
            flags = []
            if r2 < OUTLIER_THRESHOLDS["fit_min_r2"]:
                flags.append(f"R2={r2:.2f}<{OUTLIER_THRESHOLDS['fit_min_r2']}")
            h_lo, h_hi = OUTLIER_THRESHOLDS["fit_hill_range"]
            if not (h_lo <= hill <= h_hi):
                flags.append(f"Hill={hill:.2f}")
            if top < OUTLIER_THRESHOLDS["fit_top_min"]:
                flags.append(f"top={top:.2f}")
            if bottom > OUTLIER_THRESHOLDS["fit_bottom_max"]:
                flags.append(f"bottom={bottom:.2f}")
            flag_str = "; ".join(flags)
            results.append({
                "rep": rep, "ic50": ic50, "popt": popt, "r2": r2, "responses": resp,
                "ic50_ci_lo": ic50_ci_lo, "ic50_ci_hi": ic50_ci_hi,
                "top": top, "bottom": bottom, "hill": hill, "flag": flag_str,
            })
        except Exception as exc:
            warnings.warn(f"Fit failed for {rep}: {exc}")
            results.append({
                "rep": rep, "ic50": np.nan, "popt": None, "r2": np.nan, "responses": resp,
                "ic50_ci_lo": np.nan, "ic50_ci_hi": np.nan,
                "top": np.nan, "bottom": np.nan, "hill": np.nan, "flag": "FIT_FAILED",
            })

    ic50_vals = [r["ic50"] for r in results if not np.isnan(r["ic50"]) and r["flag"] == ""]
    mean_ic50 = float(np.mean(ic50_vals))           if ic50_vals          else np.nan
    sd_ic50   = float(np.std(ic50_vals, ddof=1))    if len(ic50_vals) > 1 else 0.0
    return results, mean_ic50, sd_ic50


# ─────────────────────────────────────────────────────────────
# 6. Individual dose-response plot
# ─────────────────────────────────────────────────────────────
def plot_results(ab_name: str, plate_id, concentrations: np.ndarray,
                 rep_results: list, mean_ic50: float, sd_ic50: float,
                 save_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 6))
    xfit       = np.logspace(np.log10(concentrations.min()), np.log10(concentrations.max()), 300)
    fit_curves = []

    for i, res in enumerate(rep_results):
        color = REP_PALETTE[i % len(REP_PALETTE)]
        ax.scatter(concentrations, res["responses"], color=color, label=res["rep"], zorder=5, s=55)
        ax.plot(concentrations, res["responses"], color=color, alpha=0.45, lw=1.5)
        if res["popt"] is not None:
            yf = four_param_logistic(xfit, *res["popt"])
            ax.plot(xfit, yf, color=color, lw=2.5)
            fit_curves.append(yf)

    if fit_curves:
        ymean = np.nanmean(fit_curves, axis=0)
        ystd  = np.nanstd(fit_curves, axis=0)
        ax.plot(xfit, ymean, "k--", lw=2.5, label="Mean 4PL fit")
        ax.fill_between(xfit, ymean - ystd, ymean + ystd, color="gray", alpha=0.18, label="SD of fits")

    if not np.isnan(mean_ic50):
        ax.axvline(mean_ic50, color="red", ls=":", lw=2,
                   label=f"IC50 = {mean_ic50:.4g} ug/mL +/- {sd_ic50:.4g}")

    ax.set_xscale("log")
    ax.set_xlabel("Concentration (ug/mL)", fontsize=12)
    ax.set_ylabel("Relative growth (%)", fontsize=12)
    ax.set_ylim(bottom=0)
    ax.set_title(f"{ab_name}  [Plate {plate_id}]", fontsize=14, fontweight="bold")
    ax.grid(True, which="both", ls="--", lw=0.6, alpha=0.6)
    ax.legend(fontsize=10)
    plt.tight_layout()

    path = os.path.join(save_dir, f"plate{plate_id}_{ab_name.replace(' ', '_')}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────
# 7. IC50 summary figure (16:9)
# ─────────────────────────────────────────────────────────────
def plot_summary(all_results: list, save_dir: str) -> str:
    n = len(all_results)
    if n == 0:
        return ""

    ncols   = max(4, round(np.sqrt(n * 16 / 9)))
    nrows   = int(np.ceil(n / ncols))
    fig_w   = ncols * 4.5
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_w * 9 / 16))
    axes    = np.array(axes).flatten()

    for idx, res in enumerate(all_results):
        ax   = axes[idx]
        conc = res["concentrations"]
        xfit = np.logspace(np.log10(conc.min()), np.log10(conc.max()), 300)

        for i, rep in enumerate(res["rep_results"]):
            color = REP_PALETTE[i % len(REP_PALETTE)]
            ax.scatter(conc, rep["responses"], color=color, s=18, zorder=5)
            if rep["popt"] is not None:
                ax.plot(xfit, four_param_logistic(xfit, *rep["popt"]), color=color, lw=1.8)

        if not np.isnan(res["mean_ic50"]):
            ax.axvline(res["mean_ic50"], color="red", ls=":", lw=1.5,
                       label=f"IC50 = {res['mean_ic50']:.3g}")
            ax.legend(fontsize=7)

        ax.set_xscale("log")
        ax.set_title(f"{res['ab_name']}\n[Plate {res['plate_id']}]", fontsize=8, fontweight="bold")
        ax.set_xlabel("Conc. (ug/mL)", fontsize=7)
        ax.set_ylabel("Growth (%)", fontsize=7)
        ax.set_ylim(bottom=0)
        ax.grid(True, which="both", ls="--", lw=0.4, alpha=0.5)
        ax.tick_params(labelsize=6)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("IC50 Summary", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    path = os.path.join(save_dir, "IC50_summary.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────
# 8. DMSO growth curve (aggregated across all plates)
# ─────────────────────────────────────────────────────────────
def plot_dmso_summary(dmso_data: list, save_dir: str) -> str:
    """
    dmso_data: [{"plate_id": int, "growth_pct": array(10,)}, ...]
    x: % DMSO linear 0-5,  y: % growth normalised to growth control
    """
    # concentrations generated high→low; flip to ascending for left→right x-axis
    dmso_conc = np.array([DMSO_MAX_PCT / 2 ** i for i in range(N_DILUTIONS)])[::-1]
    matrix    = np.array([d["growth_pct"][::-1] for d in dmso_data])

    fig, ax = plt.subplots(figsize=(8, 6))

    for d in dmso_data:
        ax.plot(dmso_conc, d["growth_pct"][::-1], color="steelblue", alpha=0.35, lw=1.2)

    ymean = np.nanmean(matrix, axis=0)
    ystd  = np.nanstd(matrix,  axis=0)
    ax.plot(dmso_conc, ymean, color="steelblue", lw=2.5)
    ax.fill_between(dmso_conc, ymean - ystd, ymean + ystd, color="steelblue", alpha=0.20)
    ax.axhline(100, color="gray", ls="--", lw=1, alpha=0.6)

    ax.set_xlim(0, DMSO_MAX_PCT)
    ax.set_xlabel("DMSO (%)", fontsize=12)
    ax.set_ylabel("Relative growth (%)", fontsize=12)
    ax.set_ylim(bottom=0)
    ax.set_title("DMSO growth curve – all plates", fontsize=14, fontweight="bold")
    ax.grid(True, ls="--", lw=0.6, alpha=0.6)

    legend_handles = [
        Line2D([0], [0], color="steelblue", alpha=0.4, lw=1.2, label="Individual plates"),
        Line2D([0], [0], color="steelblue", lw=2.5,            label="Mean"),
        Rectangle((0, 0), 1, 1, fc="steelblue", alpha=0.2, label="SD"),
    ]
    ax.legend(handles=legend_handles, fontsize=10)

    plt.tight_layout()
    path = os.path.join(save_dir, "DMSO_growth_curve.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────
# 9. Excel export
# ─────────────────────────────────────────────────────────────
def _autofit(ws):
    for col in ws.columns:
        width = max(len(str(cell.value or "")) for cell in col) + 4
        ws.column_dimensions[col[0].column_letter].width = min(width, 40)


def save_excel(records: list, save_dir: str) -> str:
    df      = pd.DataFrame(records)
    df_mean = (
        df[df["Replicate"] == "MEAN"]
        .drop(columns=["Replicate"])
        .reset_index(drop=True)
    )

    path = os.path.join(save_dir, "IC50_results.xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="IC50 Results")
        _autofit(writer.sheets["IC50 Results"])

        df_mean.to_excel(writer, index=False, sheet_name="Means")
        _autofit(writer.sheets["Means"])

    return path


# ─────────────────────────────────────────────────────────────
# 10. Pre-normalization QC figure
# ─────────────────────────────────────────────────────────────
def plot_raw_plate_qc(plate_df: pd.DataFrame, blank_od: float, growth_od: np.ndarray,
                      dmso_od: np.ndarray, ab1_od: dict, ab2_od: dict,
                      ab1_name: str, ab2_name: str,
                      ab1_conc: np.ndarray, ab2_conc: np.ndarray,
                      plate_id, save_dir: str) -> dict:
    """
    Pre-normalization QC figure: 96-well heatmap, absolute OD dose-response curves,
    blank/growth control distributions, DMSO raw OD.
    Returns dict with plate-level QC flags.
    """
    # Build 8x12 OD matrix
    _row_map = {r: i for i, r in enumerate("ABCDEFGH")}
    od_mat = np.full((8, 12), np.nan)
    for _, w in plate_df.iterrows():
        od_mat[_row_map[w["Well"][0]], int(w["Well"][1:]) - 1] = w["OD"]

    blank_arr   = np.array([od_mat[_row_map["A"], 0], od_mat[_row_map["A"], 11],
                             od_mat[_row_map["H"], 0], od_mat[_row_map["H"], 11]])
    blank_arr   = blank_arr[~np.isnan(blank_arr)]
    blank_mean  = float(np.nanmean(blank_arr)) if len(blank_arr) > 0 else np.nan
    growth_mean = float(np.nanmean(growth_od))
    growth_cv   = float(np.nanstd(growth_od, ddof=1) / growth_mean * 100) if growth_mean > 0 else np.nan

    flags = {
        "blank_high":     not np.isnan(blank_mean) and blank_mean > OUTLIER_THRESHOLDS["blank_max_od"],
        "growth_low":     growth_mean < OUTLIER_THRESHOLDS["growth_min_od"],
        "growth_cv_high": not np.isnan(growth_cv) and growth_cv > OUTLIER_THRESHOLDS["growth_max_cv"],
    }
    flag_msgs = []
    if flags["blank_high"]:     flag_msgs.append(f"Blank={blank_mean:.3f}>{OUTLIER_THRESHOLDS['blank_max_od']}")
    if flags["growth_low"]:     flag_msgs.append(f"GrowthCtrl={growth_mean:.3f}<{OUTLIER_THRESHOLDS['growth_min_od']}")
    if flags["growth_cv_high"]: flag_msgs.append(f"GrowthCtrl CV={growth_cv:.1f}%>{OUTLIER_THRESHOLDS['growth_max_cv']}%")
    is_flagged = any(flags.values())

    fig = plt.figure(figsize=(22, 10))
    gs  = fig.add_gridspec(2, 4, height_ratios=[1.3, 1], hspace=0.45, wspace=0.38)
    ax_heat = fig.add_subplot(gs[0, :])
    ax_ab1  = fig.add_subplot(gs[1, 0])
    ax_ab2  = fig.add_subplot(gs[1, 1])
    ax_dmso = fig.add_subplot(gs[1, 2])
    ax_qc   = fig.add_subplot(gs[1, 3])

    # 96-well heatmap
    im = ax_heat.imshow(od_mat, aspect="auto", cmap="viridis")
    plt.colorbar(im, ax=ax_heat, label="OD600", shrink=0.75)
    ax_heat.set_xticks(range(12))
    ax_heat.set_xticklabels(range(1, 13), fontsize=8)
    ax_heat.set_yticks(range(8))
    ax_heat.set_yticklabels(list("ABCDEFGH"), fontsize=9)
    ax_heat.set_title(f"Plate {plate_id} – Raw OD600 Heatmap", fontsize=12, fontweight="bold")
    _type_map: dict = {}
    for w in BLANK_WELLS:       _type_map[w] = "B"
    for w in GROWTH_CTRL_WELLS: _type_map[w] = "G"
    for w in DMSO_CTRL_WELLS:   _type_map[w] = "D"
    for rl in AB1_ROWS:
        for c in DILUTION_COLS: _type_map[f"{rl}{c}"] = "1"
    for rl in AB2_ROWS:
        for c in DILUTION_COLS: _type_map[f"{rl}{c}"] = "2"
    for well, label in _type_map.items():
        ax_heat.text(int(well[1:]) - 1, _row_map[well[0]], label,
                     ha="center", va="center", fontsize=6, color="white", fontweight="bold")

    # AB1 absolute OD dose-response
    for i, (row_letter, od_vals) in enumerate(ab1_od.items()):
        ax_ab1.plot(ab1_conc, od_vals, "o-", color=REP_PALETTE[i % len(REP_PALETTE)],
                    lw=1.5, ms=5, label=f"Row {row_letter}")
    ax_ab1.axhline(blank_od, color="gray",  ls="--", lw=1, label=f"Blank ({blank_od:.3f})")
    ax_ab1.axhline(growth_mean, color="green", ls="--", lw=1, label=f"Growth ctrl ({growth_mean:.3f})")
    ax_ab1.set_xscale("log")
    ax_ab1.set_xlabel("Concentration (µg/mL)", fontsize=9)
    ax_ab1.set_ylabel("Raw OD600", fontsize=9)
    ax_ab1.set_title(f"{ab1_name}\n(raw OD, not normalized)", fontsize=9, fontweight="bold")
    ax_ab1.legend(fontsize=7)
    ax_ab1.grid(True, which="both", ls="--", lw=0.4, alpha=0.5)

    # AB2 absolute OD dose-response
    for i, (row_letter, od_vals) in enumerate(ab2_od.items()):
        ax_ab2.plot(ab2_conc, od_vals, "o-", color=REP_PALETTE[i % len(REP_PALETTE)],
                    lw=1.5, ms=5, label=f"Row {row_letter}")
    ax_ab2.axhline(blank_od, color="gray",  ls="--", lw=1, label=f"Blank ({blank_od:.3f})")
    ax_ab2.axhline(growth_mean, color="green", ls="--", lw=1, label=f"Growth ctrl ({growth_mean:.3f})")
    ax_ab2.set_xscale("log")
    ax_ab2.set_xlabel("Concentration (µg/mL)", fontsize=9)
    ax_ab2.set_ylabel("Raw OD600", fontsize=9)
    ax_ab2.set_title(f"{ab2_name or 'AB2'}\n(raw OD, not normalized)", fontsize=9, fontweight="bold")
    ax_ab2.legend(fontsize=7)
    ax_ab2.grid(True, which="both", ls="--", lw=0.4, alpha=0.5)

    # DMSO raw OD
    dmso_conc = np.array([DMSO_MAX_PCT / 2**i for i in range(N_DILUTIONS)])[::-1]
    ax_dmso.plot(dmso_conc, dmso_od[::-1], "o-", color="steelblue", lw=1.5, ms=5, label="DMSO")
    ax_dmso.axhline(blank_od,    color="gray",  ls="--", lw=1, label=f"Blank ({blank_od:.3f})")
    ax_dmso.axhline(growth_mean, color="green", ls="--", lw=1, label=f"Growth ctrl ({growth_mean:.3f})")
    ax_dmso.set_xlabel("DMSO (%)", fontsize=9)
    ax_dmso.set_ylabel("Raw OD600", fontsize=9)
    ax_dmso.set_title("DMSO carrier\n(raw OD)", fontsize=9, fontweight="bold")
    ax_dmso.legend(fontsize=7)
    ax_dmso.grid(True, ls="--", lw=0.4, alpha=0.5)

    # QC boxplot: growth control + blank wells
    bp_data = [growth_od, blank_arr if len(blank_arr) > 0 else np.array([blank_od])]
    ax_qc.boxplot(bp_data, positions=[1, 2], widths=0.5, patch_artist=True,
                  boxprops=dict(facecolor="lightblue"),
                  medianprops=dict(color="navy", lw=2))
    ax_qc.axhline(OUTLIER_THRESHOLDS["growth_min_od"], color="red",    ls="--", lw=1.2,
                  label=f"Growth min ({OUTLIER_THRESHOLDS['growth_min_od']})")
    ax_qc.axhline(OUTLIER_THRESHOLDS["blank_max_od"],  color="orange", ls="--", lw=1.2,
                  label=f"Blank max ({OUTLIER_THRESHOLDS['blank_max_od']})")
    ax_qc.set_xticks([1, 2])
    ax_qc.set_xticklabels(["Growth ctrl\n(A2-A11)", "Blank\n(corners)"], fontsize=9)
    ax_qc.set_ylabel("Raw OD600", fontsize=9)
    ax_qc.set_title("Control QC", fontsize=9, fontweight="bold")
    ax_qc.legend(fontsize=7)
    ax_qc.grid(True, axis="y", ls="--", lw=0.4, alpha=0.5)

    flag_color = "red" if is_flagged else "darkgreen"
    flag_text  = "FLAGGED: " + "; ".join(flag_msgs) if is_flagged else "PASS"
    fig.suptitle(f"Plate {plate_id} – Pre-Normalization QC  [{flag_text}]",
                 fontsize=13, fontweight="bold", color=flag_color)

    fname_prefix = "FLAGGED_" if is_flagged else ""
    path = os.path.join(save_dir, f"{fname_prefix}plate{plate_id}_raw_qc.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    if is_flagged:
        print(f"       *** QC FLAG: {'; '.join(flag_msgs)}")

    return flags


# ─────────────────────────────────────────────────────────────
# 11. Helpers
# ─────────────────────────────────────────────────────────────
def _extract_plate_id(filename: str) -> int:
    """Extract numeric plate ID from filename (_P1, _P10, Plate_3, etc.)."""
    m = re.search(r'_[Pp](\d+)', filename)
    if m:
        return int(m.group(1))
    m = re.search(r'[Pp]late[_\s\-]?(\d+)', filename)
    if m:
        return int(m.group(1))
    m = re.search(r'(\d+)', filename)
    return int(m.group(1)) if m else 0


# ─────────────────────────────────────────────────────────────
# 11. Main pipeline
# ─────────────────────────────────────────────────────────────
def main():
    input_dir      = INPUT_DIR
    plate_map_path = PLATE_MAP
    output_dir     = os.path.join(input_dir, "IC50_output")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  IC50 Batch Pipeline")
    print("=" * 60)
    print(f"  Input : {input_dir}")
    print(f"  Map   : {plate_map_path}")
    print(f"  Output: {output_dir}")
    print()

    plate_map = load_plate_map(plate_map_path)
    print(f"[1/5] Plate map loaded  ({len(plate_map)} plates, {len(plate_map) * 2} antibiotics)")
    print()

    map_basename = os.path.basename(plate_map_path)
    plate_files  = sorted(
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".csv", ".xlsx"))
        and not f.startswith("~")
        and f != map_basename
    )
    print(f"[2/5] Found {len(plate_files)} plate file(s) – starting processing")
    print()

    excel_records = []
    summary_data  = []
    dmso_data     = []

    for plate_num, fname in enumerate(plate_files, start=1):
        fpath    = os.path.join(input_dir, fname)
        plate_id = _extract_plate_id(fname)
        print(f"  [{plate_num}/{len(plate_files)}] Plate {plate_id}: {fname}")

        map_row = plate_map[plate_map["plate_id"] == plate_id]
        if map_row.empty:
            print(f"       SKIP – plate ID {plate_id} not in plate map")
            continue
        map_row = map_row.iloc[0]

        try:
            plate_df  = load_plate(fpath)
            blank_mean = plate_df.loc[plate_df["Well"].isin(BLANK_WELLS), "OD"].mean()
            print(f"       {len(plate_df)} wells loaded  |  blank OD = {blank_mean:.3f}")
        except Exception as exc:
            print(f"       SKIP – could not load file: {exc}")
            continue

        blank_od, growth_od, dmso_od, ab1_od, ab2_od = parse_plate_layout(plate_df)

        # Pre-normalization QC figure
        _ab1_max = float(map_row["antibiotic_1_max_conc"]) if map_row["antibiotic_1_name"] else 1.0
        _ab2_max = float(map_row["antibiotic_2_max_conc"]) if map_row["antibiotic_2_name"] else 1.0
        plot_raw_plate_qc(
            plate_df, blank_od, growth_od, dmso_od, ab1_od, ab2_od,
            str(map_row["antibiotic_1_name"]) if map_row["antibiotic_1_name"] else "AB1",
            str(map_row["antibiotic_2_name"]) if map_row["antibiotic_2_name"] else "AB2",
            np.array([_ab1_max / 2**i for i in range(N_DILUTIONS)]),
            np.array([_ab2_max / 2**i for i in range(N_DILUTIONS)]),
            plate_id, output_dir,
        )

        dmso_data.append({
            "plate_id":  plate_id,
            "growth_pct": normalize_data(dmso_od, blank_od, growth_od),
        })

        antibiotics = []
        if map_row["antibiotic_1_name"]:
            antibiotics.append((map_row["antibiotic_1_name"], float(map_row["antibiotic_1_max_conc"]), ab1_od))
        if map_row["antibiotic_2_name"]:
            antibiotics.append((map_row["antibiotic_2_name"], float(map_row["antibiotic_2_max_conc"]), ab2_od))

        for ab_name, ab_max_conc, ab_od in antibiotics:
            conc    = np.array([ab_max_conc / 2 ** i for i in range(N_DILUTIONS)])
            ab_norm = {f"Rep {i+1}": normalize_data(od, blank_od, growth_od)
                       for i, od in enumerate(ab_od.values())}

            rep_results, mean_ic50, sd_ic50 = calculate_ic50(ab_norm, conc)

            ic50_str = f"{mean_ic50:.4g}" if not np.isnan(mean_ic50) else "N/A"
            print(f"       {ab_name:<30}  IC50 = {ic50_str} ug/mL  SD = {sd_ic50:.4g}")

            def _r(v, n=6): return round(v, n) if (v is not None and not np.isnan(v)) else "N/A"
            for res in rep_results:
                excel_records.append({
                    "Plate ID":     plate_id,
                    "Antibiotic":   ab_name,
                    "Replicate":    res["rep"],
                    "IC50 (ug/mL)": _r(res["ic50"]),
                    "IC50 CI Low":  _r(res["ic50_ci_lo"]),
                    "IC50 CI High": _r(res["ic50_ci_hi"]),
                    "Top":          _r(res["top"],    4),
                    "Bottom":       _r(res["bottom"], 4),
                    "Hill Slope":   _r(res["hill"],   4),
                    "R2":           _r(res["r2"],     4),
                    "Flag":         res.get("flag", ""),
                })
            excel_records.append({
                "Plate ID":     plate_id,
                "Antibiotic":   ab_name,
                "Replicate":    "MEAN",
                "IC50 (ug/mL)": _r(mean_ic50),
                "SD":           _r(sd_ic50),
                "Flag":         "excluded reps flagged above" if any(r["flag"] for r in rep_results) else "",
            })

            plot_results(ab_name, plate_id, conc, rep_results, mean_ic50, sd_ic50, output_dir)
            print(f"       Plot saved: plate{plate_id}_{ab_name.replace(' ', '_')}.png")

            summary_data.append({
                "ab_name":        ab_name,
                "plate_id":       plate_id,
                "concentrations": conc,
                "rep_results":    rep_results,
                "mean_ic50":      mean_ic50,
                "sd_ic50":        sd_ic50,
            })

        print()

    print(f"[3/5] Generating IC50 summary figure ({len(summary_data)} antibiotics)...")
    if summary_data:
        plot_summary(summary_data, output_dir)
        print("      Saved: IC50_summary.png")

    print(f"[4/5] Generating DMSO growth curve ({len(dmso_data)} plates)...")
    if dmso_data:
        plot_dmso_summary(dmso_data, output_dir)
        print("      Saved: DMSO_growth_curve.png")

    print(f"[5/5] Writing Excel ({len(excel_records)} rows)...")
    if excel_records:
        save_excel(excel_records, output_dir)
        print("      Saved: IC50_results.xlsx")

    print()
    print("=" * 60)
    print("  Done.")
    print(f"  Output folder: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
