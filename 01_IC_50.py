import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# IC50 Determination from Microplate Growth Assay
# Author: Felix Knote

# ─────────────────────────────────────────────
# User settings
# ─────────────────────────────────────────────
input_file = r" "  # CSV or XLSX
base_name = os.path.splitext(os.path.basename(input_file))[0]

# Maximum concentrations (µg/mL)
max_conc = {
    "antibiotic1": 1,
    "antibiotic2": 1,
}

# Display names for plots
antibiotic_names = {
    "antibiotic1": "",
    "antibiotic2": "",
}

# Blank and growth-control wells
blanks = [f"{r}{c}" for r in "ABCDEFGH" for c in [1, 12]]
growth_controls = (
    [f"A{i}" for i in range(2, 12)] +
    [f"H{i}" for i in range(2, 12)]
)

# Replicate well definitions
antibiotic1 = {
    "rep1": [f"B{i}" for i in range(2, 12)],
    "rep2": [f"C{i}" for i in range(2, 12)],
    "rep3": [f"D{i}" for i in range(2, 12)],
}
antibiotic2 = {
    "rep1": [f"E{i}" for i in range(2, 12)],
    "rep2": [f"F{i}" for i in range(2, 12)],
    "rep3": [f"G{i}" for i in range(2, 12)],
}


# ─────────────────────────────────────────────
# Functions
# ─────────────────────────────────────────────
def four_param_logistic(x, bottom, top, log_ic50, hill_slope):
    """4-parameter logistic (4PL) dose-response model."""
    return bottom + (top - bottom) / (1 + 10 ** ((log_ic50 - np.log10(x)) * hill_slope))


def fit_ic50(concentrations, responses):
    """Fit a 4PL curve and return (IC50, popt)."""
    mask = ~np.isnan(responses)
    xdata = np.array(concentrations)[mask]
    ydata = np.array(responses)[mask]

    lower_bounds = [0,    0,    np.log10(min(xdata)),      -5]
    upper_bounds = [200, 1200,  np.log10(max(xdata)) * 2,   5]

    eps = 1e-8
    p0 = [
        np.clip(min(ydata),                 lower_bounds[0] + eps, upper_bounds[0] - eps),
        np.clip(max(ydata),                 lower_bounds[1] + eps, upper_bounds[1] - eps),
        np.clip(np.median(np.log10(xdata)), lower_bounds[2] + eps, upper_bounds[2] - eps),
        np.clip(-1.0,                       lower_bounds[3] + eps, upper_bounds[3] - eps),
    ]

    popt, _ = curve_fit(
        four_param_logistic, xdata, ydata,
        p0=p0,
        bounds=(lower_bounds, upper_bounds),
        maxfev=20_000,
        method="trf",
    )
    return 10 ** popt[2], popt


def get_well_data(df, wells):
    return df.loc[df["Well"].isin(wells), "OD"].values


def reshape_plate_matrix(df):
    """Convert a raw plate matrix to a tidy (Well, OD) DataFrame."""
    if "Well" in df.columns and "OD" in df.columns:
        return df

    rows = list("ABCDEFGH")
    plate = []

    if df.shape[1] == 13:           # first column is a row label
        for i, row in enumerate(rows):
            for j in range(1, 13):
                plate.append({"Well": f"{row}{j}", "OD": df.iloc[i, j]})
    elif df.shape[1] == 12:         # no row-label column
        for i, row in enumerate(rows):
            for j in range(12):
                plate.append({"Well": f"{row}{j + 1}", "OD": df.iloc[i, j]})
    else:
        raise ValueError("Unrecognised plate format (expected 12 or 13 columns).")

    return pd.DataFrame(plate)


# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
if input_file.endswith(".csv"):
    raw_df = pd.read_csv(input_file, header=None, decimal=";")
else:
    raw_df = pd.read_excel(input_file, header=None, decimal=",")

df = reshape_plate_matrix(raw_df)
df["OD"] = df["OD"].astype(float).round(3)


# ─────────────────────────────────────────────
# Blank subtraction
# ─────────────────────────────────────────────
blank_value = df.loc[df["Well"].isin(blanks), "OD"].mean()
df["OD"] = (df["OD"] - blank_value).clip(lower=0)


# ─────────────────────────────────────────────
# Analysis & plotting
# ─────────────────────────────────────────────
antibiotics = {"antibiotic1": antibiotic1, "antibiotic2": antibiotic2}
palette = plt.cm.Blues(np.linspace(0.5, 1.0, 3))

for ab, reps in antibiotics.items():
    concentrations = np.array([max_conc[ab] / (2 ** i) for i in range(10)])

    fig, ax = plt.subplots(figsize=(8, 6))
    fit_curves = []
    ic50_list = []
    baseline_y_values = []

    # Pass 1: collect y-intercepts for normalisation
    for rep_name, wells in reps.items():
        responses = get_well_data(df, wells)
        ic50, params = fit_ic50(concentrations, responses)
        ic50_list.append(ic50)
        baseline_y_values.append(four_param_logistic(1e-10, *params))

    mean_baseline = np.mean(baseline_y_values)

    # Pass 2: plot normalised data and re-fit
    for i, (rep_name, wells) in enumerate(reps.items()):
        responses_norm = get_well_data(df, wells) / mean_baseline * 100

        ax.plot(
            concentrations, responses_norm,
            "o-", color=palette[i], label=rep_name,
            linewidth=2.5, markersize=7,
        )

        ic50, params = fit_ic50(concentrations, responses_norm)
        ic50_list[i] = ic50

        xfit = np.logspace(np.log10(min(concentrations)), np.log10(max(concentrations)), 200)
        fit_curves.append(four_param_logistic(xfit, *params))

    ic50_mean = np.mean(ic50_list)
    ic50_sd   = np.std(ic50_list)

    fit_arr = np.array(fit_curves)
    y_mean  = np.nanmean(fit_arr, axis=0)
    y_std   = np.nanstd(fit_arr, axis=0)

    ax.plot(xfit, y_mean, "k--", linewidth=2.5, label="Mean 4PL fit")
    ax.fill_between(
        xfit, y_mean - y_std, y_mean + y_std,
        color="gray", alpha=0.2, label="±SD of fits",
    )
    ax.axvline(
        ic50_mean, color="red", linestyle=":", linewidth=2,
        label=f"Mean IC\u2085\u2080 = {ic50_mean:.6f} µg/mL\n± {ic50_sd:.6f}",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Concentration (µg/mL, log scale)", fontsize=13)
    ax.set_ylabel("Relative growth (%)", fontsize=13)
    ax.set_ylim(bottom=0)
    ax.set_title(f"Dose-response: {antibiotic_names[ab]}", fontsize=16, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.legend(fontsize=11)
    ax.tick_params(axis="both", labelsize=11)

    plt.tight_layout()

    save_path = os.path.join(
        os.path.dirname(input_file),
        f"{base_name}_{antibiotic_names[ab].replace(' ', '_')}.png",
    )
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()