# %% SVI vs estimated TC-induced mortality burden charts
# Creates a two-panel figure:
#   Panel a: county SVI vs. estimated 172-month cumulative TC mortality response
#            per 100,000 per 1 m/s exposure. This is beta_g.
#   Panel b: county SVI vs. full 172-month cumulative estimated TC-induced
#            mortality burden per 100,000 for each county-event.
#
# Interpretation:
#   Because the lag weights h_l sum to 1, the full 172-month cumulative
#   response to 1 m/s of event-county wind exposure is beta_g.
#   For county-event (e,c), full cumulative rate burden is:
#       beta_g(c) * W_ec
#
# Outputs:
#   output/figures/svi_tc_mortality_response_scatter.png
#   output/figures/svi_tc_mortality_response_scatter.pdf
#   output/figures/svi_tc_county_event_burden_data.csv
#   output/figures/svi_county_beta_response_data.csv
#   output/figures/svi_panel_b_fit_results.csv

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# =============================================================================
# User settings
# =============================================================================

from pathlib import Path
base_dir = Path(__file__).resolve().parents[1]
data_dir = os.path.join(base_dir, "data")
output_dir = os.path.join(base_dir, "output")
beta_dir = os.path.join(output_dir, "beta")
fig_output_dir = os.path.join(output_dir, "figures")
os.makedirs(fig_output_dir, exist_ok=True)

storm_panel_file = os.path.join(
    data_dir,
    "panel_by_storm__NA_USA_density_8_yr_1930_2018.csv"
)

regression_data_file = os.path.join(output_dir, "full_data_for_regression.dta")

coef_file_dta = os.path.join(beta_dir, "coefficients_county_nature_future.dta")
coef_file_csv = os.path.join(beta_dir, "coefficients_region_poly.csv")

fig_path_png = os.path.join(fig_output_dir, "svi_tc_mortality_response_scatter.png")
fig_path_pdf = os.path.join(fig_output_dir, "svi_tc_mortality_response_scatter.pdf")

county_beta_output = os.path.join(fig_output_dir, "svi_county_beta_response_data.csv")
county_event_output = os.path.join(fig_output_dir, "svi_tc_county_event_burden_data.csv")
fit_results_output = os.path.join(fig_output_dir, "svi_panel_b_fit_results.csv")

# =============================================================================
# Analysis settings
# =============================================================================

STORM_START_YEAR = 1953
END_YEAR = 2018

svi_cutoff = 0.75

USE_REGRESSION_COUNTIES_ONLY = True
DROP_ZERO_EXPOSURE = True

# Optional: cap y-axis in panel b to reduce domination by extreme county-events.
# Set to None to show all values.
PANEL_B_YMAX_QUANTILE = None
# Example:
# PANEL_B_YMAX_QUANTILE = 0.99

# =============================================================================
# Curve-fitting settings
# =============================================================================

FIT_PANEL_B_CURVES = True

# Choose any subset:
#   ["linear", "quadratic", "exponential", "logistic"]
PANEL_B_FITS_TO_SHOW = ["logistic"]

# Fit to raw county-event points or binned means.
FIT_ON_BINNED_MEANS = True
N_BINS_FOR_FIT = 20
MIN_POINTS_PER_BIN = 3

PRINT_FIT_FORMULAS = True

# I recommend keeping this False and writing the formula in the LaTeX caption.
# Setting it True puts the fit formula inside the PNG/PDF.
SHOW_FORMULA_IN_FIGURE = False

# =============================================================================
# Figure settings
# =============================================================================

FIG_WIDTH = 13.5
FIG_HEIGHT = 6.2

TITLE_SIZE = 16
PANEL_TITLE_SIZE = 14
AXIS_LABEL_SIZE = 13
TICK_LABEL_SIZE = 11
LEGEND_SIZE = 11
PANEL_LABEL_SIZE = 16
ANNOTATION_SIZE = 10
CAPTION_SIZE = 10

POINT_SIZE_A = 55
POINT_SIZE_B = 24
ALPHA_A = 0.90
ALPHA_B = 0.45

FIT_LINEWIDTH = 2.0
FIT_ALPHA = 0.95

LOWER_COLOR = "#4C78A8"
HIGH_COLOR = "#E45756"
CUT_LINE_COLOR = "black"
FIT_COLOR = "black"
STEP_COLOR = "black"

# Legend/spacing controls
LEGEND_Y = 0.015
BOTTOM_MARGIN_NO_FORMULA = 0.17
BOTTOM_MARGIN_WITH_FORMULA = 0.21

# =============================================================================
# Helper functions
# =============================================================================

def zfill5(series):
    return (
        series.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
    )


def clean_stata_parm(parm):
    parm = str(parm).strip()
    parm = parm.replace("b.", ".").replace("o.", ".")
    parm = re.sub(r"^(\d+)[a-z]+\.", r"\1.", parm)
    return parm


def read_coefficients(dta_path, csv_path):
    if os.path.exists(dta_path):
        df = pd.read_stata(dta_path)
        source = dta_path
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        source = csv_path
    else:
        raise FileNotFoundError(
            "No coefficient file found. Expected one of:\n"
            f"{dta_path}\n{csv_path}"
        )

    if "parm" not in df.columns or "estimate" not in df.columns:
        raise KeyError(
            f"Coefficient file must contain 'parm' and 'estimate'. "
            f"Columns found: {df.columns.tolist()}"
        )

    df = df.copy()
    df["parm_clean"] = df["parm"].apply(clean_stata_parm)
    print(f"Loaded coefficient file: {source}")
    return df


def extract_svi_betas(df_coef):
    """
    Extracts:
        0.svi_high#c.X_nature -> Lower-SVI beta
        1.svi_high#c.X_nature -> High-SVI beta
    """
    beta = {}

    for _, row in df_coef.iterrows():
        p = row["parm_clean"]
        m = re.match(r"([01])\.svi_high#c\.X_nature$", p)
        if m:
            beta[int(m.group(1))] = float(row["estimate"])

    missing = [g for g in [0, 1] if g not in beta]
    if missing:
        print("Available coefficient names:")
        print(df_coef["parm_clean"].to_string(index=False))
        raise ValueError(
            f"Missing X_nature beta(s) for svi_high={missing}. "
            "Expected coefficients like 0.svi_high#c.X_nature and "
            "1.svi_high#c.X_nature."
        )

    return beta


def infer_col(df, candidates, required=True, label="column"):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(
            f"Could not find {label}. Tried: {candidates}. "
            f"Available columns: {df.columns.tolist()}"
        )
    return None


def assign_svi_group(svi_value, cutoff):
    return np.where(svi_value > cutoff, 1, 0).astype(int)


def format_coef(x, digits=4):
    return f"{x:.{digits}f}"


# =============================================================================
# Curve-fitting functions
# =============================================================================

def exp_func(x, a, b, c):
    return a * np.exp(b * x) + c


def logistic_func(x, L, k, x0, c):
    return c + L / (1.0 + np.exp(-k * (x - x0)))


def prepare_fit_data(df, x_col, y_col, use_binned=True, n_bins=20, min_points_per_bin=3):
    d = df[[x_col, y_col]].dropna().copy()
    d = d[np.isfinite(d[x_col]) & np.isfinite(d[y_col])]

    if d.empty:
        raise ValueError("No valid observations available for curve fitting.")

    if not use_binned:
        return d[x_col].to_numpy(), d[y_col].to_numpy()

    d["svi_bin"] = pd.cut(d[x_col], bins=n_bins)

    binned = (
        d.groupby("svi_bin", observed=True)
        .agg(
            x_mean=(x_col, "mean"),
            y_mean=(y_col, "mean"),
            n=(y_col, "size")
        )
        .reset_index()
    )

    binned = binned[binned["n"] >= min_points_per_bin].copy()

    if len(binned) < 4:
        raise ValueError(
            f"Only {len(binned)} bins available after filtering. "
            "Reduce N_BINS_FOR_FIT or MIN_POINTS_PER_BIN."
        )

    return binned["x_mean"].to_numpy(), binned["y_mean"].to_numpy()


def make_fit_predictions(x_fit, y_fit, x_grid, fit_name):
    if fit_name == "linear":
        coef = np.polyfit(x_fit, y_fit, deg=1)
        y_grid = np.polyval(coef, x_grid)
        y_fit_hat = np.polyval(coef, x_fit)
        params = {"a": coef[0], "b": coef[1]}

    elif fit_name == "quadratic":
        coef = np.polyfit(x_fit, y_fit, deg=2)
        y_grid = np.polyval(coef, x_grid)
        y_fit_hat = np.polyval(coef, x_fit)
        params = {"a": coef[0], "b": coef[1], "c": coef[2]}

    elif fit_name == "exponential":
        p0 = [
            max(float(y_fit.max() - y_fit.min()), 1.0),
            1.0,
            float(y_fit.min())
        ]

        bounds = (
            [0.0, -20.0, -np.inf],
            [np.inf, 20.0, np.inf]
        )

        popt, _ = curve_fit(
            exp_func,
            x_fit,
            y_fit,
            p0=p0,
            bounds=bounds,
            maxfev=20000
        )

        y_grid = exp_func(x_grid, *popt)
        y_fit_hat = exp_func(x_fit, *popt)
        params = {"a": popt[0], "b": popt[1], "c": popt[2]}

    elif fit_name == "logistic":
        y_min = float(np.nanmin(y_fit))
        y_max = float(np.nanmax(y_fit))
        y_range = max(y_max - y_min, 1.0)

        p0 = [
            y_range,      # L: vertical range
            12.0,         # k: slope
            svi_cutoff,   # x0: midpoint
            y_min         # c: lower asymptote
        ]

        bounds = (
            [0.0, -100.0, 0.0, -np.inf],
            [np.inf, 100.0, 1.0, np.inf]
        )

        popt, _ = curve_fit(
            logistic_func,
            x_fit,
            y_fit,
            p0=p0,
            bounds=bounds,
            maxfev=50000
        )

        y_grid = logistic_func(x_grid, *popt)
        y_fit_hat = logistic_func(x_fit, *popt)
        params = {"L": popt[0], "k": popt[1], "x0": popt[2], "c": popt[3]}

    else:
        raise ValueError(f"Unknown fit_name: {fit_name}")

    r2 = r2_score(y_fit, y_fit_hat)
    return y_grid, r2, params


def print_fit_formula(fit_name, params, r2):
    if fit_name == "linear":
        print("\nLinear fit:")
        print(f"  y = {format_coef(params['a'])} SVI + {format_coef(params['b'])}")

    elif fit_name == "quadratic":
        print("\nQuadratic fit:")
        print(
            f"  y = {format_coef(params['a'])} SVI^2 "
            f"+ {format_coef(params['b'])} SVI "
            f"+ {format_coef(params['c'])}"
        )

    elif fit_name == "exponential":
        print("\nExponential fit:")
        print(
            f"  y = {format_coef(params['a'])} "
            f"exp({format_coef(params['b'])} SVI) "
            f"+ {format_coef(params['c'])}"
        )

    elif fit_name == "logistic":
        print("\nLogistic fit:")
        print(
            f"  y = {format_coef(params['c'])} "
            f"+ {format_coef(params['L'])} / "
            f"[1 + exp(-{format_coef(params['k'])}"
            f"(SVI - {format_coef(params['x0'])}))]"
        )

    print(f"  R^2 = {r2:.4f}")


def formula_text_for_fit(fit_name, params, r2, digits=3):
    if fit_name == "logistic":
        return (
            f"Panel (b) logistic fit: "
            f"$y={params['c']:.{digits}f}"
            f"+\\frac{{{params['L']:.{digits}f}}}"
            f"{{1+\\exp[-{params['k']:.{digits}f}(\\mathrm{{SVI}}-{params['x0']:.{digits}f})]}}$, "
            f"$R^2={r2:.2f}$"
        )

    if fit_name == "linear":
        return (
            f"Panel (b) linear fit: "
            f"$y={params['a']:.{digits}f}\\,\\mathrm{{SVI}}"
            f"{params['b']:+.{digits}f}$, "
            f"$R^2={r2:.2f}$"
        )

    if fit_name == "quadratic":
        return (
            f"Panel (b) quadratic fit: "
            f"$y={params['a']:.{digits}f}\\,\\mathrm{{SVI}}^2"
            f"{params['b']:+.{digits}f}\\,\\mathrm{{SVI}}"
            f"{params['c']:+.{digits}f}$, "
            f"$R^2={r2:.2f}$"
        )

    if fit_name == "exponential":
        return (
            f"Panel (b) exponential fit: "
            f"$y={params['a']:.{digits}f}\\exp({params['b']:.{digits}f}\\,\\mathrm{{SVI}})"
            f"{params['c']:+.{digits}f}$, "
            f"$R^2={r2:.2f}$"
        )

    return ""


# =============================================================================
# 1. Load regression data and define county-level SVI groups
# =============================================================================

reg = pd.read_stata(regression_data_file)

required_reg_cols = ["countyfips", "year", "month", "svi", "totalpop"]
missing = [c for c in required_reg_cols if c not in reg.columns]
if missing:
    raise KeyError(f"Regression data missing columns: {missing}")

reg = reg.copy()
reg["countyfips"] = zfill5(reg["countyfips"])
reg["year"] = reg["year"].astype(int)
reg["month"] = reg["month"].astype(int)
reg["svi"] = pd.to_numeric(reg["svi"], errors="coerce")
reg["totalpop"] = pd.to_numeric(reg["totalpop"], errors="coerce")

county_svi = (
    reg.groupby("countyfips", as_index=False)
    .agg(
        mean_svi_county=("svi", "mean"),
        mean_totalpop=("totalpop", "mean")
    )
    .dropna(subset=["mean_svi_county"])
)

county_svi["svi_high"] = assign_svi_group(county_svi["mean_svi_county"], svi_cutoff)
county_svi["svi_group"] = np.where(
    county_svi["svi_high"] == 1,
    "High-SVI",
    "Lower-SVI"
)

regression_counties = set(county_svi["countyfips"])

print("\n================ County SVI groups ================")
print("Number of counties:", len(regression_counties))
print(county_svi["svi_group"].value_counts())

# =============================================================================
# 2. Load SVI-group coefficients
# =============================================================================

coef = read_coefficients(coef_file_dta, coef_file_csv)
beta_by_group = extract_svi_betas(coef)

county_svi["beta_group"] = county_svi["svi_high"].map(beta_by_group)

beta_lower = beta_by_group[0]
beta_high = beta_by_group[1]

print("\n================ SVI-group beta coefficients ================")
print(f"Lower-SVI beta: {beta_lower:.6f}")
print(f"High-SVI beta:  {beta_high:.6f}")

county_svi.to_csv(county_beta_output, index=False)

# =============================================================================
# 3. Load event-county wind exposure
# =============================================================================

wind = pd.read_csv(
    storm_panel_file,
    dtype={
        "countyfips": str,
        "stfips": str,
        "storm_serial_id": str,
        "storm_serial_ID": str,
        "storm_id": str,
        "storm_name": str,
    },
    low_memory=False
)

county_col = infer_col(wind, ["countyfips", "county_fips", "fips"], label="county FIPS")
state_col = infer_col(wind, ["stfips", "statefips", "state_fips"], required=False, label="state FIPS")
year_col = infer_col(wind, ["year"], label="year")
month_col = infer_col(wind, ["month"], label="month")
maxs_col = infer_col(wind, ["maxs", "wind", "wind_speed"], label="max wind speed")
storm_name_col = infer_col(wind, ["storm_name", "storm_nam", "storm_na", "name"], required=False, label="storm name")
storm_serial_col = infer_col(
    wind,
    ["storm_serial_id", "storm_serial_ID", "storm_ser", "storm_serial", "serial_id", "storm_id"],
    required=False,
    label="storm serial id"
)

wind = wind.copy()
wind[county_col] = zfill5(wind[county_col])
wind[year_col] = wind[year_col].astype(int)
wind[month_col] = wind[month_col].astype(int)
wind[maxs_col] = pd.to_numeric(wind[maxs_col], errors="coerce").fillna(0.0)

# Texas filter
if state_col is not None:
    wind[state_col] = wind[state_col].astype(str).str.zfill(2)
    wind_tx = wind[wind[state_col] == "48"].copy()
else:
    wind_tx = wind[wind[county_col].str.startswith("48")].copy()

wind_tx = wind_tx[
    (wind_tx[year_col] >= STORM_START_YEAR) &
    (wind_tx[year_col] <= END_YEAR)
].copy()

if USE_REGRESSION_COUNTIES_ONLY:
    wind_tx = wind_tx[wind_tx[county_col].isin(regression_counties)].copy()

if DROP_ZERO_EXPOSURE:
    wind_tx = wind_tx[wind_tx[maxs_col] > 0].copy()

# Create storm key
if storm_serial_col is None:
    if storm_name_col is None:
        wind_tx["storm_key"] = (
            "UNKNOWN_"
            + wind_tx[year_col].astype(str)
            + "_"
            + wind_tx[month_col].astype(str).str.zfill(2)
        )
    else:
        wind_tx["storm_key"] = (
            wind_tx[storm_name_col].astype(str).fillna("UNKNOWN")
            + "_"
            + wind_tx[year_col].astype(str)
            + "_"
            + wind_tx[month_col].astype(str).str.zfill(2)
        )
else:
    wind_tx["storm_key"] = wind_tx[storm_serial_col].astype(str)

if storm_name_col is not None:
    wind_tx["storm_name_plot"] = wind_tx[storm_name_col].astype(str)
else:
    wind_tx["storm_name_plot"] = wind_tx["storm_key"]

wind_tx = wind_tx.rename(columns={
    county_col: "countyfips",
    year_col: "event_year",
    month_col: "event_month",
    maxs_col: "maxs"
})

# Merge county SVI and beta
county_event = wind_tx.merge(
    county_svi[
        [
            "countyfips",
            "mean_svi_county",
            "svi_high",
            "svi_group",
            "beta_group",
            "mean_totalpop"
        ]
    ],
    on="countyfips",
    how="left"
)

missing_svi = county_event["svi_high"].isna().sum()
if missing_svi > 0:
    print(f"Warning: dropping {missing_svi:,} county-event rows with missing SVI.")
    county_event = county_event.dropna(subset=["svi_high"]).copy()

county_event["svi_high"] = county_event["svi_high"].astype(int)

# Full 172-month cumulative estimated mortality rate burden per 100,000.
# Since sum_l h_l = 1, this is beta_g * maxs.
county_event["tc_mortality_per_100k_172mo"] = (
    county_event["beta_group"] * county_event["maxs"]
)

# Optional count version, using average county population.
county_event["tc_excess_deaths_172mo_meanpop"] = (
    county_event["tc_mortality_per_100k_172mo"]
    * county_event["mean_totalpop"]
    / 100000.0
)

county_event.to_csv(county_event_output, index=False)

print("\n================ County-event burden data ================")
print("County-event rows:", len(county_event))
print("Counties:", county_event["countyfips"].nunique())
print("Storms:", county_event["storm_key"].nunique())
print("Min burden per 100k:", county_event["tc_mortality_per_100k_172mo"].min())
print("Max burden per 100k:", county_event["tc_mortality_per_100k_172mo"].max())

# =============================================================================
# 4. Plot
# =============================================================================

fig, axes = plt.subplots(
    1, 2,
    figsize=(FIG_WIDTH, FIG_HEIGHT),
    constrained_layout=False
)

ax_a, ax_b = axes

color_map = {
    "Lower-SVI": LOWER_COLOR,
    "High-SVI": HIGH_COLOR
}

# -----------------------
# Panel a: SVI vs beta_g and stepwise response
# -----------------------

x_min = max(0.0, county_svi["mean_svi_county"].min() - 0.04)
x_max = min(1.0, county_svi["mean_svi_county"].max() + 0.04)

# Stepwise function implied by the SVI-group model:
# beta(SVI) = beta_L if SVI <= cutoff, beta_H if SVI > cutoff
ax_a.hlines(
    beta_lower,
    x_min,
    svi_cutoff,
    color=STEP_COLOR,
    linestyle="-",
    linewidth=1.6,
    alpha=0.85,
    label="Stepwise response"
)

ax_a.hlines(
    beta_high,
    svi_cutoff,
    x_max,
    color=STEP_COLOR,
    linestyle="-",
    linewidth=1.6,
    alpha=0.85
)

ax_a.vlines(
    svi_cutoff,
    beta_lower,
    beta_high,
    color=STEP_COLOR,
    linestyle="-",
    linewidth=1.2,
    alpha=0.85
)

for group, df_g in county_svi.groupby("svi_group"):
    ax_a.scatter(
        df_g["mean_svi_county"],
        df_g["beta_group"],
        s=POINT_SIZE_A,
        alpha=ALPHA_A,
        color=color_map[group],
        edgecolor="black",
        linewidth=0.4,
        label=group,
        zorder=3
    )

ax_a.axvline(
    svi_cutoff,
    color=CUT_LINE_COLOR,
    linestyle="--",
    linewidth=1.2,
    zorder=2
)

ax_a.text(
    svi_cutoff + 0.01,
    beta_lower + 0.12 * (beta_high - beta_lower),
    f"SVI cutoff = {svi_cutoff}",
    rotation=90,
    fontsize=ANNOTATION_SIZE,
    va="bottom",
    ha="left"
)

ax_a.set_xlim(x_min, x_max)

beta_margin = 0.10 * abs(beta_high - beta_lower)
ax_a.set_ylim(
    min(beta_lower, beta_high) - beta_margin,
    max(beta_lower, beta_high) + beta_margin
)

# Show exact beta values on y-axis
ax_a.set_yticks([beta_lower, beta_high])
ax_a.set_yticklabels(
    [
        rf"$\hat{{\beta}}_L={beta_lower:.3f}$",
        rf"$\hat{{\beta}}_H={beta_high:.3f}$"
    ],
    fontsize=TICK_LABEL_SIZE
)

ax_a.set_title(
    "SVI and estimated TC mortality response",
    fontsize=PANEL_TITLE_SIZE
)
ax_a.set_xlabel("County mean SVI", fontsize=AXIS_LABEL_SIZE)
ax_a.set_ylabel(
    "Estimated 172-month response\n"
    "(deaths per 100,000 per 1 m/s)",
    fontsize=AXIS_LABEL_SIZE
)

ax_a.text(
    -0.12, 1.03, "a",
    transform=ax_a.transAxes,
    fontsize=PANEL_LABEL_SIZE,
    fontweight="bold",
    va="bottom"
)

ax_a.tick_params(axis="x", labelsize=TICK_LABEL_SIZE)
ax_a.spines["top"].set_visible(False)
ax_a.spines["right"].set_visible(False)

# -----------------------
# Panel b: SVI vs county-event cumulative burden
# -----------------------

for group, df_g in county_event.groupby("svi_group"):
    ax_b.scatter(
        df_g["mean_svi_county"],
        df_g["tc_mortality_per_100k_172mo"],
        s=POINT_SIZE_B,
        alpha=ALPHA_B,
        color=color_map[group],
        edgecolor="none",
        label=group
    )

ax_b.axvline(
    svi_cutoff,
    color=CUT_LINE_COLOR,
    linestyle="--",
    linewidth=1.2
)

# Optional fitted curves for Panel b
fit_results = []
caption_fit_texts = []

if FIT_PANEL_B_CURVES:
    x_fit, y_fit = prepare_fit_data(
        county_event,
        x_col="mean_svi_county",
        y_col="tc_mortality_per_100k_172mo",
        use_binned=FIT_ON_BINNED_MEANS,
        n_bins=N_BINS_FOR_FIT,
        min_points_per_bin=MIN_POINTS_PER_BIN
    )

    x_grid = np.linspace(
        county_event["mean_svi_county"].min(),
        county_event["mean_svi_county"].max(),
        300
    )

    fit_styles = {
        "linear": {"linestyle": "-",  "label": "Linear fit"},
        "quadratic": {"linestyle": "--", "label": "Quadratic fit"},
        "exponential": {"linestyle": "-.", "label": "Exponential fit"},
        "logistic": {"linestyle": ":", "label": "Logistic fit"},
    }

    print("\n================ Panel b fitted curve formulas ================")
    print(f"Fit uses binned means: {FIT_ON_BINNED_MEANS}")
    print(f"Number of fit points: {len(x_fit)}")

    for fit_name in PANEL_B_FITS_TO_SHOW:
        try:
            y_grid, r2, params = make_fit_predictions(
                x_fit,
                y_fit,
                x_grid,
                fit_name
            )

            ax_b.plot(
                x_grid,
                y_grid,
                color=FIT_COLOR,
                linestyle=fit_styles[fit_name]["linestyle"],
                linewidth=FIT_LINEWIDTH,
                alpha=FIT_ALPHA,
                label=f"{fit_styles[fit_name]['label']} ($R^2={r2:.2f}$)"
            )

            fit_results.append({
                "fit": fit_name,
                "r2": r2,
                "fit_on_binned_means": FIT_ON_BINNED_MEANS,
                "n_fit_points": len(x_fit),
                **params
            })

            if PRINT_FIT_FORMULAS:
                print_fit_formula(fit_name, params, r2)

            caption_fit_texts.append(
                formula_text_for_fit(fit_name, params, r2, digits=3)
            )

        except Exception as e:
            print(f"Could not fit {fit_name}: {e}")

if fit_results:
    fit_results_df = pd.DataFrame(fit_results)
    print("\n================ Panel b fitted curve results ================")
    print(fit_results_df)
    fit_results_df.to_csv(fit_results_output, index=False)

# Panel b axis formatting
if PANEL_B_YMAX_QUANTILE is not None:
    ymax = county_event["tc_mortality_per_100k_172mo"].quantile(PANEL_B_YMAX_QUANTILE)
    ax_b.set_ylim(bottom=0, top=ymax)

ax_b.set_title(
    "SVI and county-event TC mortality burden",
    fontsize=PANEL_TITLE_SIZE
)
ax_b.set_xlabel("County mean SVI", fontsize=AXIS_LABEL_SIZE)
ax_b.set_ylabel(
    "Cumulative estimated mortality burden\n"
    "(deaths per 100,000)",
    fontsize=AXIS_LABEL_SIZE
)

ax_b.text(
    -0.12, 1.03, "b",
    transform=ax_b.transAxes,
    fontsize=PANEL_LABEL_SIZE,
    fontweight="bold",
    va="bottom"
)

ax_b.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
ax_b.spines["top"].set_visible(False)
ax_b.spines["right"].set_visible(False)

# =============================================================================
# Legends and optional formula note
# =============================================================================

legend_elements = [
    Line2D(
        [0], [0],
        marker="o",
        color="w",
        label="Lower-SVI",
        markerfacecolor=LOWER_COLOR,
        markeredgecolor="black",
        markersize=8
    ),
    Line2D(
        [0], [0],
        marker="o",
        color="w",
        label="High-SVI",
        markerfacecolor=HIGH_COLOR,
        markeredgecolor="black",
        markersize=8
    ),
    Line2D(
        [0], [0],
        color=CUT_LINE_COLOR,
        linestyle="--",
        label=f"SVI cutoff = {svi_cutoff}"
    ),
    Line2D(
        [0], [0],
        color=STEP_COLOR,
        linestyle="-",
        linewidth=1.6,
        label="Stepwise response"
    )
]

if FIT_PANEL_B_CURVES and fit_results:
    for fit_name in PANEL_B_FITS_TO_SHOW:
        if fit_name in [r["fit"] for r in fit_results]:
            linestyle = {
                "linear": "-",
                "quadratic": "--",
                "exponential": "-.",
                "logistic": ":"
            }[fit_name]

            label = {
                "linear": "Linear fit",
                "quadratic": "Quadratic fit",
                "exponential": "Exponential fit",
                "logistic": "Logistic fit"
            }[fit_name]

            r2_val = [r["r2"] for r in fit_results if r["fit"] == fit_name][0]

            legend_elements.append(
                Line2D(
                    [0], [0],
                    color=FIT_COLOR,
                    linestyle=linestyle,
                    linewidth=FIT_LINEWIDTH,
                    label=f"{label} ($R^2={r2_val:.2f}$)"
                )
            )

if SHOW_FORMULA_IN_FIGURE and caption_fit_texts:
    fig.text(
        0.5,
        0.035,
        " ".join(caption_fit_texts),
        ha="center",
        va="bottom",
        fontsize=CAPTION_SIZE
    )
    bottom_margin = BOTTOM_MARGIN_WITH_FORMULA
    legend_y = 0.010
else:
    bottom_margin = BOTTOM_MARGIN_NO_FORMULA
    legend_y = LEGEND_Y

fig.legend(
    handles=legend_elements,
    loc="lower center",
    bbox_to_anchor=(0.5, legend_y),
    ncol=5,
    frameon=False,
    fontsize=LEGEND_SIZE
)

fig.suptitle(
    "Social vulnerability and estimated TC-induced mortality burden",
    fontsize=TITLE_SIZE,
    y=0.98
)

plt.subplots_adjust(
    left=0.08,
    right=0.98,
    top=0.88,
    bottom=bottom_margin,
    wspace=0.17
)

plt.savefig(fig_path_png, dpi=300, bbox_inches="tight")
plt.savefig(fig_path_pdf, bbox_inches="tight")
plt.show()

print("\nSaved figure:")
print(fig_path_png)
print(fig_path_pdf)

print("\nSaved data:")
print(county_beta_output)
print(county_event_output)

if fit_results:
    print(fit_results_output)

# =============================================================================
# Suggested LaTeX caption text
# =============================================================================

print("\n================ Suggested figure caption text ================")
print(
    "Panel (a) shows the SVI-group-specific 172-month cumulative mortality-rate "
    "response per 1 m/s of TC wind exposure. Because the empirical model estimates "
    "separate coefficients for lower- and high-SVI counties, the fitted response is "
    f"a step function with beta_L = {beta_lower:.3f} and beta_H = {beta_high:.3f}. "
    "Panel (b) shows county-event cumulative fitted TC mortality burden per 100,000, "
    "computed as the SVI-group coefficient multiplied by event-county wind exposure."
)

if caption_fit_texts:
    print(" " + " ".join(caption_fit_texts))

# =============================================================================
# 5. Diagnostics
# =============================================================================

print("\n================ Panel a diagnostics ================")
print(
    county_svi
    .groupby("svi_group")
    .agg(
        n_counties=("countyfips", "nunique"),
        mean_svi=("mean_svi_county", "mean"),
        beta=("beta_group", "first")
    )
)

print("\n================ Panel b diagnostics ================")
print(
    county_event
    .groupby("svi_group")
    .agg(
        n_county_events=("storm_key", "size"),
        n_counties=("countyfips", "nunique"),
        n_storms=("storm_key", "nunique"),
        mean_burden_per_100k=("tc_mortality_per_100k_172mo", "mean"),
        median_burden_per_100k=("tc_mortality_per_100k_172mo", "median"),
        max_burden_per_100k=("tc_mortality_per_100k_172mo", "max")
    )
)

print("\nTop 20 county-events by full 172-month estimated TC burden per 100,000:")
print(
    county_event
    .sort_values("tc_mortality_per_100k_172mo", ascending=False)
    .head(20)
    [
        [
            "countyfips",
            "storm_name_plot",
            "storm_key",
            "event_year",
            "event_month",
            "mean_svi_county",
            "svi_group",
            "maxs",
            "beta_group",
            "tc_mortality_per_100k_172mo",
            "tc_excess_deaths_172mo_meanpop"
        ]
    ]
)