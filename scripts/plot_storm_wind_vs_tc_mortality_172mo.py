# %% Storm wind exposure vs. estimated 172-month TC-induced mortality in Texas
#
# Creates a storm-level scatter plot with:
#   x-axis: sum_c W_{c,t_s}
#   y-axis: sum_c sum_{ell=0}^{172}
#           beta_{g(c)} * h_ell * W_{c,t_s} * Pop_{c,t_s+ell} / 100000
#
# If t_s + ell is beyond 2018, the script uses the corresponding
# calendar-month population from 2018.
#
# Outputs:
#   output/figures/storm_wind_vs_tc_mortality_full172mo.png
#   output/figures/storm_wind_vs_tc_mortality_full172mo.pdf
#   output/figures/storm_wind_vs_tc_mortality_full172mo_data.csv
#   output/figures/storm_wind_vs_tc_mortality_full172mo_county_lag_data.csv

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =============================================================================
# User settings
# =============================================================================

base_dir = r"C:\Users\toplu\Documents\GithubRepositories\social_cost_of_storms"
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

fig_path_png = os.path.join(fig_output_dir, "storm_wind_vs_tc_mortality_full172mo.png")
fig_path_pdf = os.path.join(fig_output_dir, "storm_wind_vs_tc_mortality_full172mo.pdf")
plot_data_output = os.path.join(fig_output_dir, "storm_wind_vs_tc_mortality_full172mo_data.csv")
county_lag_output = os.path.join(fig_output_dir, "storm_wind_vs_tc_mortality_full172mo_county_lag_data.csv")

# =============================================================================
# Analysis settings
# =============================================================================

LMAX = 172
svi_cutoff = 0.75

STORM_START_YEAR = 1953
END_YEAR = 2018

# If response year > 2018, use same calendar-month population from 2018.
POP_CAP_YEAR = 2018

USE_REGRESSION_COUNTIES_ONLY = True
DROP_ZERO_EXPOSURE = True

# Required x-axis:
#   sum over counties {W_{c,t_s}} for each storm s
WIND_METRIC_FOR_X = "storm_total_maxs"
LABEL_TOP_N_BY_EXPOSURE = 16


DROP_ZERO_OR_NEGATIVE_STORM_BURDENS_FOR_PLOT = False

LABEL_TOP_N_BY_DEATHS = 12
LABEL_TOP_N_BY_WIND = 6
MIN_LABEL_GAP_X_FRACTION = 0.035
MIN_LABEL_GAP_Y_FRACTION = 0.040

SHOW_LINEAR_FIT = True
FIT_THROUGH_ORIGIN = False

# =============================================================================
# Figure settings
# =============================================================================

FIG_WIDTH = 10.5
FIG_HEIGHT = 7.0
TITLE_SIZE = 15
AXIS_LABEL_SIZE = 13
TICK_LABEL_SIZE = 11
LEGEND_SIZE = 10
ANNOTATION_SIZE = 9

POINT_SIZE_MIN = 45
POINT_SIZE_MAX = 230
ALPHA = 0.78
EDGE_COLOR = "black"

decade_colors = {
    1950: "#e4572e",
    1960: "#f2b01e",
    1970: "#7b3294",
    1980: "#5bc0eb",
    1990: "#b2182b",
    2000: "#70ad47",
    2010: "#253494",
}

# =============================================================================
# Helper functions
# =============================================================================

def nature_shape(x):
    return 0.0237 + 0.000535 * x - 0.0000039 * (x ** 2)


def make_nature_lag_weights(max_lag=172):
    lags = np.arange(0, max_lag + 1)
    raw = nature_shape(lags)

    if (raw < 0).any():
        bad = lags[raw < 0]
        raise ValueError(f"Nature lag shape is negative at lags: {bad}")

    return pd.DataFrame({
        "lag": lags,
        "nature_shape_raw": raw,
        "nature_weight": raw / raw.sum(),
    })


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
        0.svi_high#c.X_nature -> lower-SVI beta
        1.svi_high#c.X_nature -> high-SVI beta
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


def month_add(year, month, lag):
    base = year.astype(int) * 12 + month.astype(int) - 1 + lag.astype(int)
    out_year = base // 12
    out_month = base % 12 + 1
    return out_year, out_month


def assign_decade(year):
    return int(year // 10 * 10)


def rescale_marker_sizes(values, min_size=45, max_size=230):
    values = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy()

    if values.max() <= values.min():
        return np.repeat((min_size + max_size) / 2, len(values))

    scaled = (values - values.min()) / (values.max() - values.min())
    return min_size + scaled * (max_size - min_size)


def select_nonoverlapping_labels(df, x_col, y_col, top_n_deaths, top_n_wind):
    candidates = pd.concat(
        [
            df.sort_values("total_excess_deaths_172mo", ascending=False).head(top_n_deaths),
            df.sort_values(x_col, ascending=False).head(top_n_wind),
        ],
        ignore_index=True
    ).drop_duplicates("storm_key")

    candidates = candidates.sort_values("total_excess_deaths_172mo", ascending=False).copy()

    x_range = max(df[x_col].max() - df[x_col].min(), 1e-9)
    y_range = max(df[y_col].max() - df[y_col].min(), 1e-9)

    min_dx = MIN_LABEL_GAP_X_FRACTION * x_range
    min_dy = MIN_LABEL_GAP_Y_FRACTION * y_range

    selected = []

    for _, row in candidates.iterrows():
        overlaps = False

        for kept in selected:
            if abs(row[x_col] - kept[x_col]) < min_dx and abs(row[y_col] - kept[y_col]) < min_dy:
                overlaps = True
                break

        if not overlaps:
            selected.append(row)

    if selected:
        return pd.DataFrame(selected)

    return pd.DataFrame(columns=df.columns)


# =============================================================================
# 1. Load regression data: population, SVI group, and estimation counties
# =============================================================================

reg = pd.read_stata(regression_data_file)

needed = ["countyfips", "year", "month", "totalpop", "svi"]
missing = [c for c in needed if c not in reg.columns]
if missing:
    raise KeyError(f"Regression data missing columns: {missing}")

reg = reg.copy()
reg["countyfips"] = zfill5(reg["countyfips"])
reg["year"] = reg["year"].astype(int)
reg["month"] = reg["month"].astype(int)
reg["totalpop"] = pd.to_numeric(reg["totalpop"], errors="coerce")
reg["svi"] = pd.to_numeric(reg["svi"], errors="coerce")

county_svi = (
    reg.groupby("countyfips", as_index=False)
    .agg(
        mean_svi_county=("svi", "mean"),
        mean_totalpop=("totalpop", "mean"),
    )
    .dropna(subset=["mean_svi_county"])
)

county_svi["svi_high"] = np.where(
    county_svi["mean_svi_county"] > svi_cutoff,
    1,
    0
).astype(int)

county_svi["svi_group"] = np.where(
    county_svi["svi_high"] == 1,
    "High-SVI",
    "Lower-SVI"
)

pop = (
    reg[["countyfips", "year", "month", "totalpop"]]
    .dropna(subset=["totalpop"])
    .drop_duplicates(["countyfips", "year", "month"])
    .copy()
)

regression_counties = set(county_svi["countyfips"].unique())

print("\n================ Regression sample ================")
print("Regression counties:", len(regression_counties))
print("SVI group county counts:")
print(county_svi["svi_group"].value_counts())


# =============================================================================
# 2. Load coefficients and lag weights
# =============================================================================

coef = read_coefficients(coef_file_dta, coef_file_csv)
beta_by_group = extract_svi_betas(coef)

print("\n================ SVI-group X_nature betas ================")
print(f"Lower-SVI beta: {beta_by_group[0]:.6f}")
print(f"High-SVI beta:  {beta_by_group[1]:.6f}")

lag_weights = make_nature_lag_weights(LMAX)


# =============================================================================
# 3. Load Texas county-storm wind panel
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
    low_memory=False,
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
    label="storm serial id",
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

wind_tx = wind_tx.rename(
    columns={
        county_col: "countyfips",
        year_col: "event_year",
        month_col: "event_month",
        maxs_col: "maxs",
    }
)

wind_tx = wind_tx.merge(
    county_svi[["countyfips", "mean_svi_county", "svi_high", "svi_group", "mean_totalpop"]],
    on="countyfips",
    how="left",
)

if wind_tx["svi_high"].isna().any():
    n_missing = wind_tx["svi_high"].isna().sum()
    print(f"Warning: dropping {n_missing:,} wind rows with missing SVI group.")
    wind_tx = wind_tx.dropna(subset=["svi_high"]).copy()

wind_tx["svi_high"] = wind_tx["svi_high"].astype(int)
wind_tx["beta_group"] = wind_tx["svi_high"].map(beta_by_group)
wind_tx["event_decade"] = wind_tx["event_year"].apply(assign_decade)
wind_tx["pop_weight_for_wind"] = pd.to_numeric(wind_tx["mean_totalpop"], errors="coerce")

print("\n================ Texas wind data after filters ================")
print("County-storm rows:", len(wind_tx))
print("Counties:", wind_tx["countyfips"].nunique())
print("Storms:", wind_tx["storm_key"].nunique())


# =============================================================================
# 4. Compute storm-level x-axis wind exposure:
#    sum over counties {W_{c,t_s}} for each TC s
# =============================================================================

def weighted_mean_maxs(df):
    w = df["pop_weight_for_wind"].fillna(0.0).to_numpy()
    x = df["maxs"].to_numpy()

    if w.sum() <= 0:
        return float(np.mean(x))

    return float(np.average(x, weights=w))


wind_summary_base = (
    wind_tx.groupby(
        ["storm_key", "storm_name_plot", "event_year", "event_month", "event_decade"],
        as_index=False
    )
    .agg(
        storm_total_maxs=("maxs", "sum"),
        storm_mean_maxs=("maxs", "mean"),
        storm_median_maxs=("maxs", "median"),
        storm_max_county_maxs=("maxs", "max"),
        n_exposed_counties=("countyfips", "nunique"),
        mean_svi_exposed_counties=("mean_svi_county", "mean"),
        high_svi_county_events=("svi_high", "sum"),
        mean_county_population=("mean_totalpop", "mean"),
    )
)

wind_popw = (
    wind_tx.groupby(
        ["storm_key", "storm_name_plot", "event_year", "event_month", "event_decade"]
    )
    .apply(weighted_mean_maxs)
    .reset_index(name="storm_popw_mean_maxs")
)

wind_summary = wind_summary_base.merge(
    wind_popw,
    on=["storm_key", "storm_name_plot", "event_year", "event_month", "event_decade"],
    how="left",
)

wind_summary["share_high_svi_county_events"] = (
    wind_summary["high_svi_county_events"] / wind_summary["n_exposed_counties"]
)


# =============================================================================
# 5. Compute full 172-month total estimated excess deaths by storm
# =============================================================================

event_rows = wind_tx[
    [
        "countyfips",
        "storm_key",
        "storm_name_plot",
        "event_year",
        "event_month",
        "event_decade",
        "maxs",
        "svi_high",
        "svi_group",
        "beta_group",
    ]
].copy()

event_rows["event_row_id"] = np.arange(len(event_rows))

# Expand each county-storm event over lags ell=0,...,172.
expanded = event_rows.merge(lag_weights, how="cross")

expanded["response_year"], expanded["response_month"] = month_add(
    expanded["event_year"],
    expanded["event_month"],
    expanded["lag"]
)

# Population lookup:
# if response_year > 2018, use same calendar-month population from 2018.
expanded["pop_lookup_year"] = np.where(
    expanded["response_year"] > POP_CAP_YEAR,
    POP_CAP_YEAR,
    expanded["response_year"]
).astype(int)

expanded["pop_lookup_month"] = expanded["response_month"].astype(int)

pop_lookup_exact = pop[["countyfips", "year", "month", "totalpop"]].copy()
pop_lookup_exact = pop_lookup_exact.rename(
    columns={
        "year": "pop_lookup_year",
        "month": "pop_lookup_month",
        "totalpop": "response_totalpop_exact",
    }
)

expanded = expanded.merge(
    pop_lookup_exact,
    on=["countyfips", "pop_lookup_year", "pop_lookup_month"],
    how="left",
    validate="m:1",
)

# Fallback for missing population:
# nearest available year within the same county-calendar-month.
missing_mask = expanded["response_totalpop_exact"].isna()

if missing_mask.any():
    pop_by_county_month = {
        (c, m): g[["year", "totalpop"]].dropna().sort_values("year").reset_index(drop=True)
        for (c, m), g in pop.groupby(["countyfips", "month"])
    }

    fallback_values = []
    fallback_years = []

    for c, m, y in expanded.loc[
        missing_mask,
        ["countyfips", "pop_lookup_month", "pop_lookup_year"]
    ].itertuples(index=False):

        g = pop_by_county_month.get((c, int(m)))

        if g is None or g.empty:
            fallback_values.append(np.nan)
            fallback_years.append(np.nan)
            continue

        years = g["year"].to_numpy(dtype=int)
        pops = g["totalpop"].to_numpy(dtype=float)

        pos = int(np.searchsorted(years, int(y)))

        candidates = []
        if pos < len(years):
            candidates.append(pos)
        if pos > 0:
            candidates.append(pos - 1)

        best = min(candidates, key=lambda j: abs(years[j] - int(y)))

        fallback_values.append(pops[best])
        fallback_years.append(years[best])

    expanded.loc[missing_mask, "response_totalpop_fallback"] = fallback_values
    expanded.loc[missing_mask, "response_pop_fallback_year"] = fallback_years

else:
    expanded["response_totalpop_fallback"] = np.nan
    expanded["response_pop_fallback_year"] = np.nan

expanded["response_totalpop"] = expanded["response_totalpop_exact"].fillna(
    expanded["response_totalpop_fallback"]
)

missing_pop = expanded["response_totalpop"].isna().sum()
if missing_pop > 0:
    print(f"Warning: dropping {missing_pop:,} county-storm-lag rows with no usable population.")
    expanded = expanded.dropna(subset=["response_totalpop"]).copy()

expanded["used_capped_2018_population"] = expanded["response_year"] > POP_CAP_YEAR
expanded["used_population_fallback"] = expanded["response_totalpop_exact"].isna()

# Core calculation:
# deaths_{c,s,ell} =
# beta_{g(c)} * h_ell * W_{c,t_s} * Pop_{c,t_s+ell} / 100000
expanded["excess_deaths"] = (
    expanded["beta_group"]
    * expanded["nature_weight"]
    * expanded["maxs"]
    * expanded["response_totalpop"]
    / 100000.0
)

expanded["excess_deaths_lower_svi"] = np.where(
    expanded["svi_high"] == 0,
    expanded["excess_deaths"],
    0.0
)

expanded["excess_deaths_high_svi"] = np.where(
    expanded["svi_high"] == 1,
    expanded["excess_deaths"],
    0.0
)

# County-storm totals.
county_storm_deaths = (
    expanded.groupby(
        [
            "event_row_id",
            "countyfips",
            "storm_key",
            "storm_name_plot",
            "event_year",
            "event_month",
            "event_decade",
            "maxs",
            "svi_high",
            "svi_group",
            "beta_group",
        ],
        as_index=False
    )
    .agg(
        county_storm_excess_deaths_172mo=("excess_deaths", "sum"),
        county_storm_lower_svi_deaths_172mo=("excess_deaths_lower_svi", "sum"),
        county_storm_high_svi_deaths_172mo=("excess_deaths_high_svi", "sum"),
        n_lags_used=("lag", "size"),
        n_capped_2018_pop_lags=("used_capped_2018_population", "sum"),
        n_fallback_pop_lags=("used_population_fallback", "sum"),
    )
)

# Storm totals = sum over counties.
storm_deaths = (
    county_storm_deaths.groupby(["storm_key"], as_index=False)
    .agg(
        total_excess_deaths_172mo=("county_storm_excess_deaths_172mo", "sum"),
        lower_svi_excess_deaths_172mo=("county_storm_lower_svi_deaths_172mo", "sum"),
        high_svi_excess_deaths_172mo=("county_storm_high_svi_deaths_172mo", "sum"),
        n_county_event_rows=("event_row_id", "nunique"),
        n_capped_2018_pop_lags=("n_capped_2018_pop_lags", "sum"),
        n_fallback_pop_lags=("n_fallback_pop_lags", "sum"),
    )
)

plot_df = wind_summary.merge(storm_deaths, on="storm_key", how="left")

plot_df[
    [
        "total_excess_deaths_172mo",
        "lower_svi_excess_deaths_172mo",
        "high_svi_excess_deaths_172mo",
    ]
] = plot_df[
    [
        "total_excess_deaths_172mo",
        "lower_svi_excess_deaths_172mo",
        "high_svi_excess_deaths_172mo",
    ]
].fillna(0.0)

plot_df["event_date"] = pd.to_datetime(
    plot_df["event_year"].astype(str)
    + "-"
    + plot_df["event_month"].astype(str).str.zfill(2)
    + "-01"
)

plot_df.to_csv(plot_data_output, index=False)
expanded.to_csv(county_lag_output, index=False)


# =============================================================================
# 6. Plot
# =============================================================================

if WIND_METRIC_FOR_X not in plot_df.columns:
    raise ValueError(
        f"WIND_METRIC_FOR_X='{WIND_METRIC_FOR_X}' is not in plot_df. "
        f"Available columns include: {plot_df.columns.tolist()}"
    )

plot_data = plot_df.copy()

if DROP_ZERO_OR_NEGATIVE_STORM_BURDENS_FOR_PLOT:
    plot_data = plot_data[plot_data["total_excess_deaths_172mo"] > 0].copy()

x_col = WIND_METRIC_FOR_X
y_col = "total_excess_deaths_172mo"

x_label_map = {
    "storm_total_maxs": "TC wind exposure: summed county-event maximum wind speed (m/s)",
    "storm_mean_maxs": "TC wind exposure: mean exposed-county maximum wind speed (m/s)",
    "storm_max_county_maxs": "TC wind exposure: maximum county wind speed (m/s)",
    "storm_popw_mean_maxs": "TC wind exposure: population-weighted mean county wind speed (m/s)",
}

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

sizes = np.repeat(90, len(plot_data))

colors = plot_data["event_decade"].map(lambda d: decade_colors.get(d, "gray"))

ax.scatter(
    plot_data[x_col],
    plot_data[y_col],
    s=90,
    c=colors,
    alpha=ALPHA,
    edgecolor=EDGE_COLOR,
    linewidth=0.5,
)

# Optional descriptive fit.
fit_note = ""

if SHOW_LINEAR_FIT and len(plot_data) >= 2:
    x = plot_data[x_col].to_numpy(dtype=float)
    y = plot_data[y_col].to_numpy(dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) >= 2 and np.nanmax(x) > np.nanmin(x):
        x_grid = np.linspace(0, x.max() * 1.03, 200)

        if FIT_THROUGH_ORIGIN:
            slope = float(np.sum(x * y) / np.sum(x * x))
            y_grid = slope * x_grid
            y_hat = slope * x
            fit_note = rf"Descriptive fit: $y={slope:.3f}x$"
        else:
            slope, intercept = np.polyfit(x, y, deg=1)
            y_grid = slope * x_grid + intercept
            y_hat = slope * x + intercept
            fit_note = rf"Descriptive fit: $y={slope:.3f}x {intercept:+.3f}$"

        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        fit_note = fit_note + rf", $R^2={r2:.2f}$"

        ax.plot(
            x_grid,
            y_grid,
            color="black",
            linestyle="--",
            linewidth=1.5,
            alpha=0.85,
        )

# Label storms with the largest TC wind exposure.
labels_df = (
    plot_data
    .sort_values(x_col, ascending=False)
    .head(LABEL_TOP_N_BY_EXPOSURE)
    .copy()
)

for _, row in labels_df.iterrows():
    label = str(row["storm_name_plot"]).title()

    ax.annotate(
        label,
        xy=(row[x_col], row[y_col]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=ANNOTATION_SIZE,
        ha="left",
        va="bottom",
        arrowprops=dict(
            arrowstyle="-",
            linewidth=0.4,
            color="black",
            alpha=0.45,
        ),
    )

ax.set_xlabel(x_label_map.get(x_col, x_col), fontsize=AXIS_LABEL_SIZE)
ax.set_ylabel(
    "Total estimated TC-induced mortality over 172 months",
    fontsize=AXIS_LABEL_SIZE,
)

ax.set_title(
    "Estimated long-run excess deaths for each tropical cyclone in Texas\n"
    "42-county estimation sample",
    fontsize=TITLE_SIZE,
)

ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.20, linewidth=0.7)

ax.set_xlim(left=0)
ax.set_ylim(bottom=0)


# Decade legend placed inside the axes so it is not cropped.
decades_in_plot = sorted(plot_data["event_decade"].dropna().astype(int).unique())

decade_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        color="none",
        label=f"{int(d)}s",
        markerfacecolor=decade_colors.get(int(d), "gray"),
        markeredgecolor=EDGE_COLOR,
        markersize=7,
    )
    for d in decades_in_plot
]

ax.legend(
    handles=decade_handles,
    title="Event decade",
    loc="upper left",
    frameon=False,
    fontsize=LEGEND_SIZE,
    title_fontsize=LEGEND_SIZE,
)

plt.tight_layout()
plt.savefig(fig_path_png, dpi=300, bbox_inches="tight")
plt.savefig(fig_path_pdf, bbox_inches="tight")
plt.show()

print("\nSaved figure:")
print(fig_path_png)
print(fig_path_pdf)

print("\nSaved data:")
print(plot_data_output)
print(county_lag_output)


# =============================================================================
# 7. Diagnostics
# =============================================================================

print("\n================ Plot diagnostics ================")
print("X-axis wind metric:", WIND_METRIC_FOR_X)
print("Number of plotted storms:", len(plot_data))

print("\nDeaths min/mean/max:")
print(plot_data[y_col].agg(["min", "mean", "max"]))

print("\nWind metric min/mean/max:")
print(plot_data[x_col].agg(["min", "mean", "max"]))

print("\nTop storms by total estimated 172-month excess deaths:")
print(
    plot_data.sort_values(y_col, ascending=False)
    .head(20)
    [
        [
            "storm_name_plot",
            "event_year",
            "event_month",
            "storm_total_maxs",
            "n_exposed_counties",
            "total_excess_deaths_172mo",
            "lower_svi_excess_deaths_172mo",
            "high_svi_excess_deaths_172mo",
            "n_capped_2018_pop_lags",
        ]
    ]
)

print("\nTop storms by summed county-event wind exposure:")
print(
    plot_data.sort_values("storm_total_maxs", ascending=False)
    .head(20)
    [
        [
            "storm_name_plot",
            "event_year",
            "event_month",
            "storm_total_maxs",
            "n_exposed_counties",
            "total_excess_deaths_172mo",
            "n_capped_2018_pop_lags",
        ]
    ]
)