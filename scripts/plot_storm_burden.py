# %% Texas-only TC mortality burden plot
# Recreates a Texas-only version of Nature Fig. 3:
#   Panel a: monthly Texas TC wind incidence
#   Panel b: stacked estimated excess mortality responses by storm
#   Panel c: optional Texas official direct deaths, only if you provide a file
#
# Uses SVI-group beta coefficients from the compact X_nature model:
#   0.svi_high#c.X_nature
#   1.svi_high#c.X_nature
#
# Outcome assumed: tdths_ttpop = deaths per 100,000 population.
# Therefore estimated excess deaths are:
#   beta_g * nature_weight_lag * maxs_event_county * population_county_month / 100000

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# =============================================================================
# User settings
# =============================================================================

from pathlib import Path
base_dir = Path(__file__).resolve().parents[1]
data_dir = os.path.join(base_dir, "data")
output_dir = os.path.join(base_dir, "output")
beta_dir = os.path.join(output_dir, "beta")

storm_panel_file = os.path.join(
    data_dir,
    "panel_by_storm__NA_USA_density_8_yr_1930_2018.csv"
)

regression_data_file = os.path.join(output_dir, "full_data_for_regression.dta")

coef_file_dta = os.path.join(beta_dir, "coefficients_county_nature_future.dta")
coef_file_csv = os.path.join(beta_dir, "coefficients_region_poly.csv")

# Optional. Leave as None if you do not have Texas direct deaths yet.
# Expected columns: year, month, direct_deaths
# Optional columns: storm_name, storm_serial_id
texas_direct_deaths_file = None
# texas_direct_deaths_file = os.path.join(data_dir, "texas_direct_deaths.csv")

fig_output_dir = os.path.join(output_dir, "figures")
os.makedirs(fig_output_dir, exist_ok=True)

fig_path_png = os.path.join(fig_output_dir, "texas_tc_mortality_burden.png")
fig_path_pdf = os.path.join(fig_output_dir, "texas_tc_mortality_burden.pdf")

# =============================================================================
# Plot settings
# =============================================================================

TITLE_SIZE = 17
AXIS_LABEL_SIZE = 14
TICK_LABEL_SIZE = 12
PANEL_LABEL_SIZE = 16
STORM_LABEL_SIZE = 12
LEGEND_SIZE = 11
LEGEND_TITLE_SIZE = 12

# =============================================================================
# Modeling and time-window settings
# =============================================================================

LMAX = 172
svi_cutoff = 0.75

# Storm exposure history and mortality plotting period.
# Storms start earlier because the mortality response uses a 172-month lag window.
STORM_START_YEAR = 1953
MORTALITY_START_YEAR = 1968
END_YEAR = 2018

# Use only counties in your regression sample.
# Recommended for consistency with the estimated SVI-group betas.
USE_REGRESSION_COUNTIES_ONLY = True

# Drop zero-exposure county-storm observations before expanding lags.
DROP_ZERO_EXPOSURE = True

# Panel a aggregation:
#   "sum": sum county-level maxs across counties/storms in each month
#   "mean": average county-level maxs among exposed counties/storms in each month
PANEL_A_AGG = "sum"

# If True, panel b only plots storm responses whose total estimated deaths are positive.
DROP_ZERO_OR_NEGATIVE_STORM_RESPONSES = False

# Label settings
top_label_n = 20
min_label_gap_months = 18

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
        "nature_weight": raw / raw.sum()
    })


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
    Returns:
        {0: beta_lower_svi, 1: beta_high_svi}
    from:
        0.svi_high#c.X_nature
        1.svi_high#c.X_nature
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
            "Expected coefficients like 0.svi_high#c.X_nature and 1.svi_high#c.X_nature."
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
    """
    Vectorized month addition.
    year, month, and lag can be pandas Series or numpy arrays.
    """
    base = year.astype(int) * 12 + month.astype(int) - 1 + lag.astype(int)
    out_year = base // 12
    out_month = base % 12 + 1
    return out_year, out_month


def assign_decade(year):
    return int(year // 10 * 10)


# =============================================================================
# 1. Load regression data: population and SVI group
# =============================================================================

reg = pd.read_stata(regression_data_file)

needed = ["countyfips", "year", "month", "totalpop", "svi"]
missing = [c for c in needed if c not in reg.columns]
if missing:
    raise KeyError(f"Regression data missing columns: {missing}")

reg = reg.copy()
reg["countyfips"] = reg["countyfips"].astype(str).str.zfill(5)
reg["year"] = reg["year"].astype(int)
reg["month"] = reg["month"].astype(int)
reg["totalpop"] = pd.to_numeric(reg["totalpop"], errors="coerce")

county_svi = (
    reg.groupby("countyfips", as_index=False)
    .agg(mean_svi_county=("svi", "mean"))
)
county_svi["svi_high"] = np.where(
    county_svi["mean_svi_county"] > svi_cutoff,
    1,
    0
).astype(int)

pop = (
    reg[["countyfips", "year", "month", "totalpop"]]
    .dropna(subset=["totalpop"])
    .drop_duplicates(["countyfips", "year", "month"])
    .copy()
)

regression_counties = set(reg["countyfips"].unique())

print("Regression counties:", len(regression_counties))
print("SVI group county counts:")
print(county_svi["svi_high"].value_counts().sort_index())

# =============================================================================
# 2. Load SVI-group betas and lag weights
# =============================================================================

coef = read_coefficients(coef_file_dta, coef_file_csv)
beta_by_group = extract_svi_betas(coef)

print("\nSVI-group X_nature betas:")
for g, b in beta_by_group.items():
    print(f"  svi_high={g}: beta = {b:.6f}")

lag_weights = make_nature_lag_weights(LMAX)

# =============================================================================
# 3. Load storm-county wind panel for Texas
# =============================================================================

wind = pd.read_csv(
    storm_panel_file,
    dtype={
        "countyfips": str,
        "stfips": str,
        "storm_serial_id": str,
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
# Storm ID column: prefer the unique LICRICE storm serial ID.
storm_serial_col = infer_col(
    wind,
    [
        "storm_serial_ID",   # preferred; preserves 104 Texas events
        "storm_serial_id",
        "storm_ser",
        "storm_serial",
        "serial_id",
        "storm_id",
    ],
    required=False,
    label="storm serial id"
)

wind = wind.copy()
wind[county_col] = wind[county_col].astype(str).str.zfill(5)
wind[year_col] = wind[year_col].astype(int)
wind[month_col] = wind[month_col].astype(int)
wind[maxs_col] = pd.to_numeric(wind[maxs_col], errors="coerce").fillna(0.0)

# Texas filter
if state_col is not None:
    wind[state_col] = wind[state_col].astype(str).str.zfill(2)
    wind_tx = wind[wind[state_col] == "48"].copy()
else:
    wind_tx = wind[wind[county_col].str.startswith("48")].copy()

# Keep storm exposure history from 1953 onward.
wind_tx = wind_tx[
    (wind_tx[year_col] >= STORM_START_YEAR) &
    (wind_tx[year_col] <= END_YEAR)
].copy()

if USE_REGRESSION_COUNTIES_ONLY:
    wind_tx = wind_tx[wind_tx[county_col].isin(regression_counties)].copy()

if DROP_ZERO_EXPOSURE:
    wind_tx = wind_tx[wind_tx[maxs_col] > 0].copy()

# Create storm key.
# Use storm_serial_ID when available so storms occurring in the same month are not collapsed.
if storm_serial_col is not None:
    wind_tx["storm_key"] = wind_tx[storm_serial_col].astype(str).str.strip()
else:
    # Fallback only if no storm serial identifier exists.
    # This may collapse distinct storms in the same month, so avoid it when possible.
    if storm_name_col is None:
        wind_tx["storm_key"] = (
            "UNKNOWN_"
            + wind_tx[year_col].astype(str)
            + "_"
            + wind_tx[month_col].astype(str).str.zfill(2)
        )
    else:
        wind_tx["storm_key"] = (
            wind_tx[storm_name_col].astype(str).fillna("UNKNOWN").str.strip()
            + "_"
            + wind_tx[year_col].astype(str)
            + "_"
            + wind_tx[month_col].astype(str).str.zfill(2)
        )

if storm_name_col is not None:
    wind_tx["storm_name_plot"] = wind_tx[storm_name_col].astype(str).str.strip()
else:
    wind_tx["storm_name_plot"] = wind_tx["storm_key"]


wind_tx = wind_tx.rename(columns={
    county_col: "countyfips",
    year_col: "event_year",
    month_col: "event_month",
    maxs_col: "maxs"
})

    
print("\n================ Storm ID diagnostics ================")
print("Storm serial column used:", storm_serial_col)
print("Unique storm_key:", wind_tx["storm_key"].nunique())

if "storm_serial_ID" in wind_tx.columns:
    print("Unique storm_serial_ID:", wind_tx["storm_serial_ID"].nunique())

if "storm_serial_id" in wind_tx.columns:
    print("Unique storm_serial_id:", wind_tx["storm_serial_id"].nunique())

same_month_storms = (
    wind_tx[["storm_key", "storm_name_plot", "event_year", "event_month"]]
    .drop_duplicates()
    .groupby(["event_year", "event_month"], as_index=False)
    .agg(
        n_storms=("storm_key", "nunique"),
        storm_keys=("storm_key", lambda x: ", ".join(sorted(map(str, x)))),
        storm_names=("storm_name_plot", lambda x: ", ".join(sorted(set(map(str, x)))))
    )
    .query("n_storms > 1")
)

print("\nMonths with multiple distinct storms:")
print(same_month_storms)


# Merge SVI group
wind_tx = wind_tx.merge(
    county_svi[["countyfips", "mean_svi_county", "svi_high"]],
    on="countyfips",
    how="left"
)

if wind_tx["svi_high"].isna().any():
    n_missing = wind_tx["svi_high"].isna().sum()
    print(f"Warning: {n_missing} wind rows have missing SVI group and will be dropped.")
    wind_tx = wind_tx.dropna(subset=["svi_high"]).copy()

wind_tx["svi_high"] = wind_tx["svi_high"].astype(int)
wind_tx["beta_group"] = wind_tx["svi_high"].map(beta_by_group)

print("\nTexas wind rows after filters:", len(wind_tx))
print("Texas counties in wind panel:", wind_tx["countyfips"].nunique())
print("Texas storms in wind panel:", wind_tx["storm_key"].nunique())

# =============================================================================
# 4. Panel a: monthly Texas wind incidence
# =============================================================================

if PANEL_A_AGG == "sum":
    panel_a = (
        wind_tx.groupby(["event_year", "event_month"], as_index=False)
        .agg(wind_incidence=("maxs", "sum"))
    )
    panel_a_ylabel = "Summed county-event maximum\nwind speed (m/s)"
elif PANEL_A_AGG == "mean":
    panel_a = (
        wind_tx.groupby(["event_year", "event_month"], as_index=False)
        .agg(wind_incidence=("maxs", "mean"))
    )
    panel_a_ylabel = "Mean exposed-county maximum\nwind speed (m s$^{-1}$)"
else:
    raise ValueError("PANEL_A_AGG must be either 'sum' or 'mean'.")

panel_a["date"] = pd.to_datetime(
    panel_a["event_year"].astype(str) + "-"
    + panel_a["event_month"].astype(str).str.zfill(2)
    + "-01"
)
panel_a["decade"] = panel_a["event_year"].apply(assign_decade)

# Display panel a over the mortality period only.
panel_a_plot = panel_a[
    (panel_a["event_year"] >= MORTALITY_START_YEAR) &
    (panel_a["event_year"] <= END_YEAR)
].copy()

# =============================================================================
# 5. Panel b: storm-specific stacked mortality responses
# =============================================================================

event_rows = wind_tx[[
    "countyfips", "storm_key", "storm_name_plot",
    "event_year", "event_month", "maxs", "svi_high", "beta_group"
]].copy()

# Expand each county-storm exposure over lags 0..172.
expanded = event_rows.merge(lag_weights, how="cross")

expanded["year"], expanded["month"] = month_add(
    expanded["event_year"],
    expanded["event_month"],
    expanded["lag"]
)

# Keep only mortality response months in the mortality sample.
# This allows pre-1968 storms to contribute if their response overlaps 1968 onward.
expanded = expanded[
    (expanded["year"] >= MORTALITY_START_YEAR) &
    (expanded["year"] <= END_YEAR)
].copy()

# Merge population at response month.
expanded = expanded.merge(
    pop,
    on=["countyfips", "year", "month"],
    how="left",
    validate="m:1"
)

missing_pop = expanded["totalpop"].isna().sum()
if missing_pop > 0:
    print(f"Warning: dropping {missing_pop:,} expanded rows with missing population.")
    expanded = expanded.dropna(subset=["totalpop"]).copy()

expanded["excess_deaths"] = (
    expanded["beta_group"]
    * expanded["nature_weight"]
    * expanded["maxs"]
    * expanded["totalpop"]
    / 100000.0
)

expanded["date"] = pd.to_datetime(
    expanded["year"].astype(str) + "-"
    + expanded["month"].astype(str).str.zfill(2)
    + "-01"
)
expanded["event_decade"] = expanded["event_year"].apply(assign_decade)

# Aggregate county contributions to storm-month response.
storm_month = (
    expanded.groupby(
        ["storm_key", "storm_name_plot", "event_year", "event_month", "event_decade", "date"],
        as_index=False
    )
    .agg(excess_deaths=("excess_deaths", "sum"))
)

storm_totals = (
    storm_month.groupby(
        ["storm_key", "storm_name_plot", "event_year", "event_month", "event_decade"],
        as_index=False
    )
    .agg(total_excess_deaths=("excess_deaths", "sum"))
)

if DROP_ZERO_OR_NEGATIVE_STORM_RESPONSES:
    keep_storms = storm_totals.loc[storm_totals["total_excess_deaths"] > 0, "storm_key"]
    storm_month = storm_month[storm_month["storm_key"].isin(keep_storms)].copy()
    storm_totals = storm_totals[storm_totals["storm_key"].isin(keep_storms)].copy()

# Sort storms chronologically for stacking.
storm_order = (
    storm_totals
    .sort_values(["event_year", "event_month", "storm_key"])
    ["storm_key"]
    .tolist()
)

all_dates = pd.date_range(
    f"{MORTALITY_START_YEAR}-01-01",
    f"{END_YEAR}-12-01",
    freq="MS"
)

stack_df = (
    storm_month.pivot_table(
        index="date",
        columns="storm_key",
        values="excess_deaths",
        aggfunc="sum",
        fill_value=0.0
    )
    .reindex(all_dates, fill_value=0.0)
)

storm_order = [s for s in storm_order if s in stack_df.columns]
stack_df = stack_df[storm_order]

total_burden = stack_df.sum(axis=1)

storm_meta = (
    storm_totals.drop_duplicates("storm_key")
    .set_index("storm_key")
    .loc[storm_order]
)

print("\n================ Storm and mortality-period check ================")
print(f"Storm exposure years retained: {STORM_START_YEAR}-{END_YEAR}")
print(f"Mortality response years plotted: {MORTALITY_START_YEAR}-{END_YEAR}")
print("Number of storms retained in exposure history:", wind_tx["storm_key"].nunique())
print(
    "Number of pre-1968 storms contributing to post-1968 mortality:",
    storm_totals.loc[storm_totals["event_year"] < MORTALITY_START_YEAR, "storm_key"].nunique()
)
print(
    "Number of 1968+ storms contributing to post-1968 mortality:",
    storm_totals.loc[storm_totals["event_year"] >= MORTALITY_START_YEAR, "storm_key"].nunique()
)

# =============================================================================
# 6. Optional panel c: Texas direct deaths
# =============================================================================

has_direct = texas_direct_deaths_file is not None and os.path.exists(texas_direct_deaths_file)

if has_direct:
    direct = pd.read_csv(texas_direct_deaths_file)
    direct_year_col = infer_col(direct, ["year"], label="direct deaths year")
    direct_month_col = infer_col(direct, ["month"], label="direct deaths month")
    direct_death_col = infer_col(
        direct,
        ["direct_deaths", "texas_direct_deaths", "deaths"],
        label="direct deaths"
    )

    direct = direct.copy()
    direct[direct_year_col] = direct[direct_year_col].astype(int)
    direct[direct_month_col] = direct[direct_month_col].astype(int)
    direct[direct_death_col] = pd.to_numeric(direct[direct_death_col], errors="coerce").fillna(0)

    direct_month = (
        direct.groupby([direct_year_col, direct_month_col], as_index=False)
        .agg(direct_deaths=(direct_death_col, "sum"))
    )
    direct_month["date"] = pd.to_datetime(
        direct_month[direct_year_col].astype(str) + "-"
        + direct_month[direct_month_col].astype(str).str.zfill(2)
        + "-01"
    )

    direct_month = direct_month[
        (direct_month["date"] >= pd.Timestamp(f"{MORTALITY_START_YEAR}-01-01")) &
        (direct_month["date"] <= pd.Timestamp(f"{END_YEAR}-12-01"))
    ].copy()
else:
    direct_month = None

# =============================================================================
# 7. Plot
# =============================================================================

decade_colors = {
    1930: "#1f77b4",
    1940: "#1f77b4",
    1950: "#e4572e",
    1960: "#f2b01e",
    1970: "#7b3294",
    1980: "#5bc0eb",
    1990: "#b2182b",
    2000: "#70ad47",
    2010: "#253494",
}

for d in sorted(set(panel_a_plot["decade"]).union(set(storm_meta["event_decade"]))):
    if d not in decade_colors:
        decade_colors[d] = "gray"

if has_direct:
    fig, axes = plt.subplots(
        3, 1,
        figsize=(14.5, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [1.1, 3.2, 0.8], "hspace": 0.08}
    )
    ax_a, ax_b, ax_c = axes
else:
    fig, axes = plt.subplots(
        2, 1,
        figsize=(14.5, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.1, 3.4], "hspace": 0.08}
    )
    ax_a, ax_b = axes
    ax_c = None

# -----------------------
# Panel a
# -----------------------
bar_colors = panel_a_plot["decade"].map(decade_colors)
ax_a.bar(
    panel_a_plot["date"],
    panel_a_plot["wind_incidence"],
    width=25,
    color=bar_colors,
    edgecolor="none"
)

ax_a.set_ylabel(panel_a_ylabel, fontsize=AXIS_LABEL_SIZE)
ax_a.text(
    -0.04, 1.16, "a",
    transform=ax_a.transAxes,
    fontsize=PANEL_LABEL_SIZE,
    fontweight="bold"
)
ax_a.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
ax_a.spines["top"].set_visible(False)
ax_a.spines["right"].set_visible(False)

# -----------------------
# Panel b
# -----------------------
x = stack_df.index
bottom = np.zeros(len(stack_df))

for storm_key in stack_df.columns:
    y = stack_df[storm_key].values
    decade = storm_meta.loc[storm_key, "event_decade"]
    color = decade_colors.get(decade, "gray")

    ax_b.fill_between(
        x,
        bottom,
        bottom + y,
        color=color,
        alpha=0.28,
        linewidth=0.35,
        edgecolor=color
    )
    bottom = bottom + y

ax_b.plot(x, total_burden.values, color="black", linewidth=1.2)

ax_b.set_ylabel("Estimated excess deaths per month", fontsize=AXIS_LABEL_SIZE)
ax_b.text(
    -0.04, -0.01, "b",
    transform=ax_b.transAxes,
    fontsize=PANEL_LABEL_SIZE,
    fontweight="bold",
    va="bottom",
    ha="left"
)
ax_b.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
ax_b.spines["top"].set_visible(False)
ax_b.spines["right"].set_visible(False)

# -----------------------
# Storm labels: top visible high-wind storms only
# -----------------------
visible_label_source = wind_tx[
    (wind_tx["event_year"] >= MORTALITY_START_YEAR) &
    (wind_tx["event_year"] <= END_YEAR)
].copy()

top_wind_storms = (
    visible_label_source.groupby(
        ["storm_key", "storm_name_plot", "event_year", "event_month"],
        as_index=False
    )
    .agg(
        storm_total_maxs=("maxs", "sum"),
        storm_max_county_maxs=("maxs", "max"),
        n_exposed_counties=("countyfips", "nunique")
    )
    .sort_values("storm_total_maxs", ascending=False)
    .head(top_label_n)
    .copy()
)

top_wind_storms["event_date"] = pd.to_datetime(
    top_wind_storms["event_year"].astype(int).astype(str)
    + "-"
    + top_wind_storms["event_month"].astype(int).astype(str).str.zfill(2)
    + "-01"
)

selected_labels = []

for _, row in top_wind_storms.sort_values("storm_total_maxs", ascending=False).iterrows():
    keep = True

    for kept in selected_labels:
        gap_months = abs(
            (row["event_date"].year - kept["event_date"].year) * 12
            + (row["event_date"].month - kept["event_date"].month)
        )

        if gap_months < min_label_gap_months:
            keep = False
            break

    if keep:
        selected_labels.append(row)

if selected_labels:
    top_wind_storms_labeled = pd.DataFrame(selected_labels).sort_values("event_date")
else:
    top_wind_storms_labeled = pd.DataFrame(columns=top_wind_storms.columns)

print("\nTop storms labeled after overlap filtering:")
if not top_wind_storms_labeled.empty:
    print(
        top_wind_storms_labeled[
            [
                "storm_name_plot",
                "storm_key",
                "event_year",
                "event_month",
                "storm_total_maxs",
                "storm_max_county_maxs",
                "n_exposed_counties",
            ]
        ]
    )
else:
    print("No labels selected.")

for _, row in top_wind_storms_labeled.iterrows():
    event_date = row["event_date"]
    label = str(row["storm_name_plot"]).title()

    ax_b.axvline(
        event_date,
        color="black",
        linestyle=":",
        linewidth=0.6,
        alpha=0.45
    )

    ax_b.text(
        event_date,
        ax_b.get_ylim()[1] * 0.92,
        label,
        rotation=90,
        fontsize=STORM_LABEL_SIZE,
        ha="center",
        va="top"
    )

print("\n================ Storm count reconciliation ================")

print("Positive-exposure storms in wind_tx:", wind_tx["storm_key"].nunique())
print("Storms in expanded after mortality-window filter:", expanded["storm_key"].nunique())
print("Storms in storm_totals:", storm_totals["storm_key"].nunique())
print("Storms stacked in stack_df:", stack_df.shape[1])

missing_from_expanded = sorted(
    set(wind_tx["storm_key"].unique()) - set(expanded["storm_key"].unique())
)

missing_from_stack = sorted(
    set(wind_tx["storm_key"].unique()) - set(stack_df.columns)
)

print("\nStorms in wind_tx but not in expanded:")
print(missing_from_expanded)

print("\nStorms in wind_tx but not stacked:")
print(missing_from_stack)

if missing_from_stack:
    print("\nDetails for missing storms:")
    print(
        wind_tx[wind_tx["storm_key"].isin(missing_from_stack)]
        [["storm_key", "storm_name_plot", "event_year", "event_month", "countyfips", "maxs"]]
        .drop_duplicates()
        .sort_values(["event_year", "event_month", "storm_key", "countyfips"])
    )
    
    
    
# -----------------------
# Panel c, optional
# -----------------------
if has_direct:
    ax_c.bar(
        direct_month["date"],
        direct_month["direct_deaths"],
        width=25,
        color="black",
        alpha=0.75
    )
    ax_c.set_ylabel("Direct deaths", fontsize=AXIS_LABEL_SIZE)
    ax_c.text(
        -0.065, 1.02, "c",
        transform=ax_c.transAxes,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold"
    )
    ax_c.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

# Shared x-axis
last_ax = ax_c if has_direct else ax_b
last_ax.set_xlabel("Year", fontsize=AXIS_LABEL_SIZE)
last_ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)

xticks = pd.to_datetime([
    f"{y}-01-01" for y in range(MORTALITY_START_YEAR, END_YEAR + 1, 10)
])
last_ax.set_xticks(xticks)
last_ax.set_xticklabels([str(y.year) for y in xticks])

# Legend by decade
legend_decades = sorted(set(panel_a_plot["decade"]).union(set(storm_meta["event_decade"])))
legend_handles = [
    Patch(facecolor=decade_colors[d], edgecolor="none", label=f"{d}s", alpha=0.7)
    for d in legend_decades
]

ax_a.legend(
    handles=legend_handles,
    loc="upper left",
    bbox_to_anchor=(1.005, 1.0),
    frameon=False,
    fontsize=LEGEND_SIZE,
    title="Event decade",
    title_fontsize=LEGEND_TITLE_SIZE
)

sample_note = "42-county estimation sample" if USE_REGRESSION_COUNTIES_ONLY else "all Texas counties in wind panel"

fig.suptitle(
    f"Estimated tropical-cyclone mortality burden in Texas ({sample_note})",
    fontsize=TITLE_SIZE,
    y=0.985
)

plt.tight_layout(rect=[0, 0, 0.88, 0.96])

plt.savefig(fig_path_png, dpi=300, bbox_inches="tight")
plt.savefig(fig_path_pdf, bbox_inches="tight")
plt.show()

print(f"\nSaved figure:\n  {fig_path_png}\n  {fig_path_pdf}")

# =============================================================================
# 8. Save plotting data for reproducibility
# =============================================================================

panel_a_plot.to_csv(
    os.path.join(fig_output_dir, "texas_fig3_panel_a_wind_incidence.csv"),
    index=False
)

storm_month.to_csv(
    os.path.join(fig_output_dir, "texas_fig3_panel_b_storm_month_excess_deaths.csv"),
    index=False
)

total_burden.rename("total_excess_deaths").to_frame().to_csv(
    os.path.join(fig_output_dir, "texas_fig3_panel_b_total_monthly_burden.csv")
)

storm_totals.to_csv(
    os.path.join(fig_output_dir, "texas_fig3_storm_total_excess_deaths.csv"),
    index=False
)

top_wind_storms_labeled.to_csv(
    os.path.join(fig_output_dir, "texas_fig3_labeled_storms.csv"),
    index=False
)

if has_direct:
    direct_month.to_csv(
        os.path.join(fig_output_dir, "texas_fig3_panel_c_direct_deaths.csv"),
        index=False
    )

print("\nSaved plotting data to:", fig_output_dir)

# =============================================================================
# 9. Basic diagnostics
# =============================================================================

print("\n================ Figure diagnostics ================")
print("Panel a max monthly wind incidence, displayed period:")
print(panel_a_plot.sort_values("wind_incidence", ascending=False).head(10))

print("\nTop storms by total estimated excess deaths over plotted mortality period:")
print(
    storm_totals.sort_values("total_excess_deaths", ascending=False)
    .head(20)
    [["storm_name_plot", "storm_key", "event_year", "event_month", "event_decade", "total_excess_deaths"]]
)

print("\nTotal estimated excess deaths over plotted mortality period:")
print(total_burden.sum())

print("\nAverage estimated excess deaths per month over plotted mortality period:")
print(total_burden.mean())

# =============================================================================
# 10. Population coverage of the 42-county estimation sample
# =============================================================================

full_texas_population_candidates = [
    os.path.join(output_dir, "full_data_before_filtering.dta"),
    os.path.join(output_dir, "full_data_all_texas_counties.dta"),
    os.path.join(output_dir, "full_data_for_regression_all_counties.dta"),
    os.path.join(data_dir, "DATA_hurricane_mortality_temp_month_county_19482018.dta"),
    os.path.join(data_dir, "inputs for stata", "DATA_hurricane_mortality_temp_month_county_19482018.dta"),
]

full_pop_file = None
for f in full_texas_population_candidates:
    if os.path.exists(f):
        full_pop_file = f
        break

if full_pop_file is None:
    print(
        "\nCould not find a full Texas county population panel. "
        "Skipping population coverage calculation."
    )
else:
    print(f"\nLoaded full Texas population panel from:\n{full_pop_file}")

    if full_pop_file.endswith(".dta"):
        pop_all = pd.read_stata(full_pop_file)
    else:
        pop_all = pd.read_csv(full_pop_file, low_memory=False)

    required_cols = ["countyfips", "year", "month", "totalpop"]
    missing = [c for c in required_cols if c not in pop_all.columns]
    if missing:
        raise KeyError(
            f"Full population file is missing columns: {missing}. "
            f"Available columns: {pop_all.columns.tolist()}"
        )

    pop_all = pop_all[required_cols].copy()
    pop_all["countyfips"] = pop_all["countyfips"].astype(str).str.zfill(5)
    pop_all["year"] = pop_all["year"].astype(int)
    pop_all["month"] = pop_all["month"].astype(int)
    pop_all["totalpop"] = pd.to_numeric(pop_all["totalpop"], errors="coerce")

    pop_all_tx = pop_all[pop_all["countyfips"].str.startswith("48")].copy()

    pop_all_tx = pop_all_tx[
        (pop_all_tx["year"] >= MORTALITY_START_YEAR) &
        (pop_all_tx["year"] <= END_YEAR)
    ].copy()

    pop_all_tx = (
        pop_all_tx
        .dropna(subset=["totalpop"])
        .drop_duplicates(["countyfips", "year", "month"])
        .copy()
    )

    pop_all_tx["in_estimation_sample"] = pop_all_tx["countyfips"].isin(regression_counties)
    pop_all_tx["sample_pop_component"] = np.where(
        pop_all_tx["in_estimation_sample"],
        pop_all_tx["totalpop"],
        0.0
    )

    monthly_pop_share = (
        pop_all_tx.groupby(["year", "month"], as_index=False)
        .agg(
            texas_total_pop=("totalpop", "sum"),
            sample_42_pop=("sample_pop_component", "sum"),
            n_texas_counties=("countyfips", "nunique"),
            n_sample_counties=("in_estimation_sample", "sum"),
        )
    )

    monthly_pop_share["sample_pop_share"] = (
        monthly_pop_share["sample_42_pop"] / monthly_pop_share["texas_total_pop"]
    )

    monthly_pop_share["date"] = pd.to_datetime(
        monthly_pop_share["year"].astype(str) + "-"
        + monthly_pop_share["month"].astype(str).str.zfill(2)
        + "-01"
    )

    annual_pop_share = (
        monthly_pop_share.groupby("year", as_index=False)
        .agg(
            texas_total_pop=("texas_total_pop", "mean"),
            sample_42_pop=("sample_42_pop", "mean"),
            sample_pop_share=("sample_pop_share", "mean"),
            n_texas_counties=("n_texas_counties", "max"),
            n_sample_counties=("n_sample_counties", "max"),
        )
    )

    overall_share = (
        monthly_pop_share["sample_42_pop"].mean()
        / monthly_pop_share["texas_total_pop"].mean()
    )

    print("\n================ Population coverage of 42-county sample ================")
    print(f"Period: {MORTALITY_START_YEAR}-{END_YEAR}")
    print(f"Number of estimation counties: {len(regression_counties)}")
    print(f"Average monthly Texas population: {monthly_pop_share['texas_total_pop'].mean():,.0f}")
    print(f"Average monthly 42-county population: {monthly_pop_share['sample_42_pop'].mean():,.0f}")
    print(f"Average population share covered by 42 counties: {overall_share:.3%}")

    print("\nPopulation share in first and last available years:")
    print(
        annual_pop_share[
            annual_pop_share["year"].isin([
                annual_pop_share["year"].min(),
                annual_pop_share["year"].max()
            ])
        ][["year", "texas_total_pop", "sample_42_pop", "sample_pop_share"]]
    )

    print("\nSelected annual shares:")
    print(
        annual_pop_share[
            annual_pop_share["year"].isin([1968, 1980, 1990, 2000, 2010, 2018])
        ][["year", "texas_total_pop", "sample_42_pop", "sample_pop_share"]]
    )

    monthly_pop_share.to_csv(
        os.path.join(fig_output_dir, "texas_42county_population_share_monthly.csv"),
        index=False
    )

    annual_pop_share.to_csv(
        os.path.join(fig_output_dir, "texas_42county_population_share_annual.csv"),
        index=False
    )