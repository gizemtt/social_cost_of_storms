# %% Texas TC mortality burden map: mortality rate and total deaths
# Creates a Texas map showing:
#   Panel a: estimated TC excess mortality burden per 100,000
#   Panel b: estimated TC excess deaths
#
# All Texas counties are shown for context.
# Only the 42 estimation-sample counties are colored.
#
# Output:
#  ...\output\figures\tx_mortality_map.png

import os
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path

# =============================================================================
# User settings
# =============================================================================

repo_dir = Path(__file__).resolve().parents[1]
data_dir = os.path.join(repo_dir, "data")
output_dir = os.path.join(repo_dir, "output")
beta_dir = os.path.join(output_dir, "beta")
fig_output_dir = os.path.join(output_dir, "figures")
os.makedirs(fig_output_dir, exist_ok=True)

storm_panel_file = os.path.join(
    data_dir,
    "panel_by_storm__NA_USA_density_8_yr_1930_2018.csv"
)

regression_data_file = os.path.join(output_dir, "full_data_for_regression.dta")

stata_panel_file = os.path.join(
    data_dir,
    "DATA_hurricane_mortality_temp_month_county_19482018.dta"
)

coef_file_dta = os.path.join(beta_dir, "coefficients_county_nature_future.dta")
coef_file_csv = os.path.join(beta_dir, "coefficients_region_poly.csv")

county_shapefile = os.path.join(
    data_dir,
    "county_boundaries",
    "tl_2018_us_county",
    "tl_2018_us_county.shp"
)

map_output_path = os.path.join(fig_output_dir, "tx_mortality_map.png")
map_output_pdf = os.path.join(fig_output_dir, "tx_mortality_map.pdf")
map_data_output = os.path.join(fig_output_dir, "tx_mortality_map_data.csv")

# =============================================================================
# Time windows and model settings
# =============================================================================

STORM_START_YEAR = 1953
MORTALITY_START_YEAR = 1968
END_YEAR = 2018
LMAX = 172

svi_cutoff = 0.75

# Observed total mortality rate variable for panel b.
# Usually tdths_ttpop = monthly all-cause deaths per 100,000.
# If your current reg file stores this rate in dths_tot, change this to "dths_tot".
TOTAL_MORTALITY_RATE_COL = "tdths_ttpop"

HIGHLIGHT_COUNTY = "48201"   # Harris County
HIGHLIGHT_EDGE = "dodgerblue"
HIGHLIGHT_LINEWIDTH = 2.6
HIGHLIGHT_LABEL = "Harris County"

# =============================================================================
# Font and map appearance settings
# =============================================================================

TITLE_SIZE = 17
PANEL_TITLE_SIZE = 15
PANEL_LABEL_SIZE = 16
LEGEND_TITLE_SIZE = 13
LEGEND_TICK_SIZE = 11

BASE_COUNTY_FACE = "#f2f2f2"
BASE_COUNTY_EDGE = "#d0d0d0"
SELECTED_EDGE = "black"
STATE_EDGE = "black"

RATE_CMAP = "OrRd"
DEATHS_CMAP = "PuRd"

# Color scale settings
# If True, each colorbar starts at the observed minimum among the 42 counties.
# If False, each colorbar starts at zero.
USE_OBSERVED_MIN_AS_VMIN = True

# =============================================================================
# Helper functions
# =============================================================================

def zfill5(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)


def nature_shape(x):
    return 0.0237 + 0.000535 * x - 0.0000039 * (x ** 2)


def make_nature_lag_weights(max_lag=172):
    lags = np.arange(0, max_lag + 1)
    raw = nature_shape(lags)

    if (raw < 0).any():
        raise ValueError(f"Nature lag shape is negative at lags: {lags[raw < 0]}")

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
            f"Found columns: {df.columns.tolist()}"
        )

    df = df.copy()
    df["parm_clean"] = df["parm"].apply(clean_stata_parm)
    print(f"Loaded coefficient file: {source}")
    return df


def extract_svi_betas(df_coef):
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
            "Expected 0.svi_high#c.X_nature and 1.svi_high#c.X_nature."
        )

    return beta


def infer_col(df, candidates, required=True, label="column"):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(
            f"Could not find {label}. Tried {candidates}. "
            f"Available columns: {df.columns.tolist()}"
        )
    return None


def month_add(year, month, lag):
    base = year.astype(int) * 12 + month.astype(int) - 1 + lag.astype(int)
    out_year = base // 12
    out_month = base % 12 + 1
    return out_year, out_month


def get_vmin_vmax(values, use_observed_min=True):
    values = pd.Series(values).dropna()
    if values.empty:
        raise ValueError("No values available for color scale.")

    vmin = float(values.min()) if use_observed_min else 0.0
    vmax = float(values.max())

    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0

    return vmin, vmax


def add_colorbar(fig, ax, cmap, vmin, vmax, label):
    sm = ScalarMappable(
        norm=Normalize(vmin=vmin, vmax=vmax),
        cmap=cmap
    )
    sm._A = []

    cbar = fig.colorbar(
        sm,
        ax=ax,
        orientation="vertical",
        fraction=0.046,
        pad=0.015
    )
    cbar.set_label(label, fontsize=LEGEND_TITLE_SIZE)
    cbar.ax.tick_params(labelsize=LEGEND_TICK_SIZE)
    return cbar


# =============================================================================
# 1. Load regression panel to define 42 counties, SVI groups, and population
# =============================================================================

if os.path.exists(regression_data_file):
    reg = pd.read_stata(regression_data_file)
    print(f"Loaded regression data: {regression_data_file}")
elif os.path.exists(stata_panel_file):
    reg = pd.read_stata(stata_panel_file)
    print(f"Loaded fallback Stata panel: {stata_panel_file}")
else:
    raise FileNotFoundError(
        "Could not find regression panel. Expected one of:\n"
        f"{regression_data_file}\n{stata_panel_file}"
    )

required_reg_cols = ["countyfips", "year", "month", "svi", TOTAL_MORTALITY_RATE_COL,"totalpop"]
missing_reg = [c for c in required_reg_cols if c not in reg.columns]
if missing_reg:
    raise KeyError(f"Regression panel missing columns: {missing_reg}")

reg = reg.copy()
reg["countyfips"] = zfill5(reg["countyfips"])
reg["year"] = reg["year"].astype(int)
reg["month"] = reg["month"].astype(int)
reg["totalpop"] = pd.to_numeric(reg["totalpop"], errors="coerce")
reg[TOTAL_MORTALITY_RATE_COL] = pd.to_numeric(
    reg[TOTAL_MORTALITY_RATE_COL],
    errors="coerce"
)

# Use only mortality-period population for converting rate burden to deaths
reg_mort = reg[
    (reg["year"] >= MORTALITY_START_YEAR) &
    (reg["year"] <= END_YEAR)
].copy()

county_svi = (
    reg_mort.groupby("countyfips", as_index=False)
    .agg(
        mean_svi_county=("svi", "mean"),
        total_mortality_per_100k=(TOTAL_MORTALITY_RATE_COL, "sum")
    )
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

estimation_counties = set(county_svi["countyfips"])

print("\n================ Estimation-sample counties ================")
print("Number of counties:", len(estimation_counties))
print(county_svi["svi_group"].value_counts())

# =============================================================================
# 2. Load betas and lag weights
# =============================================================================

coef = read_coefficients(coef_file_dta, coef_file_csv)
beta_by_group = extract_svi_betas(coef)
lag_weights = make_nature_lag_weights(LMAX)

print("\nSVI-group betas:")
print(f"Lower-SVI beta: {beta_by_group[0]:.6f}")
print(f"High-SVI beta:  {beta_by_group[1]:.6f}")

# =============================================================================
# 3. Load storm-county wind panel
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

if state_col is not None:
    wind[state_col] = wind[state_col].astype(str).str.zfill(2)
    wind_tx = wind[wind[state_col] == "48"].copy()
else:
    wind_tx = wind[wind[county_col].str.startswith("48")].copy()

wind_tx = wind_tx[
    (wind_tx[year_col] >= STORM_START_YEAR) &
    (wind_tx[year_col] <= END_YEAR)
].copy()

# Only 42 estimation counties receive mortality estimates
wind_tx = wind_tx[wind_tx[county_col].isin(estimation_counties)].copy()

# Only positive exposure events matter
wind_tx = wind_tx[wind_tx[maxs_col] > 0].copy()

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

wind_tx = wind_tx.merge(
    county_svi[["countyfips", "mean_svi_county", "svi_high", "svi_group"]],
    on="countyfips",
    how="left"
)

if wind_tx["svi_high"].isna().any():
    print("Dropping rows with missing SVI group:", wind_tx["svi_high"].isna().sum())
    wind_tx = wind_tx.dropna(subset=["svi_high"]).copy()

wind_tx["svi_high"] = wind_tx["svi_high"].astype(int)
wind_tx["beta_group"] = wind_tx["svi_high"].map(beta_by_group)

print("\n================ Wind data ================")
print("Wind rows for 42 counties:", len(wind_tx))
print("Counties with positive exposure:", wind_tx["countyfips"].nunique())
print("Storms with positive exposure:", wind_tx["storm_key"].nunique())

# =============================================================================
# 4. Compute county-level TC mortality burden
# =============================================================================

event_rows = wind_tx[[
    "countyfips", "storm_key", "storm_name_plot",
    "event_year", "event_month", "maxs", "svi_high", "svi_group", "beta_group"
]].copy()

expanded = event_rows.merge(lag_weights, how="cross")

expanded["response_year"], expanded["response_month"] = month_add(
    expanded["event_year"],
    expanded["event_month"],
    expanded["lag"]
)

# Keep only mortality response months in the mortality sample.
expanded = expanded[
    (expanded["response_year"] >= MORTALITY_START_YEAR) &
    (expanded["response_year"] <= END_YEAR)
].copy()

# Mortality-rate contribution, per 100,000.
expanded["tc_mortality_rate_contribution"] = (
    expanded["beta_group"]
    * expanded["nature_weight"]
    * expanded["maxs"]
)

county_burden = (
    expanded.groupby(["countyfips"], as_index=False)
    .agg(
        tc_mortality_per_100k=("tc_mortality_rate_contribution", "sum"),
        n_storms_contributing=("storm_key", "nunique"),
        max_county_event_wind=("maxs", "max")
    )
)

county_burden["tc_mortality_per_100k"] = county_burden["tc_mortality_per_100k"].fillna(0.0)
county_burden["n_storms_contributing"] = county_burden["n_storms_contributing"].fillna(0).astype(int)
county_burden["max_county_event_wind"] = county_burden["max_county_event_wind"].fillna(0.0)
# Ensure all 42 counties appear, even if no positive wind contribution exists.
county_burden = county_svi.merge(
    county_burden,
    on="countyfips",
    how="left"
)

print("\n================ County burden summary ================")
print(
    county_burden
    .groupby("svi_group")
    .agg(
        n_counties=("countyfips", "nunique"),
        mean_tc_burden=("tc_mortality_per_100k", "mean"),
        min_tc_burden=("tc_mortality_per_100k", "min"),
        max_tc_burden=("tc_mortality_per_100k", "max"),
        mean_total_mortality=("total_mortality_per_100k", "mean"),
        min_total_mortality=("total_mortality_per_100k", "min"),
        max_total_mortality=("total_mortality_per_100k", "max"),
        mean_svi=("mean_svi_county", "mean"),
    )
)

print("\nOverall across 42 counties:")
print(f" TC excess mortality burden per 100,000: {county_burden['tc_mortality_per_100k'].mean():,.2f}")
print(f" observed total mortality per 100,000: {county_burden['total_mortality_per_100k'].mean():,.2f}")

# =============================================================================
# 5. Load Texas county shapefile and merge burden data
# =============================================================================

tx_counties = gpd.read_file(county_shapefile)

tx_counties["STATEFP"] = tx_counties["STATEFP"].astype(str).str.zfill(2)
tx_counties["GEOID"] = tx_counties["GEOID"].astype(str).str.zfill(5)

tx_counties = tx_counties[tx_counties["STATEFP"] == "48"].copy()

try:
    tx_counties = tx_counties.to_crs(epsg=3083)  # Texas Centric Albers Equal Area
except Exception:
    print("Could not project to EPSG:3083; using original CRS.")

state_boundary = tx_counties.dissolve(by="STATEFP", as_index=False)

map_gdf = tx_counties.merge(
    county_burden,
    left_on="GEOID",
    right_on="countyfips",
    how="left"
)

map_gdf["in_estimation_sample"] = map_gdf["countyfips"].notna()
selected_gdf = map_gdf[map_gdf["in_estimation_sample"]].copy()

if selected_gdf["GEOID"].nunique() != len(estimation_counties):
    missing_from_shape = sorted(estimation_counties - set(selected_gdf["GEOID"]))
    print("Warning: estimation counties missing from shapefile:", missing_from_shape)

harris_gdf = map_gdf[map_gdf["GEOID"] == HIGHLIGHT_COUNTY].copy()

if harris_gdf.empty:
    print(f"Warning: highlight county {HIGHLIGHT_COUNTY} not found in shapefile.")
else:
    print("\n================ Highlight county ================")
    print(
        harris_gdf[
            ["GEOID", "NAME", "tc_mortality_per_100k", "total_mortality_per_100k"]
        ]
    )
# =============================================================================
# 6. Plot two-panel Texas map: rate burden and total deaths
# =============================================================================

rate_values = selected_gdf["tc_mortality_per_100k"].dropna()
total_mort_values = selected_gdf["total_mortality_per_100k"].dropna()

if rate_values.empty:
    raise ValueError("No mortality-rate burden values available to plot.")
if total_mort_values.empty:
    raise ValueError("No observed total mortality values available to plot.")

rate_vmin, rate_vmax = get_vmin_vmax(
    rate_values,
    use_observed_min=USE_OBSERVED_MIN_AS_VMIN
)
total_mort_vmin, total_mort_vmax = get_vmin_vmax(
    total_mort_values,
    use_observed_min=USE_OBSERVED_MIN_AS_VMIN
)

print("\n================ Map color scales ================")
print(f"Panel a rate burden scale: {rate_vmin:.2f} to {rate_vmax:.2f}")
print(f"Panel b total mortality scale: {total_mort_vmin:.2f} to {total_mort_vmax:.2f}")

fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

panel_specs = [
    {
        "title": "Cumulative estimated TC excess mortality",
        "column": "tc_mortality_per_100k",
        "cmap": RATE_CMAP,
        "vmin": rate_vmin,
        "vmax": rate_vmax,
        "colorbar_label": "Estimated TC excess mortality\n(per 100,000)",
        "panel_label": "a",
    },
    {
        "title": "Cumulative observed all-cause mortality",
        "column": "total_mortality_per_100k",
        "cmap": DEATHS_CMAP,
        "vmin": total_mort_vmin,
        "vmax": total_mort_vmax,
        "colorbar_label": "Observed total mortality\n(per 100,000)",
        "panel_label": "b",
    },
]

for ax, spec in zip(axes, panel_specs):
    # Base: all Texas counties
    tx_counties.plot(
        ax=ax,
        color=BASE_COUNTY_FACE,
        edgecolor=BASE_COUNTY_EDGE,
        linewidth=0.25,
        zorder=1
    )

    # Colored: all 42 estimation counties
    selected_gdf.plot(
        column=spec["column"],
        cmap=spec["cmap"],
        vmin=spec["vmin"],
        vmax=spec["vmax"],
        ax=ax,
        edgecolor=SELECTED_EDGE,
        linewidth=0.8,
        zorder=3
    )

    # Highlight Harris County
    if not harris_gdf.empty:
        harris_gdf.boundary.plot(
            ax=ax,
            color=HIGHLIGHT_EDGE,
            linewidth=HIGHLIGHT_LINEWIDTH,
            zorder=6
        )

        harris_point = harris_gdf.geometry.representative_point().iloc[0]

        ax.annotate(
            "Harris County",
            xy=(harris_point.x, harris_point.y),
            xytext=(-55, 24),
            textcoords="offset points",
            fontsize=10.5,
            fontweight="bold",
            color=HIGHLIGHT_EDGE,
            ha="right",
            va="bottom",
            zorder=8,
            clip_on=False,
            arrowprops=dict(
                arrowstyle="-",
                color=HIGHLIGHT_EDGE,
                linewidth=1.2,
                shrinkA=0,
                shrinkB=2
            ),
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                edgecolor=HIGHLIGHT_EDGE,
                linewidth=0.8,
                alpha=0.90
            )
        )

    # Texas state boundary
    state_boundary.boundary.plot(
        ax=ax,
        color=STATE_EDGE,
        linewidth=1.2,
        zorder=5
    )

    # Texas state boundary
    state_boundary.boundary.plot(
        ax=ax,
        color=STATE_EDGE,
        linewidth=1.2,
        zorder=5
    )

    ax.set_title(spec["title"], fontsize=PANEL_TITLE_SIZE)

    ax.text(
        0.01,
        0.98,
        spec["panel_label"],
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        va="top",
        ha="left"
    )

    ax.axis("off")

    add_colorbar(
        fig=fig,
        ax=ax,
        cmap=spec["cmap"],
        vmin=spec["vmin"],
        vmax=spec["vmax"],
        label=spec["colorbar_label"]
    )

fig.suptitle(
    "Cumulative mortality burden across Texas counties, 1968–2018",
    fontsize=TITLE_SIZE,
    y=0.98
)

plt.tight_layout(rect=[0, 0.02, 0.96, 0.94])

plt.savefig(map_output_path, dpi=300, bbox_inches="tight")
plt.savefig(map_output_pdf, bbox_inches="tight")
plt.show()

print("\nSaved map:")
print(map_output_path)
print(map_output_pdf)