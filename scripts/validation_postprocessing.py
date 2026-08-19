"""
Validation / postprocessing for the SVI-group X_nature + compact future-placebo model.

This script matches the Stata model:

    reghdfe tdths_ttpop \
        ibn.svi_high#c.X_nature \
        ibn.svi_high#c.X_future72 \
        [aw=totalpop], \
        absorb(...) vce(cluster countyfips) resid

Important:
    - Coefficients are SVI-group-specific, not county-specific.
    - Expected coefficient names after cleaning are:
          0.svi_high#c.X_nature
          1.svi_high#c.X_nature
          0.svi_high#c.X_future72
          1.svi_high#c.X_future72
    - Stata may export base levels as 0bn.svi_high#c.X_nature. This script
      canonicalizes those names in both the coefficient table and VCV matrix.
    - Stata prep code sets FMAX = 12. Therefore X_future72 is
      currently a compact 1- to 12-month future placebo unless you change FMAX
      to 72 in Stata.
"""

# =============================================================================
# Imports
# =============================================================================

import os
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import chi2
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.graphics.tsaplots import plot_acf
from numpy.linalg import inv, pinv, eigvalsh, LinAlgError

warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# User settings
# =============================================================================

base_dir = r"C:\Users\toplu\Documents\GithubRepositories\social_cost_of_storms"
output_dir = os.path.join(base_dir, "output")
beta_dir = os.path.join(output_dir, "beta")

# Stata model settings
group_var = "svi_high"
cluster_var = "countyfips"
svi_cutoff = 0.75

LMAX = 172
# Match your posted Stata code. If you change Stata FMAX to 72, also change this.
FMAX = 12

# Files exported by Stata
coef_file_dta = os.path.join(beta_dir, "coefficients_county_nature_future.dta")
coef_file_csv = os.path.join(beta_dir, "coefficients_region_poly.csv")
vcv_file_csv = os.path.join(beta_dir, "vcv_region_poly.csv")
residuals_file = os.path.join(beta_dir, "residuals_fitted.csv")
regression_data_file = os.path.join(output_dir, "full_data_for_regression.dta")

# Diagnostics
plot_each_county_ACF = False
event_wind_threshold = 25
event_window = 24
event_min_gap_months = 12


# =============================================================================
# Helper functions
# =============================================================================

def nature_shape(x):
    return 0.0237 + 0.000535 * x - 0.0000039 * (x ** 2)


def make_nature_lag_weights(max_lag=172):
    lags = np.arange(0, max_lag + 1)
    raw = nature_shape(lags)
    if (raw < 0).any():
        raise ValueError(f"Nature lag shape is negative for lags: {lags[raw < 0]}")
    return pd.DataFrame({
        "lag": lags,
        "nature_shape_raw": raw,
        "nature_weight": raw / raw.sum()
    })


def make_future_placebo_weights(fmax=12):
    leads = np.arange(1, fmax + 1)
    raw = nature_shape(leads)
    if (raw < 0).any():
        raise ValueError(f"Future placebo shape is negative for leads: {leads[raw < 0]}")
    return pd.DataFrame({
        "lead": leads,
        "lag": -leads,
        "future_shape_raw": raw,
        "future_weight": raw / raw.sum()
    })


def clean_stata_parm(parm):
    """
    Canonicalize Stata factor-variable coefficient names.

    Examples:
        0bn.svi_high#c.X_nature -> 0.svi_high#c.X_nature
        0b.svi_high#c.X_nature  -> 0.svi_high#c.X_nature
        0o.svi_high#c.X_nature  -> 0.svi_high#c.X_nature
        1.svi_high#c.X_nature   -> 1.svi_high#c.X_nature
    """
    parm = str(parm).strip()

    # Stata may mark base/omitted factor levels as b., o., 0b., 0bn., etc.
    parm = parm.replace("b.", ".").replace("o.", ".")
    parm = re.sub(r"^(\d+)[a-z]+\.", r"\1.", parm)
    return parm


def read_coefficients(dta_path, csv_path):
    """
    Prefer the .dta file saved by parmest. Fall back to the CSV exported by Stata.
    Creates both parm_raw and parm_clean, and also sets parm = parm_clean for
    compatibility with downstream code.
    """
    if os.path.exists(dta_path):
        df = pd.read_stata(dta_path)
        source = dta_path
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        source = csv_path
    else:
        raise FileNotFoundError(
            "No coefficient file found. Expected one of:\n"
            f"  {dta_path}\n"
            f"  {csv_path}"
        )

    required = {"parm", "estimate"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Coefficient file is missing {missing}. Columns: {df.columns.tolist()}")

    df = df.copy()
    df["parm_raw"] = df["parm"].astype(str)
    df["parm_clean"] = df["parm_raw"].apply(clean_stata_parm)
    df["parm"] = df["parm_clean"]

    print(f"Loaded coefficient file: {source}")
    return df


def parse_svi_group_parm(parm, group_name="svi_high"):
    """
    Parses Stata coefficient names from:

        0.svi_high#c.X_nature
        1.svi_high#c.X_nature
        0.svi_high#c.X_future72
        1.svi_high#c.X_future72

    Returns:
        group_id, term
        term in {"nature", "future"}
    """
    p = clean_stata_parm(parm)

    m = re.match(fr"([0-9]+)\.{group_name}#c\.X_nature$", p)
    if m:
        return int(m.group(1)), "nature"

    m = re.match(fr"([0-9]+)\.{group_name}#c\.X_future72$", p)
    if m:
        return int(m.group(1)), "future"

    return None, None


def read_vcv(path):
    if not os.path.exists(path):
        print(f"VCV file not found, skipping Wald tests: {path}")
        return None

    vcv = pd.read_csv(path, index_col=0)

    # Make VCV names match df_coef["parm_clean"].
    vcv.index = [clean_stata_parm(x) for x in vcv.index.astype(str)]
    vcv.columns = [clean_stata_parm(x) for x in vcv.columns.astype(str)]

    if vcv.index.duplicated().any() or pd.Index(vcv.columns).duplicated().any():
        dup_rows = vcv.index[vcv.index.duplicated()].tolist()
        dup_cols = pd.Index(vcv.columns)[pd.Index(vcv.columns).duplicated()].tolist()
        raise ValueError(
            "Duplicate VCV parameter names after cleaning. "
            f"Duplicated rows: {dup_rows}; duplicated columns: {dup_cols}"
        )

    print(f"Loaded VCV file: {path}")
    return vcv


def get_param_vcov(vcv, parms):
    if vcv is None:
        raise ValueError("VCV is None. Cannot compute standard errors or Wald tests.")

    parms = [clean_stata_parm(p) for p in parms]
    missing = [p for p in parms if p not in vcv.index or p not in vcv.columns]
    if missing:
        raise KeyError(
            f"These parameters are missing from VCV: {missing}\n"
            f"Available VCV parameters: {list(vcv.index)}"
        )

    V = vcv.loc[parms, parms].values
    return (V + V.T) / 2


def wald_zero(beta, V):
    beta = np.asarray(beta).reshape(-1, 1)
    V = (V + V.T) / 2

    evals = eigvalsh(V)
    if evals.min() < 0:
        eigvals, eigvecs = np.linalg.eigh(V)
        eigvals = np.clip(eigvals, a_min=0, a_max=None)
        V = eigvecs @ np.diag(eigvals) @ eigvecs.T

    try:
        Vinv = inv(V)
    except LinAlgError:
        Vinv = pinv(V)

    W = (beta.T @ Vinv @ beta).item()
    W = max(W, 0.0)
    df = len(beta)
    p = 1 - chi2.cdf(W, df)
    return W, df, p


def lincom(df_coef, vcv, test_name, weights, df_resid=None):
    """
    Linear combination test:
        H0: w'beta = 0

    Adds 95% confidence interval. Uses a t critical value if df_resid is
    available; otherwise uses the normal approximation.
    """
    if vcv is None:
        raise ValueError("VCV is required for lincom tests, but vcv is None.")

    df_coef = df_coef.copy()
    if "parm_clean" not in df_coef.columns:
        if "parm" not in df_coef.columns:
            raise KeyError("df_coef must contain either parm_clean or parm.")
        df_coef["parm_clean"] = df_coef["parm"].apply(clean_stata_parm)

    parms = [clean_stata_parm(p) for p in weights.keys()]
    weights_clean = {clean_stata_parm(p): w for p, w in weights.items()}
    w = np.array([weights_clean[p] for p in parms]).reshape(1, -1)

    coef_indexed = df_coef.set_index("parm_clean")
    missing_coef = [p for p in parms if p not in coef_indexed.index]
    if missing_coef:
        raise KeyError(
            f"These parameters are missing from coefficient table: {missing_coef}\n"
            f"Available coefficient parameters: {coef_indexed.index.tolist()}"
        )

    b = coef_indexed.loc[parms, "estimate"].values.reshape(-1, 1)
    V = get_param_vcov(vcv, parms)

    est = (w @ b).item()
    var = (w @ V @ w.T).item()
    se = np.sqrt(max(var, 0.0))
    t_stat = est / se if se > 0 else np.nan

    if df_resid is not None and not pd.isna(df_resid):
        p_two = 2 * stats.t.sf(abs(t_stat), df_resid)
        p_one_gt0 = stats.t.sf(t_stat, df_resid)
        crit = stats.t.ppf(0.975, df_resid)
    else:
        p_two = 2 * stats.norm.sf(abs(t_stat))
        p_one_gt0 = stats.norm.sf(t_stat)
        crit = stats.norm.ppf(0.975)

    ci_low = est - crit * se
    ci_high = est + crit * se

    return {
        "test": test_name,
        "estimate": est,
        "se": se,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "t": t_stat,
        "p_two_sided": p_two,
        "p_one_sided_gt0": p_one_gt0,
        "df_resid": df_resid
    }


def standardize_residuals(res_df):
    res_df = res_df.copy()
    res_df["log_resid"] = np.sign(res_df["resid"]) * np.log1p(np.abs(res_df["resid"]))
    res_df["std_resid"] = (res_df["resid"] - res_df["resid"].mean()) / res_df["resid"].std()
    res_df["outlier"] = np.abs(res_df["std_resid"]) > 3
    return res_df


def safe_to_stata(df, path):
    """Write to Stata if possible; otherwise print a warning and continue."""
    try:
        df.to_stata(path, write_index=False)
    except Exception as e:
        print(f"Warning: could not write Stata file {path}: {e}")
        
# =============================================================================
# Residual diagnostic plot output directory
# =============================================================================

resid_plot_dir = r"C:\Users\toplu\Documents\GithubRepositories\social_cost_of_storms\output\actual\resid_plots"
os.makedirs(resid_plot_dir, exist_ok=True)

def save_current_fig(filename):
    path = os.path.join(resid_plot_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {path}")


# Plot formatting for exported diagnostic figures
AXIS_LABEL_SIZE = 14
TICK_LABEL_SIZE = 12
PANEL_TITLE_SIZE = 15
SINGLE_FIGSIZE = (7, 5)


def format_axis_text(ax, xlabel=None, ylabel=None):
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_SIZE)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)


# =============================================================================
# 1. Load coefficients and construct SVI-group lag-response curves
# =============================================================================

df_coef = read_coefficients(coef_file_dta, coef_file_csv)
df_coef = df_coef[~df_coef["parm_clean"].astype(str).str.contains("_cons", na=False)].copy()

df_coef[[group_var, "term"]] = df_coef["parm_clean"].apply(
    lambda x: pd.Series(parse_svi_group_parm(x, group_var))
)

df_svi = df_coef[df_coef[group_var].notna()].copy()
df_svi[group_var] = df_svi[group_var].astype(int)

df_actual = df_svi[df_svi["term"] == "nature"].copy()
df_future = df_svi[df_svi["term"] == "future"].copy()

if df_actual.empty:
    print("Available coefficient names:")
    print(df_coef["parm_clean"].head(50).to_string(index=False))
    raise ValueError(
        "No SVI-group X_nature coefficients found. Expected names like "
        "'0.svi_high#c.X_nature' and '1.svi_high#c.X_nature'."
    )

dup_actual = df_actual.groupby(group_var).size()
if (dup_actual > 1).any():
    raise ValueError(f"Duplicate actual coefficients for groups: {dup_actual[dup_actual > 1]}")

df_actual = df_actual.rename(columns={"estimate": "beta_nature"})
df_future = df_future.rename(columns={"estimate": "beta_future"})

lag_weights = make_nature_lag_weights(LMAX)
future_weights = make_future_placebo_weights(FMAX)

df_lag_beta = (
    df_actual[[group_var, "parm_clean", "beta_nature"]]
    .merge(lag_weights, how="cross")
)
df_lag_beta["beta"] = df_lag_beta["beta_nature"] * df_lag_beta["nature_weight"]
df_lag_beta["type"] = "actual_nature"

df_cum_beta = (
    df_lag_beta
    .groupby(group_var, as_index=False)
    .agg(
        cumulative_beta=("beta", "sum"),
        beta_nature=("beta_nature", "first"),
        mean_beta=("beta", "mean"),
        min_beta=("beta", "min"),
        max_beta=("beta", "max"),
        beta_lag0=("beta", "first")
    )
)

idx_peak = df_lag_beta.groupby(group_var)["beta"].idxmax()
df_peak = df_lag_beta.loc[idx_peak, [group_var, "lag", "beta"]].rename(
    columns={"lag": "lag_peak", "beta": "beta_peak"}
)
df_cum_beta = df_cum_beta.merge(df_peak, on=group_var, how="left")
df_cum_beta["cum_minus_beta_nature"] = df_cum_beta["cumulative_beta"] - df_cum_beta["beta_nature"]

print("\n================ SVI-group actual Nature coefficients ================")
print(df_cum_beta)
print(
    "Max absolute difference cumulative_beta - beta_nature:",
    df_cum_beta["cum_minus_beta_nature"].abs().max()
)

# Future placebo distributed over negative lags for plotting only.
# The actual Stata coefficient is beta_future on compact X_future72.
if not df_future.empty:
    df_future_lag_beta = (
        df_future[[group_var, "parm_clean", "beta_future"]]
        .merge(future_weights, how="cross")
    )
    df_future_lag_beta["beta"] = df_future_lag_beta["beta_future"] * df_future_lag_beta["future_weight"]
    df_future_lag_beta["type"] = "future_placebo"
else:
    df_future_lag_beta = pd.DataFrame(
        columns=[group_var, "parm_clean", "beta_future", "lead", "lag", "beta", "type"]
    )
    print("No SVI-group X_future72 coefficients found.")


# =============================================================================
# 2. Save coefficient outputs
# =============================================================================

os.makedirs(output_dir, exist_ok=True)

df_lag_beta.to_csv(os.path.join(output_dir, "beta_lag_nature_shape_by_svi.csv"), index=False)
df_cum_beta.to_csv(os.path.join(output_dir, "beta_cumulative_nature_shape_by_svi.csv"), index=False)
safe_to_stata(df_lag_beta, os.path.join(output_dir, "beta_lag_nature_shape_by_svi.dta"))
safe_to_stata(df_cum_beta, os.path.join(output_dir, "beta_cumulative_nature_shape_by_svi.dta"))

df_plot = pd.concat(
    [
        df_lag_beta[[group_var, "lag", "beta", "type"]],
        df_future_lag_beta[[group_var, "lag", "beta", "type"]]
    ],
    ignore_index=True
).sort_values([group_var, "lag"])

df_plot.to_csv(os.path.join(output_dir, "beta_with_lags_plot_data_svi.csv"), index=False)


# =============================================================================
# 3. Plot SVI-group lag-response curves
# =============================================================================

label_map = {0: "Lower-SVI", 1: "High-SVI"}

plt.figure(figsize=(8, 5))
for g, temp in df_lag_beta.groupby(group_var):
    plt.plot(
        temp["lag"],
        temp["beta"],
        linewidth=2,
        label=f"{label_map.get(g, g)}: beta_nature={temp['beta_nature'].iloc[0]:.4g}"
    )
plt.axhline(0, linestyle="--", linewidth=1)
plt.xlabel("Lag after TC exposure (months)")
plt.ylabel("Lag-specific beta")
plt.title("Nature-shaped TC mortality response by SVI group")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
for g, temp in df_plot.groupby(group_var):
    actual = temp[temp["type"] == "actual_nature"]
    placebo = temp[temp["type"] == "future_placebo"]
    if not actual.empty:
        plt.plot(actual["lag"], actual["beta"], linewidth=2, label=f"{label_map.get(g, g)} actual")
    if not placebo.empty:
        plt.plot(placebo["lag"], placebo["beta"], linestyle="--", linewidth=2, label=f"{label_map.get(g, g)} future placebo")
plt.axhline(0, linestyle="--", linewidth=1)
plt.axvline(0, linestyle="--", linewidth=1)
plt.xlabel("Lag relative to TC exposure month")
plt.ylabel("Lag-specific beta")
plt.title("Actual Nature response and compact future-placebo response by SVI group")
plt.legend()
plt.tight_layout()
plt.show()


# =============================================================================
# 4. Wald / lincom-style tests from exported VCV
# =============================================================================

vcv = read_vcv(vcv_file_csv)

test_summary = pd.DataFrame()
wald_summary = pd.DataFrame()

if vcv is not None:
    test_rows = []

    # Individual actual coefficients and group contrast
    for _, row in df_actual.iterrows():
        p = row["parm_clean"]
        test_rows.append(
            lincom(df_coef, vcv, f"actual beta for svi_high={int(row[group_var])}", {p: 1.0})
        )

    if len(df_actual) == 2:
        p_low = df_actual.loc[df_actual[group_var] == 0, "parm_clean"].iloc[0] if (df_actual[group_var] == 0).any() else None
        p_high = df_actual.loc[df_actual[group_var] == 1, "parm_clean"].iloc[0] if (df_actual[group_var] == 1).any() else None
        if p_low is not None and p_high is not None:
            test_rows.append(
                lincom(df_coef, vcv, "actual high SVI - lower SVI", {p_high: 1.0, p_low: -1.0})
            )

    # Individual future-placebo coefficients and group contrast
    for _, row in df_future.iterrows():
        p = row["parm_clean"]
        test_rows.append(
            lincom(df_coef, vcv, f"future placebo beta for svi_high={int(row[group_var])}", {p: 1.0})
        )

    if len(df_future) == 2:
        p_low = df_future.loc[df_future[group_var] == 0, "parm_clean"].iloc[0] if (df_future[group_var] == 0).any() else None
        p_high = df_future.loc[df_future[group_var] == 1, "parm_clean"].iloc[0] if (df_future[group_var] == 1).any() else None
        if p_low is not None and p_high is not None:
            test_rows.append(
                lincom(df_coef, vcv, "future placebo high SVI - lower SVI", {p_high: 1.0, p_low: -1.0})
            )

    test_summary = pd.DataFrame(test_rows)
    print("\n================ Lincom-style coefficient tests ================")
    print(test_summary)
    test_summary.to_csv(os.path.join(output_dir, "svi_group_coefficient_tests.csv"), index=False)

    # Joint Wald tests for actual and placebo coefficients
    wald_rows = []
    if not df_actual.empty:
        parms = df_actual["parm_clean"].tolist()
        beta = df_actual["beta_nature"].values
        W, df_w, p = wald_zero(beta, get_param_vcov(vcv, parms))
        wald_rows.append({"test": "joint actual X_nature coefficients = 0", "W": W, "df": df_w, "p_value": p})

    if not df_future.empty:
        parms = df_future["parm_clean"].tolist()
        beta = df_future["beta_future"].values
        W, df_w, p = wald_zero(beta, get_param_vcov(vcv, parms))
        wald_rows.append({"test": "joint future X_future72 coefficients = 0", "W": W, "df": df_w, "p_value": p})

    wald_summary = pd.DataFrame(wald_rows)
    print("\n================ Joint Wald tests ================")
    print(wald_summary)
    if not wald_summary.empty:
        wald_summary.to_csv(os.path.join(output_dir, "svi_group_joint_wald_tests.csv"), index=False)
else:
    print("Skipping lincom and joint Wald tests because the VCV file was not found.")


# =============================================================================
# 5. Load residuals and merge with regression data
# =============================================================================

if not os.path.exists(residuals_file):
    raise FileNotFoundError(
        f"Residuals file not found:\n{residuals_file}\n\n"
        "Make sure Stata exported output/beta/residuals_fitted.csv."
    )

res_df = pd.read_csv(residuals_file)
res_df = res_df.dropna(subset=["resid", "fitted"]).copy()
res_df = standardize_residuals(res_df)

if not os.path.exists(regression_data_file):
    raise FileNotFoundError(f"Regression data file not found:\n{regression_data_file}")

regression_df = pd.read_stata(regression_data_file)

needed_cols = [
    "countyfips", "year", "month", "stfips", "state",
    "tavg", "maxs", "totalpop", "dths_tot", "tdths_ttpop",
    "svi", "X_nature", "X_future72"
]
keep_cols = [c for c in needed_cols if c in regression_df.columns]
missing = [c for c in needed_cols if c not in regression_df.columns]
if missing:
    print("Warning: missing expected columns from regression data:", missing)

regression_df = regression_df[keep_cols].copy()

for df in [res_df, regression_df]:
    df["countyfips"] = df["countyfips"].astype(int)
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)

res_df = res_df.merge(
    regression_df,
    on=["countyfips", "year", "month"],
    how="left",
    validate="m:1"
)

# Recreate svi_high exactly as in Stata if svi is available.
if "svi" in res_df.columns:
    county_svi = (
        res_df.groupby("countyfips", as_index=False)
        .agg(mean_svi_county=("svi", "mean"))
    )
    county_svi[group_var] = np.where(
        county_svi["mean_svi_county"].isna(),
        np.nan,
        np.where(county_svi["mean_svi_county"] > svi_cutoff, 1, 0)
    )
    res_df = res_df.drop(columns=[group_var, "mean_svi_county"], errors="ignore")
    res_df = res_df.merge(
        county_svi[["countyfips", "mean_svi_county", group_var]],
        on="countyfips",
        how="left"
    )
else:
    print("SVI column not found; residual diagnostics cannot be stratified by svi_high.")

res_df["date"] = pd.to_datetime(
    res_df["year"].astype(str) + "-" + res_df["month"].astype(str) + "-01"
)
res_df = res_df.sort_values(["countyfips", "date"]).copy()

print("\n================ Merge check ================")
print("Rows after merge:", len(res_df))
for c in ["X_nature", "X_future72", "maxs", "svi", group_var]:
    print(f"Missing {c}:", res_df[c].isna().sum() if c in res_df.columns else "column missing")
print("Number of counties:", res_df["countyfips"].nunique())
if group_var in res_df.columns:
    print("County counts by SVI group:")
    print(res_df.drop_duplicates("countyfips")[group_var].value_counts(dropna=False).sort_index())


# =============================================================================
# 6. Residual diagnostics: exposure, fitted values, seasonality
# =============================================================================

print("\n================ Residual summary ================")
print(res_df[["resid", "std_resid", "fitted"]].describe())
print("Number of residual observations:", len(res_df))
print("Number of |standardized residual| > 3:", int(res_df["outlier"].sum()))

# Residuals vs X_nature
if "X_nature" in res_df.columns:
    tmp = res_df.dropna(subset=["X_nature", "std_resid"]).copy()

    plt.figure(figsize=(7, 5))
    if group_var in tmp.columns:
        for g, gg in tmp.groupby(group_var, dropna=False):
            plt.scatter(gg["X_nature"], gg["std_resid"], alpha=0.35, s=12, label=label_map.get(g, str(g)))
        plt.legend(title=group_var)
    else:
        plt.scatter(tmp["X_nature"], tmp["std_resid"], alpha=0.35, s=12)
    plt.axhline(0, linestyle="--", linewidth=1)
    for k in [-3, -2, -1, 1, 2, 3]:
        plt.axhline(k, linestyle="--", linewidth=0.7)
    plt.xlabel("X_nature")
    plt.ylabel("Standardized residual")
    plt.title("Standardized residuals vs X_nature")
    plt.tight_layout()
    plt.show()

    # Binned residuals by group
    binned_frames = []
    if group_var in tmp.columns:
        iterable = tmp.groupby(group_var, dropna=False)
    else:
        iterable = [(None, tmp)]

    for key, gg in iterable:
        gg = gg.copy()
        gg["xbin"] = pd.qcut(gg["X_nature"], q=20, duplicates="drop")
        b = (
            gg.groupby("xbin", observed=False)
            .agg(
                x_mid=("X_nature", "mean"),
                mean_resid=("resid", "mean"),
                sd_resid=("resid", "std"),
                n=("resid", "size")
            )
            .reset_index()
        )
        b["se"] = b["sd_resid"] / np.sqrt(b["n"])
        if group_var in tmp.columns:
            b[group_var] = key[0] if isinstance(key, tuple) else key
        binned_frames.append(b)

    binned_x = pd.concat(binned_frames, ignore_index=True)

    plt.figure(figsize=(7, 5))
    if group_var in binned_x.columns:
        for g, gg in binned_x.groupby(group_var, dropna=False):
            plt.errorbar(
                gg["x_mid"], gg["mean_resid"], yerr=1.96 * gg["se"],
                fmt="o-", capsize=3, label=label_map.get(g, str(g))
            )
        plt.legend(title=group_var)
    else:
        plt.errorbar(
            binned_x["x_mid"], binned_x["mean_resid"],
            yerr=1.96 * binned_x["se"], fmt="o-", capsize=3
        )

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Binned X_nature")
    plt.ylabel("Mean residual")
    plt.title("Binned mean residuals vs X_nature")
    plt.tight_layout()
    plt.show()

    print("\n================ Binned residuals vs X_nature ================")
    print(binned_x[[c for c in [group_var, "x_mid", "mean_resid", "se", "n"] if c in binned_x.columns]].head(40))

# Residuals vs fitted
fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE)
ax.scatter(res_df["fitted"], res_df["std_resid"], alpha=0.35, s=12)
ax.axhline(0, linestyle="--", linewidth=1)
for k in [-3, -2, -1, 1, 2, 3]:
    ax.axhline(k, linestyle="--", linewidth=0.7)
format_axis_text(ax, xlabel="Fitted values", ylabel="Standardized residual")
fig.tight_layout()
save_current_fig("standardized_residuals_vs_fitted.png")
plt.show()

# Residual variance by fitted bin
res_df_filtered = res_df.dropna(subset=["fitted", "resid"]).copy()
created_fitted_bin = False
for n_bins in reversed(range(3, 21)):
    try:
        res_df_filtered["fitted_bin"] = pd.qcut(res_df_filtered["fitted"], q=n_bins, duplicates="drop")
        created_fitted_bin = True
        break
    except ValueError:
        continue

if created_fitted_bin:
    std_by_bin = (
        res_df_filtered.groupby("fitted_bin", observed=False)
        .agg(fitted_mid=("fitted", "mean"), resid_sd=("resid", "std"), n=("resid", "size"))
        .reset_index()
    )
else:
    std_by_bin = pd.DataFrame(columns=["fitted_mid", "resid_sd", "n"])
    print("Could not create fitted-value bins; skipping residual-variance-by-fitted plot.")

if not std_by_bin.empty:
    plt.figure(figsize=(7, 5))
    plt.plot(std_by_bin["fitted_mid"], std_by_bin["resid_sd"], marker="o")
    plt.axhline(res_df_filtered["resid"].std(), linestyle="--", label="Overall residual SD")
    plt.xlabel("Fitted values, bin mean")
    plt.ylabel("Residual standard deviation")
    plt.title("Residual variance vs fitted values")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Seasonality
season_group_cols = [group_var, "month"] if group_var in res_df.columns else ["month"]
season_df = (
    res_df.groupby(season_group_cols, as_index=False)
    .agg(mean_resid=("resid", "mean"), sd_resid=("resid", "std"), n=("resid", "size"))
)
season_df["se"] = season_df["sd_resid"] / np.sqrt(season_df["n"])

plt.figure(figsize=(7, 5))
if group_var in season_df.columns:
    for g, gg in season_df.groupby(group_var, dropna=False):
        plt.errorbar(
            gg["month"], gg["mean_resid"], yerr=1.96 * gg["se"],
            fmt="o-", capsize=3, label=label_map.get(g, str(g))
        )
    plt.legend(title=group_var)
else:
    plt.errorbar(season_df["month"], season_df["mean_resid"], yerr=1.96 * season_df["se"], fmt="o-", capsize=3)
plt.axhline(0, linestyle="--", linewidth=1)
plt.xlabel("Calendar month")
plt.ylabel("Mean residual")
plt.title("Average residual by calendar month")
plt.xticks(range(1, 13))
plt.tight_layout()
plt.show()


# =============================================================================
# 7. Within-county residual autocorrelation
# =============================================================================

def county_acf_summary(data, max_lag=40, plot_each_county=False):
    records = []
    for c, g in data.groupby("countyfips"):
        g = g.sort_values("date")
        x = g["resid"].dropna().values

        if len(x) <= max_lag + 5:
            continue

        acf_vals = acf(x, nlags=max_lag, fft=True)
        for lag in range(1, max_lag + 1):
            records.append({"countyfips": c, "lag": lag, "acf": acf_vals[lag]})

        if plot_each_county:
            lags = np.arange(0, max_lag + 1)
            plt.figure(figsize=(8, 4.5))
            plt.stem(lags, acf_vals, basefmt=" ")
            conf = 1.96 / np.sqrt(len(x))
            plt.axhline(0, linestyle="--", linewidth=1)
            plt.axhline(conf, linestyle="--", linewidth=1)
            plt.axhline(-conf, linestyle="--", linewidth=1)
            plt.xlabel("Lag in months")
            plt.ylabel("Residual ACF")
            plt.title(f"Residual ACF for county {c}")
            plt.tight_layout()
            plt.show()

    acf_df_local = pd.DataFrame(records)
    if acf_df_local.empty:
        return acf_df_local, pd.DataFrame()

    acf_sum_local = (
        acf_df_local.groupby("lag", as_index=False)
        .agg(
            mean_acf=("acf", "mean"),
            median_acf=("acf", "median"),
            p10=("acf", lambda x: x.quantile(0.10)),
            p90=("acf", lambda x: x.quantile(0.90))
        )
    )

    plt.figure(figsize=(8, 5))
    plt.plot(acf_sum_local["lag"], acf_sum_local["mean_acf"], marker="o", label="Mean ACF")
    plt.plot(acf_sum_local["lag"], acf_sum_local["median_acf"], marker="s", label="Median ACF")
    plt.fill_between(acf_sum_local["lag"], acf_sum_local["p10"], acf_sum_local["p90"], alpha=0.2)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Lag in months")
    plt.ylabel("Within-county residual ACF")
    plt.title("Distribution of within-county residual autocorrelation")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return acf_df_local, acf_sum_local


acf_df, acf_sum = county_acf_summary(res_df, max_lag=40, plot_each_county=plot_each_county_ACF)

print("\n================ Within-county ACF summary ================")
print(acf_sum.head(15) if not acf_sum.empty else "No ACF results.")


# =============================================================================
# 8. Event-time residual check around high-wind exposure months
# =============================================================================

def residual_event_study_check(data, exposure_col="maxs", threshold=25, window=24, min_gap_months=12):
    df = data.copy().sort_values(["countyfips", "date"])

    if exposure_col not in df.columns:
        print(f"{exposure_col} not available. Skipping event-time residual check.")
        return pd.DataFrame(), pd.DataFrame()

    events = df[df[exposure_col] >= threshold][["countyfips", "date", exposure_col]].copy()
    events = events.rename(columns={"date": "event_date", exposure_col: "event_exposure"})

    if events.empty:
        print(f"No events found with {exposure_col} >= {threshold}.")
        return pd.DataFrame(), pd.DataFrame()

    thinned_events = []
    for c, g in events.groupby("countyfips"):
        g = g.sort_values("event_date")
        last_event_date = None
        for _, row in g.iterrows():
            current_date = row["event_date"]
            if last_event_date is None:
                thinned_events.append(row)
                last_event_date = current_date
            else:
                gap = (current_date.year - last_event_date.year) * 12 + (current_date.month - last_event_date.month)
                if gap >= min_gap_months:
                    thinned_events.append(row)
                    last_event_date = current_date

    events = pd.DataFrame(thinned_events)
    records = []

    for _, ev in events.iterrows():
        c = ev["countyfips"]
        event_date = ev["event_date"]
        g = df[df["countyfips"] == c].copy()
        g["event_time"] = (
            (g["date"].dt.year - event_date.year) * 12
            + (g["date"].dt.month - event_date.month)
        )
        g = g[(g["event_time"] >= -window) & (g["event_time"] <= window)].copy()
        g["event_date"] = event_date
        g["event_exposure"] = ev["event_exposure"]
        records.append(g[["countyfips", "date", "event_date", "event_exposure", "event_time", "resid", "std_resid"]])

    event_df = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    if event_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    event_summary = (
        event_df.groupby("event_time", as_index=False)
        .agg(mean_resid=("resid", "mean"), sd_resid=("resid", "std"), n=("resid", "size"))
    )
    event_summary["se"] = event_summary["sd_resid"] / np.sqrt(event_summary["n"])

    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE)
    ax.errorbar(
        event_summary["event_time"],
        event_summary["mean_resid"],
        yerr=1.96 * event_summary["se"],
        fmt="o-",
        capsize=3
    )
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.axvline(0, linestyle="--", linewidth=1)
    format_axis_text(
        ax,
        xlabel="Months relative to high-wind exposure",
        ylabel="Mean residual"
    )
    fig.tight_layout()
    save_current_fig("residual_event_time_check_maxs25.png")
    plt.show()

    print("\n================ Event-time residual summary ================")
    print("Exposure threshold:", threshold)
    print("Number of thinned events:", events.shape[0])
    print("Number of counties with events:", events["countyfips"].nunique())

    return event_df, event_summary


event_resid_df, event_summary_df = residual_event_study_check(
    res_df,
    exposure_col="maxs",
    threshold=event_wind_threshold,
    window=event_window,
    min_gap_months=event_min_gap_months
)


# =============================================================================
# 9. Residual distribution and aggregate residual time series
# =============================================================================

for col in ["resid", "log_resid"]:
    plt.figure(figsize=(6, 4))
    plt.hist(res_df[col].dropna(), bins=50, density=True, alpha=0.75)
    x = res_df[col].dropna()
    if len(x) > 1:
        xs = np.linspace(x.quantile(0.01), x.quantile(0.99), 200)
        try:
            kde = stats.gaussian_kde(x)
            plt.plot(xs, kde(xs))
        except Exception as e:
            print(f"KDE failed for {col}: {e}")
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    stats.probplot(res_df[col].dropna(), dist="norm", plot=plt)
    plt.title(f"Normal Q-Q plot: {col}")
    plt.tight_layout()
    plt.show()

agg_df = (
    res_df.groupby("date", as_index=False)
    .agg(mean_resid=("resid", "mean"))
    .sort_values("date")
)

fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE)
ax.plot(agg_df["date"], agg_df["mean_resid"])
ax.axhline(0, linestyle="--", linewidth=1)
format_axis_text(
    ax,
    xlabel="Date",
    ylabel="Average residual across counties"
)
fig.tight_layout()
save_current_fig("aggregate_monthly_mean_residual.png")
plt.show()

try:
    adf_result = adfuller(agg_df["mean_resid"].dropna())
    print("\n================ Aggregate residual ADF test ================")
    print("ADF Statistic:", adf_result[0])
    print("p-value:", adf_result[1])
    for key, value in adf_result[4].items():
        print(f"Critical Value ({key}): {value}")
except Exception as e:
    print("ADF test failed:", e)

fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE)

plot_acf(
    agg_df["mean_resid"].dropna(),
    lags=40,
    ax=ax,
    title=""
)

ax.set_title("")
ax.set_xlabel("Lag")
ax.set_ylabel("Autocorrelation")

plt.tight_layout()
save_current_fig("acf_aggregate_monthly_mean_residuals.png")
plt.show()


# =============================================================================
# 10. County diagnostic flags
# =============================================================================

agg_dict = {
    "n": ("resid", "size"),
    "mean_resid": ("resid", "mean"),
    "sd_resid": ("resid", "std"),
    "max_abs_std_resid": ("std_resid", lambda x: np.max(np.abs(x))),
    "outlier_share": ("outlier", "mean"),
    "mean_pop": ("totalpop", "mean"),
    "min_pop": ("totalpop", "min"),
    "mean_deaths": ("dths_tot", "mean"),
    "min_deaths": ("dths_tot", "min"),
}
if "X_nature" in res_df.columns:
    agg_dict["max_xnature"] = ("X_nature", "max")

county_diag = res_df.groupby("countyfips").agg(**agg_dict).reset_index()

if group_var in res_df.columns:
    county_groups = res_df.drop_duplicates("countyfips")[["countyfips", group_var]]
    county_diag = county_diag.merge(county_groups, on="countyfips", how="left")

county_diag["flag_high_resid_sd"] = county_diag["sd_resid"] > county_diag["sd_resid"].quantile(0.90)
county_diag["flag_many_outliers"] = county_diag["outlier_share"] > 0.05
county_diag["flag_extreme_outlier"] = county_diag["max_abs_std_resid"] > 7
county_diag["flag_low_population"] = county_diag["mean_pop"] < 50_000
county_diag["flag_low_deaths"] = county_diag["mean_deaths"] < 50

if not acf_df.empty:
    acf_lag1 = acf_df[acf_df["lag"] == 1].rename(columns={"acf": "acf_lag1"})[["countyfips", "acf_lag1"]]
    acf_wide = acf_df.pivot(index="countyfips", columns="lag", values="acf").reset_index()
    acf_wide["mean_acf_1_12"] = acf_wide[[l for l in range(1, 13) if l in acf_wide.columns]].mean(axis=1)
    acf_wide["mean_acf_13_24"] = acf_wide[[l for l in range(13, 25) if l in acf_wide.columns]].mean(axis=1)

    county_diag = (
        county_diag
        .merge(acf_lag1, on="countyfips", how="left")
        .merge(acf_wide[["countyfips", "mean_acf_1_12", "mean_acf_13_24"]], on="countyfips", how="left")
    )
    county_diag["flag_persistent_acf"] = (
        (county_diag["acf_lag1"] > 0.60) |
        (county_diag["mean_acf_1_12"] > 0.25) |
        (county_diag["mean_acf_13_24"] > 0.15)
    )
else:
    county_diag["flag_persistent_acf"] = False

county_diag["bad_county"] = (
    (
        county_diag["flag_low_population"] |
        county_diag["flag_low_deaths"]
    )
    &
    (
        county_diag["flag_high_resid_sd"] |
        county_diag["flag_many_outliers"] |
        county_diag["flag_extreme_outlier"] |
        county_diag["flag_persistent_acf"]
    )
)

print("\n================ County residual diagnostics ================")
print(county_diag.sort_values("sd_resid", ascending=False).head(30))

bad_counties = county_diag[county_diag["bad_county"]].copy()


# =============================================================================
# 11. Save diagnostics
# =============================================================================

res_df.to_csv(os.path.join(output_dir, "residual_diagnostics_svi_nature_future.csv"), index=False)
season_df.to_csv(os.path.join(output_dir, "residual_seasonality_svi_nature_future.csv"), index=False)
std_by_bin.to_csv(os.path.join(output_dir, "residual_variance_by_fitted_svi_nature_future.csv"), index=False)
county_diag.to_csv(os.path.join(output_dir, "county_residual_diagnostics_svi_nature_future.csv"), index=False)
bad_counties[["countyfips"]].to_csv(os.path.join(output_dir, "bad_counties_svi_nature_future_rule.csv"), index=False)

if not acf_df.empty:
    acf_df.to_csv(os.path.join(output_dir, "within_county_residual_acf_svi_nature_future.csv"), index=False)
if not acf_sum.empty:
    acf_sum.to_csv(os.path.join(output_dir, "within_county_residual_acf_summary_svi_nature_future.csv"), index=False)
if not event_resid_df.empty:
    event_resid_df.to_csv(os.path.join(output_dir, "event_time_residuals_svi_nature_future.csv"), index=False)
if not event_summary_df.empty:
    event_summary_df.to_csv(os.path.join(output_dir, "event_time_residual_summary_svi_nature_future.csv"), index=False)

agg_df.to_csv(os.path.join(output_dir, "aggregate_residual_time_series_svi_nature_future.csv"), index=False)


# =============================================================================
# 12. Create 2x2 model diagnostics panel
# =============================================================================

import matplotlib.image as mpimg

panel_files = {
    "a": os.path.join(resid_plot_dir, "standardized_residuals_vs_fitted.png"),
    "b": os.path.join(resid_plot_dir, "aggregate_monthly_mean_residual.png"),
    "c": os.path.join(resid_plot_dir, "acf_aggregate_monthly_mean_residuals.png"),
    "d": os.path.join(resid_plot_dir, "residual_event_time_check_maxs25.png"),
}

panel_titles = {
    "a": "(a) Standardized residuals vs fitted values",
    "b": "(b) Aggregate monthly mean residual",
    "c": "(c) ACF of aggregate monthly mean residuals",
    "d": "(d) Event-time residual check: maxs ≥ 25",
}

missing_panel_files = [path for path in panel_files.values() if not os.path.exists(path)]
if missing_panel_files:
    print("Warning: could not create combined model diagnostics panel. Missing files:")
    for path in missing_panel_files:
        print(f"  {path}")
else:
    imgs = {k: mpimg.imread(v) for k, v in panel_files.items()}

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for ax, key in zip(axes.ravel(), ["a", "b", "c", "d"]):
        ax.imshow(imgs[key])
        ax.set_title(panel_titles[key], fontsize=PANEL_TITLE_SIZE, pad=10)
        ax.axis("off")

    fig.tight_layout()

    panel_output_path = os.path.join(resid_plot_dir, "model_diagnostics.png")
    plt.savefig(panel_output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()

    print(f"Saved combined model diagnostics panel: {panel_output_path}")

print("\nDone. Validation diagnostics for the SVI-group X_nature + X_future72 model completed.")
