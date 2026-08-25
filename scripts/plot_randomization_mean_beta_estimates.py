"""
Create county-level Figure-2-style TC mortality response plots by SVI group.

This version matches the folder structure:

    output/
      actual/
        beta/
          coefficients_county_nature_future.dta
          vcv_region_poly.csv
      placebo/
        beta/
          coefficients_county_nature_future.dta
          vcv_region_poly.csv

Important distinction
---------------------
- The ACTUAL folder is used only for the actual TC exposure estimates:
      ibn.svi_high#c.X_nature
- The PLACEBO folder is used only for the compact future-exposure placebo estimates:
      ibn.svi_high#c.X_future72
- Other randomization experiments are not used or expected.

The script treats X_nature as a normalized Nature-shaped distributed exposure index.
Thus the estimated coefficient beta_g is interpreted as the cumulative response over
lags 0,...,LMAX, and is distributed over monthly lags using the normalized Nature
quadratic weights.

Similarly, X_future72 is treated as a compact future-exposure placebo index over
leads 1,...,FMAX, and is distributed over negative months using the same quadratic
weights normalized over the selected lead window.

Outputs
-------
    output/figures/randomization/figure2_panel_c_split_svi_with_randomizations.png
    output/figures/randomization/figure2_panel_c_split_svi_with_randomizations.pdf

Randomization design mean estimates are manually entered from
Table~\ref{tbl:randomization_group_tests}. No randomization CSV is read.

Run example
-----------
python scripts/plot_randomization_mean_beta_estimates.py

python scripts/plot_randomization_mean_beta_estimates.py ^
  --base-dir "..." ^
  --output-subdir output ^
  --actual-folder actual ^
  --placebo-folder placebo ^
  --group-var svi_high ^
  --lmax 172 ^
  --fmax 12 ^
  --df-resid 41
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from pathlib import Path

# =============================================================================
# Plot style settings
# =============================================================================

MONTHLY_CI_ALPHA = 0.12
CUMULATIVE_CI_ALPHA = 0.09
HETEROGENEITY_CI_ALPHA = 0.12

AXIS_LABEL_FONTSIZE = 13
TICK_LABEL_FONTSIZE = 11
LEGEND_FONTSIZE = 11

# =============================================================================
# Helpers
# =============================================================================


def nature_shape(x: np.ndarray | float) -> np.ndarray | float:
    """Quadratic lag shape from Young and Hsiang's TC mortality response curve."""
    return 0.0237 + 0.000535 * x - 0.0000039 * (x ** 2)


def make_nature_lag_weights(lmax: int = 172) -> pd.DataFrame:
    lags = np.arange(0, lmax + 1)
    raw = np.asarray(nature_shape(lags), dtype=float)
    if np.any(raw < 0):
        raise ValueError(f"Nature lag shape is negative for lags: {lags[raw < 0].tolist()}")
    w = raw / raw.sum()
    return pd.DataFrame({"lag": lags, "raw_shape": raw, "weight": w})


def make_future_placebo_weights(fmax: int = 12) -> pd.DataFrame:
    leads = np.arange(1, fmax + 1)
    raw = np.asarray(nature_shape(leads), dtype=float)
    if np.any(raw < 0):
        raise ValueError(f"Future placebo shape is negative for leads: {leads[raw < 0].tolist()}")
    w = raw / raw.sum()
    return pd.DataFrame({"lead": leads, "lag": -leads, "raw_shape": raw, "weight": w})


def clean_stata_parm(parm: object) -> str:
    p = str(parm).strip()
    p = p.replace("b.", ".").replace("o.", ".")
    p = re.sub(r"^(\d+)[a-z]+\.", r"\1.", p)
    return p


def parse_group_term(parm: str, group_var: str = "svi_high") -> Tuple[Optional[int], Optional[str]]:
    p = clean_stata_parm(parm)
    m = re.match(fr"([0-9]+)\.{re.escape(group_var)}#c\.X_nature$", p)
    if m:
        return int(m.group(1)), "nature"
    m = re.match(fr"([0-9]+)\.{re.escape(group_var)}#c\.X_future72$", p)
    if m:
        return int(m.group(1)), "future"
    return None, None


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".dta":
        return pd.read_stata(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def find_coef_file(beta_dir: Path) -> Path:
    candidates = [
        beta_dir / "coefficients_county_nature_future.dta",
        beta_dir / "coefficients_county_nature_future.csv",
        beta_dir / "coefficients_region_poly.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    files = sorted(beta_dir.glob("coefficients*.dta")) + sorted(beta_dir.glob("coefficients*.csv"))
    if not files:
        raise FileNotFoundError(f"No coefficient file found in {beta_dir}")
    return files[0]


def normalize_coef_table(df: pd.DataFrame, source: Path, group_var: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "parm" not in df.columns:
        alternatives = ["parameter", "term", "coef_name", "name", "varname"]
        found = next((c for c in alternatives if c in df.columns), None)
        if found is None:
            raise KeyError(f"No parm column found in {source}. Columns: {df.columns.tolist()}")
        df = df.rename(columns={found: "parm"})

    if "estimate" not in df.columns:
        alternatives = ["coef", "coefficient", "b", "beta"]
        found = next((c for c in alternatives if c in df.columns), None)
        if found is None:
            raise KeyError(f"No estimate column found in {source}. Columns: {df.columns.tolist()}")
        df = df.rename(columns={found: "estimate"})

    se_col = None
    for c in ["stderr", "std_error", "se", "std_err"]:
        if c in df.columns:
            se_col = c
            break

    df["parm_raw"] = df["parm"].astype(str)
    df["parm_clean"] = df["parm_raw"].apply(clean_stata_parm)
    df[["group", "term"]] = df["parm_clean"].apply(lambda p: pd.Series(parse_group_term(p, group_var)))
    df = df[df["group"].notna() & df["term"].notna()].copy()
    df["group"] = df["group"].astype(int)
    df["estimate"] = pd.to_numeric(df["estimate"], errors="coerce")
    df["se_exported"] = pd.to_numeric(df[se_col], errors="coerce") if se_col else np.nan
    df["source_file"] = str(source)
    return df


def read_vcv(vcv_path: Path) -> Optional[pd.DataFrame]:
    if not vcv_path.exists():
        return None
    vcv = pd.read_csv(vcv_path, index_col=0)
    vcv.index = [clean_stata_parm(x) for x in vcv.index.astype(str)]
    vcv.columns = [clean_stata_parm(x) for x in vcv.columns.astype(str)]
    return (vcv + vcv.T) / 2


def get_var(vcv: Optional[pd.DataFrame], parm: str) -> float:
    if vcv is None:
        return np.nan
    p = clean_stata_parm(parm)
    if p not in vcv.index or p not in vcv.columns:
        return np.nan
    return float(vcv.loc[p, p])


def get_cov(vcv: Optional[pd.DataFrame], parm1: str, parm2: str) -> float:
    if vcv is None:
        return np.nan
    p1, p2 = clean_stata_parm(parm1), clean_stata_parm(parm2)
    if p1 not in vcv.index or p2 not in vcv.columns:
        return np.nan
    return float(vcv.loc[p1, p2])


def group_label(g: int) -> str:
    return {0: "Lower-SVI", 1: "High-SVI", 999: "high-lower difference"}.get(int(g), f"Group {g}")


def recompute_future_cumulative_from_event(df):
    """
    Recompute future-placebo cumulative effects so they are anchored at lag 0
    and cumulated backward from lead 1 to lead FMAX.

    For lead k:
        cumulative effect at lag -k = beta_future * sum_{j=1}^k h_j
        cumulative SE     at lag -k = se_future   * sum_{j=1}^k h_j
    """
    out = []

    for keys, g in df.groupby(["group", "curve", "term"], dropna=False):
        g = g.copy()

        if "lead" not in g.columns:
            g["lead"] = -g["lag"]

        g = g.sort_values("lead").copy()
        g["cum_weight_from_event"] = g["weight"].cumsum()

        g["cum_effect"] = g["beta_total"] * g["cum_weight_from_event"]
        g["cum_se"] = g["se_total"] * g["cum_weight_from_event"]

        crit = (g["cum_ci_high"].iloc[0] - g["cum_effect"].iloc[0]) / g["cum_se"].iloc[0] \
            if g["cum_se"].iloc[0] > 0 else 1.96

        g["cum_ci_low"] = g["cum_effect"] - crit * g["cum_se"]
        g["cum_ci_high"] = g["cum_effect"] + crit * g["cum_se"]

        # Plot in x-axis order, from -FMAX to -1.
        g = g.sort_values("lag").copy()
        out.append(g)

    return pd.concat(out, ignore_index=True)

# =============================================================================
# Curve construction
# =============================================================================


def build_curves_for_term(
    coef: pd.DataFrame,
    vcv: Optional[pd.DataFrame],
    weights: pd.DataFrame,
    term_to_use: str,
    curve_label: str,
    df_resid: Optional[float],
    sort_for_cumulative: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build monthly and cumulative response curves for one coefficient term."""
    rows = []
    summary_rows = []
    sub = coef[coef["term"] == term_to_use].copy()
    if sub.empty:
        raise ValueError(f"No coefficients found for term={term_to_use} in {coef['source_file'].iloc[0] if len(coef) else 'file'}")

    crit = stats.t.ppf(0.975, df_resid) if df_resid is not None else stats.norm.ppf(0.975)

    for _, r in sub.iterrows():
        g = int(r["group"])
        parm = r["parm_clean"]
        beta = float(r["estimate"])
        var = get_var(vcv, parm)
        se = np.sqrt(max(var, 0.0)) if not np.isnan(var) else float(r.get("se_exported", np.nan))

        tmp = weights.copy()
        tmp["group"] = g
        tmp["group_label"] = group_label(g)
        tmp["curve"] = curve_label
        tmp["term"] = term_to_use
        tmp["parm"] = parm
        tmp["beta_total"] = beta
        tmp["se_total"] = se
        tmp["effect"] = beta * tmp["weight"]
        tmp["se"] = se * tmp["weight"]
        tmp["ci_low"] = tmp["effect"] - crit * tmp["se"]
        tmp["ci_high"] = tmp["effect"] + crit * tmp["se"]

        if sort_for_cumulative:
            tmp = tmp.sort_values("lag").copy()
        tmp["cum_weight"] = tmp["weight"].cumsum()
        tmp["cum_effect"] = beta * tmp["cum_weight"]
        tmp["cum_se"] = se * tmp["cum_weight"]
        tmp["cum_ci_low"] = tmp["cum_effect"] - crit * tmp["cum_se"]
        tmp["cum_ci_high"] = tmp["cum_effect"] + crit * tmp["cum_se"]
        rows.append(tmp)

        t_stat = beta / se if se and not np.isnan(se) else np.nan
        summary_rows.append({
            "group": g,
            "group_label": group_label(g),
            "curve": curve_label,
            "parm": parm,
            "estimate": beta,
            "se": se,
            "ci95_low": beta - crit * se if not np.isnan(se) else np.nan,
            "ci95_high": beta + crit * se if not np.isnan(se) else np.nan,
            "t": t_stat,
            "p_two_sided": 2 * stats.t.sf(abs(t_stat), df_resid) if df_resid is not None and not np.isnan(t_stat) else np.nan,
            "p_one_sided_gt0": stats.t.sf(t_stat, df_resid) if df_resid is not None and not np.isnan(t_stat) else np.nan,
            "source_file": r["source_file"],
        })

    return pd.concat(rows, ignore_index=True), pd.DataFrame(summary_rows)


def build_difference_curve(
    coef: pd.DataFrame,
    vcv: Optional[pd.DataFrame],
    weights: pd.DataFrame,
    term_to_use: str,
    curve_label: str,
    df_resid: Optional[float],
    sort_for_cumulative: bool = True,
) -> Optional[pd.DataFrame]:
    sub = coef[coef["term"] == term_to_use].copy()
    if not set(sub["group"]).issuperset({0, 1}):
        return None

    low = sub.loc[sub["group"] == 0].iloc[0]
    high = sub.loc[sub["group"] == 1].iloc[0]
    beta_delta = float(high["estimate"] - low["estimate"])

    var_h = get_cov(vcv, high["parm_clean"], high["parm_clean"])
    var_l = get_cov(vcv, low["parm_clean"], low["parm_clean"])
    cov_hl = get_cov(vcv, high["parm_clean"], low["parm_clean"])
    if not np.isnan(var_h) and not np.isnan(var_l) and not np.isnan(cov_hl):
        se_delta = np.sqrt(max(var_h + var_l - 2 * cov_hl, 0.0))
    else:
        se_h = float(high.get("se_exported", np.nan))
        se_l = float(low.get("se_exported", np.nan))
        se_delta = np.sqrt(se_h ** 2 + se_l ** 2) if not np.isnan(se_h) and not np.isnan(se_l) else np.nan

    crit = stats.t.ppf(0.975, df_resid) if df_resid is not None else stats.norm.ppf(0.975)
    tmp = weights.copy()
    tmp["group"] = 999
    tmp["group_label"] = group_label(999)
    tmp["curve"] = curve_label
    tmp["term"] = term_to_use
    tmp["beta_total"] = beta_delta
    tmp["se_total"] = se_delta
    tmp["effect"] = beta_delta * tmp["weight"]
    tmp["se"] = se_delta * tmp["weight"]
    tmp["ci_low"] = tmp["effect"] - crit * tmp["se"]
    tmp["ci_high"] = tmp["effect"] + crit * tmp["se"]
    if sort_for_cumulative:
        tmp = tmp.sort_values("lag").copy()
    tmp["cum_weight"] = tmp["weight"].cumsum()
    tmp["cum_effect"] = beta_delta * tmp["cum_weight"]
    tmp["cum_se"] = se_delta * tmp["cum_weight"]
    tmp["cum_ci_low"] = tmp["cum_effect"] - crit * tmp["cum_se"]
    tmp["cum_ci_high"] = tmp["cum_effect"] + crit * tmp["cum_se"]
    return tmp

def set_response_xaxis(ax: plt.Axes, fmax: int, lmax: int) -> None:
    """
    Use a shared x-axis that automatically adapts to the placebo lead window
    and the post-exposure response window.

    Examples:
        fmax=12, lmax=172 -> ticks include -12, 0, ..., 172
        fmax=72, lmax=240 -> ticks include -72, 0, ..., 240
    """
    left_pad = max(3, int(round(0.03 * max(fmax, 1))))
    right_pad = max(3, int(round(0.03 * max(lmax, 1))))
    ax.set_xlim(-(fmax + left_pad), lmax + right_pad)

    positive_ticks = [t for t in range(25, lmax + 1, 25)]
    ticks = [-fmax, 0] + positive_ticks
    if lmax not in ticks:
        ticks.append(lmax)
    ticks = sorted(dict.fromkeys(ticks))

    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def add_line_and_ci_legends(
    ax: plt.Axes,
    line_bbox: tuple[float, float] = (0.36, 0.04),
    ci_bbox: tuple[float, float] = (0.76, 0.04),
    fontsize: int = LEGEND_FONTSIZE,
) -> None:
    """
    Add two side-by-side legends: one for response curves and one for
    color-matched 95% confidence-interval bands.
    """
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return

    line_legend = ax.legend(
        handles,
        labels,
        frameon=False,
        fontsize=fontsize,
        loc="lower center",
        bbox_to_anchor=line_bbox,
    )
    ax.add_artist(line_legend)

    ci_handles = []
    ci_labels = []
    for handle, label in zip(handles, labels):
        color = handle.get_color() if hasattr(handle, "get_color") else "gray"
        ci_handles.append(Patch(facecolor=color, edgecolor="none", alpha=0.18))
        ci_labels.append(f"{label} 95% CI")

    ax.legend(
        handles=ci_handles,
        labels=ci_labels,
        frameon=False,
        fontsize=fontsize,
        loc="lower center",
        bbox_to_anchor=ci_bbox,
    )

def format_axis_text(ax: plt.Axes) -> None:
    """Increase axis-label and tick-label font sizes while keeping titles unchanged."""
    ax.xaxis.label.set_size(AXIS_LABEL_FONTSIZE)
    ax.yaxis.label.set_size(AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)



# =============================================================================
# Randomization curves
# =============================================================================


MANUAL_RANDOMIZATION_MEANS = [
    # Values copied manually from Table~\ref{tbl:randomization_group_tests}.
    # group=0 is Lower-SVI; group=1 is High-SVI.
    {"randtype": "global_exposure_shuffle", "rand_label": "Global exposure shuffle", "group": 0, "beta_total": -0.016, "nsim": 1000},
    {"randtype": "global_exposure_shuffle", "rand_label": "Global exposure shuffle", "group": 1, "beta_total": -0.014, "nsim": 1000},

    {"randtype": "within_county_time_shuffle", "rand_label": "Within-county time shuffle", "group": 0, "beta_total": -0.013, "nsim": 1000},
    {"randtype": "within_county_time_shuffle", "rand_label": "Within-county time shuffle", "group": 1, "beta_total": -0.020, "nsim": 1000},

    {"randtype": "within_month_county_shuffle", "rand_label": "Within-month county shuffle", "group": 0, "beta_total": -1.047, "nsim": 1000},
    {"randtype": "within_month_county_shuffle", "rand_label": "Within-month county shuffle", "group": 1, "beta_total":  0.456, "nsim": 1000},

    {"randtype": "cross_county_history_reassignment", "rand_label": "Cross-county history reassignment", "group": 0, "beta_total": -0.904, "nsim": 1000},
    {"randtype": "cross_county_history_reassignment", "rand_label": "Cross-county history reassignment", "group": 1, "beta_total":  0.273, "nsim": 1000},

    {"randtype": "pooled_all", "rand_label": "Pooled all", "group": 0, "beta_total": -0.495, "nsim": 4000},
    {"randtype": "pooled_all", "rand_label": "Pooled all", "group": 1, "beta_total":  0.174, "nsim": 4000},
]




RANDOMIZATION_STYLE = {
    "Global exposure shuffle": {"color": "black", "linestyle": (0, (1, 1.6)), "linewidth": 2.2, "marker": "o"},
    "Within-county time shuffle": {"color": "tab:red", "linestyle": (0, (6, 2)), "linewidth": 1.9, "marker": "s"},
    "Within-month county shuffle": {"color": "tab:purple", "linestyle": (0, (6, 3)), "linewidth": 1.9, "marker": "^"},
    "Cross-county history reassignment": {"color": "tab:brown", "linestyle": (0, (7, 2.5)), "linewidth": 1.9, "marker": "D"},
    "Pooled all": {"color": "tab:pink", "linestyle": (0, (6, 2)), "linewidth": 1.9, "marker": "P"},
}

ACTUAL_COLOR = "tab:blue"
FUTURE_COLOR = "tab:orange"

def read_randomization_means_manual() -> pd.DataFrame:
    """Return manually entered mean randomized estimates from the paper table."""
    out = pd.DataFrame(MANUAL_RANDOMIZATION_MEANS)
    out["group_label"] = out["group"].apply(group_label)
    out["source_file"] = "manual_from_table_randomization_group_tests"
    return out


def build_randomization_cumulative_curves(
    rand_means: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.DataFrame:
    """Expand each randomization-design mean beta into a cumulative lag curve."""
    if rand_means is None or rand_means.empty:
        return pd.DataFrame()

    rows = []
    w = weights.sort_values("lag").copy()
    w["cum_weight"] = w["weight"].cumsum()

    for _, r in rand_means.iterrows():
        tmp = w.copy()
        tmp["randtype"] = r["randtype"]
        tmp["rand_label"] = r["rand_label"]
        tmp["group"] = int(r["group"])
        tmp["group_label"] = r["group_label"]
        tmp["beta_total"] = float(r["beta_total"])
        tmp["cum_effect"] = tmp["beta_total"] * tmp["cum_weight"]
        tmp["nsim"] = int(r["nsim"])
        tmp["source_file"] = r["source_file"]
        rows.append(tmp)

    return pd.concat(rows, ignore_index=True)


def compute_shared_ylim_for_group_cumulative(
    actual_curve: pd.DataFrame,
    future_curve: pd.DataFrame,
    randomization_curve: pd.DataFrame,
    groups: tuple[int, ...] = (0, 1),
    pad_frac: float = 0.08,
) -> tuple[float, float]:
    """Use one y-scale for both SVI panels."""
    vals = []
    for g in groups:
        a = actual_curve[actual_curve["group"] == g]
        f = future_curve[future_curve["group"] == g]
        if not a.empty:
            vals.extend(a[["cum_ci_low", "cum_ci_high", "cum_effect"]].to_numpy().ravel())
        if not f.empty:
            vals.extend(f[["cum_ci_low", "cum_ci_high", "cum_effect"]].to_numpy().ravel())
        if randomization_curve is not None and not randomization_curve.empty:
            r = randomization_curve[randomization_curve["group"] == g]
            if not r.empty:
                vals.extend(r["cum_effect"].to_numpy().ravel())

    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (-1, 1)
    lo, hi = float(vals.min()), float(vals.max())
    if np.isclose(lo, hi):
        lo -= 1
        hi += 1
    pad = pad_frac * (hi - lo)
    return lo - pad, hi + pad

# =============================================================================
# Plotting
# =============================================================================


def _plot_group_actual_placebo(
    ax: plt.Axes,
    actual_curve: pd.DataFrame,
    future_curve: pd.DataFrame,
    group_id: int,
    fmax: int,
    lmax: int,
    title: str,
) -> None:
    """One SVI-group panel: actual post-TC monthly response + future-placebo leads."""
    future = future_curve[future_curve["group"] == group_id].sort_values("lag")
    actual = actual_curve[actual_curve["group"] == group_id].sort_values("lag")

    if not future.empty:
        future_line = ax.plot(
            future["lag"],
            future["effect"],
            linestyle="--",
            linewidth=2,
            label="Future placebo",
        )[0]
        ax.fill_between(
            future["lag"].to_numpy(),
            future["ci_low"].to_numpy(),
            future["ci_high"].to_numpy(),
            color=future_line.get_color(),
            alpha=MONTHLY_CI_ALPHA,
        )

    if not actual.empty:
        actual_line = ax.plot(
            actual["lag"],
            actual["effect"],
            linewidth=2.2,
            label="Actual TC exposure",
        )[0]
        ax.fill_between(
            actual["lag"].to_numpy(),
            actual["ci_low"].to_numpy(),
            actual["ci_high"].to_numpy(),
            color=actual_line.get_color(),
            alpha=MONTHLY_CI_ALPHA,
        )
    set_response_xaxis(ax, fmax, lmax)
    ax.set_ylim(-0.02, 0.03)
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linestyle=":", linewidth=1)
    ax.axvline(lmax, linestyle=":", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Time relative to TC exposure (months)")
    ax.set_ylabel("Monthly mortality-rate response\n(deaths per 100,000 per m/s)")
    format_axis_text(ax)
    add_line_and_ci_legends(ax)


def _plot_cumulative_response(
    ax: plt.Axes,
    actual_curve: pd.DataFrame,
    future_curve: pd.DataFrame,
    fmax: int,
    lmax: int,
    title: str,
) -> None:
    """Cumulative response panel: keep actual and future placebo together."""
    for g, temp in future_curve.groupby("group"):
        temp = temp.sort_values("lag")
        line = ax.plot(
            temp["lag"],
            temp["cum_effect"],
            linestyle="--",
            linewidth=2,
            label=f"{group_label(g)} future placebo",
        )[0]
        ax.fill_between(
            temp["lag"].to_numpy(),
            temp["cum_ci_low"].to_numpy(),
            temp["cum_ci_high"].to_numpy(),
            color=line.get_color(),
            alpha=CUMULATIVE_CI_ALPHA,
        )

    for g, temp in actual_curve.groupby("group"):
        temp = temp.sort_values("lag")
        line = ax.plot(
            temp["lag"],
            temp["cum_effect"],
            linewidth=2.2,
            label=f"{group_label(g)} actual",
        )[0]
        ax.fill_between(
            temp["lag"].to_numpy(),
            temp["cum_ci_low"].to_numpy(),
            temp["cum_ci_high"].to_numpy(),
            color=line.get_color(),
            alpha=CUMULATIVE_CI_ALPHA,
        )
    set_response_xaxis(ax, fmax, lmax)
    ax.set_ylim(-4, 4)
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linestyle=":", linewidth=1)
    ax.axvline(lmax, linestyle=":", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Time relative to TC exposure (months)")
    ax.set_ylabel("Cumulative mortality response\n(deaths per 100,000 per m/s)")
    format_axis_text(ax)
    add_line_and_ci_legends(
        ax,
        line_bbox=(0.29, 0.04),
        ci_bbox=(0.73, 0.04),
        fontsize=LEGEND_FONTSIZE,
    )


def _plot_heterogeneity_contrast(
    ax: plt.Axes,
    diff_actual: Optional[pd.DataFrame],
    diff_future: Optional[pd.DataFrame],
    fmax: int,
    lmax: int,
    title: str,
) -> None:
    """Merged contrast panel: actual high-minus-lower and future-placebo high-minus-lower."""
    if diff_future is not None:
        temp = diff_future.sort_values("lag")
        line = ax.plot(
            temp["lag"],
            temp["cum_effect"],
            linestyle="--",
            linewidth=2,
            label="Future placebo $\Delta$",
        )[0]
        ax.fill_between(
            temp["lag"].to_numpy(),
            temp["cum_ci_low"].to_numpy(),
            temp["cum_ci_high"].to_numpy(),
            color=line.get_color(),
            alpha=HETEROGENEITY_CI_ALPHA,
        )

    if diff_actual is not None:
        temp = diff_actual.sort_values("lag")
        line = ax.plot(
            temp["lag"],
            temp["cum_effect"],
            linewidth=2.3,
            label="Actual $\Delta$",
        )[0]
        ax.fill_between(
            temp["lag"].to_numpy(),
            temp["cum_ci_low"].to_numpy(),
            temp["cum_ci_high"].to_numpy(),
            color=line.get_color(),
            alpha=HETEROGENEITY_CI_ALPHA,
        )
    set_response_xaxis(ax, fmax, lmax)
    ax.set_ylim(-4, 4)
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linestyle=":", linewidth=1)
    ax.axvline(lmax, linestyle=":", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Time relative to TC exposure (months)")
    ax.set_ylabel("High-lower difference cumulative mortality \nresponse (deaths per 100,000 per m/s)")
    format_axis_text(ax)
    add_line_and_ci_legends(ax)


def plot_main_figure(
    actual_curve: pd.DataFrame,
    future_curve: pd.DataFrame,
    diff_actual: Optional[pd.DataFrame],
    diff_future: Optional[pd.DataFrame],
    output_fig_dir: Path,
    lmax: int,
    fmax: int,
) -> None:
    """
    Four-panel dissertation figure:
      a. Lower-SVI actual monthly response + future-placebo leads.
      b. High-SVI actual monthly response + future-placebo leads.
      c. Cumulative response by SVI group, kept as before.
      d. Merged actual and future-placebo high-minus-lower cumulative contrasts.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.6))
    ax_a, ax_b, ax_c, ax_d = axes.flat

    _plot_group_actual_placebo(
        ax=ax_a,
        actual_curve=actual_curve,
        future_curve=future_curve,
        group_id=0,
        fmax=fmax,
        lmax=lmax,
        title="a.  Lower-SVI monthly response",
    )

    _plot_group_actual_placebo(
        ax=ax_b,
        actual_curve=actual_curve,
        future_curve=future_curve,
        group_id=1,
        fmax=fmax,
        lmax=lmax,
        title="b.  High-SVI monthly response",
    )

    _plot_cumulative_response(
        ax=ax_c,
        actual_curve=actual_curve,
        future_curve=future_curve,
        fmax=fmax,
        lmax=lmax,
        title="c.  Cumulative TC mortality response",
    )

    _plot_heterogeneity_contrast(
        ax=ax_d,
        diff_actual=diff_actual,
        diff_future=diff_future,
        fmax=fmax,
        lmax=lmax,
        title="d.  Heterogeneity contrast",
    )

    fig.tight_layout()
    fig.savefig(output_fig_dir / "figure2_svi_county_actual_future.png", dpi=400, bbox_inches="tight")
    fig.savefig(output_fig_dir / "figure2_svi_county_actual_future.pdf", bbox_inches="tight")
    plt.close(fig)



def _plot_group_cumulative_with_randomizations(
    ax: plt.Axes,
    actual_curve: pd.DataFrame,
    future_curve: pd.DataFrame,
    randomization_curve: pd.DataFrame,
    group_id: int,
    fmax: int,
    lmax: int,
    title: str,
    ylim: tuple[float, float],
) -> None:
    """
    One cumulative-response panel for one SVI group. This is the split version
    of panel c: actual + actual CI, compact future placebo + CI, and the mean
    response from each randomized exposure design as dashed placebo lines.
    """
    future = future_curve[future_curve["group"] == group_id].sort_values("lag")
    actual = actual_curve[actual_curve["group"] == group_id].sort_values("lag")

    if not actual.empty:
        ax.plot(
            actual["lag"],
            actual["cum_effect"],
            linewidth=2.5,
            color=ACTUAL_COLOR,
            label="Actual TC exposure",
            zorder=3,
        )
        ax.fill_between(
            actual["lag"].to_numpy(),
            actual["cum_ci_low"].to_numpy(),
            actual["cum_ci_high"].to_numpy(),
            color=ACTUAL_COLOR,
            alpha=CUMULATIVE_CI_ALPHA,
            label="Actual 95% CI",
            zorder=1,
        )

    if not future.empty:
        ax.plot(
            future["lag"],
            future["cum_effect"],
            linewidth=2.2,
            linestyle="--",
            color=FUTURE_COLOR,
            label="Future placebo",
            zorder=5,
        )
        ax.fill_between(
            future["lag"].to_numpy(),
            future["cum_ci_low"].to_numpy(),
            future["cum_ci_high"].to_numpy(),
            color=FUTURE_COLOR,
            alpha=CUMULATIVE_CI_ALPHA,
            label="Future placebo 95% CI",
            zorder=1,
        )

    if randomization_curve is not None and not randomization_curve.empty:
        rnd = randomization_curve[randomization_curve["group"] == group_id].copy()
        for rand_label, temp in rnd.groupby("rand_label", sort=False):
            temp = temp.sort_values("lag")
            style = RANDOMIZATION_STYLE.get(rand_label, {})
            mark_idx = np.linspace(0, len(temp) - 1, 5, dtype=int) if len(temp) >= 5 else None
            ax.plot(
                temp["lag"],
                temp["cum_effect"],
                linewidth=style.get("linewidth", 1.9),
                linestyle=style.get("linestyle", (0, (6, 2))),
                color=style.get("color", None),
                marker=style.get("marker", None),
                markevery=mark_idx,
                markersize=4.2,
                markerfacecolor="white",
                markeredgewidth=1.0,
                alpha=0.98,
                zorder=6 if rand_label == "Global exposure shuffle" else 4,
                label=rand_label,
            )

    set_response_xaxis(ax, fmax, lmax)
    ax.set_ylim(*ylim)
    # Do not draw y=0 axis; this keeps the near-zero placebo oscillations visible.
    ax.axvline(0, linestyle=":", linewidth=1)
    ax.axvline(lmax, linestyle=":", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Time relative to TC exposure (months)")
    ax.set_ylabel("Cumulative mortality response\n(deaths per 100,000 per m/s)")
    format_axis_text(ax)


def plot_cumulative_split_by_group_with_randomizations(
    actual_curve: pd.DataFrame,
    future_curve: pd.DataFrame,
    randomization_curve: pd.DataFrame,
    output_fig_dir: Path,
    fmax: int,
    lmax: int,
) -> None:
    """Create the requested two-panel replacement for panel c."""
    ylim = compute_shared_ylim_for_group_cumulative(actual_curve, future_curve, randomization_curve)

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.3), sharey=True)
    _plot_group_cumulative_with_randomizations(
        ax=axes[0],
        actual_curve=actual_curve,
        future_curve=future_curve,
        randomization_curve=randomization_curve,
        group_id=0,
        fmax=fmax,
        lmax=lmax,
        title="a. Lower-SVI cumulative response",
        ylim=ylim,
    )
    _plot_group_cumulative_with_randomizations(
        ax=axes[1],
        actual_curve=actual_curve,
        future_curve=future_curve,
        randomization_curve=randomization_curve,
        group_id=1,
        fmax=fmax,
        lmax=lmax,
        title="b. High-SVI cumulative response",
        ylim=ylim,
    )
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="y", labelleft=True)

    # Force identical y-axis limits and tick locations in both panels.
    tick_min = 2 * np.floor(ylim[0] / 2)
    tick_max = 2 * np.ceil(ylim[1] / 2)
    yticks = np.arange(tick_min, tick_max + 0.01, 2)
    for ax in axes:
        ax.set_ylim(tick_min, tick_max)
        ax.set_yticks(yticks)

    legend_handles = [
        Line2D([0], [0], color=ACTUAL_COLOR, linewidth=2.5, label="Actual TC exposure"),
        Patch(facecolor=ACTUAL_COLOR, edgecolor="none", alpha=CUMULATIVE_CI_ALPHA, label="Actual 95% CI"),
        Line2D([0], [0], color=FUTURE_COLOR, linewidth=2.2, linestyle="--", label="Future placebo"),
        Patch(facecolor=FUTURE_COLOR, edgecolor="none", alpha=CUMULATIVE_CI_ALPHA, label="Future placebo 95% CI"),
    ]
    for label, style in RANDOMIZATION_STYLE.items():
        legend_handles.append(
            Line2D(
                [0], [0],
                color=style.get("color", "black"),
                linewidth=style.get("linewidth", 1.9),
                linestyle=style.get("linestyle", (0, (6, 2))),
                marker=style.get("marker", None),
                markersize=4.2,
                markerfacecolor="white",
                markeredgewidth=1.0,
                label=label,
            )
        )

    fig.legend(
        handles=legend_handles,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.05),
    )

    fig.tight_layout(rect=(0, 0.18, 1, 1))
    fig.savefig(output_fig_dir / "figure2_panel_c_split_svi_with_randomizations.png", dpi=400, bbox_inches="tight")
    fig.savefig(output_fig_dir / "figure2_panel_c_split_svi_with_randomizations.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_cumulative_only(actual_curve: pd.DataFrame, future_curve: pd.DataFrame, output_fig_dir: Path, fmax: int, lmax: int) -> None:
    fig, ax = plt.subplots(figsize=(8.7, 5.4))
    _plot_cumulative_response(
        ax=ax,
        actual_curve=actual_curve,
        future_curve=future_curve,
        fmax=fmax,
        lmax=lmax,
        title="Cumulative TC mortality response by SVI group",
    )
    fig.tight_layout()
    fig.savefig(output_fig_dir / "figure2_svi_cumulative_actual_future.png", dpi=400, bbox_inches="tight")
    fig.savefig(output_fig_dir / "figure2_svi_cumulative_actual_future.pdf", bbox_inches="tight")
    plt.close(fig)



def resolve_beta_dir(base_dir: Path, output_dir: Path, folder_or_path: str) -> Path:
    """Resolve actual/placebo beta folder flexibly after folder-name changes.

    Accepted inputs include:
      - actual                -> output/actual/beta
      - actual/beta           -> output/actual/beta
      - output/actual/beta    -> base_dir/output/actual/beta
      - absolute/path/to/beta -> absolute/path/to/beta
    """
    raw = Path(folder_or_path)
    candidates = []
    if raw.is_absolute():
        candidates.extend([raw / "beta", raw])
    else:
        candidates.extend([
            output_dir / raw / "beta",
            output_dir / raw,
            base_dir / raw / "beta",
            base_dir / raw,
        ])
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "Could not resolve beta directory for "
        f"'{folder_or_path}'. Tried:\n" + "\n".join(f"  - {c}" for c in candidates)
    )
def resolve_output_fig_dir(base_dir: Path, output_dir: Path, output_fig_dir_arg: Optional[str]) -> Path:
    """Resolve the figure output directory."""
    if output_fig_dir_arg:
        p = Path(output_fig_dir_arg)
        return p if p.is_absolute() else base_dir / p
    return output_dir / "figures" / "randomization"

# =============================================================================
# Main
# =============================================================================


def main() -> None:
    repo_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Plot actual SVI-group TC response and future-exposure placebo leads.")
    parser.add_argument("--base-dir", type=str, default=repo_dir, help="Project base directory.")
    parser.add_argument("--output-subdir", type=str, default="output")
    parser.add_argument("--actual-folder", type=str, default="actual", help="Folder name or path containing actual/beta outputs.")
    parser.add_argument("--placebo-folder", type=str, default="placebo", help="Folder name or path containing placebo/beta outputs.")
    parser.add_argument("--output-fig-dir", type=str, default=None, help="Optional figure output directory. Relative paths are resolved from base-dir.")
    parser.add_argument("--group-var", type=str, default="svi_high")
    parser.add_argument("--lmax", type=int, default=172)
    parser.add_argument("--fmax", type=int, default=12)
    parser.add_argument("--df-resid", type=float, default=41)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = base_dir / args.output_subdir
    actual_beta_dir = resolve_beta_dir(base_dir, output_dir, args.actual_folder)
    placebo_beta_dir = resolve_beta_dir(base_dir, output_dir, args.placebo_folder)
    output_fig_dir = resolve_output_fig_dir(base_dir, output_dir, args.output_fig_dir)
    output_fig_dir.mkdir(parents=True, exist_ok=True)

    actual_coef_file = find_coef_file(actual_beta_dir)
    placebo_coef_file = find_coef_file(placebo_beta_dir)
    actual_coef = normalize_coef_table(read_table(actual_coef_file), actual_coef_file, group_var=args.group_var)
    placebo_coef = normalize_coef_table(read_table(placebo_coef_file), placebo_coef_file, group_var=args.group_var)

    actual_vcv = read_vcv(actual_beta_dir / "vcv_region_poly.csv")
    placebo_vcv = read_vcv(placebo_beta_dir / "vcv_region_poly.csv")

    lag_weights = make_nature_lag_weights(args.lmax)
    future_weights = make_future_placebo_weights(args.fmax)

    # Actual estimates: use only X_nature from actual folder.
    actual_curve, actual_summary = build_curves_for_term(
        coef=actual_coef,
        vcv=actual_vcv,
        weights=lag_weights,
        term_to_use="nature",
        curve_label="actual_X_nature",
        df_resid=args.df_resid,
        sort_for_cumulative=True,
    )

    # Placebo estimates: use only X_future72 from placebo folder.
    future_curve, future_summary = build_curves_for_term(
        coef=placebo_coef,
        vcv=placebo_vcv,
        weights=future_weights,
        term_to_use="future",
        curve_label="placebo_X_future72",
        df_resid=args.df_resid,
        sort_for_cumulative=True,
    )

    # For negative exposure controls, cumulate from lead 1 backward and anchor at lag 0.
    future_curve = recompute_future_cumulative_from_event(future_curve)

    diff_actual = build_difference_curve(
        coef=actual_coef,
        vcv=actual_vcv,
        weights=lag_weights,
        term_to_use="nature",
        curve_label="actual_high_minus_lower",
        df_resid=args.df_resid,
        sort_for_cumulative=True,
    )
    diff_future = build_difference_curve(
        coef=placebo_coef,
        vcv=placebo_vcv,
        weights=future_weights,
        term_to_use="future",
        curve_label="future_placebo_high_minus_lower",
        df_resid=args.df_resid,
        sort_for_cumulative=True,
    )

    # Same convention for the placebo high-minus-lower contrast.
    if diff_future is not None:
        diff_future = recompute_future_cumulative_from_event(diff_future)

    rand_means = read_randomization_means_manual()
    randomization_curve = build_randomization_cumulative_curves(rand_means, lag_weights)

    # Create only the requested two-panel cumulative figure. No auxiliary CSVs or other plots are exported.
    plot_cumulative_split_by_group_with_randomizations(
        actual_curve, future_curve, randomization_curve, output_fig_dir, args.fmax, args.lmax
    )

    print("\nDone. Created only the requested two-panel cumulative SVI figure.")
    print(f"Actual coefficient file:  {actual_coef_file}")
    print(f"Placebo coefficient file: {placebo_coef_file}")
    print(f"Actual VCV found:        {actual_vcv is not None}")
    print(f"Placebo VCV found:       {placebo_vcv is not None}")
    print("Randomization means:     manually entered from Table \\ref{tbl:randomization_group_tests}")
    print(f"Randomization curves:    {0 if randomization_curve.empty else randomization_curve['randtype'].nunique()}")
    print("\nActual coefficient summary:")
    print(actual_summary.to_string(index=False))
    print("\nFuture placebo coefficient summary:")
    print(future_summary.to_string(index=False))
    print(f"\nOutputs saved to: {output_fig_dir}")


if __name__ == "__main__":
    main()
