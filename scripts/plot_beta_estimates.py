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
    output/figures/figure2_svi_county_actual_future.png
    output/figures/figure2_svi_county_actual_future.pdf
    output/figures/figure2_svi_cumulative_actual_future.png
    output/figures/figure2_svi_cumulative_actual_future.pdf
    output/figures/figure2_svi_actual_plot_data.csv
    output/figures/figure2_svi_future_placebo_plot_data.csv
    output/figures/figure2_svi_actual_beta_summary.csv
    output/figures/figure2_svi_future_placebo_beta_summary.csv

Run example
-----------
python scripts/plot_beta_estimates.py

python scripts/plot_beta_estimates.py ^
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
LEGEND_FONTSIZE = 10

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


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    repo_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Plot actual SVI-group TC response and future-exposure placebo leads.")
    parser.add_argument("--base-dir", type=str, default=repo_dir, help="Project base directory.")
    parser.add_argument("--output-subdir", type=str, default="output")
    parser.add_argument("--actual-folder", type=str, default="actual")
    parser.add_argument("--placebo-folder", type=str, default="placebo")
    parser.add_argument("--group-var", type=str, default="svi_high")
    parser.add_argument("--lmax", type=int, default=172)
    parser.add_argument("--fmax", type=int, default=12)
    parser.add_argument("--df-resid", type=float, default=41)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = base_dir / args.output_subdir
    actual_beta_dir = output_dir / args.actual_folder / "beta"
    placebo_beta_dir = output_dir / args.placebo_folder / "beta"
    output_fig_dir = output_dir / "figures"
    output_fig_dir.mkdir(parents=True, exist_ok=True)

    if not actual_beta_dir.exists():
        raise FileNotFoundError(f"Actual beta folder not found: {actual_beta_dir}")
    if not placebo_beta_dir.exists():
        raise FileNotFoundError(f"Placebo beta folder not found: {placebo_beta_dir}")

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

    # Save data.
    actual_curve.to_csv(output_fig_dir / "figure2_svi_actual_plot_data.csv", index=False)
    future_curve.to_csv(output_fig_dir / "figure2_svi_future_placebo_plot_data.csv", index=False)
    actual_summary.to_csv(output_fig_dir / "figure2_svi_actual_beta_summary.csv", index=False)
    future_summary.to_csv(output_fig_dir / "figure2_svi_future_placebo_beta_summary.csv", index=False)
    if diff_actual is not None:
        diff_actual.to_csv(output_fig_dir / "figure2_svi_actual_difference_plot_data.csv", index=False)
    if diff_future is not None:
        diff_future.to_csv(output_fig_dir / "figure2_svi_future_placebo_difference_plot_data.csv", index=False)

    plot_main_figure(actual_curve, future_curve, diff_actual, diff_future, output_fig_dir, args.lmax, args.fmax)
    plot_cumulative_only(actual_curve, future_curve, output_fig_dir, args.fmax, args.lmax)

    print("\nDone. Created county Figure-2-style SVI plots using actual estimates and future placebo leads only.")
    print(f"Actual coefficient file:  {actual_coef_file}")
    print(f"Placebo coefficient file: {placebo_coef_file}")
    print(f"Actual VCV found:        {actual_vcv is not None}")
    print(f"Placebo VCV found:       {placebo_vcv is not None}")
    print("\nActual coefficient summary:")
    print(actual_summary.to_string(index=False))
    print("\nFuture placebo coefficient summary:")
    print(future_summary.to_string(index=False))
    print(f"\nOutputs saved to: {output_fig_dir}")


if __name__ == "__main__":
    main()
