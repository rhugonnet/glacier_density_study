#!/usr/bin/env python3
"""Main figure 1: mean effective density function."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from glacier_density_surrogate import RhoSurrogate

STUDY_DIR = Path(__file__).resolve().parents[1]
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

from study_paths import FIGURES_DIR, PARAM_PATH, TARGET_PREDICTIONS_PATH


# Define project paths

# Define plotting constants
RHO_ICE = 900.0
PERIODS_TO_PLOT = np.array([1, 2, 4, 7, 10, 15], dtype=float)
MEMORY_LABELS = np.array([-4.0, -1.0, -0.25, 0.25, 1.0], dtype=float)
XMAX = 30.0
SIGNED_LINTHRESH = 0.05
SIGNED_ASINH_SCALE = 0.5
DPI = 300

# Keep typography consistent across model figures
mpl.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10,
    "legend.title_fontsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def add_panel_letter(ax, letter: str) -> None:
    """Add a panel label

    :param ax: Matplotlib axis to label
    :param letter: Panel letter
    """
    ax.text(0.02, 0.98, letter, transform=ax.transAxes, ha="left", va="top", fontsize=14, fontweight="bold")


def main_style_legend_handles() -> list[mpl.lines.Line2D]:
    """Return common model and binned-data legend handles"""
    return [
        mpl.lines.Line2D([], [], color="black", lw=2.5, label="Model fit"),
        mpl.lines.Line2D([], [], color="black", marker="o", lw=0, markersize=6, label="Binned estimate"),
    ]


def signed_xgrid(axis_mode: str) -> np.ndarray:
    """Build a signed x-grid without crossing zero

    :param axis_mode: Either ``linear`` or ``log``
    """
    if axis_mode == "linear":
        neg = np.linspace(-XMAX, -SIGNED_LINTHRESH, 1000)
        pos = np.linspace(SIGNED_LINTHRESH, XMAX, 1000)
    else:
        abs_grid = np.geomspace(SIGNED_LINTHRESH, XMAX, 1000)
        neg = -abs_grid[::-1]
        pos = abs_grid
    return np.concatenate([neg, pos])


def plot_signed_curve(ax, x: np.ndarray, y: np.ndarray, color, *, lw: float) -> None:
    """Plot signed branches separately to avoid connecting across zero

    :param ax: Matplotlib axis to draw into
    :param x: Signed x support
    :param y: Model values
    :param color: Matplotlib color
    :param lw: Line width
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    for mask in (ok & (x < 0), ok & (x > 0)):
        ax.plot(x[mask], y[mask], color=color, lw=lw, alpha=0.95)


def signed_asinh_forward(x: np.ndarray) -> np.ndarray:
    """Transform signed values with a smooth near-zero compression

    :param x: Values in elevation change units
    """
    return np.arcsinh(np.asarray(x, dtype=float) / SIGNED_ASINH_SCALE)


def signed_asinh_inverse(x: np.ndarray) -> np.ndarray:
    """Invert the signed asinh transform

    :param x: Transformed values
    """
    return SIGNED_ASINH_SCALE * np.sinh(np.asarray(x, dtype=float))


def set_xaxis(ax, axis_mode: str) -> None:
    """Apply the requested x-axis mode

    :param ax: Matplotlib axis to update
    :param axis_mode: Either ``linear`` or ``log``
    """
    if axis_mode == "linear":
        ticks = np.array([-30, -15, 0, 15, 30], dtype=float)
        ax.set_xscale("linear")
    else:
        ticks = np.array([-30, -10, -2, -0.5, 0, 0.5, 2, 10, 30], dtype=float)
        ax.set_xscale("function", functions=(signed_asinh_forward, signed_asinh_inverse))
    ax.set_xlim(-XMAX, XMAX)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{tick:g}" for tick in ticks])


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute a finite weighted mean

    :param values: Values to average
    :param weights: Non-negative weights
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(ok):
        return np.nan
    return float(np.sum(values[ok] * weights[ok]) / np.sum(weights[ok]))


def build_neutral_panel_table(target: pd.DataFrame, model: RhoSurrogate) -> pd.DataFrame:
    """Build the transformed neutral-memory period panel from binned targets

    :param target: Binned prediction table saved by the fit script
    :param model: Surrogate model used for the fitted memory contribution
    """
    d = target.copy()
    current = d["current_center_w"].to_numpy(float)
    memory = d["memory_center_w"].to_numpy(float)
    period = d["period_years"].to_numpy(float)
    d["rho_mem0"] = d["rho_mean_w"].to_numpy(float) - (
        model.mu_rho(current, past_dhdt=memory, dt=period)
        - model.mu_rho(current, past_dhdt=0.0, dt=period)
    )

    rows = []
    for (period_years, current_bin), group in d.groupby(["period_years", "current_bin"], sort=True):
        weights = group["weight_sum"].to_numpy(float)
        if not np.isfinite(weights).any() or np.nansum(weights) <= 0:
            continue
        current_center = weighted_mean(group["current_center_w"].to_numpy(float), weights)
        period_center = weighted_mean(group["period_years"].to_numpy(float), weights)
        rows.append(
            {
                "period_years": float(period_years),
                "period_center_w": period_center,
                "current_bin": int(current_bin),
                "current_center_w": current_center,
                "rho_obs_w": weighted_mean(group["rho_mem0"].to_numpy(float), weights),
                "rho_pred_center": float(model.mu_rho(current_center, past_dhdt=0.0, dt=period_center)),
                "weight_sum": float(np.nansum(weights)),
                "n": int(np.nansum(group["n"].to_numpy(float))) if "n" in group.columns else len(group),
                "n_eff": float(np.nansum(group["n_eff"].to_numpy(float))) if "n_eff" in group.columns else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["period_years", "current_bin"]).reset_index(drop=True)


def plot_figure(axis_mode: str, out_path: Path) -> None:
    """Plot and save the mean function figure

    :param axis_mode: Either ``linear`` or ``log``
    :param out_path: Output PNG path
    """
    target_path = TARGET_PREDICTIONS_PATH
    param_path = PARAM_PATH
    target = pd.read_csv(target_path)
    target = target.loc[target["support_ok"].astype(bool)].copy()
    model = RhoSurrogate.from_files(parameter_path=param_path, spatial_path=None, temporal_path=None)

    # Build canvas and shared model support
    xgrid = signed_xgrid(axis_mode)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.55), constrained_layout=True, sharey=True)
    ax_mem, ax_per = axes

    # Plot past elevation change rate dependence at fixed five-year period
    periods_available = np.sort(target["period_years"].dropna().unique().astype(float))
    period5_value = periods_available[int(np.argmin(np.abs(periods_available - 5.0)))]
    period5 = target.loc[np.isclose(target["period_years"], period5_value)].copy()
    memory_lookup = (
        period5.groupby("memory_bin", as_index=False)
        .agg(memory_center_ref=("memory_center_w", "mean"))
        .sort_values("memory_center_ref")
    )
    memory_bins: list[int] = []
    memory_actual: dict[int, float] = {}
    memory_labels: dict[int, float] = {}
    for label in MEMORY_LABELS:
        idx = int(np.argmin(np.abs(memory_lookup["memory_center_ref"].to_numpy(float) - label)))
        memory_bin = int(memory_lookup["memory_bin"].iloc[idx])
        if memory_bin not in memory_bins:
            memory_bins.append(memory_bin)
            memory_actual[memory_bin] = float(memory_lookup["memory_center_ref"].iloc[idx])
            memory_labels[memory_bin] = float(label)
    cmap_mem = plt.get_cmap("coolwarm_r")
    norm_mem = mpl.colors.TwoSlopeNorm(vmin=-4.0, vcenter=0.0, vmax=4.0)
    for memory_bin in memory_bins:
        label_value = memory_labels[memory_bin]
        memory_value = memory_actual[memory_bin]
        color = cmap_mem(norm_mem(label_value))
        sub = period5.loc[period5["memory_bin"].astype(int) == memory_bin].copy()
        sub = sub.loc[np.abs(sub["current_center_w"].to_numpy(float)) >= 0.2].sort_values("current_center_w")
        ax_mem.scatter(sub["current_center_w"], sub["rho_mean_w"], s=28, color=color, alpha=0.70, linewidths=0)
        plot_signed_curve(ax_mem, xgrid, model.mu_rho(xgrid, past_dhdt=memory_value, dt=5.0), color, lw=2.0)

    # Format past elevation change rate panel
    ax_mem.axhline(RHO_ICE, color="black", lw=1, ls="--")
    ax_mem.axvline(0, color="black", lw=1)
    set_xaxis(ax_mem, axis_mode)
    ax_mem.set_ylim(-100.0, 1700.0)
    ax_mem.set_xlabel(r"Elevation change $\Delta h$ (m)")
    ax_mem.set_ylabel(r"Mean of effective density $\mu_{\rho}$ (kg m$^{-3}$)")
    ax_mem.text(0.98, 0.98, r"Period length $\Delta t = 5$ yr", transform=ax_mem.transAxes, ha="right", va="top", fontsize=10)
    ax_mem.grid(alpha=0.25)

    # Add memory and model legends
    mem_handles = [
        mpl.lines.Line2D([], [], marker="o", ms=5, lw=1.8, color=cmap_mem(norm_mem(float(label))), label=rf"{label:g} m yr$^{{-1}}$")
        for label in MEMORY_LABELS
    ]
    leg_mem = ax_mem.legend(handles=mem_handles, title="Past elevation\nchange rate " + r"$\dot{h}^{\rm p}$", frameon=False, loc="lower left")
    ax_mem.add_artist(leg_mem)
    ax_mem.legend(handles=main_style_legend_handles(), frameon=False, loc="lower right")

    # Plot period length dependence at neutral past elevation change rate
    neutral = build_neutral_panel_table(target, model)
    neutral = neutral.loc[np.abs(neutral["current_center_w"].to_numpy(float)) >= 0.5].copy()
    periods_available = neutral["period_years"].dropna().unique().astype(float)
    periods = np.array([period for period in PERIODS_TO_PLOT if np.any(np.isclose(periods_available, period))], dtype=float)
    cmap_per = plt.get_cmap("viridis")
    norm_per = mpl.colors.Normalize(vmin=float(np.min(periods)), vmax=float(np.max(periods)))
    for period in periods:
        color = cmap_per(norm_per(period))
        sub = neutral.loc[np.isclose(neutral["period_years"], period)].sort_values("current_center_w")
        ax_per.scatter(sub["current_center_w"], sub["rho_obs_w"], s=24, color=color, alpha=0.75, edgecolors="none")
        plot_signed_curve(ax_per, xgrid, model.mu_rho(xgrid, past_dhdt=0.0, dt=float(period)), color, lw=1.8)

    # Format period length panel
    ax_per.axhline(RHO_ICE, color="black", lw=1, ls="--")
    ax_per.axvline(0, color="black", lw=1)
    set_xaxis(ax_per, axis_mode)
    ax_per.set_ylim(-100.0, 1700.0)
    ax_per.set_xlabel(r"Elevation change $\Delta h$ (m)")
    ax_per.text(
        0.98,
        0.98,
        "Past elevation change rate\n" + r"$\dot{h}^{\rm p} = 0$ m yr$^{-1}$",
        transform=ax_per.transAxes,
        ha="right",
        va="top",
        fontsize=10,
    )
    ax_per.grid(alpha=0.25)

    # Add period and model legends
    period_handles = [
        mpl.lines.Line2D([], [], marker="o", ms=5, lw=1.8, color=cmap_per(norm_per(float(period))), label=f"{period:g} yr")
        for period in periods
    ]
    leg_per = ax_per.legend(handles=period_handles, title=r"Period length $\Delta t$", frameon=False, loc="lower left")
    ax_per.add_artist(leg_per)
    ax_per.legend(handles=main_style_legend_handles(), frameon=False, loc="lower right")

    # Add panel labels and save
    add_panel_letter(ax_mem, "a")
    add_panel_letter(ax_per, "b")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] Wrote figure: {out_path}")


# Save both x-axis variants
plot_figure("log", FIGURES_DIR / "FIG_main_01_rho_mean_function_log_x.pdf")
plot_figure("linear", FIGURES_DIR / "FIG_main_01_rho_mean_function_linear_x.pdf")
