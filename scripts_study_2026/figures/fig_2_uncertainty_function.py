#!/usr/bin/env python3
"""Main figure 2: effective-density uncertainty function."""

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

from study_paths import FIGURES_DIR, PARAM_PATH, SIGMA_ABSDH_TABLE_PATH


# Define project paths

# Define plotting constants
PERIODS_TO_PLOT = [1, 2, 4, 7, 10, 15]
DPI = 300
MIN_LOW_TAIL_NEFF_FRACTION_FOR_DISPLAY = 0.04
MIN_HIGH_TAIL_NEFF_FRACTION_FOR_DISPLAY = 0.35
MIN_TAIL_NEFF_FOR_DISPLAY = 800.0
LINEAR_XLIM = (0.0, 30.0)
LINEAR_FIT_XLIM = (0.02, 60.0)
LOG_XLIM = (0.1, 30.0)
LOG_FIT_XLIM = (0.05, 30.0)
YLIM = (10.0, 3000.0)

# Keep typography consistent with fit-script figures
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


# Read saved binned uncertainty outputs
sigma_path = SIGMA_ABSDH_TABLE_PATH
param_path = PARAM_PATH
tab = pd.read_csv(sigma_path)
model = RhoSurrogate.from_files(parameter_path=param_path, spatial_path=None, temporal_path=None)

# Filter valid binned values
tab = tab.loc[np.isfinite(tab["abs_center_w"]) & np.isfinite(tab["resid_std_w"]) & (tab["abs_center_w"] > 0)].copy()
available_periods = np.sort(np.asarray(tab["period_years"].dropna().unique(), dtype=float))
preferred_periods = [float(v) for v in PERIODS_TO_PLOT if np.any(np.isclose(available_periods, v))]
periods = np.asarray(preferred_periods if preferred_periods else available_periods, dtype=float)
cmap = plt.get_cmap("viridis")
norm = mpl.colors.Normalize(vmin=float(np.min(periods)), vmax=float(np.max(periods)))

def set_decimal_log_ticks(ax, axis: str = "both") -> None:
    """Apply decimal labels to log axes

    :param ax: Matplotlib axis to update
    :param axis: Axis selector, either ``x``, ``y`` or ``both``
    """
    xticks = [0.1, 0.3, 1, 3, 10, 30]
    yticks = [10, 30, 100, 300, 1000, 3000]
    formatter = mpl.ticker.FuncFormatter(lambda value, pos: f"{value:g}" if value > 0 else "")
    if axis in ("x", "both"):
        ax.set_xticks(xticks)
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    if axis in ("y", "both"):
        ax.set_yticks(yticks)
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())


def plot_uncertainty(axis_mode: str, out_path: Path) -> None:
    """Plot and save the uncertainty-function figure

    :param axis_mode: Either ``linear`` or ``log``
    :param out_path: Output PNG path
    """
    if axis_mode == "linear":
        xline = np.linspace(*LINEAR_FIT_XLIM, 700)
    else:
        xline = np.geomspace(*LOG_FIT_XLIM, 700)

    # Draw binned values and model curves
    fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
    for period in periods:
        sub = tab.loc[np.isclose(tab["period_years"], period)].sort_values("abs_center_w")
        sub_plot = sub
        if "n_eff" in sub.columns and not sub.empty:
            n_eff = sub["n_eff"].to_numpy(float)
            if np.any(np.isfinite(n_eff)):
                peak_position = int(np.nanargmax(n_eff))
                max_n_eff = float(np.nanmax(n_eff))
                low_tail_threshold = max(MIN_TAIL_NEFF_FOR_DISPLAY, MIN_LOW_TAIL_NEFF_FRACTION_FOR_DISPLAY * max_n_eff)
                high_tail_threshold = max(MIN_TAIL_NEFF_FOR_DISPLAY, MIN_HIGH_TAIL_NEFF_FRACTION_FOR_DISPLAY * max_n_eff)
                keep = np.ones(len(sub), dtype=bool)
                keep[:peak_position] = n_eff[:peak_position] >= low_tail_threshold
                keep[peak_position + 1 :] = n_eff[peak_position + 1 :] >= high_tail_threshold
                sub_plot = sub.loc[keep]
        color = cmap(norm(period))
        if not sub_plot.empty:
            ax.scatter(sub_plot["abs_center_w"], sub_plot["resid_std_w"], s=24, color=color, alpha=0.75, edgecolors="none")
        ax.plot(xline, model.sigma_rho(xline, dt=float(period)), color=color, lw=1.8)

    # Format axes
    ax.set_xscale(axis_mode)
    ax.set_yscale("log")
    ax.set_xlabel(r"Absolute elevation change $|\Delta h|$ (m)")
    ax.set_ylabel(r"Uncertainty in effective density $\sigma_{\rho}$ (kg m$^{-3}$)")
    ax.set_ylim(*YLIM)
    ax.grid(alpha=0.25)
    if axis_mode == "linear":
        ax.set_xlim(*LINEAR_XLIM)
        ax.set_xticks([0, 10, 20, 30])
        set_decimal_log_ticks(ax, axis="y")
    else:
        ax.set_xlim(*LOG_XLIM)
        set_decimal_log_ticks(ax, axis="both")

    # Add period and style legends
    handles = [mpl.lines.Line2D([], [], marker="o", lw=1.8, color=cmap(norm(period)), label=f"{period:g} yr") for period in periods]
    leg1 = ax.legend(handles=handles, title=r"Period length $\Delta t$", frameon=False, loc="upper right")
    ax.add_artist(leg1)
    style_handles = [
        mpl.lines.Line2D([], [], color="black", lw=2.5, label="Model fit"),
        mpl.lines.Line2D([], [], color="black", marker="o", lw=0, markersize=6, label="Binned estimate"),
    ]
    ax.legend(handles=style_handles, frameon=False, loc="lower left")

    # Save the requested x-axis variant
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] Wrote figure: {out_path}")


# Save both x-axis variants
plot_uncertainty("log", FIGURES_DIR / "FIG_main_02_rho_uncertainty_function_log_x.pdf")
plot_uncertainty("linear", FIGURES_DIR / "FIG_main_02_rho_uncertainty_function_linear_x.pdf")
