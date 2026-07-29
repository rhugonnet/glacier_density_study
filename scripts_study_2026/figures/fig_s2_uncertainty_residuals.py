#!/usr/bin/env python3
"""Supplementary figure S2: residuals for the uncertainty function."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STUDY_DIR = Path(__file__).resolve().parents[1]
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

from study_paths import FIGURES_DIR, FACTOR_DIAGNOSTICS_PATH

# Define project paths

# Define plotting constants
PERIODS_TO_PLOT = [1, 2, 4, 7, 10, 15]
SIGNED_LINTHRESH = 0.05
DPI = 300

# Keep typography consistent with the supplemental diagnostics
mpl.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "legend.title_fontsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def set_signed_log_xaxis(ax) -> None:
    """Apply the signed-log x axis used for elevation change

    :param ax: Matplotlib axis to update
    """
    ticks = np.array([-50, -20, -10, -5, -2, -1, -0.2, 0, 0.2, 1, 2, 5, 10, 20, 50], dtype=float)
    ax.set_xscale("symlog", linthresh=SIGNED_LINTHRESH, linscale=0.06, base=10)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])


def add_panel_letter(ax, letter: str) -> None:
    """Add a panel label

    :param ax: Matplotlib axis to label
    :param letter: Panel letter
    """
    ax.text(0.02, 0.98, letter, transform=ax.transAxes, ha="left", va="top", fontsize=14, fontweight="bold")


def plot_panel(ax, data: pd.DataFrame, ycol: str, periods: np.ndarray, xlabel: str, ylabel: str, *, hline=None, signed=False, logx=False, ylim=None):
    """Plot one diagnostic panel

    :param ax: Matplotlib axis to draw into
    :param data: Binned diagnostic table
    :param ycol: Column to plot on the y axis
    :param periods: Periods to show
    :param xlabel: X-axis label
    :param ylabel: Y-axis label
    """
    cmap = plt.get_cmap("viridis")
    norm = mpl.colors.Normalize(vmin=float(np.min(periods)), vmax=float(np.max(periods)))
    for period in periods:
        sub = data.loc[np.isclose(data["period_years"], period)].sort_values("factor_center_w")
        color = cmap(norm(period))
        if signed:
            for mask in [sub["factor_center_w"] < 0, sub["factor_center_w"] > 0]:
                ss = sub.loc[mask]
                ax.plot(ss["factor_center_w"], ss[ycol], marker="o", ms=3, lw=1.0, color=color)
        else:
            ax.plot(sub["factor_center_w"], sub[ycol], marker="o", ms=3, lw=1.0, color=color)
    if hline is not None:
        ax.axhline(hline, color="black", lw=0.9, ls="--")
    if signed:
        ax.axvline(0, color="black", lw=0.8)
        set_signed_log_xaxis(ax)
    if logx:
        ax.set_xscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    return mpl.cm.ScalarMappable(cmap=cmap, norm=norm)


# Read saved factor diagnostics
diag = pd.read_csv(FACTOR_DIAGNOSTICS_PATH)
diag["factor"] = diag["factor"].replace(
    {
        "Model memory dh": "Past elevation change rate",
        "Past elevation change": "Past elevation change rate",
    }
)
specs = [
    ("Current signed dh", r"Elevation change $\Delta h$ (m)", True, False),
    ("Past elevation change rate", r"Past elevation change rate $\dot{h}^{\rm p}$ (m yr$^{-1}$)", False, False),
    ("Area", r"Glacier area $A$", False, True),
]

# Filter diagnostics to displayed factors
diag = diag.loc[diag["factor"].isin([s[0] for s in specs])].copy()
diag = diag.loc[~(diag["factor"].eq("Current signed dh") & (np.abs(diag["factor_center_w"]) < 0.2))]
available_periods = np.sort(np.asarray(diag["period_years"].dropna().unique(), dtype=float))
preferred_periods = [float(v) for v in PERIODS_TO_PLOT if np.any(np.isclose(available_periods, v))]
periods = np.asarray(preferred_periods if preferred_periods else available_periods, dtype=float)
fig, axes = plt.subplots(2, 3, figsize=(12.6, 6.6), constrained_layout=True, sharey="row")

# Draw raw spread and standardized spread panels
sm = None
for j, (factor, xlabel, signed, logx) in enumerate(specs):
    sub = diag.loc[diag["factor"] == factor]
    ylab_top = "Binned uncertainty in\neffective density " + r"$\sigma_{\rho}$" + "\n" + r"(kg m$^{-3}$)" if j == 0 else ""
    sm = plot_panel(axes[0, j], sub, "raw_std_w", periods, "", ylab_top, signed=signed, logx=logx)
    ycol = "z_after_std_w" if "z_after_std_w" in sub.columns else "resid_std_w"
    ylab_bottom = "STD after standardization\nby fitted model " + r"$z_{\rho}$" if j == 0 else ""
    plot_panel(axes[1, j], sub, ycol, periods, xlabel, ylab_bottom, hline=1, signed=signed, logx=logx, ylim=(0, 5))
    add_panel_letter(axes[0, j], chr(ord("a") + j))
    add_panel_letter(axes[1, j], chr(ord("d") + j))

# Add colorbar and save
if sm is not None:
    fig.colorbar(sm, ax=axes, pad=0.015, shrink=0.95).set_label(r"Period length $\Delta t$ (yr)")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
out = FIGURES_DIR / "FIG_supp_02_uncertainty_residuals.pdf"
fig.savefig(out, dpi=DPI, bbox_inches="tight")
plt.close(fig)
print(f"[done] Wrote figure: {out}")
