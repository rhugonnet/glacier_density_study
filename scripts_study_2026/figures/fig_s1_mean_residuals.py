#!/usr/bin/env python3
"""Supplementary figure S1: residuals for the mean function."""

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
RHO_ICE = 900.0
SIGNED_LINTHRESH = 0.05
DPI = 300

# Keep typography consistent with fit-script figures
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
    ticks = np.array([-50.0, -10.0, -2.0, -0.5, 0.0, 0.5, 2.0, 10.0, 50.0], dtype=float)
    ax.set_xscale("symlog", linthresh=SIGNED_LINTHRESH, linscale=0.06, base=10)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])


def add_panel_letter(ax, letter: str) -> None:
    """Add a panel label

    :param ax: Matplotlib axis to label
    :param letter: Panel letter
    """
    ax.text(0.02, 0.98, letter, transform=ax.transAxes, ha="left", va="top", fontsize=14, fontweight="bold")


def apply_area_log_ticks(ax, values: np.ndarray) -> None:
    """Apply readable log ticks for glacier area

    :param ax: Matplotlib axis to update
    :param values: Area values shown on the axis
    """
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if len(vals) == 0:
        return
    ticks = []
    for pwr in range(int(np.floor(np.log10(vals.min()))), int(np.ceil(np.log10(vals.max()))) + 1):
        for mult in (1, 2, 5):
            tick = mult * 10 ** pwr
            if vals.min() * 0.99 <= tick <= vals.max() * 1.01:
                ticks.append(tick)
    ax.set_xticks(sorted(set(ticks)))
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: f"{x:g}" if x > 0 else ""))
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())


def plot_panel(ax, data: pd.DataFrame, ycol: str, periods: np.ndarray, xlabel: str, ylabel: str, *, hline=None, ylim=None, signed=False, logx=False):
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
                ss = sub.loc[mask].sort_values("factor_center_w")
                if len(ss) >= 2:
                    ax.plot(ss["factor_center_w"], ss[ycol], marker="o", ms=3.0, lw=1.0, color=color)
        elif len(sub) >= 2:
            ax.plot(sub["factor_center_w"], sub[ycol], marker="o", ms=3.0, lw=1.0, color=color)
    if hline is not None:
        ax.axhline(hline, color="black", lw=0.9, ls="--")
    if signed:
        ax.axvline(0, color="black", lw=0.8)
        set_signed_log_xaxis(ax)
    if logx:
        ax.set_xscale("log")
        apply_area_log_ticks(ax, data["factor_center_w"].to_numpy(float))
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
diag = diag.loc[~(diag["factor"].eq("Current signed dh") & (np.abs(diag["factor_center_w"]) < 0.2))].copy()
available_periods = np.sort(np.asarray(diag["period_years"].dropna().unique(), dtype=float))
preferred_periods = [float(v) for v in PERIODS_TO_PLOT if np.any(np.isclose(available_periods, v))]
periods = np.asarray(preferred_periods if preferred_periods else available_periods, dtype=float)

# Use matched top and bottom y extents around their reference levels
raw = diag["raw_mean_w"].to_numpy(float)
raw = raw[np.isfinite(raw)]
resid = diag["resid_mean_w"].to_numpy(float)
resid = resid[np.isfinite(resid)]
halfspan = 100.0
if len(raw):
    halfspan = max(halfspan, float(np.nanmax(np.abs(raw - RHO_ICE))))
if len(resid):
    halfspan = max(halfspan, float(np.nanmax(np.abs(resid))))
halfspan *= 1.05
ylim_top = (RHO_ICE - halfspan, RHO_ICE + halfspan)
ylim_bottom = (-halfspan, halfspan)

# Draw raw and residual panels
fig, axes = plt.subplots(2, 3, figsize=(12.6, 6.6), constrained_layout=True, sharey="row")
sm = None
letters = list("abcdef")
letter_i = 0
for j, (factor, xlabel, signed, logx) in enumerate(specs):
    sub = diag.loc[diag["factor"] == factor]
    ylab_top = "Binned mean of\neffective density " + r"$\mu_{\rho}$" + "\n" + r"(kg m$^{-3}$)" if j == 0 else ""
    sm = plot_panel(axes[0, j], sub, "raw_mean_w", periods, "", ylab_top, hline=RHO_ICE, ylim=ylim_top, signed=signed, logx=logx)
    ylab_bottom = "Residual mean after\nfitted model " + r"(kg m$^{-3}$)" if j == 0 else ""
    plot_panel(axes[1, j], sub, "resid_mean_w", periods, xlabel, ylab_bottom, hline=0, ylim=ylim_bottom, signed=signed, logx=logx)
    add_panel_letter(axes[0, j], letters[letter_i]); letter_i += 1
    add_panel_letter(axes[1, j], letters[letter_i]); letter_i += 1

# Add colorbar and save
if sm is not None:
    sm.set_array([])
    fig.colorbar(sm, ax=axes, pad=0.015, shrink=0.95).set_label(r"Period length $\Delta t$ (yr)")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
out = FIGURES_DIR / "FIG_supp_01_mean_residuals.pdf"
fig.savefig(out, dpi=DPI, bbox_inches="tight")
plt.close(fig)
print(f"[done] Wrote figure: {out}")
