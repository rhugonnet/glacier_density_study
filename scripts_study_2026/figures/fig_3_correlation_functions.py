#!/usr/bin/env python3
"""Main figure 3: spatial and temporal correlation functions."""

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

from study_paths import (
    FIGURES_DIR,
    PARAM_PATH,
    CORRELATION_DIAGNOSTICS_DIR,
    SPATIAL_PARAM_PATH,
    TEMPORAL_PARAM_PATH,
)

# Define project paths and file names

CORR_LABEL = (
    "rho_error_correlation_standardized_residuals_"
    "directcorr_constantspace_temporaldiagnostic"
)

# Define plotting constants
PERIODS_TO_PLOT = [1, 2, 4, 7, 10, 15]
DPI = 300

# Keep typography consistent with the main model figures
mpl.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 10,
        "legend.title_fontsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def add_panel_letter(ax: mpl.axes.Axes, letter: str) -> None:
    """Add a panel label.

    Parameters
    ----------
    ax
        Matplotlib axis to label.
    letter
        Panel letter.
    """
    ax.text(
        0.02,
        0.98,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
    )


def as_boolean(series: pd.Series) -> pd.Series:
    """Convert common CSV boolean representations to Boolean values."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def read_temporal_display_parameters(path: Path) -> tuple[float, float]:
    """Read the fitted positive-lag temporal correlation parameters."""
    params = pd.read_csv(path).iloc[0]
    if "empirical_sill" in params and "empirical_range_yr" in params:
        return float(params["empirical_sill"]), float(params["empirical_range_yr"])
    if "empirical_amplitude_after_nugget" in params and "empirical_timescale_yr" in params:
        return float(params["empirical_amplitude_after_nugget"]), float(params["empirical_timescale_yr"])
    nugget = float(params.get("empirical_nugget", params.get("nugget", 0.70)))
    return 1.0 - nugget, float(params.get("empirical_timescale_yr", 3.0))


# Read saved correlation outputs
corr_dir = CORRELATION_DIAGNOSTICS_DIR

spatial = pd.read_csv(
    corr_dir / f"{CORR_LABEL}_spatial_empirical_aggregated.csv"
)
temporal = pd.read_csv(
    corr_dir / f"{CORR_LABEL}_temporal_empirical_aggregated.csv"
)
temporal_sill, temporal_range = read_temporal_display_parameters(
    TEMPORAL_PARAM_PATH
)

model = RhoSurrogate.from_files(
    PARAM_PATH,
    SPATIAL_PARAM_PATH,
    TEMPORAL_PARAM_PATH,
)


# Select period-length bins to display
available_periods = np.sort(
    spatial["period_years"].dropna().astype(float).unique()
)

preferred_periods = [
    float(period)
    for period in PERIODS_TO_PLOT
    if np.any(np.isclose(available_periods, period))
]

periods = np.asarray(
    preferred_periods if preferred_periods else available_periods,
    dtype=float,
)

if periods.size == 0:
    raise ValueError("No finite period-length bins were found in the spatial output.")


# Prepare axes and colors
cmap = plt.get_cmap("viridis")
norm = mpl.colors.Normalize(
    vmin=float(np.min(periods)),
    vmax=float(np.max(periods)),
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12.2, 5.0),
    constrained_layout=True,
)

ax_s, ax_t = axes


# ---------------------------------------------------------------------
# Spatial correlation
# ---------------------------------------------------------------------

if "used_for_corr" in spatial.columns:
    used_for_corr = as_boolean(spatial["used_for_corr"])
else:
    used_for_corr = pd.Series(True, index=spatial.index)

# Plot empirical estimates separately for each period-length bin
for period in periods:
    sub = spatial.loc[
        np.isclose(
            spatial["period_years"].to_numpy(dtype=float),
            period,
        )
        & used_for_corr
    ].sort_values("lag_center_km")

    if sub.empty:
        continue

    ax_s.scatter(
        sub["lag_center_km"],
        sub["corr"],
        s=24,
        color=cmap(norm(period)),
        alpha=0.75,
        edgecolors="none",
    )

# Plot the single spatial model shared by all periods
x_spatial = np.geomspace(5.0, 10000.0, 700)

ax_s.plot(
    x_spatial,
    model.spatial_corr(x_spatial),
    color="black",
    lw=2.2,
    zorder=5,
)

# Format spatial panel
ax_s.axhline(0, color="black", lw=0.8, ls=":")
ax_s.set_xscale("log")
ax_s.set_xlim(5.0, 10000.0)
ax_s.set_ylim(-0.08, 1.02)
ax_s.set_xlabel(r"Spatial lag $d$ (km)")
ax_s.set_ylabel(r"Spatial error correlation $r_{\rho}^{\rm s}$")
ax_s.grid(alpha=0.25)

# Period legend represents empirical bins only
period_handles = [
    mpl.lines.Line2D(
        [],
        [],
        marker="o",
        linestyle="None",
        markersize=5,
        color=cmap(norm(period)),
        label=f"{period:g} yr",
    )
    for period in periods
]

period_legend = ax_s.legend(
    handles=period_handles,
    title=r"Period length $\Delta t$",
    frameon=False,
    loc="upper right",
)

ax_s.add_artist(period_legend)

ax_s.legend(
    handles=[
        mpl.lines.Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            color="black",
            label="Binned estimates",
        ),
        mpl.lines.Line2D(
            [],
            [],
            lw=2.2,
            color="black",
            label="Common model fit",
        ),
    ],
    frameon=False,
    loc="center right",
)


# ---------------------------------------------------------------------
# Temporal correlation
# ---------------------------------------------------------------------

if "used_for_corr" in temporal.columns:
    temporal_used = as_boolean(temporal["used_for_corr"])
    temporal_plot = temporal.loc[temporal_used].copy()
else:
    temporal_plot = temporal.copy()

temporal_plot = temporal_plot.sort_values("lag_center_yr")

ax_t.scatter(
    temporal_plot["lag_center_yr"],
    temporal_plot["corr"],
    s=32,
    color="black",
    alpha=0.75,
    edgecolors="none",
)

max_temporal_lag = max(
    14.0,
    float(np.nanmax(temporal_plot["lag_center_yr"])),
)

x_temporal = np.linspace(0.0, max_temporal_lag, 700)

ax_t.plot(
    x_temporal,
    temporal_sill * np.exp(-3.0 * x_temporal / temporal_range),
    color="black",
    lw=2.2,
)

# Format temporal panel
ax_t.axhline(0, color="black", lw=0.8, ls=":")
ax_t.set_xlim(0.0, max_temporal_lag)
ax_t.set_ylim(-0.08, 1.02)
ax_t.set_xlabel(r"Temporal lag $\tau$ (yr)")
ax_t.set_ylabel(r"Temporal error correlation $r_{\rho}^{\rm t}$")
ax_t.grid(alpha=0.25)

ax_t.legend(
    handles=[
        mpl.lines.Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            color="black",
            label="Binned estimates",
        ),
        mpl.lines.Line2D(
            [],
            [],
            lw=2.2,
            color="black",
            label="Model fit",
        ),
    ],
    frameon=False,
    loc="upper right",
)


# Add panel labels and save
add_panel_letter(ax_s, "a")
add_panel_letter(ax_t, "b")

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

out = FIGURES_DIR / "FIG_main_03_spatial_temporal_correlation.pdf"
fig.savefig(out, dpi=DPI, bbox_inches="tight")
plt.close(fig)

print(f"[done] Wrote figure: {out}")
