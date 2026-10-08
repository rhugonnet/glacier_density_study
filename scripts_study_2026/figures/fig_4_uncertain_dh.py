#!/usr/bin/env python3
"""Main figure 4: surrogate integration over uncertain elevation change."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from glacier_density_surrogate import RhoSurrogate

STUDY_DIR = Path(__file__).resolve().parents[1]
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

from study_paths import FIGURES_DIR, PARAM_PATH


# Define project paths
OUTDIR = FIGURES_DIR
PARAM_CSV = PARAM_PATH

# Embed editable TrueType fonts in PDF outputs
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

# Define the evaluated scenario
DT_YR = 1.0
MEASURED_PAST_DH = 0.0
SIGMA_PAST_DH = 0.0

# Define grid support
DH_MIN = 0.1
DH_MAX = 100.0
SIGMA_DH_MIN = 0.1
SIGMA_DH_MAX = 100.0
LOG_DH_N = 1800
LOG_SIGMA_N = 1800
GH_ORDER = 256

# Define contour levels
MEAN_CONTOUR_LEVELS = np.array([550, 575, 600, 625, 650, 700, 750, 800, 850, 900], dtype=float)
SIGMA_CONTOUR_LEVELS = np.array([15, 25, 50, 75, 100, 150, 200, 300, 500, 700, 1000, 1500, 2000, 3000], dtype=float)

# Define output names
FIGURE_SIZE = (11.5, 4.8)
FIGURE_NAME = "FIG_main_04_uncertain_dh.pdf"

# Define plot typography
CONTOUR_LINEWIDTH = 1.4
CONTOUR_LABEL_FONTSIZE = 10
CONTOUR_LABEL_HALO_LINEWIDTH = 2.4
AXIS_LABEL_FONTSIZE = 13
TICK_LABEL_FONTSIZE = 12
PANEL_LABEL_FONTSIZE = 14
PARAMETER_TEXT_FONTSIZE = 10
COLORBAR_LABEL_FONTSIZE = 12
COLORBAR_TICK_FONTSIZE = 8.5
GRID_LINEWIDTH = 0.35
GRID_ALPHA = 0.35
MEAN_PLOT_SMOOTHING_SIGMA = 2.0

# Define color scales
MEAN_CMAP_NAME = "Blues"
SIGMA_CMAP_NAME = "OrRd"
MEAN_CMIN = 0.35
MEAN_CMAX = 0.95
MEAN_GAMMA = 0.80
SIGMA_CMIN = 0.30
SIGMA_CMAX = 0.95
SIGMA_GAMMA = 0.80
MEAN_LABEL_COLOR = "#1F4e79"
SIGMA_LABEL_COLOR = "#8C2d04"


def sampled_contour_colors(cmap_name: str, n: int, *, cmin: float, cmax: float, gamma: float):
    """Sample contour colors

    :param cmap_name: Matplotlib colormap name
    :param n: Number of colors
    :param cmin: Minimum colormap coordinate
    :param cmax: Maximum colormap coordinate
    :param gamma: Nonlinear sampling exponent
    """
    cmap = plt.get_cmap(cmap_name)
    coords = cmin + (cmax - cmin) * (np.linspace(0.0, 1.0, n) ** gamma)
    return [cmap(value) for value in coords]


def compute_grid(
    model: RhoSurrogate,
    dh_values: np.ndarray,
    sigma_values: np.ndarray,
    dh_p: float,
    sigma_dh_p: float,
    dt: float,
) -> dict[str, np.ndarray]:
    """Compute the uncertain-dh integration grid

    :param model: Surrogate model instance
    :param dh_values: Elevation change support
    :param sigma_values: Elevation change uncertainty support
    :param dh_p: Past elevation change rate
    :param sigma_dh_p: Past elevation change rate uncertainty
    :param dt: Period length in years
    """
    mean = np.empty((sigma_values.size, dh_values.size), dtype=float)
    sigma_rho = np.empty_like(mean)
    for iy, sigma_dh in enumerate(sigma_values):
        for ix, dh in enumerate(dh_values):
            mean[iy, ix] = model.integrated_mu(
                dh=dh,
                sigma_dh=float(sigma_dh),
                dh_p=dh_p,
                sigma_dh_p=sigma_dh_p,
                dt=dt,
            )
            sigma_rho[iy, ix] = model.integrated_sigma(
                dh=dh,
                sigma_dh=float(sigma_dh),
                dt=dt,
            )
    return {"dh": dh_values, "sigma_dh": sigma_values, "mu": mean, "sigma": sigma_rho}


def _normalize_contour_x(x: np.ndarray) -> np.ndarray:
    return np.log10(x / DH_MIN) / math.log10(DH_MAX / DH_MIN)


def _automatic_contour_label_positions(contour_set, *, vertical_fraction_overrides=None):
    log_ymin = math.log10(SIGMA_DH_MIN)
    log_ymax = math.log10(SIGMA_DH_MAX)
    overrides = vertical_fraction_overrides or {}
    positions = {}
    for level, segments in zip(contour_set.levels, contour_set.allsegs):
        candidates = []
        for segment in segments:
            if segment.shape[0] < 2:
                continue
            x = segment[:, 0]
            y = segment[:, 1]
            valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0) & (x >= DH_MIN) & (x <= DH_MAX)
            if valid.sum() < 2:
                continue
            x = x[valid]
            y = y[valid]
            xn = _normalize_contour_x(x)
            yn = (np.log10(y) - log_ymin) / (log_ymax - log_ymin)
            prominence = max(float(np.ptp(xn)), float(np.ptp(yn))) + 0.25 * float(np.sum(np.hypot(np.diff(xn), np.diff(yn))))
            candidates.append((prominence, x, y, xn, yn))
        if not candidates:
            continue
        _, x, y, xn, yn = max(candidates, key=lambda item: item[0])
        level_value = float(level)
        if level_value in overrides:
            fraction = float(np.clip(overrides[level_value], 0.0, 1.0))
            target_y = yn.min() + fraction * (yn.max() - yn.min())
            target_x = 0.5 * (xn.min() + xn.max())
            distance = np.abs(yn - target_y) + 0.15 * np.abs(xn - target_x)
        else:
            target_x = 0.5 * (xn.min() + xn.max())
            target_y = 0.5 * (yn.min() + yn.max())
            distance = np.hypot(xn - target_x, yn - target_y)
        edge_penalty = np.clip(0.06 - xn, 0, None) + np.clip(xn - 0.94, 0, None) + np.clip(0.06 - yn, 0, None) + np.clip(yn - 0.94, 0, None)
        positions[level_value] = (float(x[int(np.argmin(distance + 4.0 * edge_penalty))]), float(y[int(np.argmin(distance + 4.0 * edge_penalty))]))
    return positions


def _format_log_ticks(values: list[float]) -> list[str]:
    return [f"{value:g}" for value in values]


def plot_figure(grid: dict[str, np.ndarray], dt: float, dh_p: float, sigma_dh_p: float, outdir: Path) -> Path:
    dh = grid["dh"]
    sigma_dh = grid["sigma_dh"]
    dh_mesh, sigma_mesh = np.meshgrid(dh, sigma_dh)
    mean = gaussian_filter(grid["mu"], sigma=MEAN_PLOT_SMOOTHING_SIGMA)
    sigma = grid["sigma"]

    mean_colors = sampled_contour_colors(MEAN_CMAP_NAME, len(MEAN_CONTOUR_LEVELS), cmin=MEAN_CMIN, cmax=MEAN_CMAX, gamma=MEAN_GAMMA)
    sigma_colors = sampled_contour_colors(SIGMA_CMAP_NAME, len(SIGMA_CONTOUR_LEVELS), cmin=SIGMA_CMIN, cmax=SIGMA_CMAX, gamma=SIGMA_GAMMA)

    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE, constrained_layout=True)
    panels = [
        (axes[0], mean, MEAN_CONTOUR_LEVELS, mean_colors, MEAN_LABEL_COLOR, r"Integrated mean of effective density $\bar{\mu}_{\rho}$ (kg m$^{-3}$)", "a", "linear"),
        (axes[1], sigma, SIGMA_CONTOUR_LEVELS, sigma_colors, SIGMA_LABEL_COLOR, "Integrated uncertainty of effective\n"
                                                                                r"density $\bar{\sigma}_{\rho}$ (kg m$^{-3}$)", "b", "log"),
    ]

    for ax, values, levels, colors, label_color, cbar_label, panel_label, cbar_scale in panels:
        cmap = mcolors.ListedColormap(colors)
        if cbar_scale == "log":
            norm = mcolors.LogNorm(vmin=float(np.min(levels)), vmax=float(np.max(levels)))
        else:
            norm = mcolors.Normalize(vmin=float(np.min(levels)), vmax=float(np.max(levels)))
        contour = ax.contour(dh_mesh, sigma_mesh, values, levels=levels, cmap=cmap, norm=norm, linewidths=CONTOUR_LINEWIDTH)
        vertical_overrides = None
        if panel_label == "b":
            vertical_overrides = {1000.0: 0.30, 1500.0: 0.46, 2000.0: 0.62, 3000.0: 0.82}
        positions = _automatic_contour_label_positions(contour, vertical_fraction_overrides=vertical_overrides)
        labels = ax.clabel(
            contour,
            levels=contour.levels,
            manual=list(positions.values()),
            fmt=lambda value: f"{value:.0f}",
            fontsize=CONTOUR_LABEL_FONTSIZE,
            inline=False,
        )
        for text in labels:
            text.set_path_effects([pe.withStroke(linewidth=CONTOUR_LABEL_HALO_LINEWIDTH, foreground="white")])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(DH_MIN, DH_MAX)
        ax.set_ylim(SIGMA_DH_MIN, SIGMA_DH_MAX)
        ax.set_xlabel(r"Absolute elevation change $|\Delta h|$ (m)", fontsize=AXIS_LABEL_FONTSIZE)
        if panel_label == "a":
            ax.set_ylabel(r"Uncertainty in elevation change $\sigma_{\Delta h}$ (m)", fontsize=AXIS_LABEL_FONTSIZE)
        ax.grid(which="both", linewidth=GRID_LINEWIDTH, alpha=GRID_ALPHA)
        ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
        ax.text(0.02, 0.97, panel_label, transform=ax.transAxes, va="top", ha="left", fontsize=PANEL_LABEL_FONTSIZE, fontweight="bold")
        scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        scalar.set_array([])
        cbar = fig.colorbar(scalar, ax=ax, ticks=levels)
        cbar.set_label(cbar_label, fontsize=COLORBAR_LABEL_FONTSIZE)
        cbar.ax.set_yticklabels([f"{level:g}" for level in levels])
        cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)

    for ax in axes:
        ticks = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(_format_log_ticks(ticks))
        ax.set_yticklabels(_format_log_ticks(ticks))

    axes[0].text(
        0.98,
        0.04,
        rf"Period length $\Delta t={dt:g}$ yr"
        + "\n\n"
        + rf"Past elevation change rate $\dot{{h}}^{{\rm p}}={dh_p:g}$ m yr$^{{-1}}$"
        + "\n\n"
        + "Uncertainty in past elevation\n"
        + rf"change rate $\sigma_{{\dot{{h}}^{{\rm p}}}}={sigma_dh_p:g}$ m yr$^{{-1}}$",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=PARAMETER_TEXT_FONTSIZE,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5),
        linespacing=1.15,
    )
    axes[1].text(
        0.98,
        0.04,
        rf"Period length $\Delta t={dt:g}$ yr",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=PARAMETER_TEXT_FONTSIZE,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5),
    )

    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / FIGURE_NAME
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    model = RhoSurrogate.from_files(
        parameter_path=PARAM_CSV,
        spatial_path=None,
        temporal_path=None,
        gh_order_current=GH_ORDER,
        gh_order_past=GH_ORDER,
    )
    dh_values = np.geomspace(DH_MIN, DH_MAX, LOG_DH_N)
    sigma_values = np.geomspace(SIGMA_DH_MIN, SIGMA_DH_MAX, LOG_SIGMA_N)
    print(
        f"[grid] Computing {LOG_DH_N} x {LOG_SIGMA_N} grid "
        f"for dt={DT_YR}, dh_p={MEASURED_PAST_DH}, sigma_dh_p={SIGMA_PAST_DH}"
    )
    grid = compute_grid(model, dh_values, sigma_values, MEASURED_PAST_DH, SIGMA_PAST_DH, DT_YR)
    figure_path = plot_figure(grid, DT_YR, MEASURED_PAST_DH, SIGMA_PAST_DH, OUTDIR)
    print(f"[done] Wrote figure to: {figure_path}")
