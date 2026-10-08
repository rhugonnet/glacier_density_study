#!/usr/bin/env python3
"""Main figure 0: surrogate model workflow flowchart."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.transforms import Bbox

STUDY_DIR = Path(__file__).resolve().parents[1]
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

from study_paths import FIGURES_DIR


CORE_COLOR = "#e8e2b7"
PROCESS_COLOR = "0.20"
DPI = 300
OUT_PDF = FIGURES_DIR / "FIG_main_00_surrogate_flowchart.pdf"

# Embed editable TrueType fonts in PDF outputs
mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


def add_node(ax, x, y, w, h, title, subtitle="", kind="data",
             fontsize=9.6, title_size=None, pad=0.012,
             title_y=0.60, subtitle_y=0.34):
    """Draw and return a flowchart node."""
    if title_size is None:
        title_size = fontsize + 0.4

    if kind == "data":
        facecolor, textcolor, lw, rounding = "white", "black", 1.15, 0.015
    elif kind == "process":
        facecolor, textcolor, lw, rounding = PROCESS_COLOR, "white", 1.10, 0.035
    elif kind == "core":
        facecolor, textcolor, lw, rounding = CORE_COLOR, "black", 1.25, 0.020
    else:
        raise ValueError(f"Unknown node kind: {kind}")

    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={pad},rounding_size={rounding}",
        linewidth=lw,
        facecolor=facecolor,
        edgecolor="black",
        transform=ax.transAxes,
        clip_on=False,
        zorder=3,
    )
    ax.add_patch(patch)

    if subtitle:
        ax.text(
            x + w / 2, y + h * title_y, title,
            ha="center", va="center",
            fontsize=title_size, color=textcolor, weight="semibold",
            transform=ax.transAxes, zorder=4,
        )
        ax.text(
            x + w / 2, y + h * subtitle_y, subtitle,
            ha="center", va="center",
            fontsize=fontsize, color=textcolor,
            transform=ax.transAxes, zorder=4,
        )
    else:
        ax.text(
            x + w / 2, y + h / 2, title,
            ha="center", va="center",
            fontsize=title_size, color=textcolor, weight="semibold",
            transform=ax.transAxes, zorder=4,
        )

    return {
        "patch": patch,
        "center": (x + w / 2, y + h / 2),
        "x": x, "y": y, "w": w, "h": h,
        "kind": kind,
    }


def add_surrogate_box(ax, x, y, w, h):
    """Custom centered surrogate model box with styled text."""
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="square,pad=0.012",
        linewidth=1.25,
        facecolor=CORE_COLOR,
        edgecolor="black",
        transform=ax.transAxes,
        clip_on=False,
        zorder=3,
    )
    ax.add_patch(patch)

    # Well-centered multiline content.
    lines = [
        ("Surrogate model", dict(fontsize=10.4, weight="semibold", style="normal")),
        ("predicts effective density", dict(fontsize=8.2, weight="bold", style="italic")),
        ("for any glacier and period", dict(fontsize=8.2, weight="bold", style="italic")),
        ("with explicit error propagation", dict(fontsize=8.2, weight="bold", style="italic")),
        ("", dict(fontsize=3)),
        (r"Mean of effective density $\mu_\rho$", dict(fontsize=8.6)),
        (r"Uncertainty of effective density $\sigma_\rho$", dict(fontsize=8.6)),
        (r"Spatial error correlation $r_\rho^{\rm s}$", dict(fontsize=9.1)),
        (r"Temporal error correlation $r_\rho^{\rm t}$", dict(fontsize=9.1)),
    ]
    ys = [0.92, 0.80, 0.715, 0.63, 0.535, 0.42, 0.29, 0.16, 0.035]
    for (txt, kw), yy in zip(lines, ys):
        ax.text(
            x + w / 2, y + h * yy, txt,
            ha="center", va="center",
            color="black",
            transform=ax.transAxes,
            zorder=4,
            **kw,
        )

    return {
        "patch": patch,
        "center": (x + w / 2, y + h / 2),
        "x": x, "y": y, "w": w, "h": h,
        "kind": "core",
    }


def add_final_box(ax, x, y, w, h):
    """Custom final applicable surrogate model box."""
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.020",
        linewidth=1.25,
        facecolor=CORE_COLOR,
        edgecolor="black",
        transform=ax.transAxes,
        clip_on=False,
        zorder=3,
    )
    ax.add_patch(patch)

    ax.text(
        x + w / 2, y + h * 0.84,
        "Final surrogate model",
        ha="center", va="center",
        fontsize=10.6, weight="semibold",
        transform=ax.transAxes, zorder=4,
    )
    ax.text(
        x + w / 2, y + h * 0.66,
        "applicable to any uncertain time series",
        ha="center", va="center",
        fontsize=9.3, weight="bold", style="italic",
        transform=ax.transAxes, zorder=4,
    )
    ax.text(
        x + w / 2, y + h * 0.30,
        r"Final mean $\bar{\mu}_\rho^{\rm closed}$",
        ha="center", va="center",
        fontsize=10.0,
        transform=ax.transAxes, zorder=4,
    )
    ax.text(
        x + w / 2, y + h * 0.13,
        r"Final uncertainty $\bar{\sigma}_\rho^{\rm closed}$",
        ha="center", va="center",
        fontsize=10.0,
        transform=ax.transAxes, zorder=4,
    )

    return {
        "patch": patch,
        "center": (x + w / 2, y + h / 2),
        "x": x, "y": y, "w": w, "h": h,
        "kind": "core",
    }


def connect_nodes(ax, source, target, connectionstyle="arc3", lw=1.15, mutation_scale=12):
    """Draw arrow from source node to target node, clipped to their patch boundaries."""
    arrow = FancyArrowPatch(
        posA=source["center"],
        posB=target["center"],
        patchA=source["patch"],
        patchB=target["patch"],
        shrinkA=0,
        shrinkB=0,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=lw,
        color="black",
        connectionstyle=connectionstyle,
        transform=ax.transAxes,
        clip_on=False,
        zorder=2,
    )
    ax.add_patch(arrow)
    return arrow


def draw_predictor_inset(fig):
    """Predictor-definition schematic positioned directly below its box."""
    ax = fig.add_axes([0.245, 0.472, 0.28, 0.285])
    ax.set_zorder(1)

    years = np.arange(1998, 2011)
    annual = np.array([
        0.02, -0.02, -0.05, -0.08, -0.12, -0.18, -0.22,
        -0.20, -0.16, -0.13, -0.10, -0.08, -0.09
    ])
    elevation = np.r_[0.0, np.cumsum(annual)]
    time = np.arange(1998, 2012)

    obs_start, obs_end = 2002, 2007
    i0 = np.where(time == obs_start)[0][0]
    i1 = np.where(time == obs_end)[0][0]

    ax.plot(time, elevation, color="0.80", linewidth=1.3, zorder=1)
    ax.axvspan(time.min(), obs_start, facecolor="0.97", edgecolor="none", zorder=0)
    ax.axvspan(obs_end, time.max(), facecolor="0.97", edgecolor="none", zorder=0)
    ax.axvspan(obs_start, obs_end, facecolor="0.91", edgecolor="none", zorder=0)
    ax.plot(
        time[i0:i1 + 1], elevation[i0:i1 + 1],
        color="black", linewidth=1.9, zorder=2,
    )
    ax.scatter(
        [obs_start, obs_end],
        [elevation[i0], elevation[i1]],
        s=14, color="black", zorder=3,
    )

    # Highlight past elevation history with decreasing lookback weight.
    for j in range(i0):
        frac = (j + 1) / i0
        ax.plot(
            time[j:j + 2],
            elevation[j:j + 2],
            color=plt.cm.Blues(0.28 + 0.52 * frac),
            linewidth=2.0,
            zorder=2.2,
        )

    # Observation period label inside shading, in grey and over two lines.
    y_min, y_max = elevation.min() - 0.20, elevation.max() + 0.10
    y_obs = y_min + 0.28 * (y_max - y_min)
    ax.text(
        (obs_start + obs_end) / 2, y_obs,
        "Observation\nperiod",
        ha="center", va="center",
        color="0.45", fontsize=8.1,
    )

    # Period length at the top.
    y_top = elevation.max() + 0.055
    ax.annotate(
        "",
        xy=(obs_end, y_top),
        xytext=(obs_start, y_top),
        arrowprops={"arrowstyle": "<->", "linewidth": 1.0, "color": "black"},
    )
    ax.text(
        (obs_start + obs_end) / 2, y_top - 0.035,
        r"Period length $\Delta t$",
        ha="center", va="top", fontsize=8.1,
    )

    # Elevation change + A.
    x_dh = obs_end + 0.55
    ax.annotate(
        "",
        xy=(x_dh, elevation[i1]),
        xytext=(x_dh, elevation[i0]),
        arrowprops={"arrowstyle": "<->", "linewidth": 1.0, "color": "black"},
    )
    ax.text(
        x_dh + 1.36, 0.5 * (elevation[i0] + elevation[i1]),
        r"Elevation" "\n" r"change" "\n" r"$\Delta h$" "\n" r"for glacier" "\n" r"area $A$",
        ha="center", va="center", fontsize=7.9,
    )

    # Past elevation change rate with explicit weighting note.
    text_x = 1998.25
    text_y = y_min + 0.77 * (y_max - y_min)
    ax.annotate(
        "Past elevation\nchange rate " + r"$\dot{h}^{\rm p}$",
        xy=(2000.8, elevation[np.where(time == 2001)[0][0]]),
        xytext=(text_x, text_y),
        ha="left", va="top", fontsize=8.0,
        arrowprops={"arrowstyle": "-", "linewidth": 0.9, "color": "0.35"},
    )
    ax.text(
        text_x, text_y - 0.14 * (y_max - y_min),
        "with decreasing\nlookback weight",
        ha="left", va="top",
        fontsize=7.8, color=plt.cm.Blues(0.75),
    )

    ax.set_xlim(1997.8, 2011.3)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Time", fontsize=8.0)
    ax.set_ylabel("Cumulative elevation change", fontsize=8.0)
    ax.tick_params(labelsize=7.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(True)

    return ax


def build_figure(out_path: Path | str = OUT_PDF) -> Path:
    """Build and save the flowchart figure."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12.6, 8.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_zorder(2)
    ax.patch.set_alpha(0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Legend (stacked vertically below the left input boxes)
    legend_x = 0.24
    legend_top = 0.305
    dy = 0.041

    ax.text(legend_x, legend_top + 0.026, "Legend", fontsize=8.8, weight="bold",
            ha="left", va="bottom", transform=ax.transAxes)

    ax.add_patch(FancyBboxPatch(
        (legend_x, legend_top - 0.014), 0.036, 0.028,
        boxstyle="round,pad=0.004,rounding_size=0.005",
        linewidth=1.0, facecolor="white", edgecolor="black",
        transform=ax.transAxes,
    ))
    ax.text(legend_x + 0.048, legend_top, "Inputs", fontsize=8.3,
            ha="left", va="center", transform=ax.transAxes)

    ax.add_patch(FancyBboxPatch(
        (legend_x, legend_top - dy - 0.014), 0.036, 0.028,
        boxstyle="round,pad=0.004,rounding_size=0.010",
        linewidth=1.0, facecolor=PROCESS_COLOR, edgecolor="black",
        transform=ax.transAxes,
    ))
    ax.text(legend_x + 0.048, legend_top - dy, "Processing step", fontsize=8.3,
            ha="left", va="center", transform=ax.transAxes)

    ax.add_patch(FancyBboxPatch(
        (legend_x, legend_top - 2 * dy - 0.014), 0.036, 0.028,
        boxstyle="square,pad=0.004",
        linewidth=1.0, facecolor=CORE_COLOR, edgecolor="black",
        transform=ax.transAxes,
    ))
    ax.text(legend_x + 0.048, legend_top - 2 * dy, "Surrogate model", fontsize=8.3,
            ha="left", va="center", transform=ax.transAxes)

    # Predictor definition and figure.
    predictor = add_node(
        ax, 0.285, 0.775, 0.20, 0.075,
        "Predictor definition",
        "from observable glacier variables\n" + r"$\Delta h,\ \dot{h}^{\rm p},\ \Delta t,\ A$",
        kind="process", fontsize=8.0, title_size=9.5, pad=0.010,
        title_y=0.74, subtitle_y=0.34,
    )
    draw_predictor_inset(fig)
    inset_left = 0.245
    inset_bottom = 0.472
    inset_width = 0.28
    inset_height = 0.285
    inset_top = inset_bottom + inset_height

    ax.plot(
        [predictor["x"] + 0.07 * predictor["w"], inset_left],
        [predictor["y"], inset_top],
        color="black", linewidth=1.0, linestyle="--",
        transform=ax.transAxes, clip_on=False, zorder=2,
    )
    ax.plot(
        [predictor["x"] + 0.93 * predictor["w"], inset_left + inset_width],
        [predictor["y"], inset_top],
        color="black", linewidth=1.0, linestyle="--",
        transform=ax.transAxes, clip_on=False, zorder=2,
    )

    # Main vertical spine.
    spine_x = 0.66

    full_model = add_node(
        ax, spine_x - 0.0625, 0.81, 0.125, 0.074,
        "Full-model outputs",
        r"annual $\rho_{\Delta V}$ and $\Delta h$" "\n"
        "globally for 2000–2019",
        kind="data", fontsize=8.2, title_size=9.1, pad=0.010,
        title_y=0.69, subtitle_y=0.33,
    )
    ax.add_patch(FancyArrowPatch(
        posA=(full_model["center"][0], full_model["y"] + full_model["h"] + 0.045),
        posB=(full_model["center"][0], full_model["y"] + full_model["h"]),
        patchB=full_model["patch"],
        shrinkA=0,
        shrinkB=0,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.15,
        color="black",
        connectionstyle="arc3",
        transform=ax.transAxes,
        clip_on=False,
        zorder=2,
    ))
    ax.text(
        full_model["center"][0] - 0.095,
        full_model["y"] + full_model["h"] + 0.031,
        "Coupled mass-balance–\nfirn-densification model (i.e. full model)",
        ha="center",
        va="center",
        fontsize=8.0,
        linespacing=1.05,
        transform=ax.transAxes,
    )
    calibration = add_node(
        ax, spine_x - 0.05, 0.685, 0.10, 0.082,
        "Calibration",
        "by binning and fitting\neffective density\nwith predictors",
        kind="process", fontsize=8.2, title_size=9.8, pad=0.010,
        title_y=0.76, subtitle_y=0.31,
    )
    surrogate = add_surrogate_box(
        ax, spine_x - 0.1025, 0.462, 0.205, 0.18,
    )
    integration = add_node(
        ax, spine_x - 0.046, 0.334, 0.092, 0.080,
        "Integration",
        "by averaging with\nerror distribution",
        kind="process", fontsize=8.1, title_size=9.4, pad=0.010,
        title_y=0.69, subtitle_y=0.33,
    )
    temporal_closure = add_node(
        ax, spine_x - 0.060, 0.205, 0.120, 0.082,
        "Temporal closure",
        "by enforcing\nmass change additivity",
        kind="process", fontsize=8.4, title_size=9.4, pad=0.010,
        title_y=0.69, subtitle_y=0.33,
    )

    # Smaller input boxes.
    left_x, left_w = 0.423, 0.112
    uncertainty_inputs = add_node(
        ax, left_x, 0.341, left_w, 0.066,
        "Uncertain inputs",
        r"$\sigma_{\Delta h},\ \sigma_{\dot{h}^{\rm p}}$",
        kind="data", fontsize=8.2, title_size=9.1, pad=0.009,
    )
    time_series_input = add_node(
        ax, left_x, 0.213, left_w, 0.066,
        "Time series input",
        r"$\Delta h(t)$",
        kind="data", fontsize=8.2, title_size=9.1, pad=0.009,
    )

    # Connectors.
    connect_nodes(ax, full_model, calibration)
    connect_nodes(ax, full_model, predictor)
    predictor_to_calibration = FancyArrowPatch(
        posA=(predictor["x"] + predictor["w"], predictor["y"] + 0.40 * predictor["h"]),
        posB=(calibration["x"], calibration["y"] + 0.52 * calibration["h"]),
        patchA=predictor["patch"],
        patchB=calibration["patch"],
        shrinkA=0,
        shrinkB=0,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.15,
        color="black",
        connectionstyle="arc3",
        transform=ax.transAxes,
        clip_on=False,
        zorder=2,
    )
    ax.add_patch(predictor_to_calibration)
    connect_nodes(ax, calibration, surrogate)
    connect_nodes(ax, surrogate, integration)
    connect_nodes(ax, uncertainty_inputs, integration)
    connect_nodes(ax, integration, temporal_closure)
    connect_nodes(ax, time_series_input, temporal_closure)

    # Return arrows indicating the expanded applicability of the surrogate.
    # Integration -> Surrogate model (arrives lower on the right side)
    int_start_x = integration["x"] + integration["w"]
    int_start_y = integration["y"] + 0.52 * integration["h"]
    int_route_x = 0.80
    int_target_y = surrogate["y"] + 0.30 * surrogate["h"]

    ax.plot([int_start_x, int_route_x], [int_start_y, int_start_y],
            color="black", linewidth=1.15, transform=ax.transAxes, clip_on=False, zorder=2)
    ax.plot([int_route_x, int_route_x], [int_start_y, int_target_y],
            color="black", linewidth=1.15, transform=ax.transAxes, clip_on=False, zorder=2)
    ax.add_patch(FancyArrowPatch(
        posA=(int_route_x, int_target_y),
        posB=(surrogate["x"] + 0.50 * surrogate["w"], int_target_y),
        patchB=surrogate["patch"],
        shrinkA=0,
        shrinkB=0,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.15,
        color="black",
        connectionstyle="arc3",
        transform=ax.transAxes,
        clip_on=False,
        zorder=4,
    ))
    ax.text(
        int_start_x + 0.525 * (int_route_x - int_start_x), int_start_y + 0.028,
        "extends\napplicability to\nuncertain inputs",
        ha="center", va="center",
        fontsize=8.0,
        transform=ax.transAxes,
    )

    # Temporal closure -> Surrogate model (arrives upper on the right side)
    clo_start_x = temporal_closure["x"] + temporal_closure["w"]
    clo_start_y = temporal_closure["y"] + 0.52 * temporal_closure["h"]
    clo_route_x = 0.82
    clo_target_y = surrogate["y"] + 0.60 * surrogate["h"]

    ax.plot([clo_start_x, clo_route_x], [clo_start_y, clo_start_y],
            color="black", linewidth=1.15, transform=ax.transAxes, clip_on=False, zorder=2)
    ax.plot([clo_route_x, clo_route_x], [clo_start_y, clo_target_y],
            color="black", linewidth=1.15, transform=ax.transAxes, clip_on=False, zorder=2)
    ax.add_patch(FancyArrowPatch(
        posA=(clo_route_x, clo_target_y),
        posB=(surrogate["x"] + 0.50 * surrogate["w"], clo_target_y),
        patchB=surrogate["patch"],
        shrinkA=0,
        shrinkB=0,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.15,
        color="black",
        connectionstyle="arc3",
        transform=ax.transAxes,
        clip_on=False,
        zorder=4,
    ))
    ax.text(
        clo_start_x + 0.50 * (clo_route_x - clo_start_x), clo_start_y + 0.028,
        "extends\napplicability to\ntime series",
        ha="center", va="center",
        fontsize=8.0,
        transform=ax.transAxes,
    )

    # Crop unused outer margins without rescaling or repositioning the figure.
    # Coordinates are in physical figure inches.
    crop_box = Bbox.from_extents(2.52, 1.48, 10.68, 7.50)

    fig.savefig(out_path, dpi=DPI, bbox_inches=crop_box)
    plt.close(fig)

    return out_path


def run() -> Path:
    """Write the manuscript flowchart figure."""
    return build_figure(OUT_PDF)


if __name__ == "__main__":
    path = run()
    print(f"[done] Wrote figure: {path}")
