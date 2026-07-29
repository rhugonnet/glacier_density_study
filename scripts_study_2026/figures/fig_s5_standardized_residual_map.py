#!/usr/bin/env python3
"""Supplementary figure S5: Fig. 6-style map of full-period standardized residuals."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STUDY_DIR = Path(__file__).resolve().parents[1]
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

from study_paths import FIGURES_DIR, STANDARDIZED_RESIDUALS_PATH

INPUT_CSV = STANDARDIZED_RESIDUALS_PATH
OUT_PNG = FIGURES_DIR / "FIG_supp_05_standardized_residual_map.pdf"
FIG6_SCRIPT = Path(__file__).resolve().with_name("fig_6_tile_area.py")

REFERENCE_VARIANT = "iteration9"
RESIDUAL_BOUNDS = np.array([-2.5, -1.0, 0.0, 1.0, 2.5], dtype=float)

mpl.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load_fig6_helpers() -> types.ModuleType:
    """Load Fig. 6 helper functions without executing its plotting block."""
    source = FIG6_SCRIPT.read_text()
    module = types.ModuleType("fig6_tile_area_helpers")
    module.__file__ = str(FIG6_SCRIPT)
    exec(compile(source, str(FIG6_SCRIPT), "exec"), module.__dict__)
    return module


def residual_colormap() -> mpl.colors.LinearSegmentedColormap:
    """Build a diverging colormap for standardized residuals."""
    cb_val = np.linspace(0, 1, len(RESIDUAL_BOUNDS))
    colors = [mpl.cm.RdBu_r(v) for v in cb_val]
    coords = (RESIDUAL_BOUNDS - RESIDUAL_BOUNDS.min()) / (RESIDUAL_BOUNDS.max() - RESIDUAL_BOUNDS.min())
    return mpl.colors.LinearSegmentedColormap.from_list("z_rho_tile_cmap", list(zip(coords, colors)), N=1000)


def aggregate_tiles(fig6) -> pd.DataFrame:
    """Aggregate full-period standardized residuals to Fig. 6 adaptive tiles."""
    usecols = [
        "rho_variant",
        "period_years",
        "area",
        "abs_dV_weight",
        "z_rho",
        "lat",
        "lon",
    ]
    df = pd.read_csv(INPUT_CSV, usecols=usecols, low_memory=True, memory_map=True)
    df = df.loc[df["rho_variant"].astype(str).eq(REFERENCE_VARIANT)].copy()
    for col in ["period_years", "area", "abs_dV_weight", "z_rho", "lat", "lon"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    full_period = float(df["period_years"].max())
    df = df.loc[np.isclose(df["period_years"], full_period)].copy()
    ok = (
        np.isfinite(df["area"])
        & np.isfinite(df["abs_dV_weight"])
        & np.isfinite(df["z_rho"])
        & np.isfinite(df["lat"])
        & np.isfinite(df["lon"])
        & (df["area"] > 0)
        & (df["abs_dV_weight"] > 0)
    )
    df = df.loc[ok].copy()
    df["tile_lat"], df["tile_lon"] = fig6.representative_tile_origin(df["lat"].to_numpy(float), df["lon"].to_numpy(float))
    rows = []
    for key, g in df.groupby(["tile_lat", "tile_lon"], sort=True, observed=True):
        area = float(np.nansum(g["area"]))
        if area < fig6.AREA_MIN_KM2:
            continue
        weight = g["abs_dV_weight"].to_numpy(float)
        z = g["z_rho"].to_numpy(float)
        rows.append(
            {
                "tile_lat": key[0],
                "tile_lon": key[1],
                "area_km2": area,
                "effective_density_kg_m3": float(np.nansum(z * weight) / np.nansum(weight)),
            }
        )
    out = pd.DataFrame(rows)
    out["center_lat"], out["center_lon"] = fig6.tile_center(out["tile_lat"].to_numpy(float), out["tile_lon"].to_numpy(float))
    out["rho_dv"] = out["effective_density_kg_m3"]
    return out


def draw_fig6_layout(fig, tiles: pd.DataFrame, fig6) -> None:
    """Draw the same exploded map layout as Fig. 6."""
    base_ax = fig.add_subplot(1, 1, 1, projection=fig6.ccrs.Robinson())
    base_ax.set_global()
    fig6.set_geo_outline(base_ax, linewidth=0, visible=False)
    base_ax.patch.set_alpha(0)

    fig6.add_inset(
        fig,
        tiles,
        [-179.99, 179.99, -89.99, 89.99],
        [0.375, 0.21, 0.25, 0.25],
        bounds=[-179.99, 179.99, -89.99, 89.99],
        shades=False,
        hillshade=False,
        main=True,
    )

    poly_aw = np.array([(-158, -79), (-135, -62), (-110, -62), (-50, -62), (-50, -79.25), (-158, -79.25)])
    fig6.add_inset(fig, tiles, [-179.99, 179.99, -89.99, 89.99], [-0.4, -0.065, 2, 2], bounds=[-158, -50, -62.5, -79], label="Antarctica_West", polygon=poly_aw, markup_sub="a", sub_pos="mb")
    poly_ae = np.array([(135, -81.5), (152, -63.7), (165, -65), (175, -70), (175, -81.25), (135, -81.75)])
    fig6.add_inset(fig, tiles, [-179.99, 179.99, -89.99, 89.99], [-0.71, -0.045, 2, 2], bounds=[130, 175, -64.5, -81], label="Antarctica_East", polygon=poly_ae, markup_sub="e", sub_pos="mb")
    poly_ac = np.array([(-25, -62), (106, -62), (80, -79.25), (-25, -79.25), (-25, -62)])
    fig6.add_inset(fig, tiles, [-179.99, 179.99, -89.99, 89.99], [-0.52, -0.065, 2, 2], bounds=[-25, 106, -62.5, -79], label="Antarctica_Center", polygon=poly_ac, markup="Antarctic and Subantarctic (19)", markpos="right", markup_sub="c", sub_pos="mb")
    fig6.add_inset(fig, tiles, [-179.99, 179.99, -89.99, 89.99], [-0.68, -0.18, 2, 2], bounds=[64, 78, -48, -55], label="Antarctica_Australes", markup_sub="d", sub_pos="lt")
    fig6.add_inset(fig, tiles, [-179.99, 179.99, -89.99, 89.99], [-0.42, -0.155, 2, 2], bounds=[-40, -23, -53, -60], label="Antarctica_South_Georgia", markup_sub="b", sub_pos="rt")
    fig6.add_inset(fig, tiles, [-179.99, 179.99, -89.99, 89.99], [-0.52, -0.225, 2, 2], bounds=[-82, -65, 13, -57], label="Andes", markup="Low Latitudes (16) &\nSouthern Andes (17)", markup_sub="a", sub_pos="lm2")
    fig6.add_inset(fig, tiles, [-179.99, 179.99, -89.99, 89.99], [-0.352, -0.39, 2, 2], bounds=[-100, -95, 22, 16], label="Mexico", markup_sub="b", sub_pos="rb")
    fig6.add_inset(fig, tiles, [-179.99, 179.99, -89.99, 89.99], [-1.078, -0.24, 2, 2], bounds=[28, 42, 2, -5], label="Africa", markup_sub="c", sub_pos="rb")
    fig6.add_inset(fig, tiles, [-179.99, 179.99, -89.99, 89.99], [-1.640, -0.3, 2, 2], bounds=[133, 140, -2, -7], label="Indonesia", markup_sub="d", sub_pos="rb")
    poly_arctic = np.array([(-105, 84.5), (115, 84.5), (110, 68), (30, 68), (18, 57), (-70, 57), (-100, 75), (-105, 84.5)])
    fig6.add_inset(fig, tiles, [-179.99, 179.99, -89.99, 89.99], [-0.48, -1.003, 2, 2], bounds=[-100, 106, 57, 84], label="Arctic West", polygon=poly_arctic, markup="Arctic (03-09)")
    fig6.add_inset(fig, tiles, [-179.99, 179.99, -89.99, 89.99], [-0.92, -0.17, 2, 2], bounds=[164, 176, -47, -40], label="New Zealand", markup="New Zealand (18)", markpos="right")
    poly_na = np.array([(-170, 72), (-140, 72), (-120, 63), (-101, 35), (-126, 35), (-165, 55), (-170, 72)])
    fig6.add_inset(fig, tiles, [-179.99, 179.99, -89.99, 89.99], [-0.1, -1.22, 2, 2], bounds=[-177, -105, 36, 70], label="North America", polygon=poly_na, markup="Alaska (01) & Western\nCanada and USA (02)")

    poly_asia = np.array([(148, 49), (160, 65), (178, 65), (170, 55), (160, 49), (148, 49)])
    poly_asia_ne = np.array([(142, 71), (142, 80), (163, 80), (155, 71), (142, 71)])
    fig6.add_inset(fig, tiles, [-179.99, 179.99, -89.99, 89.99], [-0.655, -1.165, 2, 2], bounds=[142, 160, 71, 80], polygon=poly_asia_ne, label="North Asia North E", markup_sub="d", sub_pos="rt")
    fig6.add_inset(fig, tiles, [-179.99, 179.99, -89.99, 89.99], [-0.565, -1.107, 2, 2], bounds=[87, 112, 68, 77], label="North Asia North W", markup_sub="c", sub_pos="mt")
    poly_asia_e2 = np.array([(125, 58), (125, 72), (153.8, 72), (148, 58), (125, 58)])
    fig6.only_shade(fig, [-0.71, -1.142, 2, 2], [125, 148, 58, 72], polygon=poly_asia_e2, label="tmp_NAE2")
    fig6.add_inset(fig, tiles, [-179.99, 179.99, -89.99, 89.99], [-0.822, -1.218, 2, 2], bounds=[128, 179.9, 50, 64.8], label="North Asia East", polygon=poly_asia, markup_sub="f", sub_pos="lb")
    fig6.add_inset(fig, tiles, [-179.99, 179.99, -89.99, 89.99], [-0.71, -1.142, 2, 2], bounds=[125, 148, 58, 72], polygon=poly_asia_e2, label="North Asia East 2", markup_sub="e", sub_pos="lb", shades=False)
    fig6.only_shade(fig, [-0.517, -1.035, 2, 2], [53, 70, 62, 69.8], label="tmp_NAW")
    fig6.add_inset(fig, tiles, [-179.99, 179.99, -89.99, 89.99], [-0.74, -1, 2, 2], bounds=[82, 120, 45.5, 58.9], label="South Asia North", markup="North Asia (10)", markup_sub="a", sub_pos="mb")
    fig6.add_inset(fig, tiles, [-179.99, 179.99, -89.99, 89.99], [-0.512, -1.035, 2, 2], bounds=[53, 70, 62, 69.8], label="North Asia West", markup_sub="b", sub_pos="lb", shades=False)
    fig6.add_inset(fig, tiles, [-179.99, 179.99, -89.99, 89.99], [-0.685, -1.065, 2, 2], bounds=[65, 105, 46.5, 25], label="HMA", markup="High Mountain Asia (13-15)")
    fig6.add_inset(fig, tiles, [-179.99, 179.99, -89.99, 89.99], [-0.58, -0.982, 2, 2], bounds=[-4.9, 19, 38.2, 50.5], label="Europe", markup="Central Europe (11)")
    fig6.add_inset(fig, tiles, [-179.99, 179.99, -89.99, 89.99], [-0.66, -0.89, 2, 2], bounds=[38, 54, 29, 44.75], label="Middle East", markup="Caucasus (12)")


def run() -> Path:
    fig6 = load_fig6_helpers()
    fig6.DENSITY_BOUNDS = RESIDUAL_BOUNDS
    fig6.density_colormap = residual_colormap
    tiles = aggregate_tiles(fig6)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    merged_height_inch = fig6.FIG_PANEL_HEIGHT_INCH + fig6.LEGEND_STRIP_HEIGHT_INCH
    merged_fig = plt.figure(figsize=(fig6.FIG_WIDTH_INCH, merged_height_inch))
    map_fig, _legend_strip = merged_fig.subfigures(
        2,
        1,
        height_ratios=[fig6.FIG_PANEL_HEIGHT_INCH, fig6.LEGEND_STRIP_HEIGHT_INCH],
        hspace=0.0,
    )
    draw_fig6_layout(map_fig, tiles, fig6)

    legend_background = merged_fig.add_axes(
        [0, 0, 1, fig6.LEGEND_STRIP_HEIGHT_INCH / merged_height_inch],
        label="legend_background",
        zorder=1000,
    )
    legend_background.patch.set_facecolor("white")
    legend_background.patch.set_alpha(1.0)
    legend_background.set_axis_off()
    fig6.add_legends(
        merged_fig,
        colorbar_label=r"Standardized residual $z_{\rho_{\Delta V}}$",
        panel_height_inch=fig6.FIG_PANEL_HEIGHT_INCH,
        total_height_inch=merged_height_inch,
        zorder=1001,
    )

    merged_fig.savefig(OUT_PNG, dpi=fig6.DPI)
    plt.close(merged_fig)
    return OUT_PNG


if __name__ == "__main__":
    path = run()
    print(f"[done] Wrote figure: {path.resolve()}")
