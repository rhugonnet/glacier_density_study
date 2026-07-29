#!/usr/bin/env python3
"""Supplementary figure S4: decadal changes in elevation rate and effective density."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature

STUDY_DIR = Path(__file__).resolve().parents[1]
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

from study_paths import FIGURES_DIR, INPUT_CSV

OUT_PNG = FIGURES_DIR / "FIG_supp_04_decadal_change_maps.pdf"

REFERENCE_VARIANT = "iteration9"
FIRST_DECADE = (1999, 2009)
SECOND_DECADE = (2009, 2019)
AREA_MIN_KM2 = 0.1
RHO_SENTINELS = {-99999.0, 99999.0}
FIG_WIDTH_IN = 7.4
FIG_HEIGHT_IN = 2.775 * FIG_WIDTH_IN / 1.9716
TOP_EXTRA_IN = -0.06
BOTTOM_CROP_IN = 0.35

mpl.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "pdf.compression": 9,
})


def first_existing(columns: pd.Index, candidates: list[str]) -> str:
    for col in candidates:
        if col in columns:
            return col
    raise KeyError(f"None of these columns exist: {candidates}")


def representative_tile_origin(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat_floor = np.floor(lat)
    lon_floor = np.floor(lon)
    high = np.abs(lat_floor) >= 60.0
    polar = np.abs(lat_floor) >= 74.0
    tile_lat = lat_floor.copy()
    tile_lon = lon_floor.copy()
    tile_lon[high] = np.floor((lon[high] - 0.5) / 2.0) * 2.0
    tile_lat[polar] = np.floor((lat[polar] - 0.5) / 2.0) * 2.0
    return tile_lat, tile_lon


def tile_center(tile_lat: np.ndarray, tile_lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    high = np.abs(tile_lat) >= 60.0
    polar = np.abs(tile_lat) >= 74.0
    center_lat = tile_lat + 0.5
    center_lon = tile_lon + 0.5
    center_lon[high] = tile_lon[high] + 1.0
    center_lon[polar] = tile_lon[polar] + 1.0
    center_lat[polar] = tile_lat[polar] + 1.0
    center_lon = ((center_lon + 180.0) % 360.0) - 180.0
    return center_lat, center_lon


def read_period_table() -> pd.DataFrame:
    header = pd.read_csv(INPUT_CSV, nrows=0).columns
    lat_col = first_existing(header, ["lat", "cenlat", "center_lat"])
    lon_col = first_existing(header, ["lon", "cenlon", "center_lon"])
    usecols = ["rgiid", "start_date", "end_date", "rho", "b", "area", lat_col, lon_col]
    if "rho_variant" in header:
        usecols.append("rho_variant")
    df = pd.read_csv(INPUT_CSV, usecols=usecols, low_memory=True, memory_map=True)
    df = df.rename(columns={lat_col: "lat", lon_col: "lon"})
    if "rho_variant" in df.columns:
        df = df.loc[df["rho_variant"].astype(str).eq(REFERENCE_VARIANT)].copy()
    for col in ["start_date", "end_date", "rho", "b", "area", "lat", "lon"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    ok = (
        np.isfinite(df["rho"])
        & np.isfinite(df["b"])
        & np.isfinite(df["area"])
        & np.isfinite(df["lat"])
        & np.isfinite(df["lon"])
        & (df["area"] > 0)
        & (~df["rho"].isin(RHO_SENTINELS))
    )
    return df.loc[ok].copy()


def aggregate_tiles() -> pd.DataFrame:
    df = read_period_table()
    first = df.loc[df["start_date"].eq(FIRST_DECADE[0]) & df["end_date"].eq(FIRST_DECADE[1])].copy()
    second = df.loc[df["start_date"].eq(SECOND_DECADE[0]) & df["end_date"].eq(SECOND_DECADE[1])].copy()
    cols = ["rgiid", "rho", "b", "area", "lat", "lon"]
    d = first[cols].merge(second[cols], on="rgiid", suffixes=("_first", "_second"), validate="one_to_one")
    d["area"] = d["area_second"].fillna(d["area_first"])
    d["lat"] = d["lat_second"].fillna(d["lat_first"])
    d["lon"] = d["lon_second"].fillna(d["lon_first"])
    d["dhdt_change_m_yr"] = d["b_second"] / d["rho_second"] - d["b_first"] / d["rho_first"]
    d["dV_first"] = d["area"] * 1.0e6 * (SECOND_DECADE[0] - FIRST_DECADE[0]) * d["b_first"] / d["rho_first"]
    d["dV_second"] = d["area"] * 1.0e6 * (SECOND_DECADE[1] - SECOND_DECADE[0]) * d["b_second"] / d["rho_second"]
    d["rho_change_kg_m3"] = d["rho_second"] - d["rho_first"]
    d["rho_weight"] = np.abs(d["dV_first"]) + np.abs(d["dV_second"])
    d["tile_lat"], d["tile_lon"] = representative_tile_origin(d["lat"].to_numpy(float), d["lon"].to_numpy(float))
    d = d.loc[np.isfinite(d["dhdt_change_m_yr"]) & np.isfinite(d["rho_change_kg_m3"]) & (d["rho_weight"] > 0)].copy()
    grouped = []
    for key, g in d.groupby(["tile_lat", "tile_lon"], sort=True):
        area = float(np.nansum(g["area"]))
        if area < AREA_MIN_KM2:
            continue
        grouped.append(
            {
                "tile_lat": key[0],
                "tile_lon": key[1],
                "area_km2": area,
                "dhdt_change_m_yr": float(np.nansum(g["dhdt_change_m_yr"] * g["area"]) / np.nansum(g["area"])),
                "rho_change_kg_m3": float(np.nansum(g["rho_change_kg_m3"] * g["rho_weight"]) / np.nansum(g["rho_weight"])),
            }
        )
    out = pd.DataFrame(grouped)
    out["center_lat"], out["center_lon"] = tile_center(out["tile_lat"].to_numpy(float), out["tile_lon"].to_numpy(float))
    return out


def coord_transform(orig_crs, target_crs, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return target_crs.transform_points(orig_crs, x, y)


def poly_from_extent(extent: list[float]) -> np.ndarray:
    return np.array(
        [
            (extent[0], extent[2]),
            (extent[1], extent[2]),
            (extent[1], extent[3]),
            (extent[0], extent[3]),
            (extent[0], extent[2]),
        ],
        dtype=float,
    )


def robinson_projected_polygon_to_axes_verts(polygon_coords: np.ndarray) -> np.ndarray:
    robin = np.asarray(polygon_coords, dtype=float)
    x_min, x_max = ccrs.Robinson().x_limits
    y_min, y_max = ccrs.Robinson().y_limits
    verts = robin.copy()
    verts[:, 0] = (verts[:, 0] - x_min) / (x_max - x_min)
    verts[:, 1] = (verts[:, 1] - y_min) / (y_max - y_min)
    return verts[:, 0:2]


def set_geo_outline(ax, edgecolor: str | None = None, linewidth: float | None = None) -> None:
    try:
        outline = ax.outline_patch
        if edgecolor is not None:
            outline.set_edgecolor(edgecolor)
        if linewidth is not None:
            outline.set_linewidth(linewidth)
    except AttributeError:
        spine = ax.spines.get("geo")
        if spine is not None:
            if edgecolor is not None:
                spine.set_edgecolor(edgecolor)
            if linewidth is not None:
                spine.set_linewidth(linewidth)


def tile_size_degrees(tile_lat: float) -> tuple[float, float]:
    if abs(tile_lat) >= 74.0:
        return 2.0, 2.0
    if abs(tile_lat) >= 60.0:
        return 2.0, 1.0
    return 1.0, 1.0


def add_tile_markers(ax, table: pd.DataFrame, column: str, cmap, norm) -> None:
    for row in table.itertuples(index=False):
        if not np.isfinite(row.area_km2) or row.area_km2 <= AREA_MIN_KM2:
            continue
        if not np.isfinite(getattr(row, column)):
            continue
        width, height = tile_size_degrees(row.tile_lat)
        scale = float(np.clip(0.28 + 0.035 * np.sqrt(row.area_km2), 0.35, 1.0))
        color = cmap(norm(getattr(row, column)))
        ax.add_patch(
            mpatches.Rectangle(
                (row.center_lon - 0.5 * width * scale, row.center_lat - 0.5 * height * scale),
                width * scale,
                height * scale,
                facecolor=color,
                edgecolor="0.18",
                linewidth=0.10,
                transform=ccrs.PlateCarree(),
                zorder=30,
            )
        )


def add_inset(
    fig: plt.Figure,
    table: pd.DataFrame,
    column: str,
    cmap,
    norm,
    position: list[float],
    bounds: list[float],
    label: str,
    polygon: np.ndarray | None = None,
    markup_sub: str | None = None,
    sub_pos: str = "lt",
    sub_adj: tuple[float, float] = (0.0, 0.0),
) -> None:
    ax = fig.add_axes(position, projection=ccrs.Robinson(), label=label)
    ax.set_global()
    ocean_artist = ax.add_feature(cfeature.NaturalEarthFeature("physical", "ocean", "50m", facecolor="gainsboro"), zorder=0)
    land_artist = ax.add_feature(cfeature.NaturalEarthFeature("physical", "land", "50m", facecolor="dimgrey"), zorder=1)
    coast_artist = ax.coastlines(resolution="50m", linewidth=0.18, color="0.25", zorder=3)
    ocean_artist.set_rasterized(True)
    land_artist.set_rasterized(True)
    coast_artist.set_rasterized(True)
    if polygon is None:
        polygon = poly_from_extent(bounds)
    ax.set_boundary(mpath.Path(robinson_projected_polygon_to_axes_verts(polygon)), transform=ax.transAxes)
    add_tile_markers(ax, table, column, cmap, norm)
    set_geo_outline(ax, edgecolor="white", linewidth=0.45)

    if markup_sub is not None:
        lon_min, lon_max = np.min(polygon[:, 0]), np.max(polygon[:, 0])
        lat_min, lat_max = np.min(polygon[:, 1]), np.max(polygon[:, 1])
        lon_mid = 0.5 * (lon_min + lon_max)
        lat_mid = 0.5 * (lat_min + lat_max)
        pos_lookup = {
            "lb": (lon_min, lat_min, "left", "bottom"),
            "lm": (lon_min, lat_mid, "left", "center"),
            "lt": (lon_min, lat_max, "left", "top"),
            "mb": (lon_mid, lat_min, "center", "bottom"),
            "mt": (lon_mid, lat_max, "center", "top"),
            "rb": (lon_max, lat_min, "right", "bottom"),
            "rm": (lon_max, lat_mid, "right", "center"),
            "rt": (lon_max, lat_max, "right", "top"),
        }
        x, y, ha, va = pos_lookup[sub_pos]
        if sub_pos[0] == "r":
            x -= 100000
        elif sub_pos[0] == "l":
            x += 100000
        if sub_pos[1] == "b":
            y += 100000
        elif sub_pos[1] == "t":
            y -= 100000
        x += sub_adj[0]
        y += sub_adj[1]
        ax.text(
            x,
            y,
            markup_sub,
            horizontalalignment=ha,
            verticalalignment=va,
            transform=ccrs.Robinson(),
            color="black",
            fontsize=5.5,
            bbox=dict(facecolor="white", alpha=1.0, linewidth=0.25, pad=1.2),
            fontweight="bold",
            zorder=40,
        )


def add_compartment_world_map(
    fig: plt.Figure,
    table: pd.DataFrame,
    position: list[float],
    column: str,
    cmap,
    norm,
    cbar_label: str,
    panel_label: str,
) -> None:
    yr = position[3]
    yb = position[1] + 0.01

    add_inset(fig, table, column, cmap, norm, [-0.735, yb - 0.06 * yr, 2.7, 2.7 * yr], [-5500000, -3400000, -8000000, -5900000], "AP", markup_sub="19a", sub_pos="lt", sub_adj=(30000, -170000))
    add_inset(fig, table, column, cmap, norm, [-0.4205, yb - 0.1513 * yr, 2.7, 2.7 * yr], [-9500000, -5600000, -7930000, -7320000], "AW", markup_sub="19b", sub_pos="lt", sub_adj=(70000, -15000))
    add_inset(fig, table, column, cmap, norm, [-0.7087, yb - 0.1872 * yr, 2.7, 2.7 * yr], [-1960000, 2250000, -7700000, -7080000], "AE1", markup_sub="19c", sub_pos="lt", sub_adj=(0, -25000))
    add_inset(fig, table, column, cmap, norm, [-1.0585, yb - 0.113 * yr, 2.7, 2.7 * yr], [2450000, 7500000, -7570000, -6720000], "AE2", markup_sub="19d", sub_pos="rt", sub_adj=(-1075000, -45000))
    add_inset(fig, table, column, cmap, norm, [-1.2960, yb - 0.109 * yr, 2.7, 2.7 * yr], [9430000, 11900000, -8200000, -6770000], "AE3", markup_sub="19e", sub_pos="lm", sub_adj=(15000, 0))
    add_inset(fig, table, column, cmap, norm, [-0.765, yb - 0.4694 * yr, 2.7, 2.7 * yr], [-7340000, -5100000, -5900000, 0], "South America", markup_sub="16-17", sub_pos="lm", sub_adj=(20000, -2000000))
    add_inset(fig, table, column, cmap, norm, [-0.865, yb - 1.645 * yr, 2.7, 2.7 * yr], [0, 1500000, 4500000, 5400000], "Europe", markup_sub="11", sub_pos="lt", sub_adj=(20000, -135000))
    add_inset(fig, table, column, cmap, norm, [-1.122, yb - 1.685 * yr, 2.7, 2.7 * yr], [3200000, 4800000, 3300000, 4800000], "Caucasus", markup_sub="12", sub_pos="lb", sub_adj=(60000, 20000))
    add_inset(fig, table, column, cmap, norm, [-1.56, yb - 0.3235 * yr, 2.7, 2.7 * yr], [13750000, 15225000, -5400000, -3800000], "New Zealand", markup_sub="18", sub_pos="lt", sub_adj=(135000, -100000))
    add_inset(fig, table, column, cmap, norm, [-1.3991, yb - 1.73 * yr, 2.7, 2.7 * yr], [11500000, 13200000, 5100000, 6700000], "Kamchatka", markup_sub="10f", sub_pos="lb", sub_adj=(365000, 15000))
    add_inset(fig, table, column, cmap, norm, [-1.216, yb - 1.5835 * yr, 2.7, 2.7 * yr], [5750000, 9550000, 2650000, 5850000], "HMA", markup_sub="13-15\n& 10a", sub_pos="rt", sub_adj=(-20000, -100000))
    poly_arctic = np.array([(-6050000, 7650000), (-5400000, 6800000), (-4950000, 6400000), (-3870000, 5710000), (-2500000, 5710000), (-2000000, 5720000), (1350000, 5720000), (2300000, 6600000), (6500000, 6600000), (6500000, 8400000), (-6050000, 8400000), (-6050000, 7650000)])
    add_inset(fig, table, column, cmap, norm, [-0.8675, yb - 1.715 * yr, 2.7, 2.7 * yr], [-6060000, 6420000, 6100000, 8400000], "Arctic", polygon=poly_arctic, markup_sub="03-09\n& 10b", sub_pos="mt", sub_adj=(-200000, -820000))
    poly_na = np.array([(-13600000, 5600000), (-13600000, 6000000), (-12900000, 6000000), (-12900000, 6800000), (-12500000, 6800000), (-11500000, 7420000), (-9000000, 7420000), (-9000000, 3750000), (-11000000, 3750000), (-11000000, 5600000), (-13600000, 5600000)])
    add_inset(fig, table, column, cmap, norm, [-0.15, yb - 1.885 * yr, 2.7, 2.7 * yr], [-13600000, -9000000, 3700000, 7350000], "North America", polygon=poly_na, markup_sub="01-02", sub_pos="rm", sub_adj=(-1970000, 230000))

    title_ax = fig.add_axes([0.18, position[1], 0.08, 0.001])
    title_ax.axis("off")
    title_ax.text(0.02, 2.3, panel_label, transform=title_ax.transAxes, fontsize=9, fontweight="bold", ha="left", va="bottom")
    cax = fig.add_axes([0.19, position[1] - 0.035, 0.62, 0.018])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="both")
    cbar.set_label(cbar_label, fontsize=8, labelpad=2.0)
    cbar.ax.tick_params(labelsize=7, length=2.0, width=0.5)


def run() -> Path:
    table = aggregate_tiles()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    fig.text(
        0.5,
        0.986,
        "Decadal change between 2000–2009 and 2010–2019",
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
    )
    panel_height = 1.0 / 2.775
    add_compartment_world_map(
        fig,
        table,
        [0, 0.605, 1, panel_height],
        "dhdt_change_m_yr",
        plt.get_cmap("RdBu"),
        mpl.colors.TwoSlopeNorm(vmin=-1.2, vcenter=0.0, vmax=1.2),
        r"Decadal change in elevation change rate (m yr$^{-1}$)",
        "a",
    )
    add_compartment_world_map(
        fig,
        table,
        [0, 0.115, 1, panel_height],
        "rho_change_kg_m3",
        plt.get_cmap("PuOr_r"),
        mpl.colors.TwoSlopeNorm(vmin=-250.0, vcenter=0.0, vmax=250.0),
        r"Decadal change in effective density (kg m$^{-3}$)",
        "b",
    )
    out_bbox = Bbox.from_bounds(
        0.0,
        BOTTOM_CROP_IN,
        FIG_WIDTH_IN,
        FIG_HEIGHT_IN + TOP_EXTRA_IN - BOTTOM_CROP_IN,
    )
    fig.savefig(OUT_PNG, dpi=300, bbox_inches=out_bbox)
    plt.close(fig)
    return OUT_PNG


if __name__ == "__main__":
    path = run()
    print(f"[done] Wrote figure: {path.resolve()}")
