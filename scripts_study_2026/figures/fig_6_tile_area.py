#!/usr/bin/env python3
"""Main figure 6: tile-level glacierized area and effective density."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import ShapelyFeature
from cartopy.io.shapereader import Reader
from scipy.spatial import ConvexHull

STUDY_DIR = Path(__file__).resolve().parents[1]
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

from study_paths import FIGURES_DIR, INPUT_CSV


# Define project paths
MAIN_GLACIER_OUTLINES = Path("/home/atom/data/inventory_products/RGI/buffered/rgi60_buff_diss.shp")
HILLSHADE_RASTER = Path("/home/atom/documents/paper/Hugonnet_2021/figures/world_robin_rs.tif")

# Define input and period controls
VARIANT_COL = "rho_variant"
VARIANT_TO_USE = "iteration9"
START_DATE = None
END_DATE = None

# Define map and marker controls
DPI = 300
FIG_WIDTH_INCH = 19.0
FIG_PANEL_ASPECT = 1.9716
FIG_PANEL_HEIGHT_INCH = FIG_WIDTH_INCH / FIG_PANEL_ASPECT
LEGEND_STRIP_HEIGHT_INCH = 2.0
AREA_MIN_KM2 = 0.2
MARKER_RADIUS_BASE = 12000.0
MARKER_RADIUS_SCALE = 1000.0
DENSITY_BOUNDS = np.array([700, 775, 850, 925, 1000], dtype=float)
DENSITY_CMAP_ENDPOINT_TRIM = 0.14
HILLSHADE_CROP_PAD_PX = 2
HILLSHADE_DOWNSAMPLE = 3
BACKGROUND_FEATURE_SCALE = "50m"

# Define output names
OUT_PNG = FIGURES_DIR / "FIG_main_06_tile_area_density.pdf"
_HILLSHADE_IMAGE = None
_HILLSHADE_CROPS = {}

# Embed editable TrueType fonts in PDF outputs
mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "pdf.compression": 9})


def first_existing(columns: pd.Index, candidates: list[str], required: bool = True) -> str | None:
    """Return the first matching column name

    :param columns: Available dataframe columns
    :param candidates: Candidate column names
    :param required: Raise when no column is found
    """
    for col in candidates:
        if col in columns:
            return col
    if required:
        raise KeyError(f"None of these columns exist: {candidates}")
    return None


def infer_period_filter(df: pd.DataFrame) -> tuple[float, float]:
    """Infer the representative full-period interval

    :param df: Full-model input table
    """
    if START_DATE is not None and END_DATE is not None:
        return float(START_DATE), float(END_DATE)
    spans = df[["start_date", "end_date"]].dropna().drop_duplicates().copy()
    spans["period_years"] = spans["end_date"] - spans["start_date"]
    row = spans.sort_values(["period_years", "start_date", "end_date"], ascending=[False, True, True]).iloc[0]
    return float(row["start_date"]), float(row["end_date"])


def representative_tile_origin(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return adaptive tile southwest coordinates

    :param lat: Point latitudes
    :param lon: Point longitudes
    """
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
    """Return adaptive tile center coordinates

    :param tile_lat: Tile southwest latitudes
    :param tile_lon: Tile southwest longitudes
    """
    high = np.abs(tile_lat) >= 60.0
    polar = np.abs(tile_lat) >= 74.0
    center_lat = tile_lat + 0.5
    center_lon = tile_lon + 0.5
    center_lon[high] = tile_lon[high] + 1.0
    center_lon[polar] = tile_lon[polar] + 1.0
    center_lat[polar] = tile_lat[polar] + 1.0
    center_lon = ((center_lon + 180.0) % 360.0) - 180.0
    return center_lat, center_lon


def read_full_model_input() -> pd.DataFrame:
    """Read the full-model input columns needed for aggregation"""
    header = pd.read_csv(INPUT_CSV, nrows=0).columns
    lat_col = first_existing(header, ["lat", "cenlat", "center_lat"])
    lon_col = first_existing(header, ["lon", "cenlon", "center_lon"])
    usecols = ["rgiid", "start_date", "end_date", "rho", "b", "area", lat_col, lon_col]
    if VARIANT_COL in header:
        usecols.append(VARIANT_COL)
    df = pd.read_csv(INPUT_CSV, usecols=usecols, low_memory=True, memory_map=True)
    df = df.rename(columns={lat_col: "lat", lon_col: "lon"})
    if VARIANT_COL not in df.columns:
        df[VARIANT_COL] = "single"
    return df


def aggregate_tiles() -> pd.DataFrame:
    """Aggregate glacier rows to adaptive map tiles"""
    df = read_full_model_input()
    if VARIANT_TO_USE is not None and VARIANT_TO_USE in set(df[VARIANT_COL].astype(str)):
        df = df.loc[df[VARIANT_COL].astype(str).eq(VARIANT_TO_USE)].copy()

    # Select the representative full period
    for col in ["start_date", "end_date", "rho", "b", "area", "lat", "lon"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    start_date, end_date = infer_period_filter(df)
    df = df.loc[np.isclose(df["start_date"], start_date) & np.isclose(df["end_date"], end_date)].copy()

    # Filter physically usable rows
    ok = (
        np.isfinite(df["rho"])
        & np.isfinite(df["b"])
        & np.isfinite(df["area"])
        & np.isfinite(df["lat"])
        & np.isfinite(df["lon"])
        & (df["area"] > 0)
        & (np.abs(df["rho"]) > 1.0e-6)
        & (~df["rho"].isin([-99999.0, 99999.0]))
    )
    df = df.loc[ok].copy()

    # Compute volume weights and tile origins
    df["dV_proxy"] = df["area"] * df["b"] / df["rho"]
    df["abs_dV_weight"] = np.abs(df["dV_proxy"])
    df["rho_num"] = df["rho"] * df["abs_dV_weight"]
    df["tile_lat"], df["tile_lon"] = representative_tile_origin(df["lat"].to_numpy(float), df["lon"].to_numpy(float))

    # Aggregate by tile with volume weighting
    out = (
        df.groupby(["tile_lat", "tile_lon"], sort=True, observed=True)
        .agg(
            area_km2=("area", "sum"),
            dV_proxy_m3=("dV_proxy", "sum"),
            abs_dV_weight=("abs_dV_weight", "sum"),
            rho_num=("rho_num", "sum"),
        )
        .reset_index()
    )
    out["effective_density_kg_m3"] = np.where(out["abs_dV_weight"] > 0, out["rho_num"] / out["abs_dV_weight"], np.nan)
    out["rho_dv"] = out["effective_density_kg_m3"]

    # Add tile centers and compact units
    center_lat, center_lon = tile_center(out["tile_lat"].to_numpy(float), out["tile_lon"].to_numpy(float))
    out["center_lat"] = center_lat
    out["center_lon"] = center_lon
    out["start_date"] = start_date
    out["end_date"] = end_date
    return out


def density_colormap() -> mpl.colors.LinearSegmentedColormap:
    """Build the blue-white-red density colormap with muted endpoints"""
    cb_val = np.linspace(DENSITY_CMAP_ENDPOINT_TRIM, 1.0 - DENSITY_CMAP_ENDPOINT_TRIM, len(DENSITY_BOUNDS))
    colors = [mpl.cm.bwr(v) for v in cb_val]
    coords = (DENSITY_BOUNDS - DENSITY_BOUNDS.min()) / (DENSITY_BOUNDS.max() - DENSITY_BOUNDS.min())
    return mpl.colors.LinearSegmentedColormap.from_list("rho_tile_cmap", list(zip(coords, colors)), N=1000)


def robinson_bounds_from_lonlat_polygon(polygon_coords: np.ndarray) -> tuple[float, float, float, float]:
    """Return projected Robinson bounds for a lon/lat polygon"""
    list_lat_interp = []
    list_lon_interp = []
    for i in range(len(polygon_coords) - 1):
        lon_interp = np.linspace(polygon_coords[i][0], polygon_coords[i + 1][0], 80)
        lat_interp = np.linspace(polygon_coords[i][1], polygon_coords[i + 1][1], 80)
        list_lon_interp.append(lon_interp)
        list_lat_interp.append(lat_interp)
    all_lon_interp = np.concatenate(list_lon_interp)
    all_lat_interp = np.concatenate(list_lat_interp)
    robin = coord_transform(ccrs.PlateCarree(), ccrs.Robinson(), all_lon_interp, all_lat_interp)
    x = robin[:, 0]
    y = robin[:, 1]
    ok = np.isfinite(x) & np.isfinite(y)
    if not np.any(ok):
        robin_crs = ccrs.Robinson()
        return (*robin_crs.x_limits, *robin_crs.y_limits)
    return float(np.min(x[ok])), float(np.max(x[ok])), float(np.min(y[ok])), float(np.max(y[ok]))


def add_hillshade(ax, polygon: np.ndarray | None = None) -> None:
    """Add the global Robinson hillshade used in the archived tile-area map"""
    global _HILLSHADE_IMAGE
    if not HILLSHADE_RASTER.exists():
        return
    if _HILLSHADE_IMAGE is None:
        _HILLSHADE_IMAGE = plt.imread(HILLSHADE_RASTER)

    robin = ccrs.Robinson()
    x0, x1 = robin.x_limits
    y0, y1 = robin.y_limits
    height, width = _HILLSHADE_IMAGE.shape[:2]
    if polygon is None:
        xmin, xmax = x0, x1
        ymin, ymax = y0, y1
    else:
        xmin, xmax, ymin, ymax = robinson_bounds_from_lonlat_polygon(polygon)

    # Crop in projected Robinson coordinates before adding the raster.
    # This avoids embedding the full world hillshade in every exploded inset.
    col0 = int(np.floor((xmin - x0) / (x1 - x0) * width)) - HILLSHADE_CROP_PAD_PX
    col1 = int(np.ceil((xmax - x0) / (x1 - x0) * width)) + HILLSHADE_CROP_PAD_PX
    row0 = int(np.floor((y1 - ymax) / (y1 - y0) * height)) - HILLSHADE_CROP_PAD_PX
    row1 = int(np.ceil((y1 - ymin) / (y1 - y0) * height)) + HILLSHADE_CROP_PAD_PX
    col0 = max(0, min(width - 1, col0))
    col1 = max(col0 + 1, min(width, col1))
    row0 = max(0, min(height - 1, row0))
    row1 = max(row0 + 1, min(height, row1))

    key = (row0, row1, col0, col1)
    if key not in _HILLSHADE_CROPS:
        _HILLSHADE_CROPS[key] = np.ascontiguousarray(_HILLSHADE_IMAGE[row0:row1:HILLSHADE_DOWNSAMPLE, col0:col1:HILLSHADE_DOWNSAMPLE])
    image = _HILLSHADE_CROPS[key]
    crop_x0 = x0 + col0 / width * (x1 - x0)
    crop_x1 = x0 + col1 / width * (x1 - x0)
    crop_y1 = y1 - row0 / height * (y1 - y0)
    crop_y0 = y1 - row1 / height * (y1 - y0)

    ax.imshow(
        image,
        origin="upper",
        extent=[crop_x0, crop_x1, crop_y0, crop_y1],
        transform=robin,
        cmap="Greys_r",
        interpolation="none",
        alpha=0.35,
        zorder=2,
    )


def coord_transform(orig_crs, target_crs, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Transform coordinate arrays between Cartopy projections"""
    return target_crs.transform_points(orig_crs, x, y)


def poly_from_extent(extent: list[float]) -> np.ndarray:
    """Return polygon coordinates from lon/lat bounds"""
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


def rect_units_to_verts(rect_u: list[float]) -> np.ndarray:
    """Return rectangle vertices from x, y, width, height"""
    return np.array(
        [
            [rect_u[0], rect_u[1]],
            [rect_u[0] + rect_u[2], rect_u[1]],
            [rect_u[0] + rect_u[2], rect_u[1] + rect_u[3]],
            [rect_u[0], rect_u[1] + rect_u[3]],
            [rect_u[0], rect_u[1]],
        ],
        dtype=float,
    )


def latlon_extent_to_robinson_axes_verts(polygon_coords: np.ndarray) -> np.ndarray:
    """Convert a lon/lat polygon to normalized Robinson axes vertices"""
    list_lat_interp = []
    list_lon_interp = []
    for i in range(len(polygon_coords) - 1):
        lon_interp = np.linspace(polygon_coords[i][0], polygon_coords[i + 1][0], 50)
        lat_interp = np.linspace(polygon_coords[i][1], polygon_coords[i + 1][1], 50)
        list_lon_interp.append(lon_interp)
        list_lat_interp.append(lat_interp)

    all_lon_interp = np.concatenate(list_lon_interp)
    all_lat_interp = np.concatenate(list_lat_interp)
    robin = coord_transform(ccrs.PlateCarree(), ccrs.Robinson(), all_lon_interp, all_lat_interp)
    robin_crs = ccrs.Robinson()
    x_min, x_max = robin_crs.x_limits
    y_min, y_max = robin_crs.y_limits
    ext_robin_x = x_max - x_min
    ext_robin_y = y_max - y_min
    verts = robin.copy()
    verts[:, 0] = (verts[:, 0] - x_min) / ext_robin_x
    verts[:, 1] = (verts[:, 1] - y_min) / ext_robin_y
    return verts[:, 0:2]


def set_geo_outline(ax, edgecolor: str | None = None, linewidth: float | None = None, visible: bool | None = None) -> None:
    """Set the outline of a Cartopy GeoAxes across Cartopy versions"""
    try:
        outline = ax.outline_patch
        if edgecolor is not None:
            outline.set_edgecolor(edgecolor)
        if linewidth is not None:
            outline.set_linewidth(linewidth)
        if visible is not None:
            outline.set_visible(visible)
    except AttributeError:
        spine = ax.spines.get("geo")
        if spine is not None:
            if edgecolor is not None:
                spine.set_edgecolor(edgecolor)
            if linewidth is not None:
                spine.set_linewidth(linewidth)
            if visible is not None:
                spine.set_visible(visible)


def set_robinson_extent(ax, extent: list[float]) -> None:
    """Set a Robinson map extent, using set_global for whole-world extents"""
    if extent[0] <= -179.0 and extent[1] >= 179.0 and extent[2] <= -89.0 and extent[3] >= 89.0:
        ax.set_global()
    else:
        ax.set_extent(extent, ccrs.Geodetic())


def set_polygon_boundary(ax, polygon: np.ndarray) -> None:
    """Set an inset boundary when the polygon projects to a valid path"""
    verts = latlon_extent_to_robinson_axes_verts(polygon)
    verts = verts[np.all(np.isfinite(verts), axis=1)]
    if len(verts) >= 3:
        ax.set_boundary(mpath.Path(verts), transform=ax.transAxes)


def shades_main_to_inset(
    fig: plt.Figure,
    main_pos: list[float],
    inset_pos: list[float],
    inset_verts: np.ndarray,
    label: str,
) -> None:
    """Draw the light connection shade between the central map and an inset"""
    inset_verts = inset_verts[np.all(np.isfinite(inset_verts), axis=1)]
    if len(inset_verts) < 3:
        return

    center_x = main_pos[0] + main_pos[2] / 2
    center_y = main_pos[1] + main_pos[3] / 2
    left_x = center_x - inset_pos[2] / 2
    left_y = center_y - inset_pos[3] / 2

    shade_ax = fig.add_axes([left_x, left_y, inset_pos[2], inset_pos[3]], projection=ccrs.Robinson(), label=label + "_shade")
    set_robinson_extent(shade_ax, [-179.99, 179.99, -89.99, 89.99])
    shade_ax.patch.set_alpha(0)
    set_geo_outline(shade_ax, linewidth=0)

    robin_crs = ccrs.Robinson()
    x_min, x_max = robin_crs.x_limits
    y_min, y_max = robin_crs.y_limits
    ext_robin_x = x_max - x_min
    ext_robin_y = y_max - y_min

    inset_mod_x = inset_verts[:, 0] + (inset_pos[0] - left_x) / inset_pos[2]
    inset_mod_y = inset_verts[:, 1] + (inset_pos[1] - left_y) / inset_pos[3]
    main_mod_x = (inset_verts[:, 0] * main_pos[2] - left_x + main_pos[0]) / inset_pos[2]
    main_mod_y = (inset_verts[:, 1] * main_pos[3] - left_y + main_pos[1]) / inset_pos[3]

    points = np.array(
        list(zip(np.concatenate((inset_mod_x, main_mod_x)), np.concatenate((inset_mod_y, main_mod_y)))),
        dtype=float,
    )
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 3:
        return
    chull = ConvexHull(points)
    chull_robin_x = points[chull.vertices, 0] * ext_robin_x + x_min
    chull_robin_y = points[chull.vertices, 1] * ext_robin_y + y_min

    shade_ax.plot(
        main_mod_x * ext_robin_x + x_min,
        main_mod_y * ext_robin_y + y_min,
        color="white",
        linewidth=1.5,
        transform=ccrs.Robinson(),
        zorder=2,
    )
    shade_ax.fill(chull_robin_x, chull_robin_y, transform=ccrs.Robinson(), color="indigo", alpha=0.05, zorder=1)


def only_shade(fig: plt.Figure, position: list[float], bounds: list[float], label: str, polygon: np.ndarray | None = None) -> None:
    """Draw a shade connector without adding a visible inset"""
    main_pos = [0.375, 0.21, 0.25, 0.25]
    if polygon is None and bounds is not None:
        polygon = poly_from_extent(bounds)
    shades_main_to_inset(fig, main_pos, position, latlon_extent_to_robinson_axes_verts(polygon), label=label)


def add_tile_markers(ax, tiles: pd.DataFrame, extent: list[float] | None = None, label: str | None = None) -> None:
    """Add tile circles to a Cartopy axis

    :param ax: Cartopy axis
    :param tiles: Aggregated tile table
    :param extent: Optional lon/lat extent
    """
    sub = tiles.copy()
    if extent is not None:
        lon0, lon1, lat0, lat1 = extent
        lat_min, lat_max = min(lat0, lat1), max(lat0, lat1)
        if lon0 <= lon1:
            in_lon = (sub["center_lon"] >= lon0) & (sub["center_lon"] <= lon1)
        else:
            in_lon = (sub["center_lon"] >= lon0) | (sub["center_lon"] <= lon1)
        sub = sub.loc[in_lon & (sub["center_lat"] >= lat_min) & (sub["center_lat"] <= lat_max)].copy()

    if label == "Arctic West":
        sub = sub.loc[~(((sub["center_lat"] < 71) & (sub["center_lon"] > 60)) | ((sub["center_lat"] < 76) & (sub["center_lon"] > 100)))]
    if label == "HMA":
        sub = sub.loc[sub["center_lat"] < 46]

    # Draw circles scaled by glacierized area.
    cmap = density_colormap()
    circles = []
    colors = []
    for row in sub.itertuples(index=False):
        if not np.isfinite(row.area_km2) or row.area_km2 <= AREA_MIN_KM2:
            continue
        radius = MARKER_RADIUS_BASE + np.sqrt(row.area_km2) * MARKER_RADIUS_SCALE
        if np.isfinite(row.effective_density_kg_m3):
            frac = np.clip((row.effective_density_kg_m3 - DENSITY_BOUNDS.min()) / (DENSITY_BOUNDS.max() - DENSITY_BOUNDS.min()), 0.0001, 0.9999)
            color = cmap(frac)
        else:
            color = plt.cm.Greys(0.7)
        xy = ccrs.Robinson().transform_point(row.center_lon, row.center_lat, ccrs.PlateCarree())
        circles.append(mpatches.Circle(xy=xy, radius=radius))
        colors.append(color)

    if circles:
        collection = PatchCollection(circles, facecolors=colors, edgecolors="none", linewidths=0, alpha=1.0, zorder=30)
        collection.set_transform(ccrs.Robinson()._as_mpl_transform(ax))
        ax.add_collection(collection)


def add_markup(
    fig: plt.Figure,
    position: list[float],
    extent: list[float],
    polygon: np.ndarray,
    label: str,
    markup: str | None = None,
    markpos: str = "left",
    markadj: float = 0,
    markup_sub: str | None = None,
    sub_pos: str = "lt",
) -> None:
    """Add region labels and small subregion letters"""
    if markup is not None:
        if markpos == "left":
            lon_upleft = np.min(polygon[:, 0])
            lat_upleft = np.max(polygon[:, 1])
        else:
            lon_upleft = np.max(polygon[:, 0])
            lat_upleft = np.max(polygon[:, 1])

        robin = coord_transform(ccrs.PlateCarree(), ccrs.Robinson(), np.array([lon_upleft]), np.array([lat_upleft]))
        rob_x = robin[0][0] - 50000 if markpos == "right" else robin[0][0] + 50000
        rob_y = robin[0][1]
        size_y = 200000
        size_x = 80000 * len(markup) + markadj
        sub_ax_2 = fig.add_axes(position, projection=ccrs.Robinson(), label=label + "_markup")
        set_robinson_extent(sub_ax_2, extent)
        sub_ax_2.patch.set_alpha(0)
        set_geo_outline(sub_ax_2, linewidth=0)
        sub_ax_2.text(
            rob_x,
            rob_y + 50000,
            markup,
            horizontalalignment=markpos,
            verticalalignment="bottom",
            transform=ccrs.Robinson(),
            color="black",
            fontsize=12,
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=1, edgecolor="none"),
        )

    if markup_sub is not None:
        lon_min = np.min(polygon[:, 0])
        lon_max = np.max(polygon[:, 0])
        lon_mid = 0.5 * (lon_min + lon_max)
        lat_min = np.min(polygon[:, 1])
        lat_max = np.max(polygon[:, 1])
        lat_mid = 0.5 * (lat_min + lat_max)
        lat_midup = lat_min + 0.87 * (lat_max - lat_min)
        robin = coord_transform(
            ccrs.PlateCarree(),
            ccrs.Robinson(),
            np.array([lon_min, lon_min, lon_min, lon_mid, lon_mid, lon_max, lon_max, lon_max, lon_min], dtype=float),
            np.array([lat_min, lat_mid, lat_max, lat_min, lat_max, lat_min, lat_mid, lat_max, lat_midup], dtype=float),
        )

        pos_lookup = {
            "lb": (0, "left", "bottom"),
            "lm": (1, "left", "center"),
            "lm2": (8, "left", "center"),
            "lt": (2, "left", "top"),
            "mb": (3, "center", "bottom"),
            "mt": (4, "center", "top"),
            "rb": (5, "right", "bottom"),
            "rm": (6, "right", "center"),
            "rt": (7, "right", "top"),
        }
        idx, ha, va = pos_lookup[sub_pos]
        rob_x = robin[idx][0]
        rob_y = robin[idx][1]
        if sub_pos[0] == "r":
            rob_x -= 50000
        elif sub_pos[0] == "l":
            rob_x += 50000
        if sub_pos[1] == "b":
            rob_y += 50000
        elif sub_pos[1] == "t":
            rob_y -= 50000

        sub_ax_3 = fig.add_axes(position, projection=ccrs.Robinson(), label=label + "_markup_sub")
        set_robinson_extent(sub_ax_3, extent)
        sub_ax_3.patch.set_alpha(0)
        set_geo_outline(sub_ax_3, linewidth=0)
        sub_ax_3.text(
            rob_x,
            rob_y,
            markup_sub,
            horizontalalignment=ha,
            verticalalignment=va,
            transform=ccrs.Robinson(),
            color="black",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=1, edgecolor="none"),
            fontweight="bold",
            zorder=25,
        )


def add_inset(
    fig: plt.Figure,
    tiles: pd.DataFrame,
    extent: list[float],
    position: list[float],
    bounds: list[float] | None = None,
    label: str | None = None,
    polygon: np.ndarray | None = None,
    shades: bool = True,
    hillshade: bool = True,
    main: bool = False,
    markup: str | None = None,
    markpos: str = "left",
    markadj: float = 0,
    markup_sub: str | None = None,
    sub_pos: str = "lt",
):
    """Add one map inset using the archived Fig. 6 layout"""
    main_pos = [0.375, 0.21, 0.25, 0.25]
    if polygon is None and bounds is not None:
        polygon = poly_from_extent(bounds)

    if shades and polygon is not None:
        shades_main_to_inset(fig, main_pos, position, latlon_extent_to_robinson_axes_verts(polygon), label=label or "inset")

    sub_ax = fig.add_axes(position, projection=ccrs.Robinson(), label=label)
    set_robinson_extent(sub_ax, extent)
    ocean_artist = sub_ax.add_feature(cfeature.NaturalEarthFeature("physical", "ocean", BACKGROUND_FEATURE_SCALE, facecolor="gainsboro"), zorder=0)
    land_artist = sub_ax.add_feature(cfeature.NaturalEarthFeature("physical", "land", BACKGROUND_FEATURE_SCALE, facecolor="dimgrey"), zorder=1)
    ocean_artist.set_rasterized(True)
    land_artist.set_rasterized(True)
    if hillshade:
        add_hillshade(sub_ax, polygon=polygon)
        ocean_dampener = sub_ax.add_feature(
            cfeature.NaturalEarthFeature("physical", "ocean", BACKGROUND_FEATURE_SCALE, facecolor="white"),
            alpha=0.35,
            zorder=2.5,
        )
        ocean_dampener.set_rasterized(True)
    if bounds is not None and polygon is not None:
        set_polygon_boundary(sub_ax, polygon)

    if main:
        if MAIN_GLACIER_OUTLINES.exists():
            shape_feature = ShapelyFeature(
                Reader(str(MAIN_GLACIER_OUTLINES)).geometries(),
                ccrs.PlateCarree(),
                alpha=1,
                facecolor="indigo",
                linewidth=0.5,
                edgecolor="indigo",
                zorder=30,
            )
            shape_artist = sub_ax.add_feature(shape_feature)
            shape_artist.set_rasterized(True)
    else:
        add_tile_markers(sub_ax, tiles, extent=bounds if bounds is not None else extent, label=label)
    add_markup(fig, position, extent, polygon if polygon is not None else poly_from_extent(extent), label or "inset", markup, markpos, markadj, markup_sub, sub_pos)
    set_geo_outline(sub_ax, edgecolor="lightgrey" if main else "white")
    return sub_ax


def add_legends(
    fig,
    *,
    colorbar_label: str = r"Effective density of volume change $\rho_{\Delta V}$ (kg m$^{-3}$)",
    panel_height_inch: float | None = None,
    total_height_inch: float | None = None,
    y_offset_inch: float = 0.0,
    zorder: float | None = None,
) -> None:
    """Add the archived Fig. 2 legend layout"""

    def add_panel_axes(position: list[float], **kwargs):
        if panel_height_inch is None:
            mapped_position = position
        else:
            if total_height_inch is None:
                raise ValueError("Need total_height_inch when drawing legend in a merged figure.")
            mapped_position = [
                position[0],
                (position[1] * panel_height_inch + y_offset_inch) / total_height_inch,
                position[2],
                position[3] * panel_height_inch / total_height_inch,
            ]
        ax = fig.add_axes(mapped_position, **kwargs)
        ax.patch.set_alpha(0)
        if zorder is not None:
            ax.set_zorder(zorder)
        return ax

    axleg = add_panel_axes([-0.248, -0.86, 2, 2], projection=ccrs.Robinson(), label="legend_area")
    axleg.set_extent([-179.99, 179.99, -89.99, 89.99], ccrs.Geodetic())
    set_geo_outline(axleg, linewidth=0)
    u = 0
    rad_tot = 0
    for area in [100, 1000, 10000]:
        radius = MARKER_RADIUS_BASE + np.sqrt(area) * MARKER_RADIUS_SCALE
        axleg.add_patch(
            mpatches.Circle(
                xy=[-700000 + rad_tot + u * 600000, 0],
                radius=radius,
                edgecolor="black",
                transform=ccrs.Robinson(),
                fill=False,
                zorder=30,
            )
        )
        u += 1
        rad_tot += radius
    axleg.text(0, -2.5, "100     1000     10000", transform=ccrs.Geodetic(), ha="center", va="center", fontsize=12)
    axleg.text(0, -5, r"Glacierized area (km$^2$)", transform=ccrs.Geodetic(), ha="center", va="center", fontsize=12)
    axleg.set_boundary(mpath.Path(rect_units_to_verts([-10000000, -10000000, 20000000, 20000000])), transform=axleg.transAxes)

    axleg4 = add_panel_axes([0, -0.37, 1, 1], projection=ccrs.Robinson(), label="legend_regions")
    axleg4.set_extent([-179.99, 179.99, -89.99, 89.99], ccrs.Geodetic())
    set_geo_outline(axleg4, linewidth=0)
    axleg4.set_boundary(mpath.Path(latlon_extent_to_robinson_axes_verts(poly_from_extent([-10, 10, -10, 10]))), transform=axleg4.transAxes)

    def region_text(x: float, y: float, text: str, bold: bool = False, size: int = 10, ha: str = "left") -> None:
        axleg4.text(
            x,
            y,
            text,
            fontsize=size,
            fontweight="bold" if bold else "normal",
            horizontalalignment=ha,
            verticalalignment="top",
            transform=ccrs.Robinson(),
        )

    region_text(13500000, 900000, "North\nAsia (10)", bold=True, size=11, ha="center")
    for x, y, letter, name in [
        (11000000, 250000, "a.", "Altay and\nSayan"),
        (11000000, -350000, "b.", "Ural"),
        (11000000, -650000, "c.", "North Siberia"),
        (14000000, 250000, "d.", "Bulunsky"),
        (14000000, -50000, "e.", "Cherskiy and\nSuntar Khayata"),
        (14000000, -650000, "f.", "Kamchatka Krai"),
    ]:
        region_text(x, y, letter, bold=True)
        region_text(x + 400000, y, name)

    region_text(-15000000, 900000, "Low\nLatitudes (16)", bold=True, size=11, ha="center")
    for y, letter, name in [
        (250000, "a.", "Tropical Andes"),
        (-50000, "b.", "Mexico"),
        (-350000, "c.", "East Africa"),
        (-650000, "d.", "New Guinea"),
    ]:
        region_text(-16500000, y, letter, bold=True)
        region_text(-16100000, y, name)

    region_text(-11200000, 900000, "Antarctic and\nSubantarctic (19)", bold=True, size=10, ha="center")
    for y, letter, name, dx in [
        (250000, "a.", "West and Peninsula", 400000),
        (-100000, "b.", "South Georgia\nand Central Islands", 400000),
        (-750000, "c,e.", "East", 700000),
        (-1050000, "d.", "Kerguelen and\nHeard Islands", 400000),
    ]:
        region_text(-13000000, y, letter, bold=True)
        region_text(-13000000 + dx, y, name)

    axleg2 = add_panel_axes([0, 0, 1, 1], projection=ccrs.Robinson(), label="legend_colorbar")
    set_geo_outline(axleg2, linewidth=0)
    norm = mpl.colors.Normalize(vmin=DENSITY_BOUNDS.min(), vmax=DENSITY_BOUNDS.max())
    sm = plt.cm.ScalarMappable(cmap=density_colormap(), norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=axleg2, ticks=DENSITY_BOUNDS, orientation="horizontal", extend="both", shrink=0.35)
    cb.ax.tick_params(labelsize=12)
    cb.set_label(colorbar_label, fontsize=12)

    axleg5 = add_panel_axes([-0.23, -0.365, 1, 1], projection=ccrs.Robinson(), label="legend_minimap")
    axleg5.set_extent([-179.99, 179.99, -89.99, 89.99], ccrs.Geodetic())
    set_geo_outline(axleg5, linewidth=0)
    axleg5.set_boundary(mpath.Path(latlon_extent_to_robinson_axes_verts(poly_from_extent([-10, 10, -10, 10]))), transform=axleg5.transAxes)
    axleg5.add_patch(
        mpatches.Rectangle(
            (-250000, -250000),
            500000,
            500000,
            edgecolor="black",
            linewidth=1,
            transform=ccrs.Robinson(),
            facecolor="indigo",
            alpha=1,
        )
    )
    axleg5.text(0, -1000000, "Glacier\noutlines\n(minimap)", ha="center", va="center", fontsize=12, transform=ccrs.Robinson())


def run() -> Path:
    """Write the merged exploded map and legend figure"""
    tiles = aggregate_tiles()

    # Keep the map and legend in separate original-aspect panels. The inset
    # positions and disk radii depend on this physical canvas ratio.
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    merged_height_inch = FIG_PANEL_HEIGHT_INCH + LEGEND_STRIP_HEIGHT_INCH
    merged_fig = plt.figure(figsize=(FIG_WIDTH_INCH, merged_height_inch))
    fig, _legend_strip = merged_fig.subfigures(
        2,
        1,
        height_ratios=[FIG_PANEL_HEIGHT_INCH, LEGEND_STRIP_HEIGHT_INCH],
        hspace=0.0,
    )

    # Build the archived exploded-map layout in the upper panel
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()
    set_geo_outline(ax, linewidth=0)

    add_inset(
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
    add_inset(fig, tiles, [-179.99, 179.99, -89.99, 89.99], [-0.4, -0.065, 2, 2], bounds=[-158, -50, -62.5, -79], label="Antarctica_West", polygon=poly_aw, markup_sub="a", sub_pos="mb")

    poly_ae = np.array([(135, -81.5), (152, -63.7), (165, -65), (175, -70), (175, -81.25), (135, -81.75)])
    add_inset(fig, tiles, [-179.99, 179.99, -89.99, 89.99], [-0.71, -0.045, 2, 2], bounds=[130, 175, -64.5, -81], label="Antarctica_East", polygon=poly_ae, markup_sub="e", sub_pos="mb")

    poly_ac = np.array([(-25, -62), (106, -62), (80, -79.25), (-25, -79.25), (-25, -62)])
    add_inset(
        fig,
        tiles,
        [-179.99, 179.99, -89.99, 89.99],
        [-0.52, -0.065, 2, 2],
        bounds=[-25, 106, -62.5, -79],
        label="Antarctica_Center",
        polygon=poly_ac,
        markup="Antarctic and Subantarctic (19)",
        markpos="right",
        markup_sub="c",
        sub_pos="mb",
    )

    add_inset(fig, tiles, [-179.99, 179.99, -89.99, 89.99], [-0.68, -0.18, 2, 2], bounds=[64, 78, -48, -55], label="Antarctica_Australes", markup_sub="d", sub_pos="lt")
    add_inset(fig, tiles, [-179.99, 179.99, -89.99, 89.99], [-0.42, -0.155, 2, 2], bounds=[-40, -23, -53, -60], label="Antarctica_South_Georgia", markup_sub="b", sub_pos="rt")

    add_inset(fig, tiles, [-179.99, 179.99, -89.99, 89.99], [-0.52, -0.225, 2, 2], bounds=[-82, -65, 13, -57], label="Andes", markup="Low Latitudes (16) &\nSouthern Andes (17)", markup_sub="a", sub_pos="lm2")
    add_inset(fig, tiles, [-179.99, 179.99, -89.99, 89.99], [-0.352, -0.39, 2, 2], bounds=[-100, -95, 22, 16], label="Mexico", markup_sub="b", sub_pos="rb")
    add_inset(fig, tiles, [-179.99, 179.99, -89.99, 89.99], [-1.078, -0.24, 2, 2], bounds=[28, 42, 2, -5], label="Africa", markup_sub="c", sub_pos="rb")
    add_inset(fig, tiles, [-179.99, 179.99, -89.99, 89.99], [-1.640, -0.3, 2, 2], bounds=[133, 140, -2, -7], label="Indonesia", markup_sub="d", sub_pos="rb")

    poly_arctic = np.array([(-105, 84.5), (115, 84.5), (110, 68), (30, 68), (18, 57), (-70, 57), (-100, 75), (-105, 84.5)])
    add_inset(fig, tiles, [-179.99, 179.99, -89.99, 89.99], [-0.48, -1.003, 2, 2], bounds=[-100, 106, 57, 84], label="Arctic West", polygon=poly_arctic, markup="Arctic (03-09)")

    add_inset(fig, tiles, [-179.99, 179.99, -89.99, 89.99], [-0.92, -0.17, 2, 2], bounds=[164, 176, -47, -40], label="New Zealand", markup="New Zealand (18)", markpos="right")

    poly_na = np.array([(-170, 72), (-140, 72), (-120, 63), (-101, 35), (-126, 35), (-165, 55), (-170, 72)])
    add_inset(fig, tiles, [-179.99, 179.99, -89.99, 89.99], [-0.1, -1.22, 2, 2], bounds=[-177, -105, 36, 70], label="North America", polygon=poly_na, markup="Alaska (01) & Western\nCanada and USA (02)")

    poly_asia = np.array([(148, 49), (160, 65), (178, 65), (170, 55), (160, 49), (148, 49)])
    poly_asia_ne = np.array([(142, 71), (142, 80), (163, 80), (155, 71), (142, 71)])
    add_inset(fig, tiles, [-179.99, 179.99, -89.99, 89.99], [-0.655, -1.165, 2, 2], bounds=[142, 160, 71, 80], polygon=poly_asia_ne, label="North Asia North E", markup_sub="d", sub_pos="rt")
    add_inset(fig, tiles, [-179.99, 179.99, -89.99, 89.99], [-0.565, -1.107, 2, 2], bounds=[87, 112, 68, 77], label="North Asia North W", markup_sub="c", sub_pos="mt")
    poly_asia_e2 = np.array([(125, 58), (125, 72), (153.8, 72), (148, 58), (125, 58)])
    only_shade(fig, [-0.71, -1.142, 2, 2], [125, 148, 58, 72], polygon=poly_asia_e2, label="tmp_NAE2")
    add_inset(fig, tiles, [-179.99, 179.99, -89.99, 89.99], [-0.822, -1.218, 2, 2], bounds=[128, 179.9, 50, 64.8], label="North Asia East", polygon=poly_asia, markup_sub="f", sub_pos="lb")
    add_inset(fig, tiles, [-179.99, 179.99, -89.99, 89.99], [-0.71, -1.142, 2, 2], bounds=[125, 148, 58, 72], polygon=poly_asia_e2, label="North Asia East 2", markup_sub="e", sub_pos="lb", shades=False)

    only_shade(fig, [-0.517, -1.035, 2, 2], [53, 70, 62, 69.8], label="tmp_NAW")
    add_inset(fig, tiles, [-179.99, 179.99, -89.99, 89.99], [-0.74, -1, 2, 2], bounds=[82, 120, 45.5, 58.9], label="South Asia North", markup="North Asia (10)", markup_sub="a", sub_pos="mb")
    add_inset(fig, tiles, [-179.99, 179.99, -89.99, 89.99], [-0.512, -1.035, 2, 2], bounds=[53, 70, 62, 69.8], label="North Asia West", markup_sub="b", sub_pos="lb", shades=False)

    add_inset(fig, tiles, [-179.99, 179.99, -89.99, 89.99], [-0.685, -1.065, 2, 2], bounds=[65, 105, 46.5, 25], label="HMA", markup="High Mountain Asia (13-15)")
    add_inset(fig, tiles, [-179.99, 179.99, -89.99, 89.99], [-0.58, -0.982, 2, 2], bounds=[-4.9, 19, 38.2, 50.5], label="Europe", markup="Central Europe (11)")
    add_inset(fig, tiles, [-179.99, 179.99, -89.99, 89.99], [-0.66, -0.89, 2, 2], bounds=[38, 54, 29, 44.75], label="Middle East", markup="Caucasus (12)")

    legend_background = merged_fig.add_axes(
        [0, 0, 1, LEGEND_STRIP_HEIGHT_INCH / merged_height_inch],
        label="legend_background",
        zorder=1000,
    )
    legend_background.patch.set_facecolor("white")
    legend_background.patch.set_alpha(1.0)
    legend_background.set_axis_off()
    add_legends(
        merged_fig,
        panel_height_inch=FIG_PANEL_HEIGHT_INCH,
        total_height_inch=merged_height_inch,
        y_offset_inch=0.0,
        zorder=1001,
    )

    merged_fig.savefig(OUT_PNG, dpi=DPI)
    plt.close(merged_fig)
    return OUT_PNG


if __name__ == "__main__":
    path = run()
    print(f"[done] Wrote figure: {path}")
