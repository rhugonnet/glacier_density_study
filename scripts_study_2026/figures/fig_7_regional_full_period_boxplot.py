#!/usr/bin/env python3
"""Main figure 7: full-period regional distributions of effective density."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D

STUDY_DIR = Path(__file__).resolve().parents[1]
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

from study_paths import FIGURES_DIR, INPUT_CSV


OUT_PNG = FIGURES_DIR / "FIG_main_07_regional_full_period_boxplot.pdf"

REFERENCE_VARIANT = "iteration9"
RHO_SENTINELS = {-99999.0, 99999.0}
REFERENCE_RHO = 850.0
REFERENCE_SIGMA = 60.0

RGI_REGION_NAMES = {
    1: "Alaska",
    2: "Western Canada and USA",
    3: "Arctic Canada North",
    4: "Arctic Canada South",
    5: "Greenland Periphery",
    6: "Iceland",
    7: "Svalbard and Jan Mayen",
    8: "Scandinavia",
    9: "Russian Arctic",
    10: "North Asia",
    11: "Central Europe",
    12: "Caucasus and Middle East",
    13: "Central Asia",
    14: "South Asia West",
    15: "South Asia East",
    16: "Low Latitudes",
    17: "Southern Andes",
    18: "New Zealand",
    19: "Subantarctic and Antarctic Islands",
}

mpl.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 11.5,
    "legend.fontsize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def first_existing(columns: pd.Index, candidates: list[str]) -> str:
    """Return the first existing column from a list of candidates."""
    for col in candidates:
        if col in columns:
            return col
    raise KeyError(f"None of these columns exist: {candidates}")


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    """Finite weighted quantile."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(ok):
        return np.nan
    values = values[ok]
    weights = weights[ok]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = (np.cumsum(weights) - 0.5 * weights) / np.sum(weights)
    return float(np.interp(quantile, cdf, values))


class HandlerHorizontalWhisker(HandlerBase):
    """Draw a small horizontal whisker with vertical end caps in the legend."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        """Create legend artists matching the horizontal boxplot whiskers."""
        color = orig_handle.get_color()
        lw = orig_handle.get_linewidth()
        y = ydescent + 0.5 * height
        x0 = xdescent + 0.12 * width
        x1 = xdescent + 0.88 * width
        cap = 0.28 * height
        artists = [
            Line2D([x0, x1], [y, y], color=color, lw=lw, transform=trans),
            Line2D([x0, x0], [y - cap, y + cap], color=color, lw=lw, transform=trans),
            Line2D([x1, x1], [y - cap, y + cap], color=color, lw=lw, transform=trans),
        ]
        return artists


class HandlerVerticalMedian(HandlerBase):
    """Draw a vertical median line in the legend."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        """Create legend artists matching the boxplot median line."""
        color = orig_handle.get_color()
        lw = orig_handle.get_linewidth()
        x = xdescent + 0.5 * width
        artists = [Line2D([x, x], [ydescent + 0.15 * height, ydescent + 0.85 * height], color=color, lw=lw, transform=trans)]
        return artists


class HandlerReferenceBand(HandlerBase):
    """Draw the 850 +/- 60 ribbon with its dashed centre line."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        """Create a shaded legend band with a vertical dashed center line."""
        facecolor = orig_handle.get_facecolor()
        x0 = xdescent + 0.10 * width
        y0 = ydescent + 0.22 * height
        band_width = 0.80 * width
        band_height = 0.56 * height
        xmid = x0 + 0.5 * band_width
        band = mpl.patches.Rectangle(
            (x0, y0),
            band_width,
            band_height,
            facecolor=facecolor,
            edgecolor="none",
            transform=trans,
        )
        center = Line2D(
            [xmid, xmid],
            [y0, y0 + band_height],
            color="0.35",
            lw=1.0,
            ls="--",
            transform=trans,
        )
        return [band, center]


def read_full_period_rows() -> pd.DataFrame:
    """Read reference-variant full-period rows from the full-model table."""
    header = pd.read_csv(INPUT_CSV, nrows=0).columns
    rgi_col = first_existing(header, ["rgiid", "RGIId", "RGIId_float"])
    rho_col = first_existing(header, ["rho"])
    b_col = first_existing(header, ["b"])
    area_col = first_existing(header, ["area"])
    start_col = first_existing(header, ["start_date", "start_year"])
    end_col = first_existing(header, ["end_date", "end_year"])
    usecols = [rgi_col, rho_col, b_col, area_col, start_col, end_col, "rho_variant"]
    optional = [c for c in ["region", "rgi_region", "O1Region"] if c in header]
    usecols += optional
    df = pd.read_csv(INPUT_CSV, usecols=usecols, low_memory=True, memory_map=True)
    df = df.rename(
        columns={
            rgi_col: "rgiid",
            rho_col: "rho",
            b_col: "b",
            area_col: "area",
            start_col: "start_date",
            end_col: "end_date",
        }
    )
    df = df.loc[df["rho_variant"].astype(str).eq(REFERENCE_VARIANT)].copy()
    for col in ["rho", "b", "area", "start_date", "end_date"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.loc[~df["rho"].isin(RHO_SENTINELS)].copy()
    df["period_years"] = df["end_date"] - df["start_date"]
    full_period = float(df["period_years"].max())
    df = df.loc[np.isclose(df["period_years"], full_period)].copy()
    region_col = first_existing(df.columns, ["region", "rgi_region", "O1Region"])
    df["rgi_region"] = pd.to_numeric(df[region_col], errors="coerce")
    df["signed_dh"] = df["period_years"] * df["b"] / df["rho"]
    df["volume_weight"] = np.abs(df["area"] * df["signed_dh"])
    ok = np.isfinite(df["rho"]) & np.isfinite(df["rgi_region"]) & np.isfinite(df["volume_weight"]) & (df["volume_weight"] > 0)
    return df.loc[ok].copy()


def build_box_stats(df: pd.DataFrame) -> list[dict[str, object]]:
    """Build Matplotlib boxplot stats from weighted regional quantiles."""
    stats = []
    for region in range(1, 20):
        g = df.loc[df["rgi_region"].eq(region)]
        values = g["rho"].to_numpy(float)
        weights = g["volume_weight"].to_numpy(float)
        stats.append(
            {
                "label": f"{region:02d} {RGI_REGION_NAMES[region]}",
                "whislo": weighted_quantile(values, weights, 0.10),
                "q1": weighted_quantile(values, weights, 0.25),
                "med": weighted_quantile(values, weights, 0.50),
                "q3": weighted_quantile(values, weights, 0.75),
                "whishi": weighted_quantile(values, weights, 0.90),
                "fliers": [],
            }
        )
    return stats


def plot_figure(stats: list[dict[str, object]]) -> None:
    """Plot the weighted regional boxplot."""
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.4, 8.6), constrained_layout=True)
    ax.axvspan(
        REFERENCE_RHO - REFERENCE_SIGMA,
        REFERENCE_RHO + REFERENCE_SIGMA,
        color="0.75",
        alpha=0.35,
        lw=0,
        label=r"850$\pm$60 kg m$^{-3}$",
    )
    ax.axvline(REFERENCE_RHO, color="0.35", lw=1.0, ls="--")
    ax.bxp(
        stats,
        orientation="horizontal",
        showmeans=False,
        patch_artist=True,
        widths=0.62,
        boxprops={"facecolor": "#8FB7B2", "edgecolor": "0.25", "linewidth": 1.0},
        medianprops={"color": "0.05", "linewidth": 1.4},
        whiskerprops={"color": "0.25", "linewidth": 1.0},
        capprops={"color": "0.25", "linewidth": 1.0},
    )
    ax.set_xlabel(r"Effective density $\rho_{\Delta V}$ (kg m$^{-3}$)")
    ax.set_ylabel("RGI region")
    finite_stats = np.array([v for row in stats for v in [row["whislo"], row["whishi"]] if np.isfinite(v)], dtype=float)
    xmin = max(500.0, 25.0 * np.floor((np.nanmin(finite_stats) - 20.0) / 25.0))
    xmax = min(1100.0, 25.0 * np.ceil((np.nanmax(finite_stats) + 20.0) / 25.0))
    ax.set_xlim(xmin, xmax)
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()
    median_handle = Line2D([], [], color="0.05", lw=1.4)
    whisker_handle = Line2D([], [], color="0.25", lw=1.0)
    reference_handle = mpl.patches.Patch(facecolor="0.75", edgecolor="none", alpha=0.35)
    legend_handles = [
        reference_handle,
        mpl.patches.Patch(facecolor="#8FB7B2", edgecolor="0.25", label="IQR (25–75%)"),
        median_handle,
        whisker_handle,
    ]
    legend_labels = [r"850$\pm$60 kg m$^{-3}$", "IQR (25–75%)", "Median", "Whiskers (10–90%)"]
    ax.legend(
        handles=legend_handles,
        labels=legend_labels,
        frameon=False,
        loc="center left",
        fontsize=11,
        handlelength=2.4,
        handleheight=1.15,
        labelspacing=0.45,
        handletextpad=0.7,
        handler_map={
            reference_handle: HandlerReferenceBand(),
            median_handle: HandlerVerticalMedian(),
            whisker_handle: HandlerHorizontalWhisker(),
        },
    )
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run() -> Path:
    """Generate Fig. 7."""
    df = read_full_period_rows()
    stats = build_box_stats(df)
    plot_figure(stats)
    return OUT_PNG


if __name__ == "__main__":
    path = run()
    print(f"[done] Wrote figure: {path.resolve()}")
