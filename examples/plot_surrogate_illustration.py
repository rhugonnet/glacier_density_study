#!/usr/bin/env python3
"""Plot the figure shown in the README, from the outputs of run_surrogate_example to run the surrogate predictions."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.container import ErrorbarContainer
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

from glacier_density_surrogate import DEFAULT_PARAMS


HERE = Path(__file__).resolve().parent
INPUT_DIR = HERE / "inputs"
OUTPUT_DIR = HERE / "outputs"
OUTPUT_PNG = HERE / "figures" / "surrogate_illustration.png"
BLUES = ["#277BA5", "#619ABA", "#97BDD0", "#C2D9E4"]
ORANGES = ["#C56A25", "#D88F58", "#E5B185", "#F0CEAE"]
GREENS = ["#317B66", "#679A85", "#96B9A8", "#C2D6CA"]
INK = "#243440"
LABEL_POSITIONS = {"G1": (-6.7, 2.5), "G2": (5.0, 2.1), "G3": (-6.7, -0.8), "G4": (8.2, -3.8)}


##############################
# TIME SERIES AND GLACIER MAPS
##############################


def plot_periods(
    axis: plt.Axes,
    starts: np.ndarray,
    ends: np.ndarray,
    values: np.ndarray,
    errors: np.ndarray,
    color: str,
) -> ErrorbarContainer:
    """
    Draw horizontal period values with vertical uncertainty bars at their midpoints.

    :param axis: Axes for the period series.
    :param starts: Consecutive observation start years.
    :param ends: Observation end years, matching starts.
    :param values: Changes or densities over each whole observation period.
    :param errors: One-sigma uncertainties in the same units as values.
    :param color: Color shared by the segments, points and error bars.
    :returns: The plotted error bar artists.
    """
    midpoints = (starts + ends) / 2

    # Each horizontal segment spans its observation period without connecting jumps
    axis.hlines(values, starts, ends, color=color, linewidth=2.2)
    error_bars = axis.errorbar(
        midpoints, values, yerr=errors, fmt="o", color=color,
        markersize=5.5, markeredgecolor="white", markeredgewidth=0.8,
        elinewidth=1.4, capsize=3.5, capthick=1.3, zorder=4,
    )

    # Use the same calendar range and light guides in every panel
    axis.set_xlim(starts[0] - 0.4, ends[-1] + 0.4)
    axis.set_xticks([2000, 2005, 2010, 2015, 2020])
    axis.grid(axis="y", color="#E7ECEF", linewidth=0.7, zorder=0)
    axis.tick_params(colors=INK, labelsize=10.5, length=3)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        axis.spines[side].set_color("#A6B2BB")
    return error_bars


def plot_glacier_map(
    map_axis: plt.Axes,
    features: list[dict],
    glaciers: pd.DataFrame,
    labels: list[str],
    shades: list[str],
) -> None:
    """
    Draw four glacier outlines with consistent shades for each glacier's identity.

    Positions use approximate local kilometres to avoid GIS plotting dependencies.
    Each map has the same extent, making the input and output maps easy to compare.
    G1 has the darkest fill and a thicker outline to link it to the time series.

    :param map_axis: Axes for the glacier map.
    :param features: GeoJSON features containing the simplified outlines.
    :param glaciers: Glacier coordinates in the same order as the features.
    :param labels: Glacier identifiers and values, with units stated above the map.
    :param shades: Four shades of blue, orange or green, ordered by glacier.
    """
    lon_origin = glaciers["lon"].mean()
    lat_origin = glaciers["lat"].mean()
    km_per_degree_lon = 111.2 * np.cos(np.deg2rad(lat_origin))

    # Transform the outlines and preserve any holes in their geometry
    for feature, row, label, shade in zip(features, glaciers.itertuples(index=False), labels, shades):
        rings = feature["geometry"]["coordinates"]
        for ring_index, ring in enumerate(rings):
            coordinates = np.asarray(ring, dtype=float)
            local_xy = np.column_stack((
                (coordinates[:, 0] - lon_origin) * km_per_degree_lon,
                (coordinates[:, 1] - lat_origin) * 111.2,
            ))
            map_axis.add_patch(Polygon(
                local_xy, facecolor=shade if ring_index == 0 else "white",
                edgecolor=shades[0], linewidth=1.5 if row.label == "G1" else 0.8,
            ))
        centre = ((row.lon - lon_origin) * km_per_degree_lon, (row.lat - lat_origin) * 111.2)
        map_axis.annotate(
            label.replace("-", "−"), xy=centre, xytext=LABEL_POSITIONS[row.label],
            ha="left" if row.label in ("G2", "G4") else "right", va="center", fontsize=9, color=INK,
            arrowprops={"arrowstyle": "-", "color": shades[0], "linewidth": 0.7, "alpha": 0.6},
        )

    # Leave enough room around the outlines for labels to sit entirely on white space
    map_axis.set_aspect("equal")
    map_axis.set_xlim(-11.0, 14.0)
    map_axis.set_ylim(-5.8, 5.1)
    map_axis.axis("off")


#################
# FIGURE ASSEMBLY
#################


def build_figure(
    out_path: Path | str = OUTPUT_PNG,
    *,
    input_dir: Path | str = INPUT_DIR,
    output_dir: Path | str = OUTPUT_DIR,
) -> Path:
    """
    Show glacier maps beside time series of elevation, density and mass change.

    Read observations and the results saved by run_surrogate_example.py.
    G1's original periods are drawn with plot_periods(), and the maps
    show each glacier's estimates over the full twenty years. plot_glacier_map()
    repeats the same outlines in blue, orange and green. Mass change is expressed
    in metres water equivalent per year. Mass change error bars combine input variance
    and surrogate residual variance; density errors show the residual component.

    :param out_path: Destination PNG; an SVG is saved alongside it.
    :param input_dir: Directory containing the regional CSV and glacier outlines.
    :param output_dir: Directory containing the saved glacier and regional predictions.
    :returns: Path of the generated PNG.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Read saved estimates so drawing the figure does not run the surrogate
    with (input_dir / "regional_glaciers.geojson").open() as stream:
        features = json.load(stream)["features"]
    observations = pd.read_csv(input_dir / "regional_dh_timeseries.csv").rename(columns={"glacier_id": "label"})
    predictions = pd.read_csv(output_dir / "glacier_predictions.csv").rename(
        columns={"glacier_id": "label"},
    )
    regional_periods = pd.read_csv(output_dir / "regional_predictions.csv")
    start = observations["start_year"].min()
    end = observations["end_year"].max()
    regional = regional_periods.loc[
        (regional_periods["start"] == start) & (regional_periods["end"] == end)
    ].iloc[0].to_dict()
    rho_ice = DEFAULT_PARAMS["rho_ice_fixed"]

    # Convert physical outputs to the annual rates displayed in the figure
    rate_scale = predictions["area_m2"] * predictions["period_years"] * 1000.0
    predictions["dM_mwe_yr"] = predictions["dM_kg"] / rate_scale
    predictions["sigma_dM_total_mwe_yr"] = predictions["sigma_dM_total_kg"] / rate_scale
    regional_scale = regional["area_m2"] * regional["period_years"] * 1000.0
    regional["dM_mwe_yr"] = regional["dM_kg"] / regional_scale
    regional["sigma_dM_total_mwe_yr"] = regional["sigma_dM_total_kg"] / regional_scale
    regional["dh_m_yr"] = regional["dh_m"] / regional["period_years"]
    regional["sigma_dh_m_yr"] = regional["sigma_dh_m"] / regional["period_years"]

    # Match observation periods and glacier labels to the saved predictions
    glaciers = pd.DataFrame(feature["properties"] for feature in features)
    g1_observations = observations.loc[observations["label"] == "G1"]
    starts = g1_observations["start_year"].to_numpy(float)
    ends = g1_observations["end_year"].to_numpy(float)
    g1_index = pd.MultiIndex.from_frame(g1_observations[["start_year", "end_year"]])
    g1_predictions = predictions.loc[predictions["label"] == "G1"].set_index(["start", "end"]).loc[g1_index]
    full_period = predictions.loc[
        (predictions["start"] == regional["start"]) & (predictions["end"] == regional["end"])
    ].set_index("label").loc[glaciers["label"]].reset_index()

    # Place glacier maps beside their time series for all three quantities
    fig = plt.figure(figsize=(10.4, 8.8), facecolor="white")
    input_map = fig.add_axes([0.12, 0.753, 0.36, 0.19])
    elevation_axis = fig.add_axes([0.60, 0.761, 0.365, 0.178])
    density_map = fig.add_axes([0.12, 0.457, 0.36, 0.19])
    density_axis = fig.add_axes([0.60, 0.465, 0.365, 0.178])
    mass_map = fig.add_axes([0.12, 0.207, 0.36, 0.19])
    mass_axis = fig.add_axes([0.60, 0.215, 0.365, 0.178])
    fig.text(
        0.30, 0.974, "Example of study glaciers over 2000–2020",
        ha="center", fontsize=12, weight="bold", color=INK,
    )
    fig.text(
        0.78, 0.974, "Example of observed time series for glacier G1",
        ha="center", fontsize=12, weight="bold", color=INK,
    )

    # Display the same glacier's temporal inputs and outputs with their uncertainties
    plot_periods(
        elevation_axis, starts, ends, g1_observations["dh_m"].to_numpy(float),
        g1_observations["sigma_dh_m"].to_numpy(float), BLUES[0],
    )
    plot_periods(
        density_axis, starts, ends, g1_predictions["mu_rho_kg_m3"].to_numpy(float),
        g1_predictions["sigma_rho_kg_m3"].to_numpy(float), ORANGES[0],
    )
    mass_values = g1_predictions["dM_mwe_yr"].to_numpy(float)
    mass_errors = g1_predictions["sigma_dM_total_mwe_yr"].to_numpy(float)
    plot_periods(mass_axis, starts, ends, mass_values, mass_errors, GREENS[0])

    # State units and leave enough room for the larger synthetic input errors
    elevation_axis.set_ylabel(r"$\Delta h$ (m)", fontsize=12, color=INK, labelpad=8)
    elevation_values = g1_observations["dh_m"].to_numpy(float)
    elevation_errors = g1_observations["sigma_dh_m"].to_numpy(float)
    elevation_axis.set_ylim((elevation_values - elevation_errors).min() - 0.5, 1.6)
    elevation_axis.set_yticks([-6, -4, -2, 0])
    elevation_axis.axhline(0, color="#A6B2BB", linewidth=0.8, zorder=0)
    density_axis.set_ylabel(r"$\rho_{\Delta V}$ (kg m$^{-3}$)", fontsize=12, color=INK, labelpad=8)
    density_axis.set_ylim(80, 1160)
    density_axis.set_yticks([200, 400, 600, 800, 1000])
    density_axis.axhline(rho_ice, color="#A6B2BB", linewidth=0.9, linestyle="--", zorder=0)
    density_axis.text(2019.6, rho_ice + 60, "Ice density", ha="right", fontsize=9, color=INK)
    mass_axis.set_ylabel(r"$\Delta M / A$ (m w.e. yr$^{-1}$)", fontsize=12, color=INK, labelpad=8)
    mass_axis.set_ylim((mass_values - mass_errors).min() * 1.12, (mass_values + mass_errors).max() + 0.15)
    mass_axis.axhline(0, color="#A6B2BB", linewidth=0.8, zorder=0)
    for axis in (density_axis, mass_axis):
        axis.set_xlabel("Year", fontsize=10.5, color=INK, labelpad=8)

    # Repeated outlines and shade order identify the four glaciers across quantities
    duration = regional["end"] - regional["start"]
    input_labels = [
        f"{row.label}\n{row.dh_m / duration:.2f} ± {row.sigma_dh_m / duration:.2f}"
        for row in full_period.itertuples()
    ]
    density_labels = [
        f"{row.label}\n{row.mu_rho_kg_m3:.0f} ± {row.sigma_rho_kg_m3:.0f}" for row in full_period.itertuples()
    ]
    mass_labels = [
        f"{row.label}\n{row.dM_mwe_yr:.2f} ± {row.sigma_dM_total_mwe_yr:.2f}" for row in full_period.itertuples()
    ]
    plot_glacier_map(input_map, features, glaciers, input_labels, BLUES)
    plot_glacier_map(density_map, features, glaciers, density_labels, ORANGES)
    plot_glacier_map(mass_map, features, glaciers, mass_labels, GREENS)
    for centre, title in ((0.848, "Elevation change"), (0.552, "Effective density"), (0.302, "Mass change")):
        fig.text(
            0.045, centre, title, rotation=90, ha="center", va="center",
            fontsize=15, weight="bold", color=INK,
        )
    fig.text(0.30, 0.942, r"$\Delta h / \Delta t \pm \sigma$ (m yr$^{-1}$)", ha="center", fontsize=10.5, color=BLUES[0])
    fig.text(0.30, 0.654, r"$\rho_{\Delta V} \pm \sigma$ (kg m$^{-3}$)", ha="center", fontsize=10.5, color=ORANGES[0])
    fig.text(0.30, 0.410, r"$\Delta M / A \pm \sigma$ (m w.e. yr$^{-1}$)", ha="center", fontsize=10.5, color=GREENS[0])

    # Place one vertical arrow behind the model name at the centre of the divider
    for left, right in ((0.065, 0.379), (0.611, 0.975)):
        fig.add_artist(plt.Line2D(
            [left, right], [0.705, 0.705], transform=fig.transFigure,
            color="#B7C2C9", linewidth=1.1, zorder=0.5,
        ))
    fig.add_artist(FancyArrowPatch(
        (0.495, 0.790), (0.495, 0.620), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=15, linewidth=1.5, color=INK, zorder=1,
    ))
    fig.add_artist(FancyBboxPatch(
        (0.375, 0.667), 0.240, 0.061, transform=fig.transFigure,
        boxstyle="round,pad=0.004,rounding_size=0.008",
        facecolor="white", edgecolor=INK, linewidth=1.0, zorder=2,
    ))
    fig.text(
        0.495, 0.705, "Surrogate model", ha="center", va="center",
        fontsize=14, weight="bold", color=INK, zorder=3,
    )
    fig.text(
        0.495, 0.683, "in time and space", ha="center", va="center",
        fontsize=10.5, color=INK, zorder=3,
    )

    # Place the regional input and both outputs together in one summary box
    fig.add_artist(FancyBboxPatch(
        (0.12, 0.025), 0.85, 0.147, transform=fig.transFigure,
        boxstyle="round,pad=0.004,rounding_size=0.01",
        facecolor="#F1F2F3", edgecolor="#C9CDD1", linewidth=0.9, zorder=0.5,
    ))
    fig.text(0.545, 0.145, "All glaciers during 2000–2020", ha="center", fontsize=13, weight="bold", color=INK)
    fig.text(0.545, 0.120, "Accounting for error correlation in space + time", ha="center", fontsize=9, color=INK)
    fig.text(0.25, 0.094, "Mean elevation change rate", ha="center", fontsize=10, color=INK)
    elevation_text = f"{regional['dh_m_yr']:.2f} ± {regional['sigma_dh_m_yr']:.2f}".replace("-", "−")
    fig.text(0.25, 0.060, elevation_text, ha="center", fontsize=18, weight="bold", color=BLUES[0])
    fig.text(0.25, 0.036, r"m yr$^{-1}$", ha="center", fontsize=11, color=BLUES[0])
    fig.add_artist(FancyArrowPatch(
        (0.365, 0.076), (0.430, 0.076), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=14, linewidth=1.5, color=INK,
    ))
    fig.text(0.55, 0.094, "Effective density", ha="center", fontsize=10, color=INK)
    fig.text(0.82, 0.094, "Specific mass balance rate", ha="center", fontsize=10, color=INK)
    density_text = f"{regional['mu_rho_kg_m3']:.0f} ± {regional['sigma_rho_kg_m3']:.0f}"
    mass_text = f"{regional['dM_mwe_yr']:.2f} ± {regional['sigma_dM_total_mwe_yr']:.2f}".replace("-", "−")
    fig.text(0.55, 0.060, density_text, ha="center", fontsize=18, weight="bold", color=ORANGES[0])
    fig.text(0.82, 0.060, mass_text, ha="center", fontsize=18, weight="bold", color=GREENS[0])
    fig.text(0.55, 0.036, r"kg m$^{-3}$", ha="center", fontsize=11, color=ORANGES[0])
    fig.text(0.82, 0.036, r"m w.e. yr$^{-1}$", ha="center", fontsize=11, color=GREENS[0])

    # Save a README image and a vector version for reuse
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=240, facecolor="white")
    fig.savefig(destination.with_suffix(".svg"), facecolor="white")
    plt.close(fig)
    return destination


if __name__ == "__main__":
    print(f"Wrote {build_figure()}")
