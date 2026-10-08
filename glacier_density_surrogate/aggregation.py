"""Sum glacier mass changes and propagate spatially correlated uncertainties."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .surrogate import _glacier_id_column, finite_array, positive_integer


############################
# DISTANCES AND CORRELATIONS
############################


def haversine_distance_matrix(lat_a: np.ndarray, lon_a: np.ndarray, lat_b: np.ndarray, lon_b: np.ndarray) -> np.ndarray:
    """
    Calculate great-circle distances between two sets of glacier coordinates.

    :param lat_a: First latitude vector in decimal degrees.
    :param lon_a: First longitude vector in decimal degrees.
    :param lat_b: Second latitude vector in decimal degrees.
    :param lon_b: Second longitude vector in decimal degrees.

    :returns: Distances in kilometres, as a len(lat_a) by len(lat_b) matrix.
    """

    lat_a = finite_array(lat_a, "lat_a")
    lon_a = finite_array(lon_a, "lon_a")
    lat_b = finite_array(lat_b, "lat_b")
    lon_b = finite_array(lon_b, "lon_b")
    if any(values.ndim != 1 for values in (lat_a, lon_a, lat_b, lon_b)):
        raise ValueError("Coordinates must be one-dimensional vectors")
    if lat_a.shape != lon_a.shape or lat_b.shape != lon_b.shape:
        raise ValueError("Latitude and longitude vectors must have matching lengths")
    if np.any(np.abs(lat_a) > 90) or np.any(np.abs(lat_b) > 90) or np.any(np.abs(lon_a) > 180) or np.any(np.abs(lon_b) > 180):
        raise ValueError("Coordinates must be valid latitude and longitude")
    radius_km = 6371.0

    # Broadcast both coordinate vectors to every pair before applying the haversine formula
    lat1 = np.deg2rad(lat_a)[:, None]
    lon1 = np.deg2rad(lon_a)[:, None]
    lat2 = np.deg2rad(lat_b)[None, :]
    lon2 = np.deg2rad(lon_b)[None, :]
    dlat = lat1 - lat2
    dlon = lon1 - lon2
    half_chord_squared = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * radius_km * np.arcsin(np.minimum(1.0, np.sqrt(half_chord_squared)))


def spherical_correlation(distance_km: np.ndarray, range_km: float) -> np.ndarray:
    """
    Calculate a spherical correlation that reaches zero at the supplied range.

    :param distance_km: Distances in kilometres.
    :param range_km: Correlation range in kilometres.

    :returns: Correlations with the same shape as distance_km.
    """

    finite_array(range_km, "range_km", positive=True)
    h = finite_array(distance_km, "distance_km", nonnegative=True) / float(range_km)
    out = np.zeros_like(h, dtype=float)
    inside = h < 1.0
    out[inside] = 1.0 - 1.5 * h[inside] + 0.5 * h[inside] ** 3
    return out


###########################
# SPATIAL ERROR PROPAGATION
###########################


def _validate_spatial_table(table, sigma_cols, group_col, period_cols, id_col, lat_col, lon_col) -> None:
    """Check observations before missing rows can be confused with zero variance."""
    keys = [group_col, id_col, *period_cols]
    if table[keys].isna().any().any():
        raise ValueError("Group, glacier and period identifiers must not be missing")
    if table.duplicated(keys).any():
        raise ValueError("Spatial table contains duplicate glacier periods")
    for column in (*period_cols, lat_col, lon_col):
        finite_array(table[column], column)
    finite_array(table[period_cols[1]] - table[period_cols[0]], "period length", positive=True)
    if (table[lat_col].abs() > 90).any() or (table[lon_col].abs() > 180).any():
        raise ValueError("Glacier coordinates must be valid latitude and longitude")
    for column in sigma_cols:
        finite_array(table[column], column, nonnegative=True)
    coordinates = table.groupby([group_col, id_col])[[lat_col, lon_col]].nunique()
    if (coordinates > 1).any().any():
        raise ValueError("Each glacier must have consistent coordinates across periods")


def _spatial_component_variance(sigmas, lat, lon, correlations, exact_max_items, sample_max_items, block_size, rng):
    """Sum independent variances and exact or sampled covariances between glaciers."""
    n_items = len(lat)
    diagonal = sum(np.sum(component**2, axis=0) for component in sigmas)
    sample_n = n_items if n_items <= exact_max_items else min(sample_max_items, n_items)
    selected = np.arange(n_items) if sample_n == n_items else rng.choice(n_items, sample_n, replace=False)
    off_diagonal = np.zeros_like(diagonal)

    # Share each distance block across components while excluding self covariances
    for start in range(0, sample_n, block_size):
        end = min(start + block_size, sample_n)
        distance = haversine_distance_matrix(lat[selected[start:end]], lon[selected[start:end]], lat[selected], lon[selected])
        for component, correlation in zip(sigmas, correlations):
            corr = finite_array(correlation(distance), "correlation").copy()
            if corr.shape != distance.shape or np.any(np.abs(corr) > 1):
                raise ValueError("Correlation must match the distance shape and lie between -1 and 1")
            corr[np.arange(end - start), np.arange(start, end)] = 0.0
            sampled = component[selected]
            off_diagonal += np.sum(sampled[start:end] * (corr @ sampled), axis=0)
    scale = n_items * (n_items - 1) / (sample_n * (sample_n - 1)) if sample_n > 1 else 0.0
    variance = diagonal + scale * off_diagonal
    if np.any(variance < -1e-12 * np.maximum(diagonal, 1.0)):
        raise ValueError("Spatial covariance gives a negative total variance")
    return np.maximum(variance, 0.0)


def _propagate_spatial_components(
    table, component_cols, correlations, group_col, period_cols, id_col, lat_col, lon_col,
    exact_max_items, subsample_max_items, block_size, random_seed,
) -> dict[tuple[str, float, float], float]:
    """Align all components by period and propagate them on each spatial group."""
    _validate_spatial_table(table, component_cols, group_col, period_cols, id_col, lat_col, lon_col)
    exact_max_items = positive_integer(exact_max_items, "exact_max_items")
    subsample_max_items = positive_integer(subsample_max_items, "subsample_max_items")
    block_size = positive_integer(block_size, "block_size")
    if subsample_max_items < 2:
        raise ValueError("subsample_max_items must be at least two to estimate covariances")
    rng = np.random.default_rng(random_seed)
    period_defs = sorted(table[list(period_cols)].drop_duplicates().itertuples(index=False, name=None))
    period_index = pd.MultiIndex.from_tuples(period_defs, names=list(period_cols))
    result = {}

    # An absent glacier in a period contributes no error to that period's spatial sum
    for group, observations in table.groupby(group_col, sort=False):
        glaciers = observations[[id_col, lat_col, lon_col]].drop_duplicates(id_col)
        sigmas = []
        for column in component_cols:
            aligned = observations.pivot(index=id_col, columns=list(period_cols), values=column)
            aligned = aligned.reindex(index=glaciers[id_col], columns=period_index).fillna(0.0)
            sigmas.append(aligned.to_numpy(float))
        variance = _spatial_component_variance(
            sigmas, glaciers[lat_col].to_numpy(float), glaciers[lon_col].to_numpy(float),
            correlations, exact_max_items, subsample_max_items, block_size, rng,
        )
        for (start, end), value in zip(period_defs, variance):
            result[(str(group), float(start), float(end))] = float(np.sqrt(value))
    return result


def spatially_correlated_sigma_by_group_period(
    table: pd.DataFrame,
    corr_func: Callable[[np.ndarray], np.ndarray],
    *,
    sigma_col: str,
    group_col: str = "region_group",
    period_cols: tuple[str, str] = ("start_year", "end_year"),
    id_col: str = "rgiid",
    lat_col: str = "lat",
    lon_col: str = "lon",
    exact_max_items: int = 3500,
    subsample_max_items: int = 10_000,
    block_size: int = 700,
    random_seed: int = 426,
) -> dict[tuple[str, float, float], float]:
    """
    Sum glacier uncertainties with their spatial covariance by group and period.

    Independent variances are calculated exactly.
    For groups above ``exact_max_items``, covariances are estimated from sbusampled glacier pairs, as in
    Hugonnet et al., 2022.

    :param table: One row per glacier and observation period.
    :param corr_func: Correlations from separation distances in kilometres.
    :param sigma_col: Standard deviation column, in the quantity's units.
    :param group_col: Spatial grouping column, usually RGI region.
    :param period_cols: Period start and end columns, in years.
    :param id_col: Glacier identifier column.
    :param lat_col: Latitude column, in decimal degrees.
    :param lon_col: Longitude column, in decimal degrees.
    :param exact_max_items: Largest group calculated exactly.
    :param subsample_max_items: Maximum number of glaciers sampled to estimate covariance.
    :param block_size: Number of glaciers per distance matrix block.
    :param random_seed: Seed for sampling large groups.
    :returns: Standard deviations in input units, keyed by group, start and end.
    """
    return _propagate_spatial_components(
        table, (sigma_col,), (corr_func,), group_col, period_cols, id_col, lat_col, lon_col,
        exact_max_items, subsample_max_items, block_size, random_seed,
    )


def spatially_correlated_component_sigma_by_group_period(
    table: pd.DataFrame,
    *,
    component_cols: tuple[str, ...],
    corr_ranges_km: tuple[float, ...],
    group_col: str = "region_group",
    period_cols: tuple[str, str] = ("start_year", "end_year"),
    id_col: str = "rgiid",
    lat_col: str = "lat",
    lon_col: str = "lon",
    scale_col: str = "area",
    exact_max_items: int = 3500,
    subsample_max_items: int = 10_000,
    block_size: int = 700,
    random_seed: int = 426,
) -> dict[tuple[str, float, float], float]:
    """
    Scale and sum independent error components with separate spatial ranges.

    Each observation uses its own scale, such as area to convert elevation errors
    to volume change errors. Components use spherical correlations and their variances
    add. Distance blocks are shared across components by _propagate_spatial_components().

    :param table: One row per glacier and observation period.
    :param component_cols: Standard deviation columns before scaling.
    :param corr_ranges_km: Positive spherical correlation range for each component.
    :param group_col: Spatial grouping column, usually RGI region.
    :param period_cols: Period start and end columns, in years.
    :param id_col: Glacier identifier column.
    :param lat_col: Latitude column, in decimal degrees.
    :param lon_col: Longitude column, in decimal degrees.
    :param scale_col: Positive multiplicative scale column, usually glacier area.
    :param exact_max_items: Largest group calculated exactly.
    :param subsample_max_items: Maximum number of glaciers sampled to estimate covariance.
    :param block_size: Number of glaciers per distance matrix block.
    :param random_seed: Seed for sampling large groups.
    :returns: Standard deviations after scaling, keyed by group, start and end.
    """
    if not component_cols or len(component_cols) != len(corr_ranges_km):
        raise ValueError("component_cols and corr_ranges_km must have the same nonzero length")
    scale = finite_array(table[scale_col], scale_col, positive=True)
    finite_array(corr_ranges_km, "corr_ranges_km", positive=True)
    scaled = table.copy()
    correlations = []
    for column, range_km in zip(component_cols, corr_ranges_km):
        scaled[column] = finite_array(table[column], column, nonnegative=True) * scale
        correlations.append(lambda distance, radius=range_km: spherical_correlation(distance, radius))
    return _propagate_spatial_components(
        scaled, component_cols, correlations, group_col, period_cols, id_col, lat_col, lon_col,
        exact_max_items, subsample_max_items, block_size, random_seed,
    )


#######################
# MASS CHANGE SUMMARIES
#######################


def _regional_elementary_periods(table: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """
    Select a shared elementary time grid and check that glacier changes add across it.

    Every adjacent boundary pair must be present for every glacier. This lets us
    propagate covariance over the elementary intervals instead of counting
    overlapping combined periods. Each glacier must use a constant area.
    """
    bounds = np.unique(table[["start", "end"]].to_numpy(float))
    elementary_index = pd.MultiIndex.from_arrays([bounds[:-1], bounds[1:]])
    period_index = pd.MultiIndex.from_frame(table[["start", "end"]])
    elementary = table.loc[period_index.isin(elementary_index)].copy()
    glacier_count = table[id_col].nunique()

    # Require complete coverage before assuming independent elementary residuals
    if len(elementary) != glacier_count * (len(bounds) - 1):
        raise ValueError("Regional aggregation requires the same consecutive elementary periods for every glacier")
    if (table.groupby(id_col)["area_m2"].nunique() != 1).any():
        raise ValueError("Regional aggregation requires constant area for each glacier")

    # Longer period estimates must reproduce the sums of their elementary changes
    for (start, end), period in table.groupby(["start", "end"], sort=False):
        covered = elementary.loc[(elementary["start"] >= start) & (elementary["end"] <= end)]
        expected = covered.groupby(id_col)[["dV_m3", "dM_kg"]].sum().reindex(period[id_col])
        for column in ("dV_m3", "dM_kg"):
            if not np.allclose(period[column].to_numpy(), expected[column].to_numpy(), rtol=1e-10, atol=1e-6):
                raise ValueError(f"Regional aggregation requires additive {column} for each glacier")
    return elementary


def aggregate_regions(
    predictions: pd.DataFrame,
    corr_func: Callable[[np.ndarray], np.ndarray],
    *,
    id_col: str | None = None,
    group_col: str = "region_group",
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> pd.DataFrame:
    """
    Sum glacier predictions and propagate input and residual errors for each region and period.

    _regional_elementary_periods() checks shared time coverage and additive changes.
    spatially_correlated_sigma_by_group_period() then propagates spatial covariance
    within each elementary interval. We add these independent variances for longer
    periods and divide summed mass change by summed volume change for regional density.
    The output includes only periods present for every glacier in the region.
    Input errors, when supplied, are combined across independent glaciers using each
    whole period's uncertainty, which already accounts for shared observations in time.

    :param predictions: predict_timeseries() output with glacier IDs, areas, coordinates,
        dV_m3, dM_kg and sigma_dM_rho_kg. Each glacier must have constant area and cover
        the same consecutive elementary intervals as the others in its region.
    :param corr_func: Spatial residual correlation as a function of distance in kilometres.
    :param id_col: Glacier identifier column; omitted detects glacier_id or rgiid.
    :param group_col: Region column; absent groups all glaciers into "All glaciers".
    :param lat_col: Latitude column in decimal degrees.
    :param lon_col: Longitude column in decimal degrees.
    :returns: Regional period table with glacier count, total area in m², volume change
        in m³, mass change and its uncertainties in kg, and density in kg m⁻³.
        At zero volume change, density is undefined but mass change remains available.
    """
    output_columns = [
        group_col, "start", "end", "period_years", "n_glaciers", "area_m2",
        "dV_m3", "dM_kg", "sigma_dM_rho_kg", "mu_rho_kg_m3", "sigma_rho_kg_m3",
    ]
    input_errors = "sigma_dM_dh_kg" in predictions
    volume_errors = "sigma_dV_m3" in predictions
    if input_errors:
        output_columns.extend(["sigma_dM_dh_kg", "sigma_dM_total_kg"])
    if volume_errors:
        output_columns.extend(["sigma_dV_m3", "dh_m", "sigma_dh_m"])
    if predictions.empty:
        return pd.DataFrame(columns=output_columns)
    if predictions.attrs.get("has_time_bounds") is False:
        raise ValueError("Regional aggregation requires observation start and end years")

    # Resolve identifiers and require the physical outputs needed for a regional sum
    id_col = _glacier_id_column(predictions, id_col)
    if id_col is None:
        raise ValueError("Regional aggregation requires a glacier identifier column")
    required = [id_col, "start", "end", "area_m2", "dV_m3", "dM_kg", "sigma_dM_rho_kg", lat_col, lon_col]
    missing = [column for column in required if column not in predictions.columns]
    if missing:
        raise ValueError(f"Regional aggregation requires columns: {', '.join(missing)}")
    table = predictions.copy()
    if group_col not in table:
        table[group_col] = "All glaciers"

    # Validate metadata and changes before missing values can disappear in a sum
    sigma_cols = ["sigma_dM_rho_kg"]
    if input_errors:
        sigma_cols.append("sigma_dM_dh_kg")
    if volume_errors:
        sigma_cols.append("sigma_dV_m3")
    _validate_spatial_table(table, sigma_cols, group_col, ("start", "end"), id_col, lat_col, lon_col)
    finite_array(table["area_m2"], "area_m2", positive=True)
    finite_array(table["dV_m3"], "dV_m3")
    finite_array(table["dM_kg"], "dM_kg")
    if (table.groupby(id_col)[group_col].nunique() != 1).any():
        raise ValueError("Each glacier must belong to one region")

    # Propagate spatial covariance on the elementary intervals once per region
    rows = []
    for region, glacier_periods in table.groupby(group_col, sort=False, observed=True):
        elementary = _regional_elementary_periods(glacier_periods, id_col)
        sigmas = spatially_correlated_sigma_by_group_period(
            elementary, corr_func, sigma_col="sigma_dM_rho_kg", group_col=group_col,
            period_cols=("start", "end"), id_col=id_col, lat_col=lat_col, lon_col=lon_col,
        )
        glacier_count = glacier_periods[id_col].nunique()
        for (start, end), period in glacier_periods.groupby(["start", "end"], sort=True):
            if len(period) != glacier_count:
                continue

            # Add independent interval variances rather than overlapping period errors
            variance = 0.0
            for (_, interval_start, interval_end), sigma in sigmas.items():
                if interval_start >= start and interval_end <= end:
                    variance += sigma**2
            sigma_mass = float(np.sqrt(variance))
            volume_change = float(period["dV_m3"].sum())
            mass_change = float(period["dM_kg"].sum())
            density = mass_change / volume_change if volume_change != 0 else np.nan
            sigma_density = sigma_mass / abs(volume_change) if volume_change != 0 else np.inf

            # Construct one summary from every glacier's contribution to this period
            summary = {
                group_col: region, "start": start, "end": end, "period_years": end - start,
                "n_glaciers": glacier_count, "area_m2": float(period["area_m2"].sum()),
                "dV_m3": volume_change, "dM_kg": mass_change, "sigma_dM_rho_kg": sigma_mass,
                "mu_rho_kg_m3": density, "sigma_rho_kg_m3": sigma_density,
            }

            # Input errors already include time dependence within each glacier's whole period
            if input_errors:
                sigma_input = float(np.linalg.norm(period["sigma_dM_dh_kg"]))
                summary["sigma_dM_dh_kg"] = sigma_input
                summary["sigma_dM_total_kg"] = float(np.hypot(sigma_mass, sigma_input))
            if volume_errors:
                sigma_volume = float(np.linalg.norm(period["sigma_dV_m3"]))
                summary["sigma_dV_m3"] = sigma_volume
                summary["dh_m"] = volume_change / summary["area_m2"]
                summary["sigma_dh_m"] = sigma_volume / summary["area_m2"]
            rows.append(summary)
    return pd.DataFrame(rows, columns=output_columns)


def summarize_region_period_conversions(
    periods: pd.DataFrame,
    *,
    spatial_density_sigma: dict[tuple[str, float, float], float],
    spatial_volume_sigma: dict[tuple[str, float, float], float] | None = None,
    old_rho: float = 850.0,
    old_sigma_rho: float = 60.0,
    ice_sigma_rho: float = 5.0,
) -> pd.DataFrame:
    """
    Sum glacier mass changes and uncertainty components by region and period.

    The surrogate uses the supplied spatially propagated mass change uncertainty.
    Volume change uncertainty can be supplied separately or calculated assuming independent glacier errors.
    Regional density is total mass change divided by net volume change.
    It is undefined when net volume change is zero.

    The "old" conversion uses a fixed density with fully correlated density error within each region from
    Huss et al. (2013), and is only used for comparison in the paper.

    :param periods: Glacier conversion table produced by the study application.
    :param spatial_density_sigma: Residual mass change standard deviations in kg, keyed by
        region, start and end.
    :param spatial_volume_sigma: Optional volume change standard deviations in m3 with the same keys.
    :param old_rho: Reference density in kg m-3.
    :param old_sigma_rho: Reference density standard deviation in kg m-3.
    :param ice_sigma_rho: Additional fully correlated ice density standard deviation in kg m-3.
    :returns: Regional totals and uncertainty components in Gt, rates in Gt yr-1
        and equivalent densities in kg m-3.
    """

    finite_array(old_rho, "old_rho", positive=True)
    finite_array(old_sigma_rho, "old_sigma_rho", nonnegative=True)
    finite_array(ice_sigma_rho, "ice_sigma_rho", nonnegative=True)
    for column in ("start_year", "end_year", "dV_m3", "dM_old_kg", "dM_surrogate_kg"):
        finite_array(periods[column], column)
    finite_array(periods["period_years"], "period_years", positive=True)
    finite_array(periods["area"], "area", positive=True)
    rows = []
    group_cols = ["region_group", "region_label", "start_year", "end_year", "period_years"]
    for key, group_table in periods.groupby(group_cols, sort=True):
        group, label, start, end, dt = key

        # Sum mass change and volume change before deriving the regional effective density
        volume_change = float(group_table["dV_m3"].sum())
        old_mass = float(group_table["dM_old_kg"].sum())
        new_mass = float(group_table["dM_surrogate_kg"].sum())
        new_rho_mean = new_mass / volume_change if volume_change != 0 else np.nan

        # Use supplied regional volume change uncertainty or combine independent glacier errors
        if spatial_volume_sigma is None:
            old_vol_sigma = float(np.sqrt(np.sum(group_table["sigma_dM_volume_old_kg"].to_numpy(float) ** 2)))
            new_vol_sigma = float(np.sqrt(np.sum(group_table["sigma_dM_volume_surrogate_kg"].to_numpy(float) ** 2)))
            volume_sigma = old_vol_sigma / old_rho
        else:
            volume_sigma = spatial_volume_sigma[(str(group), float(start), float(end))]
            old_vol_sigma = old_rho * volume_sigma
            new_vol_sigma = abs(new_rho_mean) * volume_sigma if np.isfinite(new_rho_mean) else np.nan
        old_rho_sigma = old_sigma_rho * abs(volume_change)
        new_rho_sigma = spatial_density_sigma[(str(group), float(start), float(end))]
        ice_sigma = ice_sigma_rho * abs(volume_change)

        # Report each uncertainty component so its contribution can be compared
        rows.append(
            {
                "region_group": group,
                "region_label": label,
                "start_year": float(start),
                "end_year": float(end),
                "period_years": float(dt),
                "n_glaciers": int(group_table["rgiid"].nunique()),
                "area_m2": float(group_table.drop_duplicates("rgiid")["area"].sum()),
                "dV_m3": volume_change,
                "sigma_dV_m3": volume_sigma,
                "old_rho_mean_kg_m3": old_rho,
                "surrogate_rho_mean_kg_m3": new_rho_mean,
                "old_mass_gt": old_mass / 1.0e12,
                "surrogate_mass_gt": new_mass / 1.0e12,
                "old_mass_rate_gt_yr": old_mass / dt / 1.0e12,
                "surrogate_mass_rate_gt_yr": new_mass / dt / 1.0e12,
                "mass_rate_difference_gt_yr": (new_mass - old_mass) / dt / 1.0e12,
                "old_sigma_mass_gt": np.sqrt(old_vol_sigma**2 + old_rho_sigma**2) / 1.0e12,
                "surrogate_sigma_mass_gt": np.sqrt(new_vol_sigma**2 + new_rho_sigma**2) / 1.0e12,
                "surrogate_sigma_mass_with_ice_gt": np.sqrt(new_vol_sigma**2 + new_rho_sigma**2 + ice_sigma**2) / 1.0e12,
                "old_density_only_sigma_gt": old_rho_sigma / 1.0e12,
                "surrogate_density_only_sigma_gt": new_rho_sigma / 1.0e12,
                "ice_density_sigma_gt": ice_sigma / 1.0e12,
                "surrogate_density_plus_ice_sigma_gt": np.sqrt(new_rho_sigma**2 + ice_sigma**2) / 1.0e12,
                "old_density_only_sigma_rho_equiv_kg_m3": (
                    old_rho_sigma / abs(volume_change) if volume_change != 0 else np.nan
                ),
                "surrogate_density_only_sigma_rho_equiv_kg_m3": (
                    new_rho_sigma / abs(volume_change) if volume_change != 0 else np.nan
                ),
                "ice_density_sigma_rho_equiv_kg_m3": ice_sigma_rho if volume_change != 0 else np.nan,
                "surrogate_density_plus_ice_sigma_rho_equiv_kg_m3": (
                    np.sqrt(new_rho_sigma**2 + ice_sigma**2) / abs(volume_change)
                    if volume_change != 0 else np.nan
                ),
                "filled_glacier_period_fraction": float(group_table["filled_from_region_period_mean"].mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_global_period_conversions(regional: pd.DataFrame) -> pd.DataFrame:
    """
    Sum regional conversions, treating errors in different regions as independent.

    :param regional: Period summaries from summarize_region_period_conversions().
    :returns: Global mass change totals, rates and standard deviations, with density equivalents.
        If both 2000–2010 and 2010–2020 are present, columns also give their rate difference.
    """

    if regional.empty:
        return regional.copy()
    finite_array(regional["period_years"], "period_years", positive=True)
    rows = []
    for key, group_table in regional.groupby(["start_year", "end_year", "period_years"], sort=True):
        start, end, dt = key

        # Sum regional mass change and volume change, assuming independent errors between regions
        old_mass = float((group_table["old_mass_gt"] * 1.0e12).sum())
        new_mass = float((group_table["surrogate_mass_gt"] * 1.0e12).sum())
        volume_change = float(group_table["dV_m3"].sum())
        row = {
            "start_year": float(start),
            "end_year": float(end),
            "period_years": float(dt),
            "dV_m3": volume_change,
            "old_rho_mean_kg_m3": old_mass / volume_change if volume_change != 0 else np.nan,
            "surrogate_rho_mean_kg_m3": new_mass / volume_change if volume_change != 0 else np.nan,
            "old_mass_gt": old_mass / 1.0e12,
            "surrogate_mass_gt": new_mass / 1.0e12,
            "old_mass_rate_gt_yr": old_mass / dt / 1.0e12,
            "surrogate_mass_rate_gt_yr": new_mass / dt / 1.0e12,
            "mass_rate_difference_gt_yr": (new_mass - old_mass) / dt / 1.0e12,
        }
        for col in [
            "old_sigma_mass_gt",
            "surrogate_sigma_mass_gt",
            "surrogate_sigma_mass_with_ice_gt",
            "old_density_only_sigma_gt",
            "surrogate_density_only_sigma_gt",
            "ice_density_sigma_gt",
            "surrogate_density_plus_ice_sigma_gt",
        ]:
            row[col] = float(np.sqrt(np.sum(group_table[col].to_numpy(float) ** 2)))
            row[col.replace("_gt", "_rate_gt_yr")] = row[col] / dt
        for col in [
            "old_density_only_sigma_gt",
            "surrogate_density_only_sigma_gt",
            "ice_density_sigma_gt",
            "surrogate_density_plus_ice_sigma_gt",
        ]:
            row[col.replace("_gt", "_rho_equiv_kg_m3")] = (
                row[col] * 1.0e12 / abs(volume_change) if volume_change != 0 else np.nan
            )
        rows.append(row)
    out = pd.DataFrame(rows)

    # Compare decadal rates when both 2000–2010 and 2010–2020 are present
    for method in ["old", "surrogate"]:
        first = out.loc[(out["start_year"].eq(2000)) & (out["end_year"].eq(2010)), f"{method}_mass_rate_gt_yr"]
        second = out.loc[(out["start_year"].eq(2010)) & (out["end_year"].eq(2020)), f"{method}_mass_rate_gt_yr"]
        if len(first) and len(second):
            out[f"{method}_mass_rate_change_2010_2020_minus_2000_2010_gt_yr"] = float(second.iloc[0] - first.iloc[0])
            out[f"{method}_loss_acceleration_gt_yr"] = float(-(second.iloc[0] - first.iloc[0]))
    return out
