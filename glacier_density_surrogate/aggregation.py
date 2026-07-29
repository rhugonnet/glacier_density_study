"""Aggregation and uncertainty propagation helpers for effective density."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def haversine_distance_matrix(lat_a: np.ndarray, lon_a: np.ndarray, lat_b: np.ndarray, lon_b: np.ndarray) -> np.ndarray:
    """Pairwise great-circle distances in kilometres.

    :param lat_a: First latitude vector in decimal degrees
    :param lon_a: First longitude vector in decimal degrees
    :param lat_b: Second latitude vector in decimal degrees
    :param lon_b: Second longitude vector in decimal degrees
    """
    radius_km = 6371.0

    # Convert to radians and use broadcasting so no Python loop is needed for
    # the pairwise distance matrix.
    lat1 = np.deg2rad(lat_a)[:, None]
    lon1 = np.deg2rad(lon_a)[:, None]
    lat2 = np.deg2rad(lat_b)[None, :]
    lon2 = np.deg2rad(lon_b)[None, :]
    dlat = lat1 - lat2
    dlon = lon1 - lon2
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * radius_km * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def spherical_correlation(distance_km: np.ndarray, range_km: float) -> np.ndarray:
    """Spherical covariance kernel with unit sill.

    :param distance_km: Distance values in kilometres
    :param range_km: Spherical-model range in kilometres
    """
    h = np.asarray(distance_km, dtype=float) / float(range_km)
    out = np.zeros_like(h, dtype=float)
    inside = h < 1.0
    out[inside] = 1.0 - 1.5 * h[inside] + 0.5 * h[inside] ** 3
    return out


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
) -> dict[tuple[str, int, int], float]:
    """Propagate spatially correlated uncertainties for each group and period.

    For small groups, the covariance double sum is exact. For large groups, the
    diagonal is exact and the off-diagonal covariance is estimated from a capped
    random glacier subsample using block matrix multiplication.

    :param table: Glacier-period table
    :param corr_func: Function returning correlation from distance in kilometres
    :param sigma_col: Per-glacier uncertainty column to propagate
    :param group_col: Spatial group column, typically RGI region
    :param period_cols: Start and end period columns
    :param id_col: Glacier identifier column
    :param lat_col: Glacier latitude column
    :param lon_col: Glacier longitude column
    :param exact_max_items: Maximum group size for exact pairwise propagation
    :param subsample_max_items: Maximum random sample size for off-diagonal terms
    :param block_size: Number of glaciers per distance-matrix block
    :param random_seed: Seed used for large-group subsampling
    """
    rng = np.random.default_rng(random_seed)
    out: dict[tuple[str, int, int], float] = {}
    period_defs = sorted(table[list(period_cols)].drop_duplicates().itertuples(index=False, name=None))
    period_index = pd.MultiIndex.from_tuples(period_defs, names=list(period_cols))
    for group, g in table.groupby(group_col, sort=False):
        meta = g[[id_col, lat_col, lon_col]].drop_duplicates(id_col).reset_index(drop=True)
        tmp = g[[id_col, *period_cols, sigma_col]].copy()
        tmp["period_key"] = list(zip(tmp[period_cols[0]].astype(int), tmp[period_cols[1]].astype(int)))
        support = (
            tmp.pivot_table(index=id_col, columns="period_key", values=sigma_col, aggfunc="first")
            .reindex(index=meta[id_col], columns=period_index)
            .fillna(0.0)
            .to_numpy(float)
        )
        lat = meta[lat_col].to_numpy(float)
        lon = meta[lon_col].to_numpy(float)
        ok = np.isfinite(lat) & np.isfinite(lon)
        support = support[ok]
        lat = lat[ok]
        lon = lon[ok]
        n_items = len(support)
        if n_items == 0:
            continue

        # The diagonal contribution is always exact. It is also the complete
        # answer when the supplied correlation function returns zero off-diagonal.
        diag = np.sum(support**2, axis=0)
        if n_items <= exact_max_items:
            quad = np.zeros(support.shape[1], dtype=float)
            for i0 in range(0, n_items, block_size):
                i1 = min(i0 + block_size, n_items)
                dist = haversine_distance_matrix(lat[i0:i1], lon[i0:i1], lat, lon)
                corr = corr_func(dist)
                corr[np.arange(i1 - i0), np.arange(i0, i1)] = 1.0
                quad += np.sum(support[i0:i1, :] * (corr @ support), axis=0)
        else:
            # For large regions, estimate only the off-diagonal term from a
            # capped random sample and rescale by the number of ordered pairs.
            sample_n = min(subsample_max_items, n_items)
            sample_ids = rng.choice(n_items, size=sample_n, replace=False)
            s_support = support[sample_ids, :]
            s_lat = lat[sample_ids]
            s_lon = lon[sample_ids]
            offdiag_sample = np.zeros(support.shape[1], dtype=float)
            for i0 in range(0, sample_n, block_size):
                i1 = min(i0 + block_size, sample_n)
                dist = haversine_distance_matrix(s_lat[i0:i1], s_lon[i0:i1], s_lat, s_lon)
                corr = corr_func(dist)
                corr[np.arange(i1 - i0), np.arange(i0, i1)] = 0.0
                offdiag_sample += np.sum(s_support[i0:i1, :] * (corr @ s_support), axis=0)
            scale = (n_items * (n_items - 1)) / max(sample_n * (sample_n - 1), 1)
            quad = diag + offdiag_sample * scale

        for (start, end), var in zip(period_defs, quad):
            out[(str(group), int(start), int(end))] = float(np.sqrt(max(var, 0.0)))
    return out


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
) -> dict[tuple[str, int, int], float]:
    """Propagate several spatially correlated components for each group and period.

    Each component is first multiplied by ``scale_col``. For volume-change
    uncertainty, the components are elevation-change errors and ``scale_col`` is
    glacier area, so the returned uncertainty is in cubic metres.

    :param table: Glacier-period table
    :param component_cols: Component uncertainty columns before area scaling
    :param corr_ranges_km: Spherical correlation ranges for each component
    :param group_col: Spatial group column, typically RGI region
    :param period_cols: Start and end period columns
    :param id_col: Glacier identifier column
    :param lat_col: Glacier latitude column
    :param lon_col: Glacier longitude column
    :param scale_col: Multiplicative scale column, usually glacier area
    :param exact_max_items: Maximum group size for exact pairwise propagation
    :param subsample_max_items: Maximum random sample size for off-diagonal terms
    :param block_size: Number of glaciers per distance-matrix block
    :param random_seed: Seed used for large-group subsampling
    """
    if len(component_cols) != len(corr_ranges_km):
        raise ValueError("component_cols and corr_ranges_km must have the same length.")

    rng = np.random.default_rng(random_seed)
    out: dict[tuple[str, int, int], float] = {}
    period_defs = sorted(table[list(period_cols)].drop_duplicates().itertuples(index=False, name=None))
    period_index = pd.MultiIndex.from_tuples(period_defs, names=list(period_cols))

    for group, g in table.groupby(group_col, sort=False):
        meta = g[[id_col, lat_col, lon_col, scale_col]].drop_duplicates(id_col).reset_index(drop=True)
        supports = []
        for component_col in component_cols:
            tmp = g[[id_col, *period_cols, component_col]].copy()
            tmp["period_key"] = list(zip(tmp[period_cols[0]].astype(int), tmp[period_cols[1]].astype(int)))
            comp = (
                tmp.pivot_table(index=id_col, columns="period_key", values=component_col, aggfunc="first")
                .reindex(index=meta[id_col], columns=period_index)
                .fillna(0.0)
                .to_numpy(float)
            )
            supports.append(comp * meta[scale_col].to_numpy(float)[:, None])

        lat = meta[lat_col].to_numpy(float)
        lon = meta[lon_col].to_numpy(float)
        ok = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(meta[scale_col].to_numpy(float))
        supports = [support[ok] for support in supports]
        lat = lat[ok]
        lon = lon[ok]
        n_items = len(lat)
        if n_items == 0:
            continue

        # Sum all component diagonals exactly before estimating the correlated
        # off-diagonal contribution.
        diag = np.zeros(len(period_defs), dtype=float)
        for support in supports:
            diag += np.sum(support**2, axis=0)

        if n_items <= exact_max_items:
            quad = np.zeros(len(period_defs), dtype=float)
            for i0 in range(0, n_items, block_size):
                i1 = min(i0 + block_size, n_items)
                dist = haversine_distance_matrix(lat[i0:i1], lon[i0:i1], lat, lon)
                for support, range_km in zip(supports, corr_ranges_km):
                    corr = spherical_correlation(dist, range_km)
                    corr[np.arange(i1 - i0), np.arange(i0, i1)] = 1.0
                    quad += np.sum(support[i0:i1, :] * (corr @ support), axis=0)
        else:
            sample_n = min(subsample_max_items, n_items)
            sample_ids = rng.choice(n_items, size=sample_n, replace=False)
            s_lat = lat[sample_ids]
            s_lon = lon[sample_ids]
            s_supports = [support[sample_ids, :] for support in supports]
            offdiag_sample = np.zeros(len(period_defs), dtype=float)
            for i0 in range(0, sample_n, block_size):
                i1 = min(i0 + block_size, sample_n)
                dist = haversine_distance_matrix(s_lat[i0:i1], s_lon[i0:i1], s_lat, s_lon)
                for support, range_km in zip(s_supports, corr_ranges_km):
                    corr = spherical_correlation(dist, range_km)
                    corr[np.arange(i1 - i0), np.arange(i0, i1)] = 0.0
                    offdiag_sample += np.sum(support[i0:i1, :] * (corr @ support), axis=0)
            scale = (n_items * (n_items - 1)) / max(sample_n * (sample_n - 1), 1)
            quad = diag + offdiag_sample * scale

        for (start, end), var in zip(period_defs, quad):
            out[(str(group), int(start), int(end))] = float(np.sqrt(max(var, 0.0)))
    return out


def summarize_region_period_conversions(
    periods: pd.DataFrame,
    *,
    spatial_density_sigma: dict[tuple[str, int, int], float],
    spatial_volume_sigma: dict[tuple[str, int, int], float] | None = None,
    old_rho: float = 850.0,
    old_sigma_rho: float = 60.0,
    ice_sigma_rho: float = 5.0,
) -> pd.DataFrame:
    """Aggregate glacier-period mass conversions to regional periods.

    The old conversion assumes ``old_rho +/- old_sigma_rho`` at glacier scale
    with fully correlated density error inside each region. The surrogate
    conversion uses the supplied spatially propagated density uncertainty, and
    can optionally use a separately propagated volume uncertainty.

    :param periods: Glacier-period conversion table
    :param spatial_density_sigma: Region-period density-related mass uncertainty
    :param spatial_volume_sigma: Optional region-period volume uncertainty
    :param old_rho: Reference density used by the old conversion
    :param old_sigma_rho: Reference density uncertainty used by the old conversion
    :param ice_sigma_rho: Additional fully correlated uncertainty in ice density
    """
    rows = []
    group_cols = ["region_group", "region_label", "start_year", "end_year", "period_years"]
    for key, g in periods.groupby(group_cols, sort=True):
        group, label, start, end, dt = key

        # Aggregate volume and mass first. The regional effective density is the
        # mass/volume ratio, equivalent to volume-change weighting.
        dV = float(g["dV_m3"].sum())
        old_mass = float(g["dM_old_kg"].sum())
        new_mass = float(g["dM_surrogate_kg"].sum())
        new_rho_mean = new_mass / dV if dV != 0 else np.nan

        # Volume-change uncertainty can either be supplied as a region-period
        # covariance result, or approximated as independent glacier terms.
        if spatial_volume_sigma is None:
            old_vol_sigma = float(np.sqrt(np.nansum(g["sigma_dM_volume_old_kg"].to_numpy(float) ** 2)))
            new_vol_sigma = float(np.sqrt(np.nansum(g["sigma_dM_volume_surrogate_kg"].to_numpy(float) ** 2)))
            volume_sigma = old_vol_sigma / old_rho
        else:
            volume_sigma = spatial_volume_sigma[(str(group), int(start), int(end))]
            old_vol_sigma = old_rho * volume_sigma
            new_vol_sigma = abs(new_rho_mean) * volume_sigma if np.isfinite(new_rho_mean) else np.nan
        old_rho_sigma = old_sigma_rho * abs(dV)
        new_rho_sigma = spatial_density_sigma[(str(group), int(start), int(end))]
        ice_sigma = ice_sigma_rho * abs(dV)

        # Keep the components explicit in the output table. This makes it easier
        # to diagnose whether volume, surrogate density, or ice density dominates.
        rows.append(
            {
                "region_group": group,
                "region_label": label,
                "start_year": int(start),
                "end_year": int(end),
                "period_years": int(dt),
                "n_glaciers": int(g["rgiid"].nunique()),
                "area_m2": float(g.drop_duplicates("rgiid")["area"].sum()),
                "dV_m3": dV,
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
                "old_density_only_sigma_rho_equiv_kg_m3": old_rho_sigma / abs(dV) if dV != 0 else np.nan,
                "surrogate_density_only_sigma_rho_equiv_kg_m3": new_rho_sigma / abs(dV) if dV != 0 else np.nan,
                "ice_density_sigma_rho_equiv_kg_m3": ice_sigma_rho if dV != 0 else np.nan,
                "surrogate_density_plus_ice_sigma_rho_equiv_kg_m3": np.sqrt(new_rho_sigma**2 + ice_sigma**2) / abs(dV) if dV != 0 else np.nan,
                "filled_glacier_period_fraction": float(g["filled_from_region_period_mean"].mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_global_period_conversions(regional: pd.DataFrame) -> pd.DataFrame:
    """Aggregate regional conversion summaries globally with independent regions.

    :param regional: Region-period table from ``summarize_region_period_conversions``
    """
    rows = []
    for key, g in regional.groupby(["start_year", "end_year", "period_years"], sort=True):
        start, end, dt = key

        # Regional mass changes are summed directly. Uncertainty is propagated
        # independently between regions, matching the Hugonnet-style comparison.
        old_mass = float((g["old_mass_gt"] * 1.0e12).sum())
        new_mass = float((g["surrogate_mass_gt"] * 1.0e12).sum())
        dV = float(g["dV_m3"].sum())
        row = {
            "start_year": int(start),
            "end_year": int(end),
            "period_years": int(dt),
            "dV_m3": dV,
            "old_rho_mean_kg_m3": old_mass / dV if dV != 0 else np.nan,
            "surrogate_rho_mean_kg_m3": new_mass / dV if dV != 0 else np.nan,
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
            row[col] = float(np.sqrt(np.nansum(g[col].to_numpy(float) ** 2)))
            row[col.replace("_gt", "_rate_gt_yr")] = row[col] / dt
        for col in [
            "old_density_only_sigma_gt",
            "surrogate_density_only_sigma_gt",
            "ice_density_sigma_gt",
            "surrogate_density_plus_ice_sigma_gt",
        ]:
            row[col.replace("_gt", "_rho_equiv_kg_m3")] = row[col] * 1.0e12 / abs(dV) if dV != 0 else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)

    # Add the decadal acceleration diagnostic when both decades are present.
    for method in ["old", "surrogate"]:
        first = out.loc[(out["start_year"].eq(2000)) & (out["end_year"].eq(2010)), f"{method}_mass_rate_gt_yr"]
        second = out.loc[(out["start_year"].eq(2010)) & (out["end_year"].eq(2020)), f"{method}_mass_rate_gt_yr"]
        if len(first) and len(second):
            out[f"{method}_mass_rate_change_2010_2020_minus_2000_2010_gt_yr"] = float(second.iloc[0] - first.iloc[0])
            out[f"{method}_loss_acceleration_gt_yr"] = float(-(second.iloc[0] - first.iloc[0]))
    return out
