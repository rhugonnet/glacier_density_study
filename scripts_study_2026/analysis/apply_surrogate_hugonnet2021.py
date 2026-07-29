#!/usr/bin/env python3
"""Apply the effective-density surrogate to Hugonnet-style glacier volume changes.

The input files are glacier-scale cumulative elevation-change time series
(`*_int_base*.csv`). We build 5-year elementary periods from endpoint
differences, then derive all contiguous 5-, 10- and 20-year periods between
2000 and 2020. The 5-year periods are the elementary periods used for temporal
closure of the surrogate mean and uncertainty.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from glacier_density_surrogate import (
    RhoSurrogate,
    integrated_mu_vectorized,
    integrated_sigma_mass_vectorized,
    integrated_sigma_vectorized,
    spatially_correlated_component_sigma_by_group_period,
    spatially_correlated_sigma_by_group_period,
    summarize_global_period_conversions,
    summarize_region_period_conversions,
    temporally_reconcile_periods,
)

STUDY_DIR = Path(__file__).resolve().parents[1]
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

from study_paths import HUGONNET_DIAGNOSTICS_DIR, HUGONNET_MAIN_PATH


INPUT_DIR = Path("/home/atom/ongoing/own/ww_tvol_study/vol_final/vol_final_shiftcorr")
OUT_DIR = HUGONNET_DIAGNOSTICS_DIR

OUT_MAIN = HUGONNET_MAIN_PATH
OUT_REGIONAL = OUT_DIR / "regional_period_conversion_summary.csv"
OUT_GLOBAL = OUT_DIR / "global_period_conversion_summary.csv"
OUT_GLACIER_CHANGES = OUT_DIR / "glacier_period_average_changes.csv"
OUT_ELEVATION_CACHE = OUT_DIR / "glacier_period_elevation_changes_5yr_10yr_20yr.csv"

TARGET_YEARS = np.array([2000, 2005, 2010, 2015, 2020], dtype=int)
PERIOD_LENGTHS_TO_KEEP = {5, 10, 20}
OLD_RHO = 850.0
OLD_SIGMA_RHO = 60.0
ICE_SIGMA_RHO = 5.0
SPATIAL_BLOCK_SIZE = 700
EXACT_SPATIAL_MAX_GLACIERS = 3500
SPATIAL_SUBSAMPLE_MAX_GLACIERS = 10_000
RANDOM_SEED = 426
VOLUME_CORR_RANGES_M = (150, 2000, 5000, 20000, 50000, 200000)
PAST_DH_METHOD = "annualized_5yr_elementary_v3_pre2000_fixed_minus0p3"
PRE2000_PAST_DH_RATE_M_PER_YR = -0.3
PRE2000_PAST_DH_SIGMA_M_PER_YR = 0.0
REBUILD_CACHE = False
CACHE_ONLY = False


def selected_region_files() -> list[tuple[str, Path]]:
    """Return input files without double-counting merged regions."""
    groups = ["01_02"] + [f"{reg:02d}" for reg in range(3, 20) if reg not in (13, 14, 15)] + ["13_14_15"]
    out = []
    for group in groups:
        path = INPUT_DIR / f"dh_{group}_rgi60_int_base.csv"
        if not path.exists() and group == "06":
            path = INPUT_DIR / "dh_06_rgi60_int_base_old.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing glacier-scale input for region group {group}: {path}")
        out.append((group, path))
    return out


def nearest_endpoint_dates(path: Path) -> dict[int, pd.Timestamp]:
    """Find the nearest available time slice for each target year."""
    times = pd.read_csv(path, usecols=["time"], parse_dates=["time"])["time"].drop_duplicates().sort_values().reset_index(drop=True)
    out = {}
    for year in TARGET_YEARS:
        target = pd.Timestamp(f"{year}-01-01")
        out[int(year)] = times.iloc[(times - target).abs().argmin()]
    return out


def parse_region_label(group: str) -> str:
    """Human-readable region group label."""
    return group.replace("_", "+")


def haversine_distance_matrix(lat_a: np.ndarray, lon_a: np.ndarray, lat_b: np.ndarray, lon_b: np.ndarray) -> np.ndarray:
    """Pairwise great-circle distances in kilometres."""
    radius_km = 6371.0
    lat1 = np.deg2rad(lat_a)[:, None]
    lon1 = np.deg2rad(lon_a)[:, None]
    lat2 = np.deg2rad(lat_b)[None, :]
    lon2 = np.deg2rad(lon_b)[None, :]
    dlat = lat1 - lat2
    dlon = lon1 - lon2
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * radius_km * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def load_endpoint_table(group: str, path: Path) -> pd.DataFrame:
    """Read endpoint rows and pivot cumulative elevation changes by target year."""
    endpoints = nearest_endpoint_dates(path)
    endpoint_dates = set(endpoints.values())
    err_corr_cols = [f"err_corr_{corr}" for corr in VOLUME_CORR_RANGES_M]
    usecols = ["rgiid", "time", "area", "dh", "err_dh", "lat", "lon", "reg", "valid_obs_py", "perc_err_cont", *err_corr_cols]
    df = pd.read_csv(path, usecols=usecols, parse_dates=["time"], low_memory=True)
    df = df.loc[df["time"].isin(endpoint_dates)].copy()
    year_by_time = {time: year for year, time in endpoints.items()}
    df["target_year"] = df["time"].map(year_by_time).astype(int)
    df = df.sort_values(["rgiid", "target_year"]).drop_duplicates(["rgiid", "target_year"], keep="last")

    meta = df.sort_values("target_year").drop_duplicates("rgiid", keep="last")[["rgiid", "area", "lat", "lon", "reg", "perc_err_cont"]].copy()
    meta["region_group"] = group
    meta["region_label"] = parse_region_label(group)
    dh_wide = df.pivot(index="rgiid", columns="target_year", values="dh").add_prefix("dh_")
    err_wide = df.pivot(index="rgiid", columns="target_year", values="err_dh").add_prefix("err_")
    valid_wide = df.pivot(index="rgiid", columns="target_year", values="valid_obs_py").add_prefix("valid_")
    err_corr_wide = [
        df.pivot(index="rgiid", columns="target_year", values=f"err_corr_{corr}").add_prefix(f"errcorr_{corr}_")
        for corr in VOLUME_CORR_RANGES_M
    ]
    out = meta.set_index("rgiid").join([dh_wide, err_wide, valid_wide, *err_corr_wide]).reset_index()
    return out


def build_periods(endpoint_table: pd.DataFrame) -> pd.DataFrame:
    """Build all 5-, 10- and 20-year periods from endpoint differences."""
    rows = []
    for i0, start in enumerate(TARGET_YEARS[:-1]):
        for end in TARGET_YEARS[i0 + 1 :]:
            dt = int(end - start)
            if dt not in PERIOD_LENGTHS_TO_KEEP:
                continue
            dh0 = pd.to_numeric(endpoint_table[f"dh_{start}"], errors="coerce")
            dh1 = pd.to_numeric(endpoint_table[f"dh_{end}"], errors="coerce")
            sig0 = pd.to_numeric(endpoint_table[f"err_{start}"], errors="coerce")
            sig1 = pd.to_numeric(endpoint_table[f"err_{end}"], errors="coerce")
            period = endpoint_table[["rgiid", "region_group", "region_label", "reg", "area", "lat", "lon", "perc_err_cont"]].copy()
            period["start_year"] = int(start)
            period["end_year"] = int(end)
            period["period_years"] = dt
            period["dh_m"] = dh1 - dh0
            period["sigma_dh_m"] = np.sqrt(sig0**2 + sig1**2)
            for corr in VOLUME_CORR_RANGES_M:
                corr0 = pd.to_numeric(endpoint_table[f"errcorr_{corr}_{start}"], errors="coerce")
                corr1 = pd.to_numeric(endpoint_table[f"errcorr_{corr}_{end}"], errors="coerce")
                period[f"sigma_dh_corr_{corr}_m"] = np.sqrt(corr0**2 + corr1**2)
            period["is_observed_period"] = np.isfinite(period["dh_m"]) & np.isfinite(period["sigma_dh_m"])
            rows.append(period)
    periods = pd.concat(rows, ignore_index=True)
    return fill_nodata_with_region_period_mean(periods)


def fill_nodata_with_region_period_mean(periods: pd.DataFrame) -> pd.DataFrame:
    """Fill missing/nodata glacier periods using the regional period mean."""
    out = periods.copy()
    for _, ids in out.groupby(["region_group", "start_year", "end_year"]).groups.items():
        idx = np.asarray(list(ids))
        sub = out.iloc[idx]
        valid = sub["is_observed_period"].to_numpy(bool)
        area = pd.to_numeric(sub["area"], errors="coerce").to_numpy(float)
        if np.any(valid):
            weights = np.where(valid & np.isfinite(area) & (area > 0), area, 0.0)
            if np.sum(weights) > 0:
                fill_dh = float(np.sum(weights * sub["dh_m"].to_numpy(float)) / np.sum(weights))
                fill_sig = float(np.sqrt(np.sum(weights * sub["sigma_dh_m"].to_numpy(float) ** 2) / np.sum(weights)))
            else:
                fill_dh = float(np.nanmean(sub.loc[valid, "dh_m"]))
                fill_sig = float(np.nanmean(sub.loc[valid, "sigma_dh_m"]))
        else:
            fill_dh = 0.0
            fill_sig = float(np.nanmedian(out["sigma_dh_m"]))
        missing = ~valid | ~np.isfinite(sub["dh_m"].to_numpy(float)) | ~np.isfinite(sub["sigma_dh_m"].to_numpy(float))
        out.loc[sub.index[missing], "dh_m"] = fill_dh
        out.loc[sub.index[missing], "sigma_dh_m"] = fill_sig
        for corr in VOLUME_CORR_RANGES_M:
            col = f"sigma_dh_corr_{corr}_m"
            vals = sub[col].to_numpy(float)
            if np.any(valid & np.isfinite(vals)):
                weights = np.where(valid & np.isfinite(vals) & np.isfinite(area) & (area > 0), area, 0.0)
                fill_corr = float(np.sqrt(np.sum(weights * vals**2) / np.sum(weights))) if np.sum(weights) > 0 else float(np.nanmean(vals[valid & np.isfinite(vals)]))
            else:
                fill_corr = 0.0
            out.loc[sub.index[missing], col] = fill_corr
    out["filled_from_region_period_mean"] = ~out["is_observed_period"]
    return out


def attach_past_dh(periods: pd.DataFrame, model: RhoSurrogate) -> pd.DataFrame:
    """Attach past elevation-change predictor from 5-year elementary rates."""
    out = periods.copy()
    tau = float(model.params.get("T_p", 5.0))
    tau_max = float(model.params.get("tau_max", 20.0))
    elementary = out.loc[out["period_years"].eq(5), ["rgiid", "start_year", "end_year", "dh_m", "sigma_dh_m"]].copy()
    elementary["dh_rate_m_per_yr"] = elementary["dh_m"] / 5.0
    elementary["sigma_rate_m_per_yr"] = elementary["sigma_dh_m"] / 5.0
    rate = elementary.pivot(index="rgiid", columns="start_year", values="dh_rate_m_per_yr")
    rate_sig = elementary.pivot(index="rgiid", columns="start_year", values="sigma_rate_m_per_yr")
    past_tables = []
    for start in sorted(out["start_year"].unique()):
        start = int(start)
        if start == int(TARGET_YEARS[0]):
            # Hugonnet et al. (2021) report a global thinning-rate acceleration
            # of about 0.10 m yr-1 per decade. Rolling back the 2000-2010
            # rate gives a defensible pre-2000 state of ca. -0.3 m yr-1,
            # here redistributed uniformly over the 1990-2000 decade.
            vals = pd.Series(PRE2000_PAST_DH_RATE_M_PER_YR, index=rate.index)
            sigs = pd.Series(PRE2000_PAST_DH_SIGMA_M_PER_YR, index=rate_sig.index)
        else:
            elem_starts = [int(s) for s in rate.columns if int(s) + 5 <= start]
            if elem_starts:
                # The past predictor is annual-equivalent. Split each 5-year
                # elementary rate into five equal annual substeps for memory
                # weighting, while keeping the five substeps from one block
                # fully correlated for uncertainty propagation.
                block_weights = []
                for elem_start in elem_starts:
                    annual_ends = np.arange(elem_start + 1, elem_start + 6, dtype=float)
                    lags = float(start) - annual_ends
                    lags = lags[(lags >= 0) & (lags <= tau_max)]
                    block_weights.append(float(np.sum(np.exp(-lags / tau))) if len(lags) else 0.0)
                weights = np.array(block_weights, dtype=float)
                weights = weights / np.sum(weights)
                vals = pd.Series(rate[elem_starts].to_numpy(float) @ weights, index=rate.index)
                sigs = pd.Series(np.sqrt((rate_sig[elem_starts].to_numpy(float) ** 2) @ (weights**2)), index=rate_sig.index)
            else:
                vals = rate.get(start, pd.Series(np.nan, index=rate.index))
                sigs = rate_sig.get(start, pd.Series(np.nan, index=rate_sig.index))
        tmp = pd.DataFrame({"rgiid": rate.index, "start_year": start, "past_dh_m": vals.to_numpy(float), "sigma_past_dh_m": sigs.to_numpy(float)})
        past_tables.append(tmp)
    past_df = pd.concat(past_tables, ignore_index=True)
    out = out.merge(past_df, on=["rgiid", "start_year"], how="left")
    fallback = out["dh_m"] / out["period_years"]
    fallback_sig = out["sigma_dh_m"] / out["period_years"]
    out["past_dh_m"] = out["past_dh_m"].fillna(fallback)
    out["sigma_past_dh_m"] = out["sigma_past_dh_m"].fillna(fallback_sig)
    return out


def build_elevation_change_cache(model: RhoSurrogate) -> pd.DataFrame:
    """Build and write the glacier-period elevation-change table."""
    all_periods = []
    for group, path in selected_region_files():
        print(f"[read] {group}: {path.name}", flush=True)
        endpoints = load_endpoint_table(group, path)
        periods = build_periods(endpoints)
        periods = attach_past_dh(periods, model)
        all_periods.append(periods)
    periods = pd.concat(all_periods, ignore_index=True)
    periods["past_dh_method"] = PAST_DH_METHOD
    periods.to_csv(OUT_ELEVATION_CACHE, index=False)
    print(f"[done] Wrote elevation-change cache: {OUT_ELEVATION_CACHE}", flush=True)
    return periods


def read_or_build_elevation_changes(model: RhoSurrogate, rebuild_cache: bool = False) -> pd.DataFrame:
    """Read cached glacier-period elevation changes, or rebuild them."""
    if OUT_ELEVATION_CACHE.exists() and not rebuild_cache:
        columns = pd.read_csv(OUT_ELEVATION_CACHE, nrows=0).columns
        required = {"perc_err_cont", "past_dh_method", *[f"sigma_dh_corr_{corr}_m" for corr in VOLUME_CORR_RANGES_M]}
        if required.issubset(columns):
            method = pd.read_csv(OUT_ELEVATION_CACHE, usecols=["past_dh_method"], nrows=1)["past_dh_method"].iloc[0]
            if method != PAST_DH_METHOD:
                print("[rebuild] Cached elevation-change table uses an older past-dh method.", flush=True)
                return build_elevation_change_cache(model)
            print(f"[read] Cached elevation-change table: {OUT_ELEVATION_CACHE}", flush=True)
            return pd.read_csv(OUT_ELEVATION_CACHE, low_memory=False)
        print("[rebuild] Cached elevation-change table lacks required method or uncertainty columns.", flush=True)
    return build_elevation_change_cache(model)


def spatial_volume_sigma(periods: pd.DataFrame) -> dict[tuple[str, int, int], float]:
    """Regional period volume-change uncertainty from original correlated components."""
    corr_sigma = spatially_correlated_component_sigma_by_group_period(
        periods,
        component_cols=tuple(f"sigma_dh_corr_{corr}_m" for corr in VOLUME_CORR_RANGES_M),
        corr_ranges_km=tuple(corr / 1000.0 for corr in VOLUME_CORR_RANGES_M),
        exact_max_items=EXACT_SPATIAL_MAX_GLACIERS,
        subsample_max_items=SPATIAL_SUBSAMPLE_MAX_GLACIERS,
        block_size=SPATIAL_BLOCK_SIZE,
        random_seed=RANDOM_SEED,
    )
    out = {}
    for key, g in periods.groupby(["region_group", "start_year", "end_year"], sort=True):
        group, start, end = key
        dV = float(g["dV_m3"].sum())
        area = g["area"].to_numpy(float)
        perc = g["perc_err_cont"].to_numpy(float)
        area_sum = np.nansum(area)
        perc_region = np.nansum(perc * area) / area_sum if area_sum > 0 else 0.0
        contour_sigma = abs(dV) * perc_region / 100.0
        key_out = (str(group), int(start), int(end))
        out[key_out] = float(np.sqrt(corr_sigma[key_out] ** 2 + contour_sigma**2))
    return out


def apply_conversions(periods: pd.DataFrame, model: RhoSurrogate) -> pd.DataFrame:
    """Apply old and surrogate conversions to glacier-period rows."""
    out = periods.copy()
    out["dV_m3"] = out["area"] * out["dh_m"]
    out["sigma_dV_m3"] = out["area"] * out["sigma_dh_m"]
    out["dM_old_kg"] = OLD_RHO * out["dV_m3"]
    out["sigma_dM_rho_old_kg"] = OLD_SIGMA_RHO * np.abs(out["dV_m3"])
    out["sigma_dM_volume_old_kg"] = OLD_RHO * np.abs(out["sigma_dV_m3"])

    chunk = 200_000
    mu = np.empty(len(out), dtype=float)
    sigma = np.empty(len(out), dtype=float)
    for i0 in range(0, len(out), chunk):
        i1 = min(i0 + chunk, len(out))
        sl = slice(i0, i1)
        mu[sl] = integrated_mu_vectorized(
            model,
            out["dh_m"].to_numpy(float)[sl],
            out["sigma_dh_m"].to_numpy(float)[sl],
            out["past_dh_m"].to_numpy(float)[sl],
            out["sigma_past_dh_m"].to_numpy(float)[sl],
            out["period_years"].to_numpy(float)[sl],
        )
        sigma[sl] = integrated_sigma_vectorized(
            model,
            out["dh_m"].to_numpy(float)[sl],
            out["sigma_dh_m"].to_numpy(float)[sl],
            out["period_years"].to_numpy(float)[sl],
        )
    out["mu_rho_independent_kg_m3"] = mu
    out["sigma_rho_independent_kg_m3"] = sigma
    out.loc[~np.isfinite(out["mu_rho_independent_kg_m3"]), "mu_rho_independent_kg_m3"] = model.rho_ice
    out["dM_independent_kg"] = out["mu_rho_independent_kg_m3"] * out["dV_m3"]
    out["sigma_dM_rho_independent_kg"] = integrated_sigma_mass_vectorized(
        model,
        out["area"].to_numpy(float),
        out["dh_m"].to_numpy(float),
        out["sigma_dh_m"].to_numpy(float),
        out["period_years"].to_numpy(float),
    )
    out = temporally_reconcile_periods(out, model)
    out["sigma_dM_volume_surrogate_kg"] = np.abs(out["mu_rho_surrogate_kg_m3"] * out["sigma_dV_m3"])
    return out


def spatial_sigma(periods: pd.DataFrame, model: RhoSurrogate, sigma_col: str) -> dict[tuple[str, int, int], float]:
    """Propagate density-related uncertainty spatially within each region and period."""
    rng = np.random.default_rng(RANDOM_SEED)
    out: dict[tuple[str, int, int], float] = {}
    period_defs = sorted(periods[["start_year", "end_year"]].drop_duplicates().itertuples(index=False, name=None))
    period_index = pd.MultiIndex.from_tuples(period_defs, names=["start_year", "end_year"])
    for group, g in periods.groupby("region_group", sort=False):
        meta = g[["rgiid", "lat", "lon"]].drop_duplicates("rgiid").reset_index(drop=True)
        support = (
            g.assign(period_key=pd.MultiIndex.from_frame(g[["start_year", "end_year"]]))
            .pivot_table(index="rgiid", columns="period_key", values=sigma_col, aggfunc="first")
            .reindex(index=meta["rgiid"], columns=period_index)
            .fillna(0.0)
            .to_numpy(float)
        )
        lat = meta["lat"].to_numpy(float)
        lon = meta["lon"].to_numpy(float)
        ok = np.isfinite(lat) & np.isfinite(lon)
        support = support[ok]
        lat = lat[ok]
        lon = lon[ok]
        n = len(support)
        if n == 0:
            continue
        diag = np.sum(support**2, axis=0)
        if n <= EXACT_SPATIAL_MAX_GLACIERS:
            quad = np.zeros(support.shape[1], dtype=float)
            for i0 in range(0, n, SPATIAL_BLOCK_SIZE):
                i1 = min(i0 + SPATIAL_BLOCK_SIZE, n)
                dist = haversine_distance_matrix(lat[i0:i1], lon[i0:i1], lat, lon)
                corr = model.spatial_corr(dist)
                corr[np.arange(i1 - i0), np.arange(i0, i1)] = 1.0
                quad += np.sum(support[i0:i1, :] * (corr @ support), axis=0)
        else:
            sample_n = min(SPATIAL_SUBSAMPLE_MAX_GLACIERS, n)
            sample_ids = rng.choice(n, size=sample_n, replace=False)
            s_support = support[sample_ids, :]
            s_lat = lat[sample_ids]
            s_lon = lon[sample_ids]
            offdiag_sample = np.zeros(support.shape[1], dtype=float)
            for i0 in range(0, sample_n, SPATIAL_BLOCK_SIZE):
                i1 = min(i0 + SPATIAL_BLOCK_SIZE, sample_n)
                dist = haversine_distance_matrix(s_lat[i0:i1], s_lon[i0:i1], s_lat, s_lon)
                corr = model.spatial_corr(dist)
                corr[np.arange(i1 - i0), np.arange(i0, i1)] = 0.0
                offdiag_sample += np.sum(s_support[i0:i1, :] * (corr @ s_support), axis=0)
            scale = (n * (n - 1)) / max(sample_n * (sample_n - 1), 1)
            quad = diag + offdiag_sample * scale
        for (start, end), var in zip(period_defs, quad):
            out[(str(group), int(start), int(end))] = float(np.sqrt(max(var, 0.0)))
    return out


def summarize_region_periods(periods: pd.DataFrame, model: RhoSurrogate) -> pd.DataFrame:
    """Aggregate glacier conversions to regional periods."""
    new_spatial = spatial_sigma(periods, model, "sigma_dM_rho_surrogate_kg")
    rows = []
    for key, g in periods.groupby(["region_group", "region_label", "start_year", "end_year", "period_years"], sort=True):
        group, label, start, end, dt = key
        dV = float(g["dV_m3"].sum())
        old_mass = float(g["dM_old_kg"].sum())
        new_mass = float(g["dM_surrogate_kg"].sum())
        old_vol_sigma = float(np.sqrt(np.nansum(g["sigma_dM_volume_old_kg"].to_numpy(float) ** 2)))
        new_vol_sigma = float(np.sqrt(np.nansum(g["sigma_dM_volume_surrogate_kg"].to_numpy(float) ** 2)))
        old_rho_sigma = OLD_SIGMA_RHO * abs(dV)
        new_rho_sigma = new_spatial[(str(group), int(start), int(end))]
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
                "old_rho_mean_kg_m3": OLD_RHO,
                "surrogate_rho_mean_kg_m3": new_mass / dV if dV != 0 else np.nan,
                "old_mass_gt": old_mass / 1.0e12,
                "surrogate_mass_gt": new_mass / 1.0e12,
                "old_mass_rate_gt_yr": old_mass / dt / 1.0e12,
                "surrogate_mass_rate_gt_yr": new_mass / dt / 1.0e12,
                "mass_rate_difference_gt_yr": (new_mass - old_mass) / dt / 1.0e12,
                "old_sigma_mass_gt": np.sqrt(old_vol_sigma**2 + old_rho_sigma**2) / 1.0e12,
                "surrogate_sigma_mass_gt": np.sqrt(new_vol_sigma**2 + new_rho_sigma**2) / 1.0e12,
                "old_sigma_rho_equiv_kg_m3": np.sqrt(old_vol_sigma**2 + old_rho_sigma**2) / abs(dV) if dV != 0 else np.nan,
                "surrogate_sigma_rho_equiv_kg_m3": np.sqrt(new_vol_sigma**2 + new_rho_sigma**2) / abs(dV) if dV != 0 else np.nan,
                "old_density_only_sigma_gt": old_rho_sigma / 1.0e12,
                "surrogate_density_only_sigma_gt": new_rho_sigma / 1.0e12,
                "filled_glacier_period_fraction": float(g["filled_from_region_period_mean"].mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_global(regional: pd.DataFrame) -> pd.DataFrame:
    """Aggregate regional summaries globally with regions independent."""
    rows = []
    for key, g in regional.groupby(["start_year", "end_year", "period_years"], sort=True):
        start, end, dt = key
        old_mass = float((g["old_mass_gt"] * 1.0e12).sum())
        new_mass = float((g["surrogate_mass_gt"] * 1.0e12).sum())
        dV = float(g["dV_m3"].sum())
        rows.append(
            {
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
                "old_sigma_mass_gt": float(np.sqrt(np.nansum(g["old_sigma_mass_gt"].to_numpy(float) ** 2))),
                "surrogate_sigma_mass_gt": float(np.sqrt(np.nansum(g["surrogate_sigma_mass_gt"].to_numpy(float) ** 2))),
                "old_density_only_sigma_gt": float(np.sqrt(np.nansum(g["old_density_only_sigma_gt"].to_numpy(float) ** 2))),
                "surrogate_density_only_sigma_gt": float(np.sqrt(np.nansum(g["surrogate_density_only_sigma_gt"].to_numpy(float) ** 2))),
                "old_sigma_rho_equiv_kg_m3": float(np.sqrt(np.nansum((g["old_sigma_mass_gt"].to_numpy(float) * 1.0e12) ** 2))) / abs(dV) if dV != 0 else np.nan,
                "surrogate_sigma_rho_equiv_kg_m3": float(np.sqrt(np.nansum((g["surrogate_sigma_mass_gt"].to_numpy(float) * 1.0e12) ** 2))) / abs(dV) if dV != 0 else np.nan,
                "old_density_only_sigma_rho_equiv_kg_m3": float(np.sqrt(np.nansum((g["old_density_only_sigma_gt"].to_numpy(float) * 1.0e12) ** 2))) / abs(dV) if dV != 0 else np.nan,
                "surrogate_density_only_sigma_rho_equiv_kg_m3": float(np.sqrt(np.nansum((g["surrogate_density_only_sigma_gt"].to_numpy(float) * 1.0e12) ** 2))) / abs(dV) if dV != 0 else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    for method in ["old", "surrogate"]:
        first = out.loc[(out["start_year"].eq(2000)) & (out["end_year"].eq(2010)), f"{method}_mass_rate_gt_yr"]
        second = out.loc[(out["start_year"].eq(2010)) & (out["end_year"].eq(2020)), f"{method}_mass_rate_gt_yr"]
        if len(first) and len(second):
            out[f"{method}_mass_rate_change_2010_2020_minus_2000_2010_gt_yr"] = float(second.iloc[0] - first.iloc[0])
            out[f"{method}_loss_acceleration_gt_yr"] = float(-(second.iloc[0] - first.iloc[0]))
    return out


def summarize_glacier_changes(periods: pd.DataFrame) -> pd.DataFrame:
    """Summarize average glacier-scale changes from 850+/-60 to the surrogate."""
    rows = []
    for key, g in periods.groupby(["period_years"], sort=True):
        key = key[0] if isinstance(key, tuple) else key
        weight = np.abs(g["dV_m3"].to_numpy(float))
        rows.append(
            {
                "period_years": int(key),
                "n_glacier_periods": int(len(g)),
                "mean_mu_change_kg_m3": float(np.nanmean(g["mu_rho_surrogate_kg_m3"] - OLD_RHO)),
                "volume_weighted_mu_change_kg_m3": float(np.nansum(weight * (g["mu_rho_surrogate_kg_m3"] - OLD_RHO)) / np.nansum(weight)),
                "mean_sigma_rho_change_kg_m3": float(np.nanmean((g["sigma_rho_surrogate_kg_m3"] - OLD_SIGMA_RHO).replace([np.inf, -np.inf], np.nan))),
                "volume_weighted_sigma_rho_change_kg_m3": float(
                    np.nansum(weight * (g["sigma_rho_surrogate_kg_m3"] - OLD_SIGMA_RHO).replace([np.inf, -np.inf], np.nan))
                    / np.nansum(weight[np.isfinite((g["sigma_rho_surrogate_kg_m3"] - OLD_SIGMA_RHO).to_numpy(float))])
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = RhoSurrogate()
    periods = read_or_build_elevation_changes(model, rebuild_cache=REBUILD_CACHE)
    if CACHE_ONLY:
        return
    periods = apply_conversions(periods, model)
    spatial_density_sigma = spatially_correlated_sigma_by_group_period(
        periods,
        model.spatial_corr,
        sigma_col="sigma_dM_rho_surrogate_kg",
        exact_max_items=EXACT_SPATIAL_MAX_GLACIERS,
        subsample_max_items=SPATIAL_SUBSAMPLE_MAX_GLACIERS,
        block_size=SPATIAL_BLOCK_SIZE,
        random_seed=RANDOM_SEED,
    )
    volume_sigma = spatial_volume_sigma(periods)
    regional = summarize_region_period_conversions(
        periods,
        spatial_density_sigma=spatial_density_sigma,
        spatial_volume_sigma=volume_sigma,
        old_rho=OLD_RHO,
        old_sigma_rho=OLD_SIGMA_RHO,
        ice_sigma_rho=ICE_SIGMA_RHO,
    )
    global_summary = summarize_global_period_conversions(regional)
    glacier_changes = summarize_glacier_changes(periods)

    regional.to_csv(OUT_REGIONAL, index=False)
    global_summary.to_csv(OUT_GLOBAL, index=False)
    global_summary.to_csv(OUT_MAIN, index=False)
    glacier_changes.to_csv(OUT_GLACIER_CHANGES, index=False)
    print(f"[done] Wrote {OUT_MAIN}")
    print(f"[done] Wrote {OUT_REGIONAL}")
    print(f"[done] Wrote {OUT_GLOBAL}")
    print(f"[done] Wrote {OUT_GLACIER_CHANGES}")


if __name__ == "__main__":
    main()
