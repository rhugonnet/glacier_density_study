#!/usr/bin/env python3
"""Compute per-glacier and regional agreement between the full model and the surrogate model."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

STUDY_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = STUDY_DIR.parent
for path in [STUDY_DIR, REPO_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from glacier_density_surrogate import RhoSurrogate, haversine_distance_matrix, temporally_reconcile_periods
from study_paths import (
    INPUT_CSV,
    PARAM_PATH,
    SPATIAL_PARAM_PATH,
    TEMPORAL_PARAM_PATH,
    AGREEMENT_DIAGNOSTICS_DIR,
    AGREEMENT_MAIN_PATH,
)


OUT_DIR = AGREEMENT_DIAGNOSTICS_DIR
OUT_MAIN_CSV = AGREEMENT_MAIN_PATH
OUT_DETAIL_CSV = OUT_DIR / "regional_period_agreement_detail.csv"
OUT_SUMMARY_CSV = OUT_DIR / "regional_period_agreement_summary.csv"
OUT_PERIOD_SUMMARY_CSV = OUT_DIR / "regional_period_agreement_by_period_length.csv"
OUT_GLACIER_DETAIL_CSV = OUT_DIR / "glacier_period_agreement_detail.csv"
OUT_GLACIER_SUMMARY_CSV = OUT_DIR / "glacier_period_agreement_summary.csv"
OUT_GLACIER_PERIOD_SUMMARY_CSV = OUT_DIR / "glacier_period_agreement_by_period_length.csv"
OUT_REGIONAL_DH_CATEGORY_SUMMARY_CSV = OUT_DIR / "regional_period_agreement_by_dh_category.csv"
OUT_REGIONAL_PERIOD_CATEGORY_SUMMARY_CSV = OUT_DIR / "regional_period_agreement_by_period_category.csv"
OUT_REGIONAL_DH_PERIOD_CATEGORY_SUMMARY_CSV = OUT_DIR / "regional_period_agreement_by_dh_period_category.csv"
OUT_GLACIER_DH_CATEGORY_SUMMARY_CSV = OUT_DIR / "glacier_period_agreement_by_dh_category.csv"
OUT_GLACIER_PERIOD_CATEGORY_SUMMARY_CSV = OUT_DIR / "glacier_period_agreement_by_period_category.csv"
OUT_GLACIER_DH_PERIOD_CATEGORY_SUMMARY_CSV = OUT_DIR / "glacier_period_agreement_by_dh_period_category.csv"

REFERENCE_VARIANT = "iteration9"
RHO_SENTINELS = {-99999.0, 99999.0}
VARIANT_COL = "rho_variant"
START_YEAR = 2004.0
END_YEAR = 2019.0
GH_ORDER = 64
SPATIAL_BLOCK_SIZE = 700
DH_CATEGORY_EDGES_M = np.array([0.0, 2.0, 10.0, np.inf], dtype=float)
DH_CATEGORY_LABELS = ["low_absdh_lt2m", "mid_absdh_2to10m", "high_absdh_ge10m"]
PERIOD_CATEGORY_EDGES_YR = np.array([0.0, 3.0, 8.0, np.inf], dtype=float)
PERIOD_CATEGORY_LABELS = ["short_1to2yr", "mid_3to7yr", "long_ge8yr"]


def first_existing(columns: pd.Index, candidates: list[str], required: bool = True) -> str | None:
    """Return the first matching column name."""
    for col in candidates:
        if col in columns:
            return col
    if required:
        raise KeyError(f"None of these columns exist: {candidates}")
    return None


def read_full_model_input(path: Path, variants: list[str] | None = None) -> pd.DataFrame:
    """Read the full model effective density calibration sample."""
    header = pd.read_csv(path, nrows=0).columns
    rgi_col = first_existing(header, ["rgiid", "RGIId", "RGIId_float"])
    rho_col = first_existing(header, ["rho"])
    b_col = first_existing(header, ["b"])
    area_col = first_existing(header, ["area"])
    start_col = first_existing(header, ["start_date", "start_year"])
    end_col = first_existing(header, ["end_date", "end_year"])
    optional = [VARIANT_COL, "lat", "lon", "cenlat", "cenlon", "region", "rgi_region", "O1Region", "O2Region"]
    usecols = [rgi_col, rho_col, b_col, area_col, start_col, end_col] + [c for c in optional if c in header]
    df = pd.read_csv(path, usecols=usecols, low_memory=True, memory_map=True)
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

    if VARIANT_COL not in df.columns:
        df[VARIANT_COL] = "single"
    if variants is not None:
        df = df.loc[df[VARIANT_COL].isin(variants)].copy()
    region_col = first_existing(df.columns, ["region", "rgi_region", "O1Region"], required=False)
    if region_col is not None:
        df["rgi_region"] = pd.to_numeric(df[region_col], errors="coerce")

    for col in ["rho", "b", "area", "start_date", "end_date"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df["period_years"] = df["end_date"] - df["start_date"]
    ok = (
        np.isfinite(df["rho"])
        & np.isfinite(df["b"])
        & np.isfinite(df["area"])
        & np.isfinite(df["period_years"])
        & (df["area"] > 0)
        & (df["period_years"] > 0)
        & (~df["rho"].isin(RHO_SENTINELS))
        & (np.abs(df["rho"]) > 1.0e-6)
    )
    df = df.loc[ok].copy()
    df["signed_dh"] = df["period_years"] * df["b"] / df["rho"]
    df["signed_dV_proxy"] = df["area"] * df["b"] / df["rho"]
    df["abs_dV_weight"] = np.abs(df["signed_dV_proxy"])

    nvar = df.groupby(["rgiid", "start_date", "end_date"], observed=True)[VARIANT_COL].transform("nunique")
    df["abs_dV_weight"] = df["abs_dV_weight"] / nvar.clip(lower=1)
    finite = np.isfinite(df["signed_dh"]) & np.isfinite(df["abs_dV_weight"]) & (df["abs_dV_weight"] > 0)
    return df.loc[finite].copy()


def weighted_mean(values: pd.Series | np.ndarray, weights: pd.Series | np.ndarray) -> float:
    """Compute a finite weighted mean."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(ok):
        return np.nan
    return float(np.sum(v[ok] * w[ok]) / np.sum(w[ok]))


def weighted_rmse(values: pd.Series | np.ndarray, weights: pd.Series | np.ndarray) -> float:
    """Compute a finite weighted root-mean-square value."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(ok):
        return np.nan
    return float(np.sqrt(np.sum(w[ok] * v[ok] ** 2) / np.sum(w[ok])))


def add_agreement_categories(table: pd.DataFrame, abs_dh_col: str) -> pd.DataFrame:
    """Attach low/mid/high |dh| and short/mid/long period labels."""
    out = table.copy()
    out["dh_category"] = pd.cut(
        out[abs_dh_col].to_numpy(float),
        bins=DH_CATEGORY_EDGES_M,
        labels=DH_CATEGORY_LABELS,
        include_lowest=True,
        right=False,
    )
    out["period_category"] = pd.cut(
        out["period_years"].to_numpy(float),
        bins=PERIOD_CATEGORY_EDGES_YR,
        labels=PERIOD_CATEGORY_LABELS,
        include_lowest=True,
        right=False,
    )
    return out


def summarize_grouped_agreement(
    table: pd.DataFrame,
    group_cols: list[str],
    *,
    abs_dh_col: str,
    count_col: str,
) -> pd.DataFrame:
    """Summarize agreement metrics for each mode and requested grouping."""
    rows = []
    required = ["mode", *group_cols]
    d = table.dropna(subset=required).copy()
    for key, g in d.groupby(required, sort=True, observed=True):
        if not isinstance(key, tuple):
            key = (key,)
        meta = dict(zip(required, key))
        residual = g["residual_pred_minus_full_kg_m3"].to_numpy(float)
        full = g["full_model_density_kg_m3"].to_numpy(float)
        weight = g["volume_weight"].to_numpy(float)
        full_mean = weighted_mean(full, weight)
        bias = weighted_mean(residual, weight)
        full_variance = weighted_mean((full - full_mean) ** 2, weight)
        residual_mse = weighted_mean(residual**2, weight)
        residual_variance = weighted_mean((residual - bias) ** 2, weight)
        row = {
            **meta,
            "n_regions": int(g["rgi_region"].nunique()),
            "n_periods": int(g[["start_date", "end_date"]].drop_duplicates().shape[0]),
            count_col: int(len(g)),
            "volume_weight_sum": float(np.nansum(weight)),
            "abs_dh_mean_m": weighted_mean(g[abs_dh_col].to_numpy(float), weight),
            "period_years_mean": weighted_mean(g["period_years"].to_numpy(float), weight),
            "weighted_rmse": weighted_rmse(residual, weight),
            "weighted_mae": weighted_mean(np.abs(residual), weight),
            "weighted_bias": bias,
            "weighted_full_model_variance": full_variance,
            "weighted_residual_mse": residual_mse,
            "weighted_residual_variance": residual_variance,
            "weighted_r2": float(1.0 - residual_mse / full_variance) if full_variance > 0 else np.nan,
        }
        if "rgiid" in g.columns:
            row["n_glaciers"] = int(g["rgiid"].nunique())
        if "ci95_intersects_full" in g.columns:
            row["ci95_intersection_fraction"] = float(g["ci95_intersects_full"].mean())
            row["ci95_intersection_weighted_fraction"] = weighted_mean(g["ci95_intersects_full"].astype(float), weight)
        if "ci95_intersects_full_independent" in g.columns:
            row["ci95_intersection_independent_fraction"] = float(g["ci95_intersects_full_independent"].mean())
            row["ci95_intersection_independent_weighted_fraction"] = weighted_mean(g["ci95_intersects_full_independent"].astype(float), weight)
        if "ci95_intersects_full_spatial" in g.columns:
            row["ci95_intersection_spatial_fraction"] = float(g["ci95_intersects_full_spatial"].mean())
            row["ci95_intersection_spatial_weighted_fraction"] = weighted_mean(g["ci95_intersects_full_spatial"].astype(float), weight)
        rows.append(row)
    return pd.DataFrame(rows)


def add_past_change(periods: pd.DataFrame, annual_source: pd.DataFrame, model: RhoSurrogate, start_year: float, end_year: float) -> pd.DataFrame:
    """Attach exponentially weighted past elevation change rate to each period."""
    annual = annual_source.loc[
        np.isclose(annual_source["period_years"].to_numpy(float), 1.0),
        ["rgiid", "start_date", "signed_dh"],
    ].copy()
    annual = annual.sort_values(["rgiid", "start_date"]).drop_duplicates(["rgiid", "start_date"], keep="first")
    annual_wide = annual.pivot(index="rgiid", columns="start_date", values="signed_dh")

    tau_years = float(model.params["memory_tau_years"])
    tau_max = int(round(float(model.params["tau_max"])))
    past_parts = []
    for start in np.arange(int(start_year), int(end_year)):
        available_lags = [lag for lag in range(1, tau_max + 1) if (start - lag) in annual_wide.columns]
        if not available_lags:
            vals = pd.Series(0.0, index=annual_wide.index, name="past_dh")
        else:
            weights = np.exp(-np.asarray(available_lags, dtype=float) / tau_years)
            arr = annual_wide.loc[:, [start - lag for lag in available_lags]].to_numpy(float)
            finite = np.isfinite(arr)
            denom = np.sum(finite * weights[None, :], axis=1)
            vals_arr = np.divide(
                np.nansum(np.where(finite, arr, 0.0) * weights[None, :], axis=1),
                denom,
                out=np.zeros(len(annual_wide), dtype=float),
                where=denom > 0,
            )
            vals = pd.Series(vals_arr, index=annual_wide.index, name="past_dh")
        part = vals.reset_index()
        part["start_date"] = float(start)
        past_parts.append(part)

    past = pd.concat(past_parts, ignore_index=True)
    out = periods.drop(columns=["past_dh", "has_past_dh", "past_start_date"], errors="ignore")
    out = out.merge(past, on=["rgiid", "start_date"], how="left", validate="many_to_one")
    out["past_dh"] = out["past_dh"].fillna(0.0)
    return out


def prepare_periods(df: pd.DataFrame, model: RhoSurrogate, start_year: float, end_year: float, variant: str) -> pd.DataFrame:
    """Prepare full model and independent surrogate estimates for each glacier and period."""
    d = df.loc[df[VARIANT_COL].astype(str).eq(variant)].copy()
    if "rgi_region" not in d.columns:
        d["rgi_region"] = pd.to_numeric(d["rgiid"].astype(str).str.extract(r"RGI60-(\d+)")[0], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(
        subset=[
            "rgiid",
            "rgi_region",
            "start_date",
            "end_date",
            "rho",
            "area",
            "signed_dh",
        ]
    )
    periods = d.loc[(d["start_date"] >= start_year) & (d["end_date"] <= end_year)].copy()
    periods = add_past_change(periods, d, model, start_year, end_year)
    periods["mu_rho_ind_kg_m3"] = model.mu_rho(
        periods["signed_dh"].to_numpy(float),
        dh_p=periods["past_dh"].to_numpy(float),
        dt=periods["period_years"].to_numpy(float),
    )
    periods["sigma_rho_ind_kg_m3"] = model.sigma_rho(
        periods["signed_dh"].to_numpy(float),
        dt=periods["period_years"].to_numpy(float),
    )
    periods["dV_m3"] = periods["area"] * periods["signed_dh"]
    periods["dM_full_kg"] = periods["rho"] * periods["dV_m3"]
    periods["dM_ind_kg"] = periods["mu_rho_ind_kg_m3"] * periods["dV_m3"]
    periods["b_ind_kg"] = (periods["mu_rho_ind_kg_m3"] - model.rho_ice) * periods["dV_m3"]
    periods["sigma_dM_ind_kg"] = periods["sigma_rho_ind_kg_m3"] * np.abs(periods["dV_m3"])
    periods["i0"] = (periods["start_date"] - start_year).astype(int)
    periods["i1"] = (periods["end_date"] - start_year).astype(int)
    periods = periods.loc[
        np.isfinite(periods["mu_rho_ind_kg_m3"])
        & np.isfinite(periods["sigma_dM_ind_kg"])
        & np.isfinite(periods["dV_m3"])
        & (periods["i1"] > periods["i0"])
    ].copy()
    return periods


def add_temporal_closure(periods: pd.DataFrame, model: RhoSurrogate, start_year: float, end_year: float) -> pd.DataFrame:
    """Add reconciled mass change and uncertainty to each glacier and observation period."""
    period_table = (
        periods[["start_date", "end_date", "i0", "i1", "period_years"]]
        .drop_duplicates()
        .sort_values(["start_date", "end_date"])
        .reset_index(drop=True)
    )
    period_table["period_id"] = np.arange(len(period_table))
    n_period = len(period_table)

    periods = periods.merge(
        period_table[["start_date", "end_date", "period_id"]],
        on=["start_date", "end_date"],
        how="left",
        validate="many_to_one",
    )
    counts = periods.groupby("rgiid", observed=True)["period_id"].nunique()
    complete_ids = counts.index[counts.eq(n_period)]
    p = periods.loc[periods["rgiid"].isin(complete_ids)].copy()
    p = p.sort_values(["rgiid", "period_id"]).reset_index(drop=True)
    if p.empty:
        return p

    closed = temporally_reconcile_periods(
        p.rename(columns={"signed_dh": "dh_m"}),
        model,
        start_col="start_date",
        end_col="end_date",
        dh_col="dh_m",
        area_col="area",
        dvol_col="dV_m3",
        mean_col="mu_rho_ind_kg_m3",
        sigma_mass_col="sigma_dM_ind_kg",
    )
    closed["dM_closed_kg"] = closed["dM_surrogate_kg"]
    closed["sigma_dM_closed_kg"] = closed["sigma_dM_rho_surrogate_kg"]
    return closed


def spatial_sigma_by_period(region: pd.DataFrame, model: RhoSurrogate, sigma_col: str) -> dict[int, float]:
    """Propagate uncertainty for all periods of one region with cached distance blocks."""
    if not {"lat", "lon", "period_id"}.issubset(region.columns):
        return {}
    meta = (
        region[["rgiid", "lat", "lon"]]
        .drop_duplicates("rgiid")
        .sort_values("rgiid")
        .reset_index(drop=True)
    )
    lat = pd.to_numeric(meta["lat"], errors="coerce").to_numpy(float)
    lon = pd.to_numeric(meta["lon"], errors="coerce").to_numpy(float)
    ok = np.isfinite(lat) & np.isfinite(lon)
    if not np.all(ok):
        meta = meta.loc[ok].reset_index(drop=True)
        lat = lat[ok]
        lon = lon[ok]
    if len(meta) == 0:
        return {}

    support = (
        region.pivot_table(
            index="rgiid",
            columns="period_id",
            values=sigma_col,
            aggfunc="first",
        )
        .reindex(meta["rgiid"])
        .sort_index(axis=1)
        .fillna(0.0)
    )
    support_values = support.to_numpy(float)
    period_ids = support.columns.to_numpy(int)
    quad = np.zeros(support_values.shape[1], dtype=float)
    for i0 in range(0, len(meta), SPATIAL_BLOCK_SIZE):
        i1 = min(i0 + SPATIAL_BLOCK_SIZE, len(meta))
        dist = haversine_distance_matrix(lat[i0:i1], lon[i0:i1], lat, lon)
        corr = model.spatial_corr(dist)
        rows = np.arange(i0, i1)
        corr[np.arange(i1 - i0), rows] = 1.0
        block_support = support_values[i0:i1, :]
        quad += np.sum(block_support * (corr @ support_values), axis=0)
    return {int(pid): float(np.sqrt(max(var, 0.0))) for pid, var in zip(period_ids, quad)}


def aggregate_region_period(periods: pd.DataFrame, mode: str, model: RhoSurrogate) -> pd.DataFrame:
    """Combine glacier estimates for each RGI region and observation period."""
    if mode == "independent":
        dM_col = "dM_ind_kg"
        sigma_col = "sigma_dM_ind_kg"
    elif mode == "closed":
        dM_col = "dM_closed_kg"
        sigma_col = "sigma_dM_closed_kg"
    else:
        raise ValueError("mode must be 'independent' or 'closed'")

    rows = []
    spatial_mass_sigma: dict[tuple[int, int], float] = {}
    for region, reg in periods.groupby("rgi_region", sort=True):
        for period_id, sigma in spatial_sigma_by_period(reg, model, sigma_col).items():
            spatial_mass_sigma[(int(region), int(period_id))] = sigma

    for (region, start, end), g in periods.groupby(["rgi_region", "start_date", "end_date"], sort=True):
        volume = float(np.nansum(g["dV_m3"]))
        if not np.isfinite(volume) or abs(volume) < 1.0e-12:
            continue
        area = float(np.nansum(g["area"]))
        regional_dh = float(volume / area) if np.isfinite(area) and area > 0 else np.nan
        full = float(np.nansum(g["dM_full_kg"]) / volume)
        surrogate = float(np.nansum(g[dM_col]) / volume)
        sigma_independent = float(np.sqrt(np.nansum(g[sigma_col].to_numpy(float) ** 2)) / abs(volume))
        period_id = int(g["period_id"].iloc[0])
        mass_sigma_spatial = spatial_mass_sigma.get((int(region), period_id), np.nan)
        sigma_spatial = float(mass_sigma_spatial / abs(volume)) if np.isfinite(mass_sigma_spatial) else np.nan
        sigma_for_ci = sigma_spatial if np.isfinite(sigma_spatial) else sigma_independent
        rows.append(
            {
                "mode": mode,
                "rgi_region": int(region),
                "start_date": float(start),
                "end_date": float(end),
                "period_years": float(end - start),
                "n_glaciers": int(g["rgiid"].nunique()),
                "regional_area": area,
                "regional_dh_m": regional_dh,
                "regional_abs_dh_m": abs(regional_dh) if np.isfinite(regional_dh) else np.nan,
                "full_model_density_kg_m3": full,
                "surrogate_density_kg_m3": surrogate,
                "surrogate_sigma_independent_kg_m3": sigma_independent,
                "surrogate_sigma_spatial_kg_m3": sigma_spatial,
                "residual_pred_minus_full_kg_m3": surrogate - full,
                "ci95_intersects_full_independent": bool(abs(surrogate - full) <= 2.0 * sigma_independent),
                "ci95_intersects_full_spatial": bool(abs(surrogate - full) <= 2.0 * sigma_for_ci),
                "volume_weight": abs(volume),
            }
        )
    return pd.DataFrame(rows)


def glacier_period_agreement(periods: pd.DataFrame) -> pd.DataFrame:
    """Build per-glacier agreement rows for independent and closed estimates."""
    dh_col = "signed_dh" if "signed_dh" in periods.columns else "dh_m"
    mode_specs = {
        "independent": ("mu_rho_ind_kg_m3", "sigma_rho_ind_kg_m3"),
        "closed": ("mu_rho_surrogate_kg_m3", "sigma_rho_surrogate_kg_m3"),
    }
    rows = []
    base_cols = [
        "rgiid",
        "rgi_region",
        "start_date",
        "end_date",
        "period_years",
        dh_col,
        "area",
        "dV_m3",
        "rho",
    ]
    missing = [col for col in base_cols if col not in periods.columns]
    if missing:
        raise KeyError(f"Missing columns for glacier-period agreement: {missing}")

    for mode, (mean_col, sigma_col) in mode_specs.items():
        if mean_col not in periods.columns or sigma_col not in periods.columns:
            raise KeyError(f"Missing columns for {mode} glacier-period agreement: {mean_col}, {sigma_col}")
        d = periods.loc[:, base_cols + [mean_col, sigma_col]].copy()
        d = d.rename(
            columns={
                dh_col: "signed_dh_m",
                "area": "area_m2",
                "rho": "full_model_density_kg_m3",
                mean_col: "surrogate_density_kg_m3",
                sigma_col: "surrogate_sigma_kg_m3",
            }
        )
        d["mode"] = mode
        d["abs_dh_m"] = np.abs(d["signed_dh_m"].to_numpy(float))
        d["volume_weight"] = np.abs(d["dV_m3"].to_numpy(float))
        d["residual_pred_minus_full_kg_m3"] = d["surrogate_density_kg_m3"] - d["full_model_density_kg_m3"]
        d["ci95_intersects_full"] = np.abs(d["residual_pred_minus_full_kg_m3"]) <= 2.0 * d["surrogate_sigma_kg_m3"]
        d = d.loc[
            np.isfinite(d["full_model_density_kg_m3"])
            & np.isfinite(d["surrogate_density_kg_m3"])
            & np.isfinite(d["surrogate_sigma_kg_m3"])
            & np.isfinite(d["volume_weight"])
            & (d["volume_weight"] > 0)
        ].copy()
        rows.append(d)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    ordered = [
        "mode",
        "rgiid",
        "rgi_region",
        "start_date",
        "end_date",
        "period_years",
        "signed_dh_m",
        "abs_dh_m",
        "area_m2",
        "dV_m3",
        "volume_weight",
        "full_model_density_kg_m3",
        "surrogate_density_kg_m3",
        "surrogate_sigma_kg_m3",
        "residual_pred_minus_full_kg_m3",
        "ci95_intersects_full",
    ]
    return out.loc[:, ordered]


def summarize_agreement(table: pd.DataFrame, start_year: float, end_year: float) -> pd.DataFrame:
    """Summarize agreement metrics for each surrogate mode."""
    rows = []
    for mode, g in table.groupby("mode", sort=True):
        residual = g["residual_pred_minus_full_kg_m3"].to_numpy(float)
        full = g["full_model_density_kg_m3"].to_numpy(float)
        weight = g["volume_weight"].to_numpy(float)
        full_mean = weighted_mean(full, weight)
        bias = weighted_mean(residual, weight)
        denom = weighted_mean((full - full_mean) ** 2, weight)
        residual_mse = weighted_mean(residual**2, weight)
        residual_variance = weighted_mean((residual - bias) ** 2, weight)
        rows.append(
            {
                "mode": mode,
                "start_year": float(start_year),
                "end_year": float(end_year),
                "n_regions": int(g["rgi_region"].nunique()),
                "n_annual_periods": int(g.loc[np.isclose(g["period_years"], 1.0), ["start_date", "end_date"]].drop_duplicates().shape[0]),
                "n_periods": int(g[["start_date", "end_date"]].drop_duplicates().shape[0]),
                "n_region_periods": int(len(g)),
                "weighted_rmse": weighted_rmse(residual, weight),
                "weighted_mae": weighted_mean(np.abs(residual), weight),
                "weighted_bias": bias,
                "weighted_full_model_variance": denom,
                "weighted_residual_mse": residual_mse,
                "weighted_residual_variance": residual_variance,
                "weighted_r2": float(1.0 - residual_mse / denom) if denom > 0 else np.nan,
                "ci95_intersection_independent_fraction": float(g["ci95_intersects_full_independent"].mean()),
                "ci95_intersection_independent_weighted_fraction": weighted_mean(g["ci95_intersects_full_independent"].astype(float), weight),
                "ci95_intersection_spatial_fraction": float(g["ci95_intersects_full_spatial"].mean()),
                "ci95_intersection_spatial_weighted_fraction": weighted_mean(g["ci95_intersects_full_spatial"].astype(float), weight),
            }
        )
    return pd.DataFrame(rows)


def summarize_agreement_by_period_length(table: pd.DataFrame) -> pd.DataFrame:
    """Summarize agreement metrics for each mode and period length."""
    return summarize_grouped_agreement(
        table,
        ["period_years"],
        abs_dh_col="regional_abs_dh_m",
        count_col="n_region_periods",
    )


def run() -> dict[str, Path]:
    """Run agreement analysis and write detail and summary outputs."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = RhoSurrogate.from_files(
        PARAM_PATH,
        SPATIAL_PARAM_PATH,
        TEMPORAL_PARAM_PATH,
        gh_order_current=GH_ORDER,
        gh_order_past=GH_ORDER,
    )
    df = read_full_model_input(INPUT_CSV, variants=[REFERENCE_VARIANT])
    periods = prepare_periods(df, model, START_YEAR, END_YEAR, REFERENCE_VARIANT)
    closed = add_temporal_closure(periods, model, START_YEAR, END_YEAR)
    detail = pd.concat(
        [
            aggregate_region_period(closed, "independent", model),
            aggregate_region_period(closed, "closed", model),
        ],
        ignore_index=True,
    )
    detail = add_agreement_categories(detail, "regional_abs_dh_m")
    glacier_detail = add_agreement_categories(glacier_period_agreement(closed), "abs_dh_m")
    summary = summarize_agreement(detail, START_YEAR, END_YEAR)
    period_summary = summarize_agreement_by_period_length(detail)
    regional_dh_category_summary = summarize_grouped_agreement(
        detail,
        ["dh_category"],
        abs_dh_col="regional_abs_dh_m",
        count_col="n_region_periods",
    )
    regional_period_category_summary = summarize_grouped_agreement(
        detail,
        ["period_category"],
        abs_dh_col="regional_abs_dh_m",
        count_col="n_region_periods",
    )
    regional_dh_period_category_summary = summarize_grouped_agreement(
        detail,
        ["dh_category", "period_category"],
        abs_dh_col="regional_abs_dh_m",
        count_col="n_region_periods",
    )
    glacier_summary = summarize_grouped_agreement(
        glacier_detail,
        [],
        abs_dh_col="abs_dh_m",
        count_col="n_glacier_periods",
    )
    glacier_period_summary = summarize_grouped_agreement(
        glacier_detail,
        ["period_years"],
        abs_dh_col="abs_dh_m",
        count_col="n_glacier_periods",
    )
    glacier_dh_category_summary = summarize_grouped_agreement(
        glacier_detail,
        ["dh_category"],
        abs_dh_col="abs_dh_m",
        count_col="n_glacier_periods",
    )
    glacier_period_category_summary = summarize_grouped_agreement(
        glacier_detail,
        ["period_category"],
        abs_dh_col="abs_dh_m",
        count_col="n_glacier_periods",
    )
    glacier_dh_period_category_summary = summarize_grouped_agreement(
        glacier_detail,
        ["dh_category", "period_category"],
        abs_dh_col="abs_dh_m",
        count_col="n_glacier_periods",
    )
    detail.to_csv(OUT_DETAIL_CSV, index=False)
    glacier_detail.to_csv(OUT_GLACIER_DETAIL_CSV, index=False)
    summary.to_csv(OUT_SUMMARY_CSV, index=False)
    summary.to_csv(OUT_MAIN_CSV, index=False)
    period_summary.to_csv(OUT_PERIOD_SUMMARY_CSV, index=False)
    regional_dh_category_summary.to_csv(OUT_REGIONAL_DH_CATEGORY_SUMMARY_CSV, index=False)
    regional_period_category_summary.to_csv(OUT_REGIONAL_PERIOD_CATEGORY_SUMMARY_CSV, index=False)
    regional_dh_period_category_summary.to_csv(OUT_REGIONAL_DH_PERIOD_CATEGORY_SUMMARY_CSV, index=False)
    glacier_summary.to_csv(OUT_GLACIER_SUMMARY_CSV, index=False)
    glacier_period_summary.to_csv(OUT_GLACIER_PERIOD_SUMMARY_CSV, index=False)
    glacier_dh_category_summary.to_csv(OUT_GLACIER_DH_CATEGORY_SUMMARY_CSV, index=False)
    glacier_period_category_summary.to_csv(OUT_GLACIER_PERIOD_CATEGORY_SUMMARY_CSV, index=False)
    glacier_dh_period_category_summary.to_csv(OUT_GLACIER_DH_PERIOD_CATEGORY_SUMMARY_CSV, index=False)
    return {
        "main": OUT_MAIN_CSV,
        "detail": OUT_DETAIL_CSV,
        "glacier_detail": OUT_GLACIER_DETAIL_CSV,
        "summary": OUT_SUMMARY_CSV,
        "period_summary": OUT_PERIOD_SUMMARY_CSV,
        "regional_dh_category_summary": OUT_REGIONAL_DH_CATEGORY_SUMMARY_CSV,
        "regional_period_category_summary": OUT_REGIONAL_PERIOD_CATEGORY_SUMMARY_CSV,
        "regional_dh_period_category_summary": OUT_REGIONAL_DH_PERIOD_CATEGORY_SUMMARY_CSV,
        "glacier_summary": OUT_GLACIER_SUMMARY_CSV,
        "glacier_period_summary": OUT_GLACIER_PERIOD_SUMMARY_CSV,
        "glacier_dh_category_summary": OUT_GLACIER_DH_CATEGORY_SUMMARY_CSV,
        "glacier_period_category_summary": OUT_GLACIER_PERIOD_CATEGORY_SUMMARY_CSV,
        "glacier_dh_period_category_summary": OUT_GLACIER_DH_PERIOD_CATEGORY_SUMMARY_CSV,
    }


if __name__ == "__main__":
    for name, path in run().items():
        print(f"[done] {name}: {path}")
