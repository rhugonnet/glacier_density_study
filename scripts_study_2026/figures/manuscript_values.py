#!/usr/bin/env python3
"""Generate manuscript-ready values from full-model outputs and the surrogate."""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STUDY_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = STUDY_DIR.parent
for path in [STUDY_DIR, REPO_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fig_5_temporal_closure import (
    AREA_KM2,
    build_periods,
    make_summary_tables,
    make_synthetic_series,
    temporal_reconcile,
)
from glacier_density_surrogate import RhoSurrogate
from study_paths import (
    INPUT_CSV,
    PARAM_PATH,
    SPATIAL_PARAM_PATH,
    TEMPORAL_PARAM_PATH,
    FIT_SUMMARY_PATH,
    STANDARDIZED_RESIDUALS_PATH,
    AGREEMENT_DIAGNOSTICS_DIR,
    HUGONNET_DIAGNOSTICS_DIR,
    MANUSCRIPT_VALUES_DIAGNOSTICS_DIR,
    MANUSCRIPT_VALUES_MAIN_PATH,
)


# Define project paths
DEFAULT_INPUT_CSV = INPUT_CSV
DEFAULT_OUTDIR = MANUSCRIPT_VALUES_DIAGNOSTICS_DIR
FIT_SUMMARY_CSV = FIT_SUMMARY_PATH
EXACT_RESIDUAL_CSV = STANDARDIZED_RESIDUALS_PATH
AGREEMENT_DIR = AGREEMENT_DIAGNOSTICS_DIR
AGREEMENT_DETAIL_CSV = AGREEMENT_DIR / "regional_period_agreement_detail.csv"
AGREEMENT_SUMMARY_CSV = AGREEMENT_DIR / "regional_period_agreement_summary.csv"
AGREEMENT_PERIOD_SUMMARY_CSV = AGREEMENT_DIR / "regional_period_agreement_by_period_length.csv"
GLACIER_AGREEMENT_SUMMARY_CSV = AGREEMENT_DIR / "glacier_period_agreement_summary.csv"
HUGONNET_APP_DIR = HUGONNET_DIAGNOSTICS_DIR
HUGONNET_GLOBAL_CSV = HUGONNET_APP_DIR / "global_period_conversion_summary.csv"
HUGONNET_REGIONAL_CSV = HUGONNET_APP_DIR / "regional_period_conversion_summary.csv"
HUGONNET_GLACIER_CHANGES_CSV = HUGONNET_APP_DIR / "glacier_period_average_changes.csv"
PARAM_CSV = PARAM_PATH
SPATIAL_PARAM_CSV = SPATIAL_PARAM_PATH
TEMPORAL_PARAM_CSV = TEMPORAL_PARAM_PATH
MANUSCRIPT_TEX: Path | None = None

# Define run controls
ALL_VARIANTS = False
VARIANTS_TO_USE = ["iteration9", "sensmin", "sensmax"]
REFERENCE_VARIANT = "iteration9"
SAMPLE_ROWS = 10000
GH_ORDER = 64

# Define input constants
RHO_SENTINELS = {-99999.0, 99999.0}
VARIANT_COL = "rho_variant"
PARAM_UNITS = {
    "rho_ice": "kg m-3",
    "H_d": "m",
    "P_d": "1",
    "B_h": "kg m-3",
    "B_0": "kg m-3",
    "B_t": "kg m-3",
    "T_t": "yr",
    "P0": "kg m-3",
    "P1": "kg m-3",
    "TP": "yr",
    "B_deltat": "kg m-3",
    "T_deltat": "yr",
    "B_p": "kg m-3 (m yr-1)^-Pp",
    "P_p": "1",
    "B_q": "kg m-3 m^Pq",
    "H_q": "m yr-1",
    "P_q": "1",
    "T_p": "yr",
    "U_h": "kg m-3 m1/2",
    "U_t": "kg m-3 m yr-1/2",
    "alpha_n": "1",
    "beta_n": "1",
    "alpha_s": "1",
    "beta_s": "1",
    "r1": "km",
    "r2": "km",
    "s_t": "1",
    "r_t": "yr",
}


def _first_existing(columns: pd.Index, candidates: list[str], required: bool = True) -> str | None:
    """Return the first matching dataframe column

    :param columns: Available column names
    :param candidates: Candidate column names
    :param required: Raise when no candidate is found
    """
    for col in candidates:
        if col in columns:
            return col
    if required:
        raise KeyError(f"None of these columns exist: {candidates}")
    return None


def read_uncommented_manuscript(path: Path | None) -> str:
    """Read a LaTeX manuscript while ignoring commented paragraphs

    :param path: Optional manuscript path
    """
    if path is None or not path.exists():
        return ""
    keep: list[str] = []
    for line in path.read_text().splitlines():
        if line.lstrip().startswith("%"):
            continue
        keep.append(line)
    return "\n".join(keep)


def read_full_model_input(path: Path, variants: list[str] | None = None) -> pd.DataFrame:
    """Read the full-model effective-density calibration sample

    :param path: Full-model input CSV
    :param variants: Optional sensitivity variants to retain
    """
    header = pd.read_csv(path, nrows=0).columns
    rgi_col = _first_existing(header, ["rgiid", "RGIId", "RGIId_float"])
    rho_col = _first_existing(header, ["rho"])
    b_col = _first_existing(header, ["b"])
    area_col = _first_existing(header, ["area"])
    start_col = _first_existing(header, ["start_date", "start_year"])
    end_col = _first_existing(header, ["end_date", "end_year"])
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

    # Normalize optional columns and variants
    if VARIANT_COL not in df.columns:
        df[VARIANT_COL] = "single"
    if variants is not None:
        df = df.loc[df[VARIANT_COL].isin(variants)].copy()
    region_col = _first_existing(df.columns, ["region", "rgi_region", "O1Region"], required=False)
    if region_col is not None:
        df["rgi_region"] = pd.to_numeric(df[region_col], errors="coerce")

    # Compute common physical predictors
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
    df["mass_proxy"] = df["area"] * df["b"]
    df["abs_dV_weight"] = np.abs(df["signed_dV_proxy"])

    # Avoid over-weighting repeated sensitivity variants
    nvar = df.groupby(["rgiid", "start_date", "end_date"], observed=True)[VARIANT_COL].transform("nunique")
    df["abs_dV_weight"] = df["abs_dV_weight"] / nvar.clip(lower=1)
    df["mass_proxy"] = df["mass_proxy"] / nvar.clip(lower=1)
    df["signed_dV_proxy"] = df["signed_dV_proxy"] / nvar.clip(lower=1)
    finite = np.isfinite(df["signed_dh"]) & np.isfinite(df["abs_dV_weight"]) & (df["abs_dV_weight"] > 0)
    return df.loc[finite].copy()


def attach_exponential_past_change(df: pd.DataFrame, tau_years: float, tau_max: float) -> pd.DataFrame:
    """Attach the final exponential past elevation-change predictor

    :param df: Full-model input rows
    :param tau_years: Exponential memory time scale in years
    :param tau_max: Maximum look-back period in years
    """
    out = df.copy()
    annual = df.loc[
        np.isclose(df["period_years"].to_numpy(float), 1.0),
        ["rgiid", VARIANT_COL, "start_date", "signed_dh", "abs_dV_weight"],
    ].copy()
    if annual.empty:
        out["past_dh"] = 0.0
        out["has_past_dh"] = False
        return out

    # Keep one annual value per glacier, variant and year. The largest support
    # row is retained if duplicated sensitivity or bookkeeping rows exist.
    annual = (
        annual.sort_values("abs_dV_weight", ascending=False)
        .drop_duplicates(["rgiid", VARIANT_COL, "start_date"], keep="first")
    )
    annual_wide = annual.pivot(index=["rgiid", VARIANT_COL], columns="start_date", values="signed_dh")
    col_by_year = {
        int(round(float(col))): col
        for col in annual_wide.columns
        if np.isfinite(float(col)) and abs(float(col) - round(float(col))) < 1.0e-6
    }

    past_parts = []
    starts = np.sort(out["start_date"].dropna().unique())
    for start in starts:
        start_year = int(round(float(start)))
        available_lags = [
            lag for lag in range(1, int(round(tau_max)) + 1)
            if (start_year - lag) in col_by_year
        ]
        if available_lags:
            weights = np.exp(-np.asarray(available_lags, dtype=float) / float(tau_years))
            cols = [col_by_year[start_year - lag] for lag in available_lags]
            arr = annual_wide.loc[:, cols].to_numpy(float)
            finite = np.isfinite(arr)
            denom = np.sum(finite * weights[None, :], axis=1)
            vals = np.divide(
                np.nansum(np.where(finite, arr, 0.0) * weights[None, :], axis=1),
                denom,
                out=np.full(len(annual_wide), np.nan, dtype=float),
                where=denom > 0,
            )
        else:
            vals = np.full(len(annual_wide), np.nan, dtype=float)

        part = pd.DataFrame(
            {
                "rgiid": annual_wide.index.get_level_values("rgiid"),
                VARIANT_COL: annual_wide.index.get_level_values(VARIANT_COL),
                "start_date": float(start),
                "past_dh": vals,
            }
        )
        past_parts.append(part)

    past = pd.concat(past_parts, ignore_index=True)
    out = out.merge(
        past,
        on=["rgiid", VARIANT_COL, "start_date"],
        how="left",
        validate="many_to_one",
    )
    out["has_past_dh"] = np.isfinite(out["past_dh"])
    out["past_dh"] = out["past_dh"].fillna(0.0)
    return out


def weighted_mean(values: pd.Series | np.ndarray, weights: pd.Series | np.ndarray) -> float:
    """Compute a finite weighted mean

    :param values: Values to average
    :param weights: Positive weights
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(ok):
        return np.nan
    return float(np.sum(v[ok] * w[ok]) / np.sum(w[ok]))


def weighted_std(values: pd.Series | np.ndarray, weights: pd.Series | np.ndarray) -> float:
    """Compute a finite weighted standard deviation

    :param values: Values to summarize
    :param weights: Positive weights
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(ok):
        return np.nan
    mean = np.sum(v[ok] * w[ok]) / np.sum(w[ok])
    return float(np.sqrt(np.sum(w[ok] * (v[ok] - mean) ** 2) / np.sum(w[ok])))


def weighted_quantile(values: pd.Series | np.ndarray, weights: pd.Series | np.ndarray, quantile: float) -> float:
    """Compute a finite weighted quantile

    :param values: Values to summarize
    :param weights: Positive weights
    :param quantile: Quantile between 0 and 1
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(ok):
        return np.nan
    order = np.argsort(v[ok])
    v = v[ok][order]
    w = w[ok][order]
    cdf = (np.cumsum(w) - 0.5 * w) / np.sum(w)
    return float(np.interp(float(quantile), cdf, v))


def weighted_quantile_skewness(values: pd.Series | np.ndarray, weights: pd.Series | np.ndarray) -> float:
    """Compute the Bowley weighted quantile skewness used in Fig. S3

    :param values: Values to summarize
    :param weights: Positive weights
    """
    q25 = weighted_quantile(values, weights, 0.25)
    q50 = weighted_quantile(values, weights, 0.50)
    q75 = weighted_quantile(values, weights, 0.75)
    denom = q75 - q25
    if denom <= 0 or not np.isfinite(denom):
        return np.nan
    return float((q75 + q25 - 2.0 * q50) / denom)


def weighted_excess_kurtosis(values: pd.Series | np.ndarray, weights: pd.Series | np.ndarray) -> float:
    """Compute weighted excess kurtosis used in Fig. S3

    :param values: Values to summarize
    :param weights: Positive weights
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(ok):
        return np.nan
    v = v[ok]
    w = w[ok]
    mu = np.average(v, weights=w)
    sd = np.sqrt(np.average((v - mu) ** 2, weights=w))
    if sd <= 0 or not np.isfinite(sd):
        return np.nan
    return float(np.average(((v - mu) / sd) ** 4, weights=w) - 3.0)


def weighted_rmse(values: pd.Series | np.ndarray, weights: pd.Series | np.ndarray) -> float:
    """Compute a finite weighted root-mean-square value

    :param values: Values to summarize
    :param weights: Positive weights
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(ok):
        return np.nan
    return float(np.sqrt(np.sum(w[ok] * v[ok] ** 2) / np.sum(w[ok])))


def aggregate_effective_density(df: pd.DataFrame) -> float:
    """Aggregate effective density by summing mass and volume proxies

    :param df: Rows to aggregate
    """
    denom = float(np.nansum(df["signed_dV_proxy"]))
    if not np.isfinite(denom) or abs(denom) < 1.0e-12:
        return np.nan
    return float(np.nansum(df["mass_proxy"]) / denom)


def past_change_weight_fraction(timescale_years: float, years: int) -> float:
    """Return the discrete exponential past-change weight within a look-back period

    :param timescale_years: Exponential decay time scale
    :param years: Look-back length in years
    """
    q = float(np.exp(-1.0 / timescale_years))
    return float(1.0 - q**int(years))


def temporal_display_parameters() -> tuple[float, float]:
    """Return the positive-lag temporal correlation parameters shown in the figure

    The manuscript equation uses the simple positive-lag exponential correlation
    curve plotted in the main figure.
    """
    params = pd.read_csv(TEMPORAL_PARAM_CSV).iloc[0]
    if "empirical_sill" in params and "empirical_range_yr" in params:
        return float(params["empirical_sill"]), float(params["empirical_range_yr"])
    if "empirical_amplitude_after_nugget" in params and "empirical_timescale_yr" in params:
        return float(params["empirical_amplitude_after_nugget"]), float(params["empirical_timescale_yr"])
    nugget = float(params.get("empirical_nugget", params.get("nugget", 0.70)))
    return 1.0 - nugget, float(params.get("empirical_timescale_yr", 3.0))


def temporal_display_corr(lag_years: float | np.ndarray) -> np.ndarray:
    """Evaluate the displayed positive-lag temporal correlation model

    :param lag_years: Temporal lag in years
    """
    sill, range_years = temporal_display_parameters()
    lag = np.asarray(lag_years, dtype=float)
    corr = sill * np.exp(-3.0 * lag / range_years)
    return np.where(lag == 0, 1.0, corr)


def temporal_closure_summary(model: RhoSurrogate) -> pd.DataFrame:
    """Recompute the synthetic temporal-closure values used in Fig. 5

    :param model: Effective-density surrogate
    """
    annual = make_synthetic_series()
    periods = build_periods(annual, model=model, area_m2=AREA_KM2 * 1e6)
    periods, _ = temporal_reconcile(periods, n_elem=len(annual), model=model)
    summary, _ = make_summary_tables(periods, int(annual["year0"].min()), int(annual["year1"].max()))
    return summary


def add_value(
    rows: list[dict[str, object]],
    key: str,
    value: object,
    formatted: str,
    units: str,
    source: str,
    context: str,
) -> None:
    """Append one manuscript value record

    :param rows: Mutable row list
    :param key: Stable manuscript value key
    :param value: Raw numeric or text value
    :param formatted: Manuscript-ready formatted value
    :param units: Value units
    :param source: Computation source
    :param context: Manuscript sentence or section
    """
    rows.append(
        {
            "key": key,
            "value": value,
            "formatted": formatted,
            "units": units,
            "source": source,
            "manuscript_context": context,
        }
    )


def manuscript_section(key: str, context: str) -> tuple[int, str]:
    """Assign each value to the manuscript section where it is used

    :param key: Stable manuscript value key
    :param context: Manuscript sentence or section
    """
    if key.startswith("abstract_"):
        return 10, "Abstract"
    if key in {"calibration_rows", "calibration_glaciers", "period_combinations", "period_length_max"}:
        return 20, "Data and Methods - calibration sample"
    correlation_parameter_keys = {
        "surrogate_parameter_spatial_corr_form",
        "surrogate_parameter_n_spatial_components",
        "surrogate_parameter_q0_nugget_fraction",
        "surrogate_parameter_q1_range_fraction",
        "surrogate_parameter_q2_range_fraction",
        "surrogate_parameter_q3_range_fraction",
        "surrogate_parameter_r1_km",
        "surrogate_parameter_r2_km",
        "surrogate_parameter_r3_km",
        "surrogate_parameter_s_t",
        "surrogate_parameter_r_t",
    }
    if key.startswith("rgi") or key.startswith("global_effective_density_"):
        return 30, "Results - regionally-specified effective densities"
    if key.startswith("surrogate_parameter_") and key not in correlation_parameter_keys:
        return 40, "Results - mean and uncertainty with observable predictors"
    if key in {"skewness_reduction", "kurtosis_reduction"}:
        return 40, "Results - mean and uncertainty with observable predictors"
    if key in correlation_parameter_keys:
        return 60, "Results - spatial and temporal correlation"
    if key.startswith("spatial_corr_") or key.startswith("temporal_corr_"):
        return 60, "Results - spatial and temporal correlation"
    if (
        key.startswith("integrated_example_")
        or key.startswith("temporal_closure_")
        or key.startswith("regional_period_agreement_")
        or key.startswith("agreement_")
    ):
        return 70, "Results - final surrogate estimates and agreement with full model"
    if key.startswith("past_dh_weight_"):
        return 80, "Discussion - practical application"
    if (
        key.startswith("mu_rho_abs")
        or key.startswith("integrated_sigma")
        or key.startswith("neutral_abs_dh_at_850")
        or key.startswith("sustained_thinning_")
        or key.startswith("short_term_")
        or key.startswith("idealized_population_")
        or key.startswith("spatial_reduction_")
    ):
        return 90, "Discussion - implications for previous assessments"
    if key.startswith("hugonnet_global_") or key.startswith("hugonnet_regional_") or key.startswith("hugonnet_per_glacier_"):
        return 100, "Discussion - revised global-scale mass change estimate"
    if key.startswith("uncommented_xx"):
        return 110, "Manuscript scan"
    return 999, context or "Other"


def order_catalog(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Create a manuscript-ordered value catalog

    :param rows: Raw value records in computation order
    """
    catalog = pd.DataFrame(rows)
    if catalog.empty:
        return catalog
    sections = [manuscript_section(str(row["key"]), str(row["manuscript_context"])) for _, row in catalog.iterrows()]
    catalog.insert(0, "section", [name for _, name in sections])
    catalog["_section_order"] = [order for order, _ in sections]
    catalog["_row_order"] = np.arange(len(catalog))
    catalog = catalog.sort_values(["_section_order", "_row_order"], kind="stable").drop(columns=["_section_order", "_row_order"])
    return catalog.reset_index(drop=True)


def summarize_input(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize the full-model calibration sample

    :param df: Evaluated full-model rows
    """
    periods = df[["start_date", "end_date"]].drop_duplicates()
    rows = [
        ("rows", len(df), f"{len(df):,}"),
        ("glaciers", df["rgiid"].nunique(), f"{df['rgiid'].nunique():,}"),
        ("variants", df[VARIANT_COL].nunique(), ", ".join(map(str, sorted(pd.unique(df[VARIANT_COL]))))),
        ("period_combinations", len(periods), f"{len(periods):,}"),
        ("period_years_min", df["period_years"].min(), f"{df['period_years'].min():.0f}"),
        ("period_years_max", df["period_years"].max(), f"{df['period_years'].max():.0f}"),
        ("signed_dh_min_m", df["signed_dh"].min(), f"{df['signed_dh'].min():.2f}"),
        ("signed_dh_max_m", df["signed_dh"].max(), f"{df['signed_dh'].max():.2f}"),
    ]
    return pd.DataFrame([{"metric": k, "value": v, "text": text} for k, v, text in rows])


def evaluate_rows(df: pd.DataFrame, model: RhoSurrogate) -> pd.DataFrame:
    """Apply the surrogate mean and uncertainty to full-model rows

    :param df: Full-model rows with predictors
    :param model: Effective-density surrogate
    """
    out = df.copy()
    out["mu_rho_kg_m3"] = model.mu_rho(
        out["signed_dh"].to_numpy(float),
        dh_p=out["past_dh"].to_numpy(float),
        dt=out["period_years"].to_numpy(float),
    )
    out["sigma_rho_kg_m3"] = model.sigma_rho(out["signed_dh"].to_numpy(float), dt=out["period_years"].to_numpy(float))
    out["rho_residual_kg_m3"] = out["rho"] - out["mu_rho_kg_m3"]
    out["z_rho"] = out["rho_residual_kg_m3"] / out["sigma_rho_kg_m3"]
    return out


def evaluate_reference_cases(model: RhoSurrogate) -> pd.DataFrame:
    """Evaluate manuscript reference cases for the surrogate

    :param model: Effective-density surrogate
    """
    rows = []
    for dt in [1, 2, 5, 10, 20]:
        for dh in [-50.0, -20.0, -10.0, -5.0, -1.0, -0.25, 0.25, 1.0, 5.0, 10.0, 20.0, 50.0]:
            rows.append(
                {
                    "dh_m": dh,
                    "period_years": dt,
                    "past_dh_m": 0.0,
                    "mu_rho_kg_m3": model.mu_rho(dh, dh_p=0.0, dt=dt).item(),
                    "sigma_rho_kg_m3": model.sigma_rho(dh, dt=dt).item(),
                }
            )
    return pd.DataFrame(rows)


def summarize_model_application(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize surrogate application by period length

    :param df: Evaluated full-model rows
    """
    rows = []
    for period, g in df.groupby("period_years", sort=True):
        rows.append(
            {
                "period_years": float(period),
                "n_rows": int(len(g)),
                "n_glaciers": int(g["rgiid"].nunique()),
                "weighted_mean_full_model_rho": weighted_mean(g["rho"], g["abs_dV_weight"]),
                "weighted_mean_surrogate_mu": weighted_mean(g["mu_rho_kg_m3"], g["abs_dV_weight"]),
                "weighted_mean_surrogate_sigma": weighted_mean(g["sigma_rho_kg_m3"], g["abs_dV_weight"]),
                "weighted_mean_residual": weighted_mean(g["rho_residual_kg_m3"], g["abs_dV_weight"]),
                "past_change_available_fraction": float(g["has_past_dh"].mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_correlations(model: RhoSurrogate) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build spatial and temporal correlation reference tables

    :param model: Effective-density surrogate
    """
    spatial_rows = []
    for distance in [0, 50, 100, 200, 500, 1000, 5000, 10000, 20000]:
        spatial_rows.append(
            {
                "distance_km": distance,
                "spatial_corr": model.spatial_corr(distance).item(),
            }
        )
    temporal_rows = []
    for lag in range(0, 15):
        temporal_rows.append({"lag_years": lag, "temporal_corr": temporal_display_corr(lag).item()})
    return pd.DataFrame(spatial_rows), pd.DataFrame(temporal_rows)


def summarize_region_long_period(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize full-period effective densities by RGI region

    :param df: Evaluated full-model rows
    """
    d = df.copy()
    period = float(d["period_years"].max())
    d = d.loc[np.isclose(d["period_years"], period)].copy()
    if REFERENCE_VARIANT in set(d[VARIANT_COL].astype(str)):
        d = d.loc[d[VARIANT_COL].astype(str).eq(REFERENCE_VARIANT)].copy()
    if "rgi_region" not in d.columns:
        d["rgi_region"] = pd.to_numeric(d["rgiid"].astype(str).str.extract(r"RGI60-(\d+)")[0], errors="coerce")

    # Compute regional mass-over-volume ratios
    rows = []
    for region, g in d.groupby("rgi_region", sort=True):
        rows.append(
            {
                "rgi_region": int(region),
                "period_years": period,
                "n_glaciers": int(g["rgiid"].nunique()),
                "effective_density_signed_kg_m3": aggregate_effective_density(g),
                "effective_density_abs_weighted_kg_m3": weighted_mean(g["rho"], g["abs_dV_weight"]),
                "effective_density_abs_dv_weighted_median_kg_m3": weighted_quantile(g["rho"], g["abs_dV_weight"], 0.50),
                "p05_kg_m3": float(np.nanpercentile(g["rho"], 5)),
                "p95_kg_m3": float(np.nanpercentile(g["rho"], 95)),
            }
        )
    return pd.DataFrame(rows)


def summarize_normality(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize raw, centered, and standardized distribution shapes

    :param df: Evaluated full-model rows
    """
    raw = df["rho"].to_numpy(float)
    centered = df["rho_residual_kg_m3"].to_numpy(float)
    z = df["z_rho"].to_numpy(float)
    if "abs_dV_weight" in df.columns:
        weights = df["abs_dV_weight"].to_numpy(float)
    else:
        weights = np.ones_like(raw, dtype=float)
    ok = np.isfinite(raw) & np.isfinite(centered) & np.isfinite(z) & np.isfinite(weights) & (weights > 0)
    values = {"raw_rho": raw[ok], "mean_removed": centered[ok], "standardized": z[ok]}
    weights = weights[ok]
    rows = []
    for name, arr in values.items():
        rows.append(
            {
                "distribution": name,
                "n": int(arr.size),
                "robust_skewness": weighted_quantile_skewness(arr, weights),
                "excess_kurtosis": weighted_excess_kurtosis(arr, weights),
            }
        )
    out = pd.DataFrame(rows)
    raw_skew = abs(out.loc[out["distribution"].eq("raw_rho"), "robust_skewness"].iloc[0])
    centered_skew = abs(out.loc[out["distribution"].eq("mean_removed"), "robust_skewness"].iloc[0])
    raw_kurt = abs(out.loc[out["distribution"].eq("raw_rho"), "excess_kurtosis"].iloc[0])
    standardized_kurt = abs(out.loc[out["distribution"].eq("standardized"), "excess_kurtosis"].iloc[0])
    out["reduction_percent"] = np.nan
    if raw_skew > 0:
        out.loc[out["distribution"].eq("mean_removed"), "reduction_percent"] = 100.0 * (1.0 - centered_skew / raw_skew)
    if raw_kurt > 0:
        out.loc[out["distribution"].eq("standardized"), "reduction_percent"] = 100.0 * (1.0 - standardized_kurt / raw_kurt)
    return out


def summarize_exact_residual_normality(path: Path) -> pd.DataFrame:
    """Summarize distribution shapes from the exact fit residual output

    :param path: Standardized residual CSV written by the fitting script
    """
    if not path.exists():
        raise FileNotFoundError(path)
    usecols = ["rho", "rho_mean_removed", "z_rho", "abs_dV_weight"]
    d = pd.read_csv(path, usecols=usecols, low_memory=True, memory_map=True)
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    d = d.loc[d["abs_dV_weight"] > 0].copy()
    weights = d["abs_dV_weight"].to_numpy(float)
    values = {
        "raw_rho": d["rho"].to_numpy(float),
        "mean_removed": d["rho_mean_removed"].to_numpy(float),
        "standardized": d["z_rho"].to_numpy(float),
    }
    rows = []
    for name, arr in values.items():
        rows.append(
            {
                "distribution": name,
                "n": int(arr.size),
                "robust_skewness": weighted_quantile_skewness(arr, weights),
                "excess_kurtosis": weighted_excess_kurtosis(arr, weights),
            }
        )
    out = pd.DataFrame(rows)
    raw_skew = abs(out.loc[out["distribution"].eq("raw_rho"), "robust_skewness"].iloc[0])
    centered_skew = abs(out.loc[out["distribution"].eq("mean_removed"), "robust_skewness"].iloc[0])
    raw_kurt = abs(out.loc[out["distribution"].eq("raw_rho"), "excess_kurtosis"].iloc[0])
    standardized_kurt = abs(out.loc[out["distribution"].eq("standardized"), "excess_kurtosis"].iloc[0])
    out["reduction_percent"] = np.nan
    if raw_skew > 0:
        out.loc[out["distribution"].eq("mean_removed"), "reduction_percent"] = 100.0 * (1.0 - centered_skew / raw_skew)
    if raw_kurt > 0:
        out.loc[out["distribution"].eq("standardized"), "reduction_percent"] = 100.0 * (1.0 - standardized_kurt / raw_kurt)
    return out


def read_fit_summary(path: Path) -> pd.Series:
    """Read the retained mean-model fit summary

    :param path: Fit summary CSV written by the fitting script
    """
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path).iloc[0]


def read_surrogate_agreement_outputs(
    detail_path: Path,
    regional_summary_path: Path,
    period_summary_path: Path,
    glacier_summary_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read regional full-model versus surrogate agreement outputs

    :param detail_path: Region-period agreement table from the analysis script
    :param regional_summary_path: Regional agreement summary table from the analysis script
    :param period_summary_path: Agreement summary by period length
    :param glacier_summary_path: Glacier-scale agreement summary table
    """
    missing = [
        path
        for path in [detail_path, regional_summary_path, period_summary_path, glacier_summary_path]
        if not path.exists()
    ]
    if missing:
        msg = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Missing surrogate agreement output. Run "
            "scripts_study_2026/analysis/surrogate_full_model_agreement.py first. "
            f"Missing: {msg}"
        )
    detail = pd.read_csv(detail_path)
    regional_summary = pd.read_csv(regional_summary_path)
    period_summary = pd.read_csv(period_summary_path)
    glacier_summary = pd.read_csv(glacier_summary_path)
    for label, table, path in [
        ("regional", regional_summary, regional_summary_path),
        ("glacier", glacier_summary, glacier_summary_path),
    ]:
        missing_modes = {"independent", "closed"} - set(table["mode"].astype(str))
        if missing_modes:
            raise ValueError(f"Missing {label} agreement mode(s) in {path}: {sorted(missing_modes)}")
    return detail, regional_summary, period_summary, glacier_summary


AGREEMENT_MODE_SLUGS = {
    "independent": "without_temporal_closure",
    "closed": "with_temporal_closure",
}
AGREEMENT_MODE_LABELS = {
    "independent": "without temporal closure",
    "closed": "with temporal closure",
}


def agreement_ci_weighted_column(summary: pd.DataFrame, scale_key: str) -> str:
    """Return the weighted CI-hit column to use for one agreement scale.

    :param summary: Agreement summary table
    :param scale_key: ``glacier_scale`` or ``regional_scale``
    """
    if scale_key == "regional_scale" and "ci95_intersection_spatial_weighted_fraction" in summary.columns:
        return "ci95_intersection_spatial_weighted_fraction"
    if "ci95_intersection_weighted_fraction" in summary.columns:
        return "ci95_intersection_weighted_fraction"
    if "ci95_intersection_independent_weighted_fraction" in summary.columns:
        return "ci95_intersection_independent_weighted_fraction"
    raise KeyError(f"No weighted CI-hit fraction column found for {scale_key}")


def add_agreement_catalog_values(
    rows: list[dict[str, object]],
    summary: pd.DataFrame,
    scale_key: str,
    scale_label: str,
) -> None:
    """Record agreement variance, bias and CI-hit values by closure mode.

    :param rows: Mutable manuscript-value rows
    :param summary: Agreement summary table with independent and closed rows
    :param scale_key: Stable key component for the scale
    :param scale_label: Human-readable scale label
    """
    ci_col = agreement_ci_weighted_column(summary, scale_key)
    for mode in ["independent", "closed"]:
        match = summary.loc[summary["mode"].astype(str).eq(mode)]
        if match.empty:
            continue
        row = match.iloc[0]
        mode_slug = AGREEMENT_MODE_SLUGS[mode]
        mode_label = AGREEMENT_MODE_LABELS[mode]
        prefix = f"agreement_{scale_key}_{mode_slug}"
        context = f"{scale_label} agreement, {mode_label}"
        variance = 100.0 * float(row["weighted_r2"])
        bias = float(row["weighted_bias"])
        ci_hits = 100.0 * float(row[ci_col])
        add_value(
            rows,
            f"{prefix}_variance_explained",
            variance,
            f"{variance:.0f}",
            "%",
            "surrogate_model",
            context,
        )
        add_value(
            rows,
            f"{prefix}_bias_pred_minus_full",
            bias,
            f"{bias:.1f}",
            "kg m-3",
            "surrogate_model",
            context,
        )
        add_value(
            rows,
            f"{prefix}_ci95_hits_weighted",
            ci_hits,
            f"{ci_hits:.0f}",
            "%",
            "surrogate_model",
            context,
        )


def read_hugonnet_application_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the global, regional and glacier-scale Hugonnet application outputs."""
    missing = [path for path in [HUGONNET_GLOBAL_CSV, HUGONNET_REGIONAL_CSV, HUGONNET_GLACIER_CHANGES_CSV] if not path.exists()]
    if missing:
        msg = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Missing Hugonnet application output. Run "
            "scripts_study_2026/analysis/apply_surrogate_hugonnet2021.py first. "
            f"Missing: {msg}"
        )
    return (
        pd.read_csv(HUGONNET_GLOBAL_CSV),
        pd.read_csv(HUGONNET_REGIONAL_CSV),
        pd.read_csv(HUGONNET_GLACIER_CHANGES_CSV),
    )


def summarize_regional_period_surrogate_agreement(
    df: pd.DataFrame,
    model: RhoSurrogate,
    start_year: float = 2004.0,
    end_year: float = 2019.0,
    variant: str = REFERENCE_VARIANT,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compare final surrogate and full-model means after regional aggregation

    The final surrogate is applied glacier by glacier to all contiguous periods
    inside the evaluation window, then temporally reconciled before aggregating
    effective density by RGI region and period. Elevation-change predictors are
    treated as exact values for this comparison, because both the surrogate and
    full-model estimates are derived from the same modelled volume changes.

    :param df: Full-model input rows
    :param model: Effective-density surrogate
    :param start_year: First annual period start year to retain
    :param end_year: Last annual period end year requested
    :param variant: Full-model sensitivity variant to retain
    """
    d = df.loc[df[VARIANT_COL].astype(str).eq(variant)].copy()
    if "rgi_region" not in d.columns:
        d["rgi_region"] = pd.to_numeric(d["rgiid"].astype(str).str.extract(r"RGI60-(\d+)")[0], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=["rgiid", "rgi_region", "start_date", "end_date", "rho", "b", "area", "signed_dh", "signed_dV_proxy", "mass_proxy"])

    # Use annual periods before the evaluation window to compute the same
    # exponentially weighted past-elevation-change predictor as in the surrogate.
    annual = d.loc[np.isclose(d["period_years"].to_numpy(float), 1.0), ["rgiid", "start_date", "signed_dh"]].copy()
    annual = annual.sort_values(["rgiid", "start_date"]).drop_duplicates(["rgiid", "start_date"], keep="first")
    annual_wide = annual.pivot(index="rgiid", columns="start_date", values="signed_dh")
    tau_years = float(model.params.get("T_p", 5.0))
    tau_max = int(round(float(model.params.get("tau_max", 20.0))))
    validation_starts = np.arange(int(start_year), int(end_year))
    past_parts = []
    for start in validation_starts:
        available_lags = [lag for lag in range(1, tau_max + 1) if (start - lag) in annual_wide.columns]
        if not available_lags:
            vals = pd.Series(0.0, index=annual_wide.index, name="past_dh")
        else:
            weights = np.exp(-np.asarray(available_lags, dtype=float) / tau_years)
            cols = [start - lag for lag in available_lags]
            arr = annual_wide.loc[:, cols].to_numpy(float)
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

    periods = d.loc[(d["start_date"] >= start_year) & (d["end_date"] <= end_year)].copy()
    periods = periods.drop(columns=["past_dh", "has_past_dh", "past_start_date"], errors="ignore")
    periods = periods.merge(past, on=["rgiid", "start_date"], how="left", validate="many_to_one")
    periods["past_dh"] = periods["past_dh"].fillna(0.0)

    # Apply the surrogate to all glacier-period rows before temporal closure.
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
    periods["dM_full"] = periods["rho"] * periods["dV_m3"]
    periods["dM_ind_kg"] = periods["mu_rho_ind_kg_m3"] * periods["dV_m3"]
    periods["b_ind_kg"] = (periods["mu_rho_ind_kg_m3"] - model.rho_ice) * periods["dV_m3"]
    periods["sigma_dM_rho_ind_kg"] = periods["sigma_rho_ind_kg_m3"] * np.abs(periods["dV_m3"])
    periods["i0"] = (periods["start_date"] - start_year).astype(int)
    periods["i1"] = (periods["end_date"] - start_year).astype(int)
    periods["year0"] = periods["start_date"].astype(int)
    periods["year1"] = periods["end_date"].astype(int)
    periods["dt_yr"] = periods["period_years"].astype(int)
    periods = periods.loc[
        np.isfinite(periods["mu_rho_ind_kg_m3"])
        & np.isfinite(periods["sigma_dM_rho_ind_kg"])
        & np.isfinite(periods["dV_m3"])
        & (periods["i1"] > periods["i0"])
    ].copy()

    # Reconcile each glacier independently with a batched version of the same
    # weighted least-squares problem used in Fig. 5.
    n_elem = int(end_year - start_year)
    period_table = (
        periods[["start_date", "end_date", "i0", "i1", "period_years"]]
        .drop_duplicates()
        .sort_values(["start_date", "end_date"])
        .reset_index(drop=True)
    )
    period_table["period_id"] = np.arange(len(period_table))
    n_period = len(period_table)
    S = np.zeros((n_period, n_elem), dtype=float)
    for row in period_table.itertuples(index=False):
        S[int(row.period_id), int(row.i0) : int(row.i1)] = 1.0

    periods = periods.merge(period_table[["start_date", "end_date", "period_id"]], on=["start_date", "end_date"], how="left", validate="many_to_one")
    counts = periods.groupby("rgiid", observed=True)["period_id"].nunique()
    complete_ids = counts.index[counts.eq(n_period)]
    p = periods.loc[periods["rgiid"].isin(complete_ids)].copy()
    p = p.sort_values(["rgiid", "period_id"]).reset_index(drop=True)
    if p.empty:
        table = pd.DataFrame()
        return table, {
            "start_year": np.nan,
            "end_year": np.nan,
            "n_regions": 0,
            "n_annual_periods": 0,
            "n_periods": 0,
            "n_region_years": 0,
            "weighted_rmse": np.nan,
            "weighted_mae": np.nan,
            "weighted_bias": np.nan,
            "weighted_r2": np.nan,
        }

    n_glacier = int(len(p) // n_period)
    b = p["b_ind_kg"].to_numpy(float).reshape(n_glacier, n_period)
    sigma = p["sigma_dM_rho_ind_kg"].to_numpy(float).reshape(n_glacier, n_period)
    finite_sigma = sigma[np.isfinite(sigma) & (sigma > 0)]
    sigma_floor = float(np.nanmedian(finite_sigma) * 1.0e-3) if finite_sigma.size else 1.0
    sigma = np.maximum(sigma, sigma_floor)
    sigma[~np.isfinite(sigma)] = sigma_floor
    w2 = 1.0 / sigma**2

    # Solve (S'WS)a = S'Wb for each glacier. A small ridge only stabilizes
    # numerically ill-conditioned near-zero-volume cases.
    lhs = np.einsum("pi,gp,pj->gij", S, w2, S, optimize=True)
    rhs = np.einsum("pi,gp,gp->gi", S, w2, b, optimize=True)
    ridge = 1.0e-12 * np.nanmedian(np.diagonal(lhs, axis1=1, axis2=2), axis=1)
    lhs += ridge[:, None, None] * np.eye(n_elem)[None, :, :]
    elementary_anomaly = np.linalg.solve(lhs, rhs[:, :, None])[:, :, 0]
    closed_b = elementary_anomaly @ S.T
    p["dM_closed_kg"] = model.rho_ice * p["dV_m3"].to_numpy(float) + closed_b.reshape(-1)
    closed_all = p

    rows = []
    for (region, start, end), g in closed_all.groupby(["rgi_region", "start_date", "end_date"], sort=True):
        volume = float(np.nansum(g["dV_m3"]))
        if not np.isfinite(volume) or abs(volume) < 1.0e-12:
            continue
        full = float(np.nansum(g["dM_full"]) / volume)
        surrogate = float(np.nansum(g["dM_closed_kg"]) / volume)
        rows.append(
            {
                "rgi_region": int(region),
                "start_date": float(start),
                "end_date": float(end),
                "period_years": float(end - start),
                "n_glaciers": int(g["rgiid"].nunique()),
                "full_model_density_kg_m3": full,
                "surrogate_density_kg_m3": surrogate,
                "residual_pred_minus_full_kg_m3": surrogate - full,
                "volume_weight": abs(volume),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table, {
            "start_year": np.nan,
            "end_year": np.nan,
            "n_regions": 0,
            "n_annual_periods": 0,
            "n_periods": 0,
            "n_region_years": 0,
            "weighted_rmse": np.nan,
            "weighted_mae": np.nan,
            "weighted_bias": np.nan,
            "weighted_r2": np.nan,
        }

    residual = table["residual_pred_minus_full_kg_m3"].to_numpy(float)
    full = table["full_model_density_kg_m3"].to_numpy(float)
    weight = table["volume_weight"].to_numpy(float)
    full_mean = weighted_mean(full, weight)
    denom = weighted_mean((full - full_mean) ** 2, weight)
    summary = {
        "start_year": float(table["start_date"].min()),
        "end_year": float(table["end_date"].max()),
        "n_regions": int(table["rgi_region"].nunique()),
        "n_annual_periods": int(table.loc[np.isclose(table["period_years"], 1.0), ["start_date", "end_date"]].drop_duplicates().shape[0]),
        "n_periods": int(table[["start_date", "end_date"]].drop_duplicates().shape[0]),
        "n_region_years": int(len(table)),
        "weighted_rmse": weighted_rmse(residual, weight),
        "weighted_mae": weighted_mean(np.abs(residual), weight),
        "weighted_bias": weighted_mean(residual, weight),
        "weighted_r2": float(1.0 - weighted_mean(residual**2, weight) / denom) if denom > 0 else np.nan,
    }
    return table, summary


def final_implications_reference_values(model: RhoSurrogate) -> dict[str, float]:
    """Compute reference values for the final implications paragraph

    :param model: Effective-density surrogate
    """
    from scipy.optimize import brentq

    out: dict[str, float] = {}
    for dt in [5.0, 10.0, 20.0]:
        f = lambda x: float(model.mu_rho(-x, dh_p=0.0, dt=dt)) - 850.0
        out[f"neutral_abs_dh_at_850_dt{dt:g}"] = float(brentq(f, 0.01, 200.0))
    out["sustained_thinning_mu_dh_minus10_dhp_minus0p5_dt20"] = float(model.mu_rho(-10.0, dh_p=-0.5, dt=20.0))
    out["sustained_thinning_mu_dh_minus50_dhp_minus0p5_dt20"] = float(model.mu_rho(-50.0, dh_p=-0.5, dt=20.0))
    out["short_term_mu_dh_minus2_dhp_minus0p5_dt1"] = float(model.mu_rho(-2.0, dh_p=-0.5, dt=1.0))
    out["short_term_mu_dh_plus2_dhp_minus0p5_dt1"] = float(model.mu_rho(2.0, dh_p=-0.5, dt=1.0))
    return out


def idealized_population_uncertainty_values(model: RhoSurrogate) -> dict[str, float]:
    """Compute reproducible uncertainty metrics for the idealized glacier population

    :param model: Effective-density surrogate
    """
    from scipy.special import erf

    rng = np.random.default_rng(42)
    rates = rng.normal(-0.5, 1.0, 5_000_000)
    dt = 20.0
    dh = rates * dt
    sigma_dh = 1.0
    uh = float(model.params["U_h"])
    ut = float(model.params["U_t"])

    # Use the analytical variance-additive integrated uncertainty form.
    eabs = (
        sigma_dh * np.sqrt(2.0 / np.pi) * np.exp(-(dh**2) / (2.0 * sigma_dh**2))
        + dh * erf(dh / (np.sqrt(2.0) * sigma_dh))
    )
    sigma_rho = np.sqrt(uh**2 * eabs + ut**2 * dt) / np.maximum(np.abs(dh), 1.0e-12)
    ok = np.isfinite(sigma_rho) & np.isfinite(dh)
    dh = dh[ok]
    sigma_rho = sigma_rho[ok]
    support_ok = np.abs(dh) > 0.25
    return {
        "mean_individual_sigma": float(np.mean(sigma_rho)),
        "median_individual_sigma": float(np.median(sigma_rho)),
        "volume_weighted_mean_sigma": float(np.sum(np.abs(dh) * sigma_rho) / np.sum(np.abs(dh))),
        "regional_uncorrelated_equivalent_sigma": float(np.sqrt(np.mean((dh * sigma_rho) ** 2)) / abs(np.mean(dh))),
        "mean_individual_sigma_absdh_gt_0p25": float(np.mean(sigma_rho[support_ok])),
        "volume_weighted_mean_sigma_absdh_gt_0p25": float(
            np.sum(np.abs(dh[support_ok]) * sigma_rho[support_ok]) / np.sum(np.abs(dh[support_ok]))
        ),
        "fraction_absdh_gt_0p25": float(np.mean(support_ok)),
    }


def haversine_distance_matrix(lat_a: np.ndarray, lon_a: np.ndarray, lat_b: np.ndarray, lon_b: np.ndarray) -> np.ndarray:
    """Compute pairwise great-circle distances in kilometres

    :param lat_a: First latitude array
    :param lon_a: First longitude array
    :param lat_b: Second latitude array
    :param lon_b: Second longitude array
    """
    radius_km = 6371.0
    lat1 = np.deg2rad(lat_a)[:, None]
    lon1 = np.deg2rad(lon_a)[:, None]
    lat2 = np.deg2rad(lat_b)[None, :]
    lon2 = np.deg2rad(lon_b)[None, :]
    dlat = lat1 - lat2
    dlon = lon1 - lon2
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * radius_km * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def spatial_reduction_examples(df: pd.DataFrame, model: RhoSurrogate, block_size: int = 500) -> pd.DataFrame:
    """Compute regional uncertainty reductions relative to full correlation

    :param df: Full-model input rows
    :param model: Effective-density surrogate
    :param block_size: Pairwise covariance block size
    """
    cases = [
        ("European Alps", [11], 20.0, 1999.0, 2019.0),
        ("European Alps", [11], 5.0, 2014.0, 2019.0),
        ("European Alps", [11], 1.0, 2018.0, 2019.0),
        ("Andes", [16, 17], 20.0, 1999.0, 2019.0),
        ("Andes", [16, 17], 5.0, 2014.0, 2019.0),
        ("Andes", [16, 17], 1.0, 2018.0, 2019.0),
    ]
    rows = []
    d = df.loc[df[VARIANT_COL].astype(str).eq(REFERENCE_VARIANT)].copy()
    for name, regions, dt, start, end in cases:
        g = d.loc[d["rgi_region"].isin(regions) & np.isclose(d["start_date"], start) & np.isclose(d["end_date"], end)].copy()
        if g.empty:
            continue
        dh = dt * g["b"].to_numpy(float) / g["rho"].to_numpy(float)
        dvol = g["area"].to_numpy(float) * g["b"].to_numpy(float) / g["rho"].to_numpy(float)
        sigma = model.sigma_rho(dh, dt=dt)
        support = dvol * sigma
        denom = abs(float(np.sum(dvol)))
        full_corr_sigma = float(np.sum(np.abs(support)) / denom)
        lat = g["lat"].to_numpy(float)
        lon = g["lon"].to_numpy(float)
        quad = 0.0
        for i0 in range(0, len(g), block_size):
            i1 = min(i0 + block_size, len(g))
            dist = haversine_distance_matrix(lat[i0:i1], lon[i0:i1], lat, lon)
            corr = model.spatial_corr(dist, period_years=dt)
            for local_i, global_i in enumerate(range(i0, i1)):
                corr[local_i, global_i] = 1.0
            quad += float(np.sum(support[i0:i1, None] * corr * support[None, :]))
        spatial_sigma = float(np.sqrt(max(quad, 0.0)) / denom)
        rows.append(
            {
                "region_name": name,
                "rgi_regions": ",".join(map(str, regions)),
                "period_years": dt,
                "start_date": start,
                "end_date": end,
                "n_glaciers": int(len(g)),
                "fully_correlated_sigma_kg_m3": full_corr_sigma,
                "spatially_correlated_sigma_kg_m3": spatial_sigma,
                "reduction_percent": 100.0 * (1.0 - spatial_sigma / full_corr_sigma),
            }
        )
    return pd.DataFrame(rows)


def build_manuscript_catalog(
    df: pd.DataFrame,
    model: RhoSurrogate,
    input_summary: pd.DataFrame,
    region_summary: pd.DataFrame,
    normality: pd.DataFrame,
    regional_period_agreement_modes: pd.DataFrame,
    glacier_period_agreement_modes: pd.DataFrame,
    final_implications: dict[str, float],
    idealized_population: dict[str, float],
    spatial_reductions: pd.DataFrame,
    hugonnet_global: pd.DataFrame,
    hugonnet_regional: pd.DataFrame,
    hugonnet_glacier_changes: pd.DataFrame,
    manuscript_text: str,
) -> pd.DataFrame:
    """Build manuscript-keyed value records

    :param df: Evaluated full-model rows
    :param model: Effective-density surrogate
    :param input_summary: Input summary table
    :param region_summary: Full-period regional table
    :param normality: Normality diagnostics
    :param regional_period_agreement_modes: Independent and closed agreement diagnostics
    :param glacier_period_agreement_modes: Glacier-scale independent and closed agreement diagnostics
    :param final_implications: Final implications paragraph values
    :param idealized_population: Idealized population uncertainty values
    :param spatial_reductions: Regional spatial propagation reduction table
    :param hugonnet_global: Global Hugonnet 2021 application summary
    :param hugonnet_regional: Regional Hugonnet 2021 application summary
    :param hugonnet_glacier_changes: Glacier-scale change summary
    :param manuscript_text: Uncommented manuscript text
    """
    rows: list[dict[str, object]] = []
    lookup = {row.metric: row for row in input_summary.itertuples(index=False)}
    period_max = float(df["period_years"].max())
    long = df.loc[np.isclose(df["period_years"], period_max)].copy()
    if REFERENCE_VARIANT in set(long[VARIANT_COL].astype(str)):
        long_ref = long.loc[long[VARIANT_COL].astype(str).eq(REFERENCE_VARIANT)].copy()
    else:
        long_ref = long

    # Record calibration-sample values
    add_value(rows, "calibration_rows", lookup["rows"].value, lookup["rows"].text, "rows", "full_model_input", "Calibration sample")
    add_value(rows, "calibration_glaciers", lookup["glaciers"].value, lookup["glaciers"].text, "glaciers", "full_model_input", "Calibration sample")
    add_value(rows, "period_combinations", lookup["period_combinations"].value, lookup["period_combinations"].text, "periods", "full_model_input", "Calibration sample")
    add_value(rows, "period_length_max", period_max, f"{period_max:.0f}", "yr", "full_model_input", "1999--2019 period length")

    # Record global and regional full-period values
    global_signed = aggregate_effective_density(long_ref)
    global_q25 = weighted_quantile(long_ref["rho"], long_ref["abs_dV_weight"], 0.25)
    global_q75 = weighted_quantile(long_ref["rho"], long_ref["abs_dV_weight"], 0.75)
    add_value(rows, "global_effective_density_signed_full_period", global_signed, f"{global_signed:.0f}", "kg m-3", "full_model_input", "Full-model calibration-sample global signed-volume effective density between 1999--2019")
    add_value(rows, "global_effective_density_iqr_p25_full_period", global_q25, f"{global_q25:.0f}", "kg m-3", "full_model_input", "Global interquartile range over 1999--2019")
    add_value(rows, "global_effective_density_iqr_p75_full_period", global_q75, f"{global_q75:.0f}", "kg m-3", "full_model_input", "Global interquartile range over 1999--2019")
    for region in [18, 19]:
        reg = region_summary.loc[region_summary["rgi_region"].eq(region)]
        if reg.empty:
            continue
        row = reg.iloc[0]
        add_value(
            rows,
            f"rgi{region:02d}_effective_density_abs_dv_weighted_median_full_period",
            row["effective_density_abs_dv_weighted_median_kg_m3"],
            f"{row['effective_density_abs_dv_weighted_median_kg_m3']:.0f}",
            "kg m-3",
            "full_model_input",
            f"Regional full-period volume-change-weighted median effective density for RGI {region:02d}",
        )

    # Record fitted surrogate parameters
    for key in ["rho_ice", "H_d", "P_d"]:
        value = float(model.params[key])
        add_value(rows, f"surrogate_parameter_{key}", value, f"{value:.3g}", PARAM_UNITS.get(key, ""), "surrogate_parameters", "Retained surrogate model parameter")
    if str(model.params.get("period_form")) == "power_param":
        value = float(model.params["B_h"]) + float(model.params["P0"])
        add_value(rows, "surrogate_parameter_B_0", value, f"{value:.3g}", PARAM_UNITS["B_0"], "surrogate_parameters", "Retained surrogate model background parameter")
        value = float(model.params["P1"])
        add_value(rows, "surrogate_parameter_B_deltat", value, f"{value:.3g}", PARAM_UNITS["B_deltat"], "surrogate_parameters", "Retained surrogate model period-length parameter")
        value = float(model.params["TP"])
        add_value(rows, "surrogate_parameter_T_deltat", value, f"{value:.3g}", PARAM_UNITS["T_deltat"], "surrogate_parameters", "Retained surrogate model period-length parameter")
    else:
        for key in ["B_h", "B_t", "T_t"]:
            value = float(model.params[key])
            add_value(rows, f"surrogate_parameter_{key}", value, f"{value:.3g}", PARAM_UNITS.get(key, ""), "surrogate_parameters", "Retained surrogate model parameter")
    for key in ["B_p", "P_p", "B_q", "H_q", "P_q", "T_p", "U_h", "U_t"]:
        value = float(model.params[key])
        add_value(rows, f"surrogate_parameter_{key}", value, f"{value:.3g}", PARAM_UNITS.get(key, ""), "surrogate_parameters", "Retained surrogate model parameter")
    add_value(rows, "surrogate_parameter_spatial_corr_form", model.params["spatial_corr_form"], str(model.params["spatial_corr_form"]), "1", "correlation_model", "Retained spatial correlation parameter")
    for key, unit in [
        ("n_spatial_components", "1"),
        ("q0_nugget_fraction", "1"),
        ("q1_range_fraction", "1"),
        ("q2_range_fraction", "1"),
        ("q3_range_fraction", "1"),
        ("r1_km", "km"),
        ("r2_km", "km"),
        ("r3_km", "km"),
    ]:
        if key not in model.params:
            continue
        value = float(model.params[key])
        if not np.isfinite(value):
            continue
        add_value(rows, f"surrogate_parameter_{key}", value, f"{value:.3g}", unit, "correlation_model", "Retained spatial correlation parameter")
    temporal_sill, temporal_range = temporal_display_parameters()
    add_value(rows, "surrogate_parameter_s_t", temporal_sill, f"{temporal_sill:.2g}", "1", "correlation_model", "Displayed temporal correlation parameter")
    add_value(rows, "surrogate_parameter_r_t", temporal_range, f"{temporal_range:.2g}", "yr", "correlation_model", "Displayed temporal correlation parameter")

    # Record abstract examples for a twenty-year period and neutral past change
    for abs_dh in [2, 20]:
        mu = model.mu_rho(float(abs_dh), dh_p=0.0, dt=20.0).item()
        sigma = model.sigma_rho(float(abs_dh), dt=20.0).item()
        add_value(rows, f"abstract_mu_rho_dt20_abs_dh_{abs_dh:g}m", mu, f"{mu:.0f}", "kg m-3", "surrogate_model", "Abstract twenty-year example with no past elevation-change rate")
        add_value(rows, f"abstract_sigma_rho_dt20_abs_dh_{abs_dh:g}m", sigma, f"{sigma:.0f}", "kg m-3", "surrogate_model", "Abstract twenty-year example with no past elevation-change rate")

    # Record integrated examples mentioned in the final-surrogate Results section
    for sigma_dh in [0.1, 3.0]:
        mu = model.integrated_mu(dh=0.5, sigma_dh=sigma_dh, dh_p=0.0, sigma_dh_p=0.0, dt=1.0)
        sigma = model.integrated_sigma(dh=0.5, sigma_dh=sigma_dh, dt=1.0)
        label = str(sigma_dh).replace(".", "p")
        add_value(
            rows,
            f"integrated_example_mu_dh0p5_sigdh{label}_dt1",
            mu,
            f"{mu:.0f}",
            "kg m-3",
            "surrogate_model",
            "Final surrogate integrated example with dh=0.5 m, past dh=0 m and dt=1 yr",
        )
        add_value(
            rows,
            f"integrated_example_sigma_dh0p5_sigdh{label}_dt1",
            sigma,
            f"{sigma:.0f}",
            "kg m-3",
            "surrogate_model",
            "Final surrogate integrated example with dh=0.5 m, past dh=0 m and dt=1 yr",
        )

    # Record uncertainty examples with uncertain elevation change
    for dt in [20, 5, 1]:
        dh = -0.5 * float(dt)
        value = model.integrated_sigma(dh=dh, sigma_dh=1.0, dt=float(dt))
        add_value(rows, f"integrated_sigma_dh_rate_0p5_dt{dt}", value, f"{value:.0f}", "kg m-3", "surrogate_model", "Discussion uncertainty examples for -0.5 m yr-1 thinning and 1 m elevation-change uncertainty")

    # Record final implications paragraph reference values
    for dt in [5, 10, 20]:
        value = final_implications[f"neutral_abs_dh_at_850_dt{dt:g}"]
        add_value(rows, f"neutral_abs_dh_at_850_dt{dt:g}", value, f"{value:.0f}", "m", "surrogate_model", "Neutral past-change elevation change where mean is 850 kg m-3")
    for key, label in [
        ("sustained_thinning_mu_dh_minus10_dhp_minus0p5_dt20", "20-year thinning example with dh=-10 m and past dh=-0.5 m"),
        ("sustained_thinning_mu_dh_minus50_dhp_minus0p5_dt20", "Large cumulative thinning example with dh=-50 m and past dh=-0.5 m"),
        ("short_term_mu_dh_minus2_dhp_minus0p5_dt1", "Annual departure example with dh=-2 m and past dh=-0.5 m"),
        ("short_term_mu_dh_plus2_dhp_minus0p5_dt1", "Annual departure example with dh=+2 m and past dh=-0.5 m"),
    ]:
        value = final_implications[key]
        add_value(rows, key, value, f"{value:.0f}", "kg m-3", "surrogate_model", label)

    # Record only the idealized population values used in the discussion.
    # The regional value is the volume-change-weighted value, not the
    # independent-error equivalent.
    for catalog_key, source_key, label in [
        ("mean_individual_sigma", "mean_individual_sigma", "Mean individual-glacier density uncertainty in idealized population"),
        (
            "regional_volume_weighted_sigma",
            "volume_weighted_mean_sigma",
            "Regional volume-change-weighted density uncertainty in idealized population",
        ),
    ]:
        value = idealized_population[source_key]
        add_value(rows, f"idealized_population_{catalog_key}", value, f"{value:.0f}", "kg m-3", "surrogate_model", label)

    # Record spatial propagation reductions for Alps and Andes
    for row in spatial_reductions.itertuples(index=False):
        slug = str(row.region_name).lower().replace(" ", "_")
        dt = int(row.period_years)
        add_value(
            rows,
            f"spatial_reduction_{slug}_dt{dt}",
            row.reduction_percent,
            f"{row.reduction_percent:.0f}",
            "%",
            "correlation_model",
            f"Uncertainty reduction for {row.region_name} relative to full spatial correlation",
        )

    # Record temporal correlation examples
    for lag in [1, 2, 5]:
        value = temporal_display_corr(lag).item()
        add_value(rows, f"temporal_corr_lag{lag}", value, f"{100 * value:.0f}", "%", "correlation_model", "Temporal error-correlation examples")

    # Record agreement metrics by scale and temporal-closure mode. The
    # manuscript catalog keeps only weighted variance explained, bias and
    # CI-hit fractions; stratified agreement diagnostics are rendered by the
    # separate LaTeX table script.
    add_agreement_catalog_values(
        rows,
        glacier_period_agreement_modes,
        "glacier_scale",
        "Glacier-scale",
    )
    add_agreement_catalog_values(
        rows,
        regional_period_agreement_modes,
        "regional_scale",
        "Regional-scale",
    )

    # Record Hugonnet 2021 application values
    h20 = hugonnet_global.loc[hugonnet_global["start_year"].eq(2000) & hugonnet_global["end_year"].eq(2020)]
    if not h20.empty:
        h20 = h20.iloc[0]
        for key, col, units, fmt, label in [
            ("hugonnet_global_old_mass_rate_2000_2020", "old_mass_rate_gt_yr", "Gt yr-1", "{:.1f}", "Global mass-change rate with 850 kg m-3"),
            ("hugonnet_global_surrogate_mass_rate_2000_2020", "surrogate_mass_rate_gt_yr", "Gt yr-1", "{:.1f}", "Global mass-change rate with the surrogate"),
            ("hugonnet_global_mass_rate_difference_2000_2020", "mass_rate_difference_gt_yr", "Gt yr-1", "{:.1f}", "Global mass-change-rate difference"),
            ("hugonnet_global_surrogate_rho_2000_2020", "surrogate_rho_mean_kg_m3", "kg m-3", "{:.0f}", "Global surrogate effective density applied to the external Hugonnet/WW-TVOL 2000--2020 volume-change product"),
            ("hugonnet_global_old_total_sigma_rate_2000_2020", "old_sigma_mass_rate_gt_yr", "Gt yr-1", "{:.1f}", "Global total uncertainty with 850 kg m-3"),
            ("hugonnet_global_surrogate_total_sigma_rate_2000_2020", "surrogate_sigma_mass_rate_gt_yr", "Gt yr-1", "{:.1f}", "Global total uncertainty with the surrogate"),
            ("hugonnet_global_surrogate_total_with_ice_sigma_rate_2000_2020", "surrogate_sigma_mass_with_ice_rate_gt_yr", "Gt yr-1", "{:.1f}", "Global total uncertainty with surrogate and ice-density floor"),
            ("hugonnet_global_old_density_sigma_rate_2000_2020", "old_density_only_sigma_rate_gt_yr", "Gt yr-1", "{:.1f}", "Global density-only uncertainty with 850 kg m-3"),
            ("hugonnet_global_surrogate_density_sigma_rate_2000_2020", "surrogate_density_only_sigma_rate_gt_yr", "Gt yr-1", "{:.1f}", "Global surrogate density-only uncertainty"),
            ("hugonnet_global_ice_density_sigma_rate_2000_2020", "ice_density_sigma_rate_gt_yr", "Gt yr-1", "{:.2f}", "Global external ice-density uncertainty"),
            ("hugonnet_global_surrogate_density_plus_ice_sigma_rate_2000_2020", "surrogate_density_plus_ice_sigma_rate_gt_yr", "Gt yr-1", "{:.1f}", "Global surrogate density plus ice-density uncertainty"),
        ]:
            value = float(h20[col])
            add_value(rows, key, value, fmt.format(value), units, "hugonnet_application", label)

        # Record abstract-facing aliases for the revised Hugonnet 2021
        # application sentence. These are duplicated from the detailed
        # application rows so that abstract numbers can be checked directly.
        old_rate = float(h20["old_mass_rate_gt_yr"])
        surrogate_rate = float(h20["surrogate_mass_rate_gt_yr"])
        mass_loss_increase = 100.0 * (abs(surrogate_rate) - abs(old_rate)) / abs(old_rate)
        old_density_sigma = float(h20["old_density_only_sigma_rate_gt_yr"])
        surrogate_density_sigma = float(h20["surrogate_density_only_sigma_rate_gt_yr"])
        density_uncertainty_reduction = 100.0 * (old_density_sigma - surrogate_density_sigma) / old_density_sigma
        add_value(rows, "hugonnet_global_fixed_effective_density_reference", 850.0, "850", "kg m-3", "hugonnet_application", "Discussion comparison with previously used conversion")
        add_value(rows, "hugonnet_global_fixed_effective_density_reference_uncertainty", 60.0, "60", "kg m-3", "hugonnet_application", "Discussion comparison with previously used conversion")
        add_value(rows, "hugonnet_global_mass_loss_increase_percent_2000_2020", mass_loss_increase, f"{mass_loss_increase:.1f}", "%", "hugonnet_application", "Global mass-loss increase from surrogate relative to 850 kg m-3")
        add_value(rows, "hugonnet_global_density_uncertainty_reduction_percent_2000_2020", density_uncertainty_reduction, f"{density_uncertainty_reduction:.0f}", "%", "hugonnet_application", "Reduction in effective-density uncertainty relative to 60 kg m-3")
        add_value(rows, "abstract_fixed_effective_density_reference", 850.0, "850", "kg m-3", "hugonnet_application", "Abstract comparison with previously used conversion")
        add_value(rows, "abstract_fixed_effective_density_reference_uncertainty", 60.0, "60", "kg m-3", "hugonnet_application", "Abstract comparison with previously used conversion")
        add_value(rows, "abstract_hugonnet_mass_loss_increase_percent", mass_loss_increase, f"{mass_loss_increase:.1f}", "%", "hugonnet_application", "Abstract global mass-loss increase from surrogate relative to 850 kg m-3")
        add_value(rows, "abstract_hugonnet_density_uncertainty_reduction_percent", density_uncertainty_reduction, f"{density_uncertainty_reduction:.0f}", "%", "hugonnet_application", "Abstract reduction in effective-density uncertainty relative to 60 kg m-3")
        add_value(rows, "abstract_hugonnet_surrogate_rho_2000_2020", float(h20["surrogate_rho_mean_kg_m3"]), f"{float(h20['surrogate_rho_mean_kg_m3']):.0f}", "kg m-3", "hugonnet_application", "Abstract global effective density applied to the external Hugonnet/WW-TVOL 2000--2020 volume-change product")
        add_value(rows, "abstract_hugonnet_surrogate_density_sigma_rho_equiv_2000_2020", float(h20["surrogate_density_only_sigma_rho_equiv_kg_m3"]), f"{float(h20['surrogate_density_only_sigma_rho_equiv_kg_m3']):.0f}", "kg m-3", "hugonnet_application", "Abstract equivalent effective-density uncertainty for the external Hugonnet/WW-TVOL 2000--2020 product")
        volume_rate = math.sqrt(max(0.0, float(h20["surrogate_sigma_mass_with_ice_rate_gt_yr"]) ** 2 - float(h20["surrogate_density_plus_ice_sigma_rate_gt_yr"]) ** 2))
        volume_share = 100.0 * volume_rate**2 / float(h20["surrogate_sigma_mass_with_ice_rate_gt_yr"]) ** 2
        density_share = 100.0 - volume_share
        add_value(rows, "hugonnet_global_surrogate_volume_sigma_rate_2000_2020", volume_rate, f"{volume_rate:.1f}", "Gt yr-1", "hugonnet_application", "Global volume-change contribution to surrogate total uncertainty")
        add_value(rows, "hugonnet_global_surrogate_volume_uncertainty_variance_share_2000_2020", volume_share, f"{volume_share:.0f}", "%", "hugonnet_application", "Global variance share from volume-change uncertainty")
        add_value(rows, "hugonnet_global_surrogate_density_uncertainty_variance_share_2000_2020", density_share, f"{density_share:.0f}", "%", "hugonnet_application", "Global variance share from density plus ice-density uncertainty")
        if "old_loss_acceleration_gt_yr" in h20.index:
            add_value(rows, "hugonnet_global_old_loss_acceleration", float(h20["old_loss_acceleration_gt_yr"]), f"{float(h20['old_loss_acceleration_gt_yr']):.1f}", "Gt yr-1 decade-1", "hugonnet_application", "Global loss acceleration between 2000--2010 and 2010--2020")
            add_value(rows, "hugonnet_global_surrogate_loss_acceleration", float(h20["surrogate_loss_acceleration_gt_yr"]), f"{float(h20['surrogate_loss_acceleration_gt_yr']):.1f}", "Gt yr-1 decade-1", "hugonnet_application", "Global loss acceleration between 2000--2010 and 2010--2020")

    reg20 = hugonnet_regional.loc[hugonnet_regional["start_year"].eq(2000) & hugonnet_regional["end_year"].eq(2020)].copy()
    if not reg20.empty:
        rho = reg20["surrogate_rho_mean_kg_m3"].astype(float)
        add_value(rows, "hugonnet_global_n_glaciers_2000_2020", int(reg20["n_glaciers"].sum()), f"{int(reg20['n_glaciers'].sum()):,}", "glaciers", "hugonnet_application", "Number of glaciers in the 2000--2020 Hugonnet application")
        add_value(rows, "hugonnet_regional_surrogate_rho_min_2000_2020", float(rho.min()), f"{rho.min():.0f}", "kg m-3", "hugonnet_application", "Minimum regional surrogate effective density")
        add_value(rows, "hugonnet_regional_surrogate_rho_max_2000_2020", float(rho.max()), f"{rho.max():.0f}", "kg m-3", "hugonnet_application", "Maximum regional surrogate effective density")

        old_loss = reg20["old_mass_rate_gt_yr"].abs().astype(float)
        surrogate_loss = reg20["surrogate_mass_rate_gt_yr"].abs().astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_change = 100.0 * (surrogate_loss - old_loss) / old_loss
        rel_change = rel_change.replace([np.inf, -np.inf], np.nan).dropna()
        if not rel_change.empty:
            add_value(rows, "hugonnet_regional_mass_loss_increase_percent_min_2000_2020", float(rel_change.min()), f"{rel_change.min():.1f}", "%", "hugonnet_application", "Minimum regional mass-loss increase from surrogate relative to 850 kg m-3")
            add_value(rows, "hugonnet_regional_mass_loss_increase_percent_max_2000_2020", float(rel_change.max()), f"{rel_change.max():.1f}", "%", "hugonnet_application", "Maximum regional mass-loss increase from surrogate relative to 850 kg m-3")

        region_names = {
            "01+02": "Alaska and Western North America",
            "03": "Arctic Canada North",
            "04": "Arctic Canada South",
            "05": "Greenland Periphery",
            "06": "Iceland",
            "07": "Svalbard and Jan Mayen",
            "08": "Scandinavia",
            "09": "Russian Arctic",
            "10": "North Asia",
            "11": "Central Europe",
            "12": "Caucasus and Middle East",
            "13": "Central Asia",
            "14": "South Asia West",
            "15": "South Asia East",
            "16": "Low Latitudes",
            "17": "Southern Andes",
            "18": "New Zealand",
            "19": "Antarctic and Subantarctic",
        }
        ranked = reg20.assign(abs_mass_rate_difference_gt_yr=reg20["mass_rate_difference_gt_yr"].abs()).sort_values(
            "abs_mass_rate_difference_gt_yr", ascending=False
        )
        for rank, row in enumerate(ranked.head(4).itertuples(index=False), start=1):
            region_label = str(row.region_label)
            region_name = region_names.get(region_label, region_label)
            add_value(rows, f"hugonnet_regional_adjustment_rank{rank}_region_2000_2020", region_name, region_name, "", "hugonnet_application", "Region with one of the largest absolute mass-rate adjustments")
            add_value(
                rows,
                f"hugonnet_regional_adjustment_rank{rank}_mass_rate_difference_2000_2020",
                float(row.mass_rate_difference_gt_yr),
                f"{row.mass_rate_difference_gt_yr:.1f}",
                "Gt yr-1",
                "hugonnet_application",
                "Largest absolute regional mass-rate adjustment from surrogate relative to 850 kg m-3",
            )

    for row in hugonnet_glacier_changes.itertuples(index=False):
        dt = int(row.period_years)
        for suffix, value, units, fmt, label in [
            (
                "volume_weighted_mu_change",
                row.volume_weighted_mu_change_kg_m3,
                "kg m-3",
                "{:.0f}",
                "Volume-change-weighted per-glacier change in effective density relative to 850 kg m-3",
            ),
            (
                "volume_weighted_sigma_rho",
                60.0 + row.volume_weighted_sigma_rho_change_kg_m3,
                "kg m-3",
                "{:.0f}",
                "Volume-change-weighted per-glacier surrogate density uncertainty",
            ),
        ]:
            add_value(rows, f"hugonnet_per_glacier_dt{dt}_{suffix}", float(value), fmt.format(float(value)), units, "hugonnet_application", label)

    # Record past-change memory and temporal-closure example values
    for years in [5, 10]:
        value = 100.0 * past_change_weight_fraction(float(model.params["T_p"]), years)
        add_value(rows, f"past_dh_weight_{years}yr", value, f"{value:.0f}", "%", "surrogate_parameters", "Past elevation-change predictor memory")
    closure = temporal_closure_summary(model)
    closure_lookup = {row.estimate: row for row in closure.itertuples(index=False)}
    direct_full = closure_lookup["direct full-period surrogate"]
    annual_sum = closure_lookup["sum of independent annual surrogate estimates"]
    closed_full = closure_lookup["temporally reconciled full period"]
    mismatch_percent = 100.0 * (annual_sum.dM_Gt - direct_full.dM_Gt) / abs(direct_full.dM_Gt)
    closed_diff_percent = 100.0 * (closed_full.dM_Gt - direct_full.dM_Gt) / abs(direct_full.dM_Gt)
    add_value(rows, "temporal_closure_independent_annual_mismatch", mismatch_percent, f"{abs(mismatch_percent):.0f}", "%", "surrogate_model", "Synthetic temporal-closure example")
    add_value(rows, "temporal_closure_closed_full_difference", closed_diff_percent, f"{abs(closed_diff_percent):.0f}", "%", "surrogate_model", "Synthetic temporal-closure example")
    add_value(rows, "temporal_closure_annual_equiv_density", annual_sum.mu_or_equiv_rho_kg_m3, f"{annual_sum.mu_or_equiv_rho_kg_m3:.0f}", "kg m-3", "surrogate_model", "Synthetic temporal-closure example")
    add_value(rows, "temporal_closure_direct_full_density", direct_full.mu_or_equiv_rho_kg_m3, f"{direct_full.mu_or_equiv_rho_kg_m3:.0f}", "kg m-3", "surrogate_model", "Synthetic temporal-closure example")
    add_value(rows, "temporal_closure_closed_full_density", closed_full.mu_or_equiv_rho_kg_m3, f"{closed_full.mu_or_equiv_rho_kg_m3:.0f}", "kg m-3", "surrogate_model", "Synthetic temporal-closure example")

    # Skewness reduction is measured after removing the mean only.  Kurtosis
    # reduction is measured after the subsequent scaling by sigma.
    skew_row = normality.loc[normality["distribution"].eq("mean_removed")]
    if not skew_row.empty:
        value = float(skew_row["reduction_percent"].iloc[0])
        add_value(rows, "skewness_reduction", value, f"{value:.0f}", "%", "surrogate_residuals", "Supplementary normality check")

    kurt_row = normality.loc[normality["distribution"].eq("standardized")]
    if not kurt_row.empty:
        value = float(kurt_row["reduction_percent"].iloc[0])
        add_value(rows, "kurtosis_reduction", value, f"{value:.0f}", "%", "surrogate_residuals", "Supplementary normality check")

    # Record uncommented manuscript placeholder count when a manuscript file is supplied
    if manuscript_text:
        xx_count = len(re.findall(r"\bXX\b", manuscript_text))
        add_value(rows, "uncommented_xx_placeholders", xx_count, f"{xx_count}", "placeholders", "manuscript_tex", "Only uncommented LaTeX text is scanned")
    return order_catalog(rows)


def write_markdown(
    out_path: Path,
    input_summary: pd.DataFrame,
    period_summary: pd.DataFrame,
    region_summary: pd.DataFrame,
    catalog: pd.DataFrame,
) -> None:
    """Write a compact Markdown report

    :param out_path: Markdown output path
    :param input_summary: Input summary table
    :param period_summary: Period summary table
    :param region_summary: Regional summary table
    :param catalog: Manuscript value catalog
    """
    def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
        frame = df.head(max_rows).copy() if max_rows is not None else df.copy()
        for col in frame.columns:
            if pd.api.types.is_float_dtype(frame[col]):
                frame[col] = frame[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
            else:
                frame[col] = frame[col].map(lambda x: "" if pd.isna(x) else str(x))
        header = "| " + " | ".join(map(str, frame.columns)) + " |"
        sep = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
        body = ["| " + " | ".join(row) + " |" for row in frame.to_numpy(dtype=str)]
        return "\n".join([header, sep] + body)

    # Write report sections
    with out_path.open("w") as f:
        f.write("# Manuscript Values\n\n")
        f.write("## Value Catalog\n\n")
        f.write(markdown_table(catalog[["section", "key", "formatted", "units", "source", "manuscript_context"]]))
        f.write("\n\n## Input Data\n\n")
        f.write(markdown_table(input_summary))
        f.write("\n\n## Period Summary\n\n")
        f.write(markdown_table(period_summary, max_rows=25))
        f.write("\n\n## Regional Full-Period Summary\n\n")
        f.write(markdown_table(region_summary))
        f.write("\n")


def run() -> dict[str, Path]:
    """Generate all manuscript value tables

    :returns: Mapping of output labels to written paths
    """
    outdir = DEFAULT_OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)
    variants = None if ALL_VARIANTS else VARIANTS_TO_USE

    # Load surrogate parameters and full-model input
    model = RhoSurrogate.from_files(
        parameter_path=PARAM_CSV,
        spatial_path=SPATIAL_PARAM_CSV,
        temporal_path=TEMPORAL_PARAM_CSV,
        gh_order_current=GH_ORDER,
        gh_order_past=GH_ORDER,
    )
    df = read_full_model_input(DEFAULT_INPUT_CSV, variants=variants)
    df = attach_exponential_past_change(
        df,
        tau_years=float(model.params.get("T_p", 5.0)),
        tau_max=float(model.params.get("tau_max", 20.0)),
    )
    evaluated = evaluate_rows(df, model)

    # Build manuscript-ready tables
    manuscript_text = read_uncommented_manuscript(MANUSCRIPT_TEX)
    input_summary = summarize_input(evaluated)
    period_summary = summarize_model_application(evaluated)
    reference_cases = evaluate_reference_cases(model)
    spatial_summary, temporal_summary = summarize_correlations(model)
    region_summary = summarize_region_long_period(evaluated)
    normality = summarize_exact_residual_normality(EXACT_RESIDUAL_CSV)
    regional_period_agreement, regional_period_agreement_modes, regional_period_agreement_by_period, glacier_period_agreement_modes = read_surrogate_agreement_outputs(
        AGREEMENT_DETAIL_CSV,
        AGREEMENT_SUMMARY_CSV,
        AGREEMENT_PERIOD_SUMMARY_CSV,
        GLACIER_AGREEMENT_SUMMARY_CSV,
    )
    hugonnet_global, hugonnet_regional, hugonnet_glacier_changes = read_hugonnet_application_outputs()
    final_implications = final_implications_reference_values(model)
    idealized_population = idealized_population_uncertainty_values(model)
    spatial_reductions = spatial_reduction_examples(df, model)
    catalog = build_manuscript_catalog(
        evaluated,
        model,
        input_summary,
        region_summary,
        normality,
        regional_period_agreement_modes,
        glacier_period_agreement_modes,
        final_implications,
        idealized_population,
        spatial_reductions,
        hugonnet_global,
        hugonnet_regional,
        hugonnet_glacier_changes,
        manuscript_text,
    )

    # Define row-level sample output
    sample_cols = [
        "rgiid",
        VARIANT_COL,
        "start_date",
        "end_date",
        "period_years",
        "signed_dh",
        "past_dh",
        "rho",
        "mu_rho_kg_m3",
        "sigma_rho_kg_m3",
        "rho_residual_kg_m3",
        "z_rho",
    ]
    paths = {
        "catalog": MANUSCRIPT_VALUES_MAIN_PATH,
        "catalog_diagnostic": outdir / "manuscript_value_catalog.csv",
        "input_summary": outdir / "input_summary.csv",
        "period_summary": outdir / "period_summary.csv",
        "reference_cases": outdir / "reference_cases.csv",
        "spatial_correlation": outdir / "spatial_correlation_reference.csv",
        "temporal_correlation": outdir / "temporal_correlation_reference.csv",
        "regional_full_period": outdir / "regional_full_period_effective_density.csv",
        "regional_period_agreement": outdir / "regional_period_surrogate_agreement.csv",
        "regional_period_agreement_summary": outdir / "regional_period_surrogate_agreement_summary.csv",
        "regional_period_agreement_by_period": outdir / "regional_period_surrogate_agreement_by_period_length.csv",
        "glacier_period_agreement_summary": outdir / "glacier_period_surrogate_agreement_summary.csv",
        "spatial_reduction_examples": outdir / "spatial_reduction_examples.csv",
        "normality": outdir / "normality_summary.csv",
        "row_sample": outdir / "evaluated_row_sample.csv",
        "markdown": outdir / "manuscript_values.md",
    }

    # Write all outputs
    catalog.to_csv(paths["catalog"], index=False)
    catalog.to_csv(paths["catalog_diagnostic"], index=False)
    input_summary.to_csv(paths["input_summary"], index=False)
    period_summary.to_csv(paths["period_summary"], index=False)
    reference_cases.to_csv(paths["reference_cases"], index=False)
    spatial_summary.to_csv(paths["spatial_correlation"], index=False)
    temporal_summary.to_csv(paths["temporal_correlation"], index=False)
    region_summary.to_csv(paths["regional_full_period"], index=False)
    regional_period_agreement.to_csv(paths["regional_period_agreement"], index=False)
    regional_period_agreement_modes.to_csv(paths["regional_period_agreement_summary"], index=False)
    regional_period_agreement_by_period.to_csv(paths["regional_period_agreement_by_period"], index=False)
    glacier_period_agreement_modes.to_csv(paths["glacier_period_agreement_summary"], index=False)
    spatial_reductions.to_csv(paths["spatial_reduction_examples"], index=False)
    normality.to_csv(paths["normality"], index=False)
    evaluated.loc[:, [c for c in sample_cols if c in evaluated.columns]].head(SAMPLE_ROWS).to_csv(paths["row_sample"], index=False)
    write_markdown(paths["markdown"], input_summary, period_summary, region_summary, catalog)
    return paths


if __name__ == "__main__":
    output_paths = run()
    print("[done] Wrote manuscript values:")
    for name, path in output_paths.items():
        print(f"  {name}: {path}")
