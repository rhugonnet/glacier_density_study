#!/usr/bin/env python3
"""Main figure 5: temporal closure and annual agreement for one RGI region."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import numpy as np
import pandas as pd

STUDY_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = STUDY_DIR.parent
for path in [STUDY_DIR, REPO_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from glacier_density_surrogate import RhoSurrogate, haversine_distance_matrix, temporally_reconcile_periods
from study_paths import FIGURES_DIR, INPUT_CSV, PARAM_PATH

OUT_PNG = FIGURES_DIR / "FIG_main_05_temporal_closure.pdf"
PARAM_CSV = PARAM_PATH

REFERENCE_VARIANT = "iteration9"
REGION = 10
START_YEAR = 1999
END_YEAR = 2019
DISPLAY_START_YEAR = 2000
PAST_ELEVATION_END_YEAR = 2005
AREA_KM2 = 100.0
SEED = 4
INTERANNUAL_SD = 0.25
UNCERTAINTY_FRACTION = 0.25
DENSITY_REF = 900.0
DENSITY_AXIS_SCALE = 400.0
DENSITY_TICKS = np.array(
    [0.0, 300.0, 600.0, 900.0, 1200.0, 2000.0, 4000.0, 8000.0, 12000.0],
    dtype=float,
)
DENSITY_YLIM = (-500.0, 13500.0)
RHO_SENTINELS = {-99999.0, 99999.0}
SPATIAL_BLOCK_SIZE = 700
REGION_NAMES = {
    3: "Arctic Canada North (RGI 03)",
    4: "Arctic Canada South (RGI 04)",
    7: "Svalbard and Jan Mayen (RGI 07)",
    10: "North Asia (RGI 10)",
    14: "South Asia West (RGI 14)",
}

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10,
    "legend.title_fontsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _first_existing(columns: pd.Index, candidates: list[str]) -> str:
    """Return the first matching dataframe column."""
    for col in candidates:
        if col in columns:
            return col
    raise KeyError(f"None of these columns exist: {candidates}")


def _rho_to_axis(value: np.ndarray | float) -> np.ndarray | float:
    """Transform density to a display coordinate centered on ice density."""
    return np.arcsinh((np.asarray(value, dtype=float) - DENSITY_REF) / DENSITY_AXIS_SCALE)


def _axis_to_rho(value: np.ndarray | float) -> np.ndarray | float:
    """Transform display coordinate back to density."""
    return DENSITY_REF + DENSITY_AXIS_SCALE * np.sinh(np.asarray(value, dtype=float))


def read_full_model_table(region: int = REGION) -> pd.DataFrame:
    """Read reference full model rows for one RGI region."""
    header = pd.read_csv(INPUT_CSV, nrows=0).columns
    rgi_col = _first_existing(header, ["rgiid", "RGIId", "RGIId_float"])
    rho_col = _first_existing(header, ["rho"])
    b_col = _first_existing(header, ["b"])
    area_col = _first_existing(header, ["area"])
    start_col = _first_existing(header, ["start_date", "start_year"])
    end_col = _first_existing(header, ["end_date", "end_year"])
    region_col = _first_existing(header, ["region", "rgi_region", "O1Region"])
    lat_col = next((col for col in ["lat", "cenlat"] if col in header), None)
    lon_col = next((col for col in ["lon", "cenlon"] if col in header), None)
    optional = [c for c in ["rho_variant"] if c in header]
    if lat_col is not None:
        optional.append(lat_col)
    if lon_col is not None:
        optional.append(lon_col)
    usecols = [rgi_col, rho_col, b_col, area_col, start_col, end_col, region_col, *optional]
    rename_cols = {
        rgi_col: "rgiid",
        rho_col: "rho",
        b_col: "b",
        area_col: "area_km2",
        start_col: "start_date",
        end_col: "end_date",
        region_col: "rgi_region",
    }
    if lat_col is not None:
        rename_cols[lat_col] = "lat"
    if lon_col is not None:
        rename_cols[lon_col] = "lon"
    df = pd.read_csv(INPUT_CSV, usecols=usecols, low_memory=True, memory_map=True)
    df = df.rename(columns=rename_cols)
    if "rho_variant" in df.columns:
        df = df.loc[df["rho_variant"].astype(str).eq(REFERENCE_VARIANT)].copy()
    for col in ["rho", "b", "area_km2", "start_date", "end_date", "rgi_region"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["lat", "lon"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.loc[df["rgi_region"].eq(region)].copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.loc[
        np.isfinite(df["rho"])
        & np.isfinite(df["b"])
        & np.isfinite(df["area_km2"])
        & (df["area_km2"] > 0)
        & (~df["rho"].isin(RHO_SENTINELS))
    ].copy()
    df["period_years"] = df["end_date"] - df["start_date"]
    df["area_m2"] = df["area_km2"] * 1.0e6
    df["signed_dh"] = df["period_years"] * df["b"] / df["rho"]
    df["dV_m3"] = df["area_m2"] * df["signed_dh"]
    df["dM_full_kg"] = df["rho"] * df["dV_m3"]
    return df.loc[np.isfinite(df["signed_dh"]) & np.isfinite(df["dV_m3"])].copy()


def complete_region_subset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep glaciers with a complete annual and nested-period record."""
    annual = df.loc[
        np.isclose(df["period_years"], 1.0)
        & (df["start_date"] >= START_YEAR)
        & (df["end_date"] <= END_YEAR)
    ].sort_values(["rgiid", "start_date"]).drop_duplicates(["rgiid", "start_date"])
    required_annual = END_YEAR - START_YEAR
    complete_annual_ids = annual.groupby("rgiid", observed=True).size()
    complete_annual_ids = complete_annual_ids.index[complete_annual_ids.eq(required_annual)]

    periods = df.loc[
        df["rgiid"].isin(complete_annual_ids)
        & (df["start_date"] >= START_YEAR)
        & (df["end_date"] <= END_YEAR)
        & (df["period_years"] > 0)
    ].copy()
    period_table = periods[["start_date", "end_date"]].drop_duplicates()
    n_period_expected = required_annual * (required_annual + 1) // 2
    complete_period_ids = periods.groupby("rgiid", observed=True).size()
    complete_period_ids = complete_period_ids.index[complete_period_ids.eq(n_period_expected)]
    periods = periods.loc[periods["rgiid"].isin(complete_period_ids)].copy()
    annual = annual.loc[annual["rgiid"].isin(complete_period_ids)].copy()
    if len(period_table) < n_period_expected:
        raise ValueError("Full-model input does not contain the complete nested-period set.")
    if periods.empty:
        raise ValueError(f"No complete RGI {region:02d} glacier records found for Fig. 5.")
    return annual, periods


def past_elevation_change_from_wide(annual_wide: pd.DataFrame, start: int, model: RhoSurrogate) -> pd.Series:
    """Compute the exponentially weighted past elevation change rate."""
    tau_years = float(model.params["memory_tau_years"])
    tau_max = int(round(float(model.params["tau_max"])))
    lags = [lag for lag in range(1, tau_max + 1) if (start - lag) in annual_wide.columns]
    if not lags:
        return pd.Series(0.0, index=annual_wide.index)
    weights = np.exp(-np.asarray(lags, dtype=float) / tau_years)
    values = annual_wide[[start - lag for lag in lags]].to_numpy(float)
    finite = np.isfinite(values)
    denom = np.sum(finite * weights[None, :], axis=1)
    out = np.divide(
        np.nansum(np.where(finite, values, 0.0) * weights[None, :], axis=1),
        denom,
        out=np.zeros(len(annual_wide), dtype=float),
        where=denom > 0,
    )
    return pd.Series(out, index=annual_wide.index)


def apply_surrogate_and_closure(annual: pd.DataFrame, periods: pd.DataFrame, model: RhoSurrogate) -> pd.DataFrame:
    """Apply direct and temporally reconciled surrogate estimates."""
    annual_wide = annual.pivot(index="rgiid", columns="start_date", values="signed_dh")
    past_parts = []
    for start in range(START_YEAR, END_YEAR):
        part = past_elevation_change_from_wide(annual_wide, start, model).rename("dh_p_m").reset_index()
        part["start_date"] = float(start)
        past_parts.append(part)
    past = pd.concat(past_parts, ignore_index=True)
    out = periods.merge(past, on=["rgiid", "start_date"], how="left", validate="many_to_one")
    out["dh_p_m"] = out["dh_p_m"].fillna(0.0)
    out["mu_rho_ind_kg_m3"] = model.mu_rho(
        out["signed_dh"].to_numpy(float),
        dh_p=out["dh_p_m"].to_numpy(float),
        dt=out["period_years"].to_numpy(float),
    )
    out["sigma_rho_ind_kg_m3"] = model.sigma_rho(out["signed_dh"].to_numpy(float), dt=out["period_years"].to_numpy(float))
    out["dM_ind_kg"] = out["mu_rho_ind_kg_m3"] * out["dV_m3"]
    out["b_ind_kg"] = (out["mu_rho_ind_kg_m3"] - model.rho_ice) * out["dV_m3"]
    out["sigma_dM_ind_kg"] = out["sigma_rho_ind_kg_m3"] * np.abs(out["dV_m3"])

    out = temporally_reconcile_periods(
        out,
        model,
        id_col="rgiid",
        start_col="start_date",
        end_col="end_date",
        dh_col="signed_dh",
        area_col="area_m2",
        dvol_col="dV_m3",
        mean_col="mu_rho_ind_kg_m3",
        sigma_mass_col="sigma_dM_ind_kg",
    )
    out["dM_closed_kg"] = out["dM_surrogate_kg"]
    out["mu_rho_closed_kg_m3"] = out["mu_rho_surrogate_kg_m3"]
    return out


def annual_spatial_mass_sigma(periods: pd.DataFrame, model: RhoSurrogate, sigma_col: str) -> dict[float, float]:
    """Propagate spatially correlated mass change uncertainty for annual regional estimates."""
    required = {"rgiid", "lat", "lon", "start_date", sigma_col}
    if not required.issubset(periods.columns):
        return {}
    meta = (
        periods[["rgiid", "lat", "lon"]]
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
        periods.pivot_table(index="rgiid", columns="start_date", values=sigma_col, aggfunc="first")
        .reindex(meta["rgiid"])
        .sort_index(axis=1)
        .fillna(0.0)
    )
    support_values = support.to_numpy(float)
    starts = support.columns.to_numpy(float)
    quad = np.zeros(support_values.shape[1], dtype=float)
    for i0 in range(0, len(meta), SPATIAL_BLOCK_SIZE):
        i1 = min(i0 + SPATIAL_BLOCK_SIZE, len(meta))
        dist = haversine_distance_matrix(lat[i0:i1], lon[i0:i1], lat, lon)
        corr = model.spatial_corr(dist)
        rows = np.arange(i0, i1)
        corr[np.arange(i1 - i0), rows] = 1.0
        block_support = support_values[i0:i1, :]
        quad += np.sum(block_support * (corr @ support_values), axis=0)
    return {float(start): float(np.sqrt(max(var, 0.0))) for start, var in zip(starts, quad)}


def regional_annual_summary(periods: pd.DataFrame, model: RhoSurrogate | None = None) -> pd.DataFrame:
    """Aggregate annual full model and surrogate estimates over the plotted RGI region."""
    annual = periods.loc[np.isclose(periods["period_years"], 1.0)].copy()
    spatial_sigma = annual_spatial_mass_sigma(annual, model, "sigma_dM_rho_surrogate_kg") if model is not None else {}
    rows = []
    for start, g in annual.groupby("start_date", sort=True):
        dV = float(np.nansum(g["dV_m3"]))
        area = float(np.nansum(g.drop_duplicates("rgiid")["area_m2"]))
        closed_sigma_mass_independent = float(np.sqrt(np.nansum(g["sigma_dM_rho_surrogate_kg"].to_numpy(float) ** 2)))
        closed_sigma_mass_spatial = spatial_sigma.get(float(start), np.nan)
        closed_sigma_mass = (
            closed_sigma_mass_spatial
            if np.isfinite(closed_sigma_mass_spatial)
            else closed_sigma_mass_independent
        )
        rows.append(
            {
                "year0": int(start),
                "year1": int(start + 1),
                "dh_m": dV / area if area > 0 else np.nan,
                "volume_weight_m3": abs(dV),
                "full_rho_kg_m3": float(np.nansum(g["dM_full_kg"]) / dV) if dV != 0 else np.nan,
                "direct_rho_kg_m3": float(np.nansum(g["dM_ind_kg"]) / dV) if dV != 0 else np.nan,
                "closed_rho_kg_m3": float(np.nansum(g["dM_closed_kg"]) / dV) if dV != 0 else np.nan,
                "closed_sigma_rho_kg_m3": closed_sigma_mass / abs(dV) if dV != 0 else np.nan,
                "closed_sigma_independent_rho_kg_m3": closed_sigma_mass_independent / abs(dV) if dV != 0 else np.nan,
                "closed_sigma_spatial_rho_kg_m3": closed_sigma_mass_spatial / abs(dV) if dV != 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def regional_period_effective_density(periods: pd.DataFrame, start_year: int, end_year: int) -> pd.Series:
    """Aggregate full-period effective density over the plotted RGI region."""
    period = periods.loc[
        np.isclose(periods["start_date"], float(start_year))
        & np.isclose(periods["end_date"], float(end_year))
    ].copy()
    if period.empty:
        raise KeyError(f"No period records found for {start_year}-{end_year}.")
    dV = float(np.nansum(period["dV_m3"]))
    if not np.isfinite(dV) or dV == 0:
        raise ValueError(f"Cannot compute effective density for {start_year}-{end_year}: zero or invalid dV.")
    full = float(np.nansum(period["dM_full_kg"]) / dV)
    direct = float(np.nansum(period["dM_ind_kg"]) / dV)
    closed = float(np.nansum(period["dM_closed_kg"]) / dV)
    return pd.Series(
        {
            "start_year": start_year,
            "end_year": end_year,
            "full_rho_kg_m3": full,
            "direct_rho_kg_m3": direct,
            "closed_rho_kg_m3": closed,
            "direct_minus_full_kg_m3": direct - full,
            "closed_minus_full_kg_m3": closed - full,
        }
    )


def format_period_density_comparison(summary: pd.Series) -> str:
    """Format the prediction-period density comparison for the run log."""
    start = int(summary["start_year"])
    end = int(summary["end_year"])
    return (
        f"[density] {start}-{end} effective density (kg m-3): "
        f"full model {summary['full_rho_kg_m3']:.1f}; "
        f"before temporal reconciliation {summary['direct_rho_kg_m3']:.1f} "
        f"({summary['direct_minus_full_kg_m3']:+.1f}); "
        f"after temporal reconciliation {summary['closed_rho_kg_m3']:.1f} "
        f"({summary['closed_minus_full_kg_m3']:+.1f})."
    )


def make_synthetic_series(
    start_year: int = 2000,
    n_years: int = 20,
    positive_years: int = 10,
    seed: int = SEED,
    pos_mean: float = 0.35,
    neg_mean: float = -0.55,
    interannual_sd: float = INTERANNUAL_SD,
    uncertainty_fraction: float = UNCERTAINTY_FRACTION,
) -> pd.DataFrame:
    """Build the previous synthetic series, retained for manuscript value compatibility."""
    rng = np.random.default_rng(seed)
    pos = np.maximum(rng.normal(pos_mean, interannual_sd, positive_years), 0.03)
    neg = np.minimum(rng.normal(neg_mean, interannual_sd, n_years - positive_years), -0.03)
    obs_dh = np.concatenate([pos, neg])
    years0 = np.arange(start_year, start_year + n_years)
    return pd.DataFrame({"i": np.arange(n_years), "year0": years0, "year1": years0 + 1, "dh_obs_m": obs_dh, "sigma_dh_m": uncertainty_fraction * interannual_sd})


def build_periods(annual: pd.DataFrame, model: RhoSurrogate, area_m2: float) -> pd.DataFrame:
    """Compatibility helper used by manuscript_values for the synthetic example."""
    dh_obs = annual["dh_obs_m"].to_numpy(float)
    sigma_annual = annual["sigma_dh_m"].to_numpy(float)
    years0 = annual["year0"].to_numpy(int)
    rows = []
    for i0 in range(len(annual)):
        for i1 in range(i0 + 1, len(annual) + 1):
            dt = i1 - i0
            dh = float(np.sum(dh_obs[i0:i1]))
            sigma_dh = float(np.sqrt(np.sum(sigma_annual[i0:i1] ** 2)))
            dh_p = 0.0 if i0 == 0 else float(np.mean(dh_obs[max(0, i0 - 5): i0]))
            dv = area_m2 * dh
            mu = model.integrated_mu(dh=dh, sigma_dh=sigma_dh, dh_p=dh_p, sigma_dh_p=0.0, dt=dt)
            sigma_rho, sigma_dM = model.integrated_sigma_equiv_density(dh=dh, sigma_dh=sigma_dh, dt=dt, area_m2=area_m2)
            rows.append({"i0": i0, "i1": i1, "year0": int(years0[i0]), "year1": int(years0[i1 - 1] + 1), "dt_yr": dt, "dh_obs_m": dh, "dV_m3": dv, "mu_rho_ind_kg_m3": mu, "sigma_rho_ind_kg_m3": sigma_rho, "sigma_dM_rho_ind_kg": sigma_dM, "dM_ind_kg": mu * dv, "b_ind_kg": (mu - model.rho_ice) * dv})
    return pd.DataFrame(rows)


def temporal_reconcile(periods: pd.DataFrame, n_elem: int, model: RhoSurrogate) -> tuple[pd.DataFrame, np.ndarray]:
    """Compatibility helper used by manuscript_values for the synthetic example."""
    S = np.zeros((len(periods), n_elem), dtype=float)
    for q, row in periods.iterrows():
        S[q, int(row.i0): int(row.i1)] = 1.0
    b = periods["b_ind_kg"].to_numpy(float)
    elem = periods.loc[np.isclose(periods["dt_yr"], 1.0)].sort_values("i0")
    elem_abs_dh = elem["dh_obs_m"].abs().to_numpy(float)
    period_dt = periods["dt_yr"].to_numpy(float)
    sigma_support = float(model.params["A0"]) ** 2 * (S @ elem_abs_dh) + float(model.params["A1"]) ** 2 * period_dt
    sigma = AREA_KM2 * 1e6 * np.sqrt(np.maximum(sigma_support, 0.0))
    floor = np.nanmedian(sigma[np.isfinite(sigma) & (sigma > 0)]) * 1.0e-3
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), np.maximum(sigma, floor), floor)
    sw = np.sqrt(1.0 / sigma**2)
    elem, *_ = np.linalg.lstsq(S * sw[:, None], b * sw, rcond=None)
    out = periods.copy()
    out["dM_closed_kg"] = model.rho_ice * out["dV_m3"].to_numpy(float) + S @ elem
    out["mu_rho_closed_kg_m3"] = np.where(out["dV_m3"] != 0, out["dM_closed_kg"] / out["dV_m3"], np.nan)
    out["sigma_rho_closed_kg_m3"] = np.where(out["dV_m3"] != 0, sigma / np.abs(out["dV_m3"].to_numpy(float)), np.inf)
    return out, elem


def period_lookup(periods: pd.DataFrame, year0: int, year1: int) -> pd.Series:
    sel = periods[(periods["year0"] == year0) & (periods["year1"] == year1)]
    if len(sel) != 1:
        raise KeyError(f"No unique period {year0}-{year1}")
    return sel.iloc[0]


def make_summary_tables(periods: pd.DataFrame, start_year: int, end_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compatibility helper used by manuscript_values for the synthetic example."""
    full = period_lookup(periods, start_year, end_year)
    annual = periods[(periods["dt_yr"] == 1) & (periods["year0"] >= start_year) & (periods["year1"] <= end_year)]
    summary = pd.DataFrame(
        [
            {"estimate": "direct full-period surrogate", "dM_Gt": full.dM_ind_kg / 1e12, "mu_or_equiv_rho_kg_m3": full.mu_rho_ind_kg_m3, "sigma_rho_kg_m3": full.sigma_rho_ind_kg_m3},
            {"estimate": "sum of independent annual surrogate estimates", "dM_Gt": annual.dM_ind_kg.sum() / 1e12, "mu_or_equiv_rho_kg_m3": annual.dM_ind_kg.sum() / annual.dV_m3.sum(), "sigma_rho_kg_m3": np.nan},
            {"estimate": "temporally reconciled full period", "dM_Gt": full.dM_closed_kg / 1e12, "mu_or_equiv_rho_kg_m3": full.mu_rho_closed_kg_m3, "sigma_rho_kg_m3": getattr(full, "sigma_rho_closed_kg_m3", np.nan)},
        ]
    )
    return summary, pd.DataFrame()


def plot_results(summary: pd.DataFrame, out_png: Path) -> None:
    """Plot annual elevation change and annual effective density."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 5.8), constrained_layout=True, sharex=True)
    display = summary.loc[
        summary["year1"].ge(DISPLAY_START_YEAR) & summary["year1"].le(END_YEAR)
    ].copy()
    years = display["year1"].to_numpy(float)
    prediction = np.ones(len(display), dtype=bool)
    major_years = [DISPLAY_START_YEAR, PAST_ELEVATION_END_YEAR, 2010, 2015, END_YEAR]

    ax = axes[0]
    ax.plot(years, display["dh_m"], color="black", marker="o", ms=3.6, lw=1.7)
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_ylabel("Annual elevation change\n" + r"$\Delta h$ (m)")
    ax.grid(alpha=0.25)

    ax = axes[1]
    closed = display.loc[prediction, "closed_rho_kg_m3"].to_numpy(float)
    closed_sigma = display.loc[prediction, "closed_sigma_rho_kg_m3"].to_numpy(float)
    closed_band = ax.fill_between(
        years[prediction],
        _rho_to_axis(closed - closed_sigma),
        _rho_to_axis(closed + closed_sigma),
        color="#F58518",
        alpha=0.22,
        linewidth=0,
    )
    full_line, = ax.plot(
        years[prediction],
        _rho_to_axis(display.loc[prediction, "full_rho_kg_m3"]),
        color="black",
        marker="o",
        ms=3.6,
        lw=1.7,
        label="Full model",
    )
    direct_line, = ax.plot(
        years[prediction],
        _rho_to_axis(display.loc[prediction, "direct_rho_kg_m3"]),
        color="#4C78A8",
        marker="o",
        ms=3.4,
        lw=1.6,
        label="Surrogate model without temporal reconciliation",
    )
    closed_line, = ax.plot(
        years[prediction],
        _rho_to_axis(closed),
        color="#F58518",
        marker="o",
        ms=3.4,
        lw=1.6,
        label="Surrogate model with temporal reconciliation",
    )
    ice_line = ax.axhline(_rho_to_axis(DENSITY_REF), color="0.3", lw=0.8, label="Ice density")
    ax.set_ylim(_rho_to_axis(DENSITY_YLIM[0]), _rho_to_axis(DENSITY_YLIM[1]))
    ax.yaxis.set_major_locator(mticker.FixedLocator(_rho_to_axis(DENSITY_TICKS)))
    ax.yaxis.set_major_formatter(mticker.FixedFormatter([f"{tick:.0f}" for tick in DENSITY_TICKS]))
    ax.tick_params(axis="y", labelsize=10)
    ax.set_ylabel("Annual effective density\n" + r"$\rho_{\Delta V}$ (kg m$^{-3}$)")
    ax.set_xlabel("Year")
    ax.legend(
        [ice_line, full_line, direct_line, (closed_band, closed_line)],
        [
            "Ice density",
            "Full model",
            "Surrogate model\nwithout temporal reconciliation",
            "Surrogate model\nwith temporal reconciliation\n" + r"($\pm$ 2$\sigma$ uncertainty)",
        ],
        handler_map={tuple: HandlerTuple(ndivide=1)},
        frameon=False,
        fontsize=9,
        handlelength=2.4,
        loc="upper right",
    )
    ax.grid(alpha=0.25)

    axes[0].text(
        0.98,
        0.92,
        REGION_NAMES.get(REGION, f"RGI {REGION:02d}"),
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.5),
    )
    axes[-1].set_xlim(DISPLAY_START_YEAR - 0.5, END_YEAR + 0.5)
    axes[-1].xaxis.set_major_locator(mticker.FixedLocator(major_years))
    axes[-1].xaxis.set_minor_locator(mticker.FixedLocator(np.arange(DISPLAY_START_YEAR, END_YEAR + 1)))
    axes[-1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):d}"))
    for letter, ax in zip("ab", axes):
        ax.text(0.02, 0.98, letter, transform=ax.transAxes, ha="left", va="top", fontsize=14, fontweight="bold")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run() -> Path:
    """Generate Fig. 5."""
    model = RhoSurrogate.from_files(parameter_path=PARAM_CSV)
    df = read_full_model_table(REGION)
    annual, periods = complete_region_subset(df)
    evaluated = apply_surrogate_and_closure(annual, periods, model)
    summary = regional_annual_summary(evaluated, model)
    period_density = regional_period_effective_density(evaluated, PAST_ELEVATION_END_YEAR, END_YEAR)
    print(format_period_density_comparison(period_density))
    plot_results(summary, OUT_PNG)
    return OUT_PNG


if __name__ == "__main__":
    path = run()
    print(f"[done] Wrote figure: {path.resolve()}")
