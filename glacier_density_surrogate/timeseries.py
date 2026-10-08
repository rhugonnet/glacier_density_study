"""Predict, integrate and reconcile effective density and mass change for glacier time series."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .surrogate import (
    RhoSurrogate,
    _glacier_id_column,
    _period_component,
    _residual_variance,
    _signed_abs_power,
    expected_abs_normal_array,
    finite_array,
    make_dh_error_correlation,
)


#####################
# GLACIER TIME SERIES
#####################


def predict_timeseries(
    model: RhoSurrogate,
    data: pd.DataFrame,
    *,
    id_col: str | None = None,
    start_col: str | None = None,
    end_col: str | None = None,
    dh_col: str | None = None,
    sigma_dh_col: str | None = None,
    dt_col: str | None = None,
    past_dhdt_col: str | None = None,
    sigma_past_dhdt_col: str | None = None,
    area_col: str | None = None,
    area_m2: float | None = None,
    expand_periods: bool = True,
    past_missing: str = "current",
    past_error_factor: float = 2.0,
    dh_error_corr: Callable[[np.ndarray], np.ndarray] | None = None,
    return_components: bool = False,
) -> pd.DataFrame:
    """
    Predict each glacier independently and combine their results in one table.

    A glacier_id or rgiid column identifies separate time series; id_col selects
    another name. _predict_glacier_timeseries() estimates each glacier's past
    elevation change rate and reconciles its periods without mixing glaciers.
    We attach identifiers and constant region_group, lat and lon metadata to
    every output period so aggregate_regions() can propagate regional errors.
    Tables without an identifier follow the single glacier workflow.
    """
    id_col = _glacier_id_column(data, id_col)

    # Send the same calculation options to each independent glacier time series
    options = {
        "start_col": start_col, "end_col": end_col, "dh_col": dh_col,
        "sigma_dh_col": sigma_dh_col, "dt_col": dt_col, "past_dhdt_col": past_dhdt_col,
        "sigma_past_dhdt_col": sigma_past_dhdt_col, "area_col": area_col,
        "area_m2": area_m2, "expand_periods": expand_periods,
        "past_missing": past_missing, "past_error_factor": past_error_factor,
        "dh_error_corr": dh_error_corr, "return_components": return_components,
    }
    if id_col is None:
        return _predict_glacier_timeseries(model, data, **options)

    # Preserve the schema for empty tables without inventing a glacier identifier
    metadata_cols = [column for column in ("region_group", "lat", "lon") if column in data and column != id_col]
    if data.empty:
        result = _predict_glacier_timeseries(model, data, **options)
        result.insert(0, id_col, pd.Series(dtype=data[id_col].dtype))
        for column in metadata_cols:
            result[column] = pd.Series(dtype=data[column].dtype)
        return result

    # Region membership and coordinates describe a glacier across all its periods
    tables = []
    for glacier_id, observations in data.groupby(id_col, sort=False, observed=True):
        for column in metadata_cols:
            if observations[column].nunique(dropna=False) > 1:
                raise ValueError(f"Each glacier must have consistent {column} metadata")
        predictions = _predict_glacier_timeseries(model, observations, **options)
        predictions.insert(0, id_col, glacier_id)
        for column in metadata_cols:
            predictions[column] = observations[column].iloc[0]
        tables.append(predictions)
    return pd.concat(tables, ignore_index=True)


def _predict_glacier_timeseries(
    model: RhoSurrogate,
    data: pd.DataFrame,
    *,
    start_col: str | None = None,
    end_col: str | None = None,
    dh_col: str | None = None,
    sigma_dh_col: str | None = None,
    dt_col: str | None = None,
    past_dhdt_col: str | None = None,
    sigma_past_dhdt_col: str | None = None,
    area_col: str | None = None,
    area_m2: float | None = None,
    expand_periods: bool = True,
    past_missing: str = "current",
    past_error_factor: float = 2.0,
    dh_error_corr: Callable[[np.ndarray], np.ndarray] | None = None,
    return_components: bool = False,
) -> pd.DataFrame:
    """
    Calculate and reconcile predictions for one glacier's observation periods.

    _prepare_input_table() resolves column names and sorts the observations.
    _expand_period_table() combines consecutive rows, and _attach_past_predictor()
    estimates the past elevation change rate. After predict() evaluates each period,
    _temporal_closure() fits additive mass change anomalies for a complete elementary grid.
    _propagate_input_uncertainty() reuses observation errors across those calculations.
    The final table selects the public outputs and any requested components.
    """

    # Remember whether physical mass change outputs can be returned
    has_area = area_m2 is not None
    if not has_area:
        area_names = ["area_m2", "area", "glacier_area_m2"]
        has_area = area_col is not None or _first_existing(data.columns, area_names, required=False) is not None
    table = _prepare_input_table(
        data,
        start_col=start_col,
        end_col=end_col,
        dh_col=dh_col,
        sigma_dh_col=sigma_dh_col,
        dt_col=dt_col,
        past_dhdt_col=past_dhdt_col,
        sigma_past_dhdt_col=sigma_past_dhdt_col,
        area_col=area_col,
        area_m2=area_m2,
    )
    has_time_bounds = table.attrs["has_time_bounds"]
    finite_array(past_error_factor, "past_error_factor", nonnegative=True)
    if past_missing not in {"current", "zero"}:
        raise ValueError("past_missing must be 'current' or 'zero'")
    # Build longer periods and estimate the history available before each start
    corr = dh_error_corr or make_dh_error_correlation("none")
    if has_time_bounds and expand_periods and _is_elementary_series(table):
        periods = _expand_period_table(table, corr)
    else:
        periods = table.copy()
    input_weights = _period_input_weights(periods, table)
    periods, history_weights = _attach_past_predictor(
        model, periods, table, corr, past_missing, past_error_factor, input_weights,
    )

    # Evaluate each period before enforcing agreement across time
    rows = []
    for row in periods.itertuples(index=False):
        rows.append(
            model.predict(
                dh=row.dh_m,
                sigma_dh=row.sigma_dh_m,
                dt=row.period_years,
                past_dhdt=row.past_dhdt_m_yr,
                sigma_past_dhdt=row.sigma_past_dhdt_m_yr,
                area_m2=row.area_m2,
                past_missing=past_missing,
                past_error_factor=past_error_factor,
            )
        )
    raw_columns = ["mu_rho_kg_m3", "sigma_rho_kg_m3", "dM_kg", "sigma_dM_rho_kg", "sigma_dM_dh_kg"]
    raw = pd.DataFrame(rows, columns=raw_columns).add_prefix("raw_")
    out = pd.concat([periods.reset_index(drop=True), raw], axis=1)
    out["mu_rho_independent_kg_m3"] = out["raw_mu_rho_kg_m3"]
    out["sigma_rho_independent_kg_m3"] = out["raw_sigma_rho_kg_m3"]
    out["dV_m3"] = out["area_m2"] * out["dh_m"]
    out["dM_independent_kg"] = out["raw_dM_kg"]
    out["sigma_dM_rho_independent_kg"] = out["raw_sigma_dM_rho_kg"]

    # Reconcile connected records; disconnected periods use their direct estimates
    if has_time_bounds and _can_temporally_reconcile(out):
        out = _temporal_closure(model, out)
    else:
        out["mu_rho_closed_kg_m3"] = out["mu_rho_independent_kg_m3"]
        out["sigma_rho_closed_kg_m3"] = out["sigma_rho_independent_kg_m3"]
        out["dM_closed_kg"] = out["dM_independent_kg"]
        out["sigma_dM_rho_closed_kg"] = out["sigma_dM_rho_independent_kg"]
        out["temporally_closed"] = False

    # Select density outputs and add mass change or comparison columns when requested
    out["mu_rho_kg_m3"] = out["mu_rho_closed_kg_m3"]
    out["sigma_rho_kg_m3"] = out["sigma_rho_closed_kg_m3"]
    out["dM_kg"] = out["dM_closed_kg"]
    out["sigma_dM_rho_kg"] = out["sigma_dM_rho_closed_kg"]
    if has_area:
        out = _propagate_input_uncertainty(model, table, out, input_weights, history_weights, corr)
    output_columns = [
        "start",
        "end",
        "period_years",
        "dh_m",
        "sigma_dh_m",
        "past_dhdt_m_yr",
        "sigma_past_dhdt_m_yr",
        "mu_rho_kg_m3",
        "sigma_rho_kg_m3",
    ]
    if has_area:
        output_columns.extend([
            "area_m2", "dV_m3", "dM_kg", "sigma_dM_rho_kg",
            "sigma_dV_m3", "sigma_dM_dh_kg", "sigma_dM_total_kg",
        ])
    if return_components:
        output_columns.extend(
            [
                "mu_rho_closed_kg_m3",
                "sigma_rho_closed_kg_m3",
                "dM_closed_kg",
                "sigma_dM_rho_closed_kg",
                "mu_rho_independent_kg_m3",
                "sigma_rho_independent_kg_m3",
                "dM_independent_kg",
                "sigma_dM_rho_independent_kg",
                "temporally_closed",
            ]
        )
    result = out[[c for c in output_columns if c in out.columns]].copy()
    result.attrs["has_time_bounds"] = has_time_bounds
    return result


#####################
# OBSERVATION PERIODS
#####################


def _first_existing(columns: pd.Index, candidates: list[str], required: bool = True) -> str | None:
    """Select the first accepted column name, raising an error if required."""
    for col in candidates:
        if col in columns:
            return col
    if required:
        raise KeyError(f"None of these columns exist: {candidates}")
    return None


def _prepare_input_table(
    data: pd.DataFrame,
    *,
    start_col: str | None,
    end_col: str | None,
    dh_col: str | None,
    sigma_dh_col: str | None,
    dt_col: str | None,
    past_dhdt_col: str | None,
    sigma_past_dhdt_col: str | None,
    area_col: str | None,
    area_m2: float | None,
) -> pd.DataFrame:
    """Resolve input columns, validate observations and sort by period bounds."""
    columns = data.columns

    # Accept common CSV column names as aliases for the period table columns
    dh_col = dh_col or _first_existing(columns, ["dh_m", "dh", "elevation_change_m", "delta_h_m"])
    sigma_dh_col = sigma_dh_col or _first_existing(
        columns, ["sigma_dh_m", "sig_dh_m", "dh_uncertainty_m", "sigma_dh"], required=False
    )
    start_col = start_col or _first_existing(columns, ["start", "start_year", "year0", "period_start"], required=False)
    end_col = end_col or _first_existing(columns, ["end", "end_year", "year1", "period_end"], required=False)
    dt_col = dt_col or _first_existing(columns, ["dt", "dt_yr", "period_years", "duration_yr"], required=False)

    # Prefer explicit rate units while accepting column names from earlier CSV files
    past_dhdt_col = past_dhdt_col or _first_existing(
        columns, ["past_dhdt_m_yr", "past_dhdt", "past_dh_m", "past_dh", "dh_p_m", "dh_p"], required=False
    )
    sigma_past_dhdt_col = sigma_past_dhdt_col or _first_existing(
        columns, ["sigma_past_dhdt_m_yr", "sigma_past_dhdt", "sigma_past_dh_m", "sig_past_dh_m", "sigma_dh_p_m"],
        required=False,
    )
    area_col = area_col or _first_existing(columns, ["area_m2", "area", "glacier_area_m2"], required=False)
    out = pd.DataFrame({"dh_m": pd.to_numeric(data[dh_col], errors="coerce")})
    out["sigma_dh_m"] = pd.to_numeric(data[sigma_dh_col], errors="coerce") if sigma_dh_col else 0.0

    # Derive period length from dates, or create bounds when only duration is supplied
    if start_col and end_col:
        out["start"] = pd.to_numeric(data[start_col], errors="coerce")
        out["end"] = pd.to_numeric(data[end_col], errors="coerce")
        out["period_years"] = out["end"] - out["start"]
    elif dt_col:
        out["period_years"] = pd.to_numeric(data[dt_col], errors="coerce")
        out["start"] = np.arange(len(out), dtype=float)
        out["end"] = out["start"] + out["period_years"]
    else:
        raise KeyError("Input must contain start/end columns or a period length column")
    out["past_dhdt_m_yr"] = pd.to_numeric(data[past_dhdt_col], errors="coerce") if past_dhdt_col else np.nan
    out["sigma_past_dhdt_m_yr"] = pd.to_numeric(data[sigma_past_dhdt_col], errors="coerce") if sigma_past_dhdt_col else np.nan
    if area_col:
        out["area_m2"] = pd.to_numeric(data[area_col], errors="coerce")
    else:
        out["area_m2"] = 1.0 if area_m2 is None else float(area_m2)

    # Missing past elevation change rate fields may be estimated; invalid measured inputs must be corrected
    for column in ("dh_m", "start", "end"):
        finite_array(out[column], column)
    finite_array(out["sigma_dh_m"], "sigma_dh_m", nonnegative=True)
    finite_array(out["period_years"], "period_years", positive=True)
    finite_array(out["area_m2"], "area_m2", positive=True)
    for column in ("past_dhdt_m_yr", "sigma_past_dhdt_m_yr"):
        supplied = out[column].dropna()
        finite_array(supplied, column, nonnegative=column.startswith("sigma"))
    if out.duplicated(["start", "end"]).any():
        raise ValueError("Input contains duplicate observation periods")
    out = out.sort_values(["start", "end"]).reset_index(drop=True)
    out.attrs["has_time_bounds"] = bool(start_col and end_col)
    return out


def _is_elementary_series(table: pd.DataFrame) -> bool:
    """Check whether sorted rows form consecutive observation periods."""
    ordered = table.sort_values(["start", "end"]).reset_index(drop=True)
    if len(ordered) <= 1:
        return False
    return bool(np.array_equal(ordered["start"].to_numpy(float)[1:], ordered["end"].to_numpy(float)[:-1]))


def _period_input_weights(periods: pd.DataFrame, observations: pd.DataFrame) -> np.ndarray:
    """Map each period to its original observation or the observations combined during expansion."""
    weights = np.zeros((len(periods), len(observations)))
    starts = observations["start"].to_numpy()
    ends = observations["end"].to_numpy()
    for index, period in enumerate(periods.itertuples(index=False)):
        original = (starts == period.start) & (ends == period.end)
        if original.any():
            weights[index, original] = 1.0
        else:
            weights[index] = (period.start <= starts) & (ends <= period.end)
    return weights


def _can_temporally_reconcile(table: pd.DataFrame) -> bool:
    """Check that each adjacent boundary pair has an elementary observation."""
    if len(table) <= 1 or not {"start", "end"}.issubset(table.columns):
        return False
    intervals = table[["start", "end"]].drop_duplicates().to_numpy(float)
    bounds = np.array(sorted(set(intervals[:, 0]).union(set(intervals[:, 1]))), dtype=float)
    if len(bounds) <= 2:
        return False
    for start, end in zip(bounds[:-1], bounds[1:]):
        covered = np.any((intervals[:, 0] == start) & (intervals[:, 1] == end))
        if not covered:
            return False
    return True


def _normal_error_covariance(
    sigmas: np.ndarray, times: np.ndarray, corr: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """Combine error standard deviations with correlations at period midpoints."""
    lag = np.abs(np.asarray(times, dtype=float)[:, None] - np.asarray(times, dtype=float)[None, :])
    correlation = finite_array(corr(lag), "elevation error correlation")
    if correlation.shape != lag.shape or np.any(np.abs(correlation) > 1):
        raise ValueError("Elevation error correlation must match the lag shape and lie between -1 and 1")
    if not np.allclose(correlation, correlation.T, rtol=1e-12, atol=1e-12):
        raise ValueError("Elevation error correlation must be symmetric")
    if not np.allclose(np.diag(correlation), 1.0, rtol=1e-12, atol=1e-12):
        raise ValueError("Elevation error correlation must equal one at zero lag")
    return correlation * np.outer(sigmas, sigmas)


def _normal_error_sigma(covariance: np.ndarray, weights: np.ndarray) -> float:
    """Propagate a weighted error and reject negative variance beyond roundoff."""
    variance = float(weights @ covariance @ weights)
    independent_variance = float(np.sum(weights**2 * np.diag(covariance)))
    if variance < -1e-12 * independent_variance:
        raise ValueError("Elevation error covariance gives a negative variance")
    return float(np.sqrt(max(variance, 0.0)))


def _expand_period_table(table: pd.DataFrame, corr: Callable[[np.ndarray], np.ndarray]) -> pd.DataFrame:
    """
    Combine consecutive observations into all contiguous periods.

    Elevation changes add directly. Their uncertainty uses the supplied correlation
    between observation midpoints. Area must be constant so volume changes add.
    """

    elementary = table.sort_values(["start", "end"]).reset_index(drop=True)
    if not np.allclose(elementary["area_m2"], elementary["area_m2"].iloc[0], rtol=1e-12, atol=0):
        raise ValueError("Period expansion requires constant glacier area")
    midpoints = 0.5 * (elementary["start"].to_numpy(float) + elementary["end"].to_numpy(float))
    sigma = elementary["sigma_dh_m"].to_numpy(float)
    covariance = _normal_error_covariance(sigma, midpoints, corr)
    rows = []

    # Sum changes and error covariances for every contiguous set of observations
    for i0 in range(len(elementary)):
        for i1 in range(i0 + 1, len(elementary) + 1):
            interval_slice = slice(i0, i1)
            observations = elementary.iloc[interval_slice]
            dh = float(observations["dh_m"].sum())
            ones = np.ones(i1 - i0)
            sigma_dh = _normal_error_sigma(covariance[interval_slice, interval_slice], ones)
            rows.append(
                {
                    "start": float(observations["start"].iloc[0]),
                    "end": float(observations["end"].iloc[-1]),
                    "period_years": float(observations["end"].iloc[-1] - observations["start"].iloc[0]),
                    "dh_m": dh,
                    "sigma_dh_m": sigma_dh,
                    "area_m2": float(observations["area_m2"].iloc[0]),
                    "past_dhdt_m_yr": float(observations["past_dhdt_m_yr"].iloc[0]),
                    "sigma_past_dhdt_m_yr": float(observations["sigma_past_dhdt_m_yr"].iloc[0]),
                }
            )
    out = pd.DataFrame(rows)
    out.attrs.update(table.attrs)
    return out


############################
# PAST ELEVATION CHANGE RATE
############################


def _attach_past_predictor(
    model: RhoSurrogate,
    periods: pd.DataFrame,
    elementary: pd.DataFrame,
    corr: Callable[[np.ndarray], np.ndarray],
    past_missing: str,
    past_error_factor: float,
    input_weights: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Fill missing past elevation change rates and their uncertainties independently from observed annual rates.

    Only elementary observations with actual dates contribute to history. Coarse
    observations are split into roughly annual steps sharing the original error.
    Exponential weights favor recent steps up to tau_max years before each start.
    Return the filled periods and the observation weights used for each estimated
    past elevation change rate, so uncertainty propagation uses the same history.
    """
    out = periods.copy()
    history_weights = np.zeros_like(input_weights)
    observations = elementary.copy()
    if not elementary.attrs["has_time_bounds"]:
        observations = observations.iloc[:0]
    elif not observations.empty:
        bounds = np.unique(observations[["start", "end"]].to_numpy(float))
        elementary_bounds = set(zip(bounds[:-1], bounds[1:]))
        selected = [(a, b) in elementary_bounds for a, b in observations[["start", "end"]].itertuples(index=False, name=None)]
        observations = observations.loc[selected]

    # Split elementary changes into annual rates, recording their shared input errors
    annual_rows = []
    for observation_idx, row in observations.iterrows():
        dt = float(row["period_years"])
        edges = np.linspace(row["start"], row["end"], max(1, int(round(dt))) + 1)
        for start, end in zip(edges[:-1], edges[1:]):
            annual_rows.append({
                "observation": observation_idx, "end": end, "mid": (start + end) / 2,
                "rate": row["dh_m"] / dt, "sigma_rate": row["sigma_dh_m"] / dt,
            })
    annual = pd.DataFrame(annual_rows)
    if not annual.empty:
        sigmas = annual["sigma_rate"].to_numpy(float)
        ids = annual["observation"].to_numpy()
        covariance = _normal_error_covariance(sigmas, annual["mid"].to_numpy(float), corr)
        same_input = ids[:, None] == ids[None, :]
        covariance = np.where(same_input, np.outer(sigmas, sigmas), covariance)

    # Use the chosen assumption where earlier observations are missing, then average available history
    for idx, row in out.iterrows():
        if pd.notna(row["past_dhdt_m_yr"]) and pd.notna(row["sigma_past_dhdt_m_yr"]):
            continue
        past_mean = row["dh_m"] / row["period_years"] if past_missing == "current" else 0.0
        past_sigma = past_error_factor * row["sigma_dh_m"] / row["period_years"]
        past_weights = input_weights[idx] / row["period_years"] if past_missing == "current" else np.zeros(len(elementary))
        if not annual.empty:
            lag = row["start"] - annual["end"].to_numpy(float)
            past_indices = np.flatnonzero((lag >= 0) & (lag <= float(model.params["tau_max"])))
            if len(past_indices):
                weights = np.exp(-lag[past_indices] / float(model.params["memory_tau_years"]))
                weights /= weights.sum()
                past_mean = float(weights @ annual["rate"].to_numpy(float)[past_indices])
                past_sigma = _normal_error_sigma(covariance[np.ix_(past_indices, past_indices)], weights)
                past_weights = np.zeros(len(elementary))
                sources = ids[past_indices]
                np.add.at(past_weights, sources, weights / elementary.loc[sources, "period_years"].to_numpy())
        if pd.isna(row["past_dhdt_m_yr"]):
            out.loc[idx, "past_dhdt_m_yr"] = past_mean
            history_weights[idx] = past_weights
        if pd.isna(row["sigma_past_dhdt_m_yr"]):
            out.loc[idx, "sigma_past_dhdt_m_yr"] = past_sigma
    return out, history_weights


####################################
# GLACIER MASS CHANGE RECONCILIATION
####################################


def _temporal_closure(model: RhoSurrogate, periods: pd.DataFrame) -> pd.DataFrame:
    """Reconcile one glacier with the batch calculation over its observation periods."""
    out = periods.copy()
    out["_glacier"] = "glacier"
    reconciled = temporally_reconcile_periods(
        out, model, id_col="_glacier", start_col="start", end_col="end", area_col="area_m2",
    )
    for source, target in (
        ("dM_surrogate_kg", "dM_closed_kg"),
        ("mu_rho_surrogate_kg_m3", "mu_rho_closed_kg_m3"),
        ("sigma_dM_rho_surrogate_kg", "sigma_dM_rho_closed_kg"),
        ("sigma_rho_surrogate_kg_m3", "sigma_rho_closed_kg_m3"),
    ):
        out[target] = reconciled[source].to_numpy()
    out["temporally_closed"] = True
    return out.drop(columns="_glacier")


##################################
# DENSITY AND RESIDUAL UNCERTAINTY
##################################


def _broadcast_predictors(dh, sigma_dh, past_dhdt, sigma_past_dhdt, dt) -> tuple[np.ndarray, ...]:
    """Validate current elevation change and past elevation change rate and give them a common array shape."""
    return tuple(np.broadcast_arrays(
        finite_array(dh, "dh"), finite_array(sigma_dh, "sigma_dh", nonnegative=True),
        finite_array(past_dhdt, "past_dhdt"), finite_array(sigma_past_dhdt, "sigma_past_dhdt", nonnegative=True),
        finite_array(dt, "dt", positive=True),
    ))


def integrated_mu_vectorized(
    model: RhoSurrogate,
    dh: np.ndarray,
    sigma_dh: np.ndarray,
    past_dhdt: np.ndarray,
    sigma_past_dhdt: np.ndarray,
    dt: np.ndarray,
) -> np.ndarray:
    """
    Calculate effective densities for many observation periods at once.

    This applies integrated_mu() to broadcast arrays using the
    model's quadrature nodes. Errors in current elevation change and past
    elevation change rate are independent.

    :param model: Effective density surrogate.
    :param dh: Elevation changes over the periods, in metres.
    :param sigma_dh: Standard deviations of elevation change in metres.
    :param past_dhdt: Past elevation change rate in m yr-1.
    :param sigma_past_dhdt: Standard deviations of the past elevation change rate in m yr-1.
    :param dt: Period lengths in years.
    :returns: Effective densities in kg m-3, with NaN where dh is zero.
    """

    params = model.params
    predictors = _broadcast_predictors(dh, sigma_dh, past_dhdt, sigma_past_dhdt, dt)
    shape = predictors[0].shape
    dh, sigma_dh, past_dhdt, sigma_past_dhdt, dt = [array.ravel() for array in predictors]

    # Average the two past elevation change rate terms before integrating current change
    y = past_dhdt[:, None] + np.sqrt(2.0) * sigma_past_dhdt[:, None] * model._gh_x_past[None, :]
    signed_power = np.sum(
        model._gh_w_past[None, :] * _signed_abs_power(y, float(model.params["etaMem"])),
        axis=1,
    ) / np.sqrt(np.pi)
    tanh_moment = np.sum(model._gh_w_past[None, :] * np.tanh(y / float(params["LA"])), axis=1) / np.sqrt(np.pi)

    # Average density times current change so density times observed volume change gives mean mass change
    x = dh[:, None] + np.sqrt(2.0) * sigma_dh[:, None] * model._gh_x_current[None, :]
    abs_dh = np.maximum(np.abs(x), 1.0e-12)
    sign_dh = np.where(x < 0, -1.0, 1.0)
    damping = np.exp(-np.clip((abs_dh / float(params["H"])) ** float(params["beta"]), 0.0, 700.0))
    finite_past = float(params["Bmem"]) * sign_dh * signed_power[:, None]
    singular_past = float(params["A"]) * sign_dh * tanh_moment[:, None] / (abs_dh ** float(params["alpha"]))
    density_correction = float(params["Bc"]) + _period_component(params, dt[:, None]) + finite_past + singular_past
    density_at_nodes = model.rho_ice + damping * density_correction
    numerator = np.sum(model._gh_w_current[None, :] * density_at_nodes * x, axis=1) / np.sqrt(np.pi)

    out = np.full_like(dh, np.nan, dtype=float)
    np.divide(numerator, dh, out=out, where=dh != 0)
    return out.reshape(shape)


def integrated_sigma_vectorized(
    model: RhoSurrogate, dh: np.ndarray, sigma_dh: np.ndarray, dt: np.ndarray
) -> np.ndarray:
    """
    Express integrated surrogate residual uncertainty as a density for many periods.

    This uses the same mass change variance as integrated_sigma(), including uncertainty
    in the absolute current elevation change. It excludes variance of the mean
    mass change caused by uncertain current elevation change and past elevation change rate.

    :param model: Effective density surrogate.
    :param dh: Vector of elevation changes in metres.
    :param sigma_dh: Vector of elevation change standard deviations in metres.
    :param dt: Vector of period lengths in years.
    :returns: Standard deviations in kg m-3, with infinity where dh is zero.
    """

    variance_over_area2 = _residual_variance(model, dh, sigma_dh, dt)
    dh, variance_over_area2 = np.broadcast_arrays(dh, variance_over_area2)
    out = np.full(dh.shape, np.inf, dtype=float)
    np.divide(np.sqrt(np.maximum(variance_over_area2, 0.0)), np.abs(dh), out=out, where=np.asarray(dh) != 0)
    return out


def integrated_sigma_mass_vectorized(
    model: RhoSurrogate,
    area_m2: np.ndarray,
    dh: np.ndarray,
    sigma_dh: np.ndarray,
    dt: np.ndarray,
) -> np.ndarray:
    """
    Calculate surrogate residual mass change uncertainty directly from its variance.

    The result stays finite at zero elevation change. Computing it directly avoids
    the undefined product of infinite density uncertainty and zero volume change.

    :param model: Effective density surrogate.
    :param area_m2: Glacier areas in square metres.
    :param dh: Elevation changes in metres.
    :param sigma_dh: Elevation change standard deviations in metres.
    :param dt: Period lengths in years.
    :returns: Mass change standard deviations in kg, with the broadcast input shape.
    """

    variance_over_area2 = _residual_variance(model, dh, sigma_dh, dt)
    return finite_array(area_m2, "area_m2", positive=True) * np.sqrt(variance_over_area2)


################################
# SHARED INPUT ERROR PROPAGATION
################################


def _propagate_input_uncertainty(
    model: RhoSurrogate,
    observations: pd.DataFrame,
    predictions: pd.DataFrame,
    input_weights: np.ndarray,
    history_weights: np.ndarray,
    corr: Callable[[np.ndarray], np.ndarray],
) -> pd.DataFrame:
    """
    Add input and total mass change errors using the prepared observation history.

    Independent periods use the quadrature variance already calculated by predict().
    With real dates, _sample_reconciled_mass() reuses observation errors across periods
    and estimated past elevation change rates. Exact inputs need no sampling.
    """
    result = predictions.copy()
    result["sigma_dM_dh_kg"] = result["raw_sigma_dM_dh_kg"]
    uncertain = (result["sigma_dh_m"] > 0) | (result["sigma_past_dhdt_m_yr"] > 0)

    # Reuse uncertain observations in dated time series, including disconnected periods
    if len(result) > 1 and observations.attrs["has_time_bounds"] and uncertain.any():
        mass_samples = _sample_reconciled_mass(
            model, observations, result, input_weights, history_weights, corr,
        )
        result["sigma_dM_dh_kg"] = mass_samples.std(axis=0, ddof=1)

    # Input and surrogate residual variances describe distinct error sources
    result["sigma_dV_m3"] = result["area_m2"] * result["sigma_dh_m"]
    result["sigma_dM_total_kg"] = np.hypot(result["sigma_dM_rho_kg"], result["sigma_dM_dh_kg"])
    return result


def _sample_reconciled_mass(
    model: RhoSurrogate,
    observations: pd.DataFrame,
    predictions: pd.DataFrame,
    input_weights: np.ndarray,
    history_weights: np.ndarray,
    corr: Callable[[np.ndarray], np.ndarray],
    standard_draws: np.ndarray | None = None,
) -> np.ndarray:
    """
    Sample mass changes using shared elevation observations and their history weights.

    Arrays input_weights and history_weights map original observations to each period's
    current change and estimated past elevation change rate. A supplied past rate is
    independent of observations. Extra history errors are shared by periods with the
    same start. We preserve the predicted past rate error magnitude when it overrides
    the contribution from observed inputs.

    Balanced Gaussian draws have exactly zero sample means and identity covariance.
    Reconciliation is repeated for each draw, giving additive sampled mass changes.
    Return an array shaped (draws, predicted periods), in kg.
    """
    sigmas = observations["sigma_dh_m"].to_numpy()
    midpoints = (observations["start"].to_numpy() + observations["end"].to_numpy()) / 2
    covariance = _normal_error_covariance(sigmas, midpoints, corr)

    # Include singular covariance, such as perfectly opposing elevation errors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tolerance = 1e-12 * np.max(np.abs(eigenvalues), initial=0.0)
    if np.any(eigenvalues < -tolerance):
        raise ValueError("Elevation error covariance must be positive semidefinite")
    if np.array_equal(covariance, np.diag(sigmas**2)):
        factor = np.diag(sigmas)
    else:
        factor = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))

    # Separate reused observations from additional uncertainty in the past rate
    history_variance = np.einsum("pi,ij,pj->p", history_weights, covariance, history_weights)
    history_sigma = np.sqrt(np.maximum(history_variance, 0.0))
    past_sigma = predictions["sigma_past_dhdt_m_yr"].to_numpy()
    history_scale = np.ones(len(predictions))
    np.divide(past_sigma, history_sigma, out=history_scale, where=history_sigma > 0)
    history_scale = np.minimum(history_scale, 1.0)
    shared_variance = (history_sigma * history_scale)**2
    extra_variance = np.maximum(past_sigma**2 - shared_variance, 0.0)
    rounding = 1e-12 * np.maximum(past_sigma**2, shared_variance)
    extra_sigma = np.sqrt(np.where(extra_variance <= rounding, 0.0, extra_variance))
    extra_starts = pd.Index(predictions.loc[extra_sigma > 0, "start"].unique())
    extra_columns = extra_starts.get_indexer(predictions["start"])
    dimension = len(observations) + len(extra_starts)

    # A fixed seed makes predictions reproducible and independent of glacier ordering
    if standard_draws is None:
        sample_count = max(8192, dimension + 1)
        rng = np.random.default_rng(2026)
        standard_draws = rng.normal(size=(sample_count, dimension))
        standard_draws -= standard_draws.mean(axis=0)
        sample_covariance = standard_draws.T @ standard_draws / (sample_count - 1)
        balance = np.linalg.cholesky(sample_covariance)
        standard_draws = np.linalg.solve(balance, standard_draws.T).T
    else:
        standard_draws = finite_array(standard_draws, "standard_draws")
        if standard_draws.ndim != 2 or standard_draws.shape[1] != dimension:
            raise ValueError("standard_draws must match the observation and extra history error count")

    # Reuse each observation error in every current and past rate estimate depending on it
    observation_errors = standard_draws[:, :len(observations)] @ factor.T
    current_errors = observation_errors @ input_weights.T
    current_errors[:, predictions["sigma_dh_m"].to_numpy() == 0] = 0.0
    sampled_changes = predictions["dh_m"].to_numpy() + current_errors
    past_errors = (observation_errors @ history_weights.T) * history_scale
    extra = extra_sigma > 0
    past_errors[:, extra] += standard_draws[:, len(observations) + extra_columns[extra]] * extra_sigma[extra]
    sampled_past = predictions["past_dhdt_m_yr"].to_numpy() + past_errors

    # Convert sampled current and past elevation changes to mass changes
    areas = predictions["area_m2"].to_numpy()
    volumes = sampled_changes * areas
    durations = predictions["period_years"].to_numpy()
    densities = model.mu_rho(sampled_changes, sampled_past, durations)
    masses = densities * volumes
    if not predictions["temporally_closed"].any():
        return masses

    # Independent overlapping measurements are fitted on their nominal additive volume grid
    fit_changes = sampled_changes
    fit_volumes = volumes
    if not _is_elementary_series(observations):
        fit_changes = np.broadcast_to(predictions["dh_m"].to_numpy(), sampled_changes.shape)
        fit_volumes = np.broadcast_to(predictions["dV_m3"].to_numpy(), volumes.shape)

    # Apply the same temporal reconciliation to each sampled mass change time series
    sample_count, period_count = sampled_changes.shape
    sampled_periods = pd.DataFrame({
        "draw": np.repeat(np.arange(sample_count), period_count),
        "start": np.tile(predictions["start"].to_numpy(), sample_count),
        "end": np.tile(predictions["end"].to_numpy(), sample_count),
        "area_m2": np.tile(areas, sample_count),
        "dh_m": fit_changes.ravel(),
        "dV_m3": fit_volumes.ravel(),
        "dM_independent_kg": masses.ravel(),
    })
    reconciled = temporally_reconcile_periods(
        sampled_periods, model, id_col="draw", start_col="start", end_col="end", area_col="area_m2",
    )
    return reconciled["dM_surrogate_kg"].to_numpy().reshape(sample_count, period_count)


#############################
# PREDICTOR ERROR PROPAGATION
#############################


def _past_term_moments_vectorized(
    model: RhoSurrogate,
    past_dhdt: np.ndarray,
    sigma_past_dhdt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate means, variances and covariance of the two past elevation change rate terms.

    Centering terms before squaring preserves small predictor uncertainties.
    We reuse these five moments at every current integration point.
    """
    y = past_dhdt[:, None] + np.sqrt(2.0) * sigma_past_dhdt[:, None] * model._gh_x_past[None, :]
    weights = model._gh_w_past[None, :] / np.sqrt(np.pi)

    # Center the terms so small variances do not subtract large squared means
    signed_power = _signed_abs_power(y, float(model.params["etaMem"]))
    tanh_term = np.tanh(y / float(model.params["LA"]))
    mean_signed_power = np.sum(weights * signed_power, axis=1)
    mean_tanh = np.sum(weights * tanh_term, axis=1)
    centered_power = signed_power - mean_signed_power[:, None]
    centered_tanh = tanh_term - mean_tanh[:, None]
    variance_power = np.sum(weights * centered_power**2, axis=1)
    variance_tanh = np.sum(weights * centered_tanh**2, axis=1)
    covariance = np.sum(weights * centered_power * centered_tanh, axis=1)
    return mean_signed_power, mean_tanh, variance_power, variance_tanh, covariance


def mean_density_volume_moments_vectorized(
    model: RhoSurrogate,
    area_m2: np.ndarray,
    dh: np.ndarray,
    sigma_dh: np.ndarray,
    past_dhdt: np.ndarray,
    sigma_past_dhdt: np.ndarray,
    dt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate expected mass change and its variance from uncertain elevation changes.

    Current elevation change X and past elevation change rate Y are independent
    normal variables. Mass change is mu_rho(X, Y) * area * X.
    _past_term_moments_vectorized() averages the past elevation change rate terms
    first. Current quadrature then combines conditional variances.
    The resulting variance excludes the surrogate residual variance.
    Inputs follow NumPy broadcasting and outputs use their common shape.

    :param model: Effective density surrogate.
    :param area_m2: Glacier areas in square metres.
    :param dh: Measured current elevation changes in metres.
    :param sigma_dh: Current elevation change standard deviations in metres.
    :param past_dhdt: Past elevation change rate in m yr-1.
    :param sigma_past_dhdt: Past elevation change rate standard deviations in m yr-1.
    :param dt: Period lengths in years.
    :returns: Expected mass change in kg and its variance in kg2, in that order.
    """

    predictors = _broadcast_predictors(dh, sigma_dh, past_dhdt, sigma_past_dhdt, dt)
    area_m2, dh, sigma_dh, past_dhdt, sigma_past_dhdt, dt = np.broadcast_arrays(
        finite_array(area_m2, "area_m2", positive=True), *predictors,
    )
    shape = dh.shape
    area = area_m2.ravel()
    dh = dh.ravel()
    sigma_dh = sigma_dh.ravel()
    past_dhdt = past_dhdt.ravel()
    sigma_past_dhdt = sigma_past_dhdt.ravel()
    dt = dt.ravel()

    params = model.params
    past_moments = _past_term_moments_vectorized(model, past_dhdt, sigma_past_dhdt)
    mean_power, mean_tanh, variance_power, variance_tanh, covariance = past_moments

    # Evaluate current nodes using past elevation change rate moments instead of every pair of quadrature nodes
    x = dh[:, None] + np.sqrt(2.0) * sigma_dh[:, None] * model._gh_x_current[None, :]
    weights_x = model._gh_w_current[None, :] / np.sqrt(np.pi)
    abs_dh = np.maximum(np.abs(x), 1.0e-12)
    sign_x = np.where(x < 0, -1.0, 1.0)
    damping = np.exp(-np.clip((abs_dh / float(params["H"])) ** float(params["beta"]), 0.0, 700.0))

    # Separate the baseline density from coefficients multiplying the two past elevation change rate terms
    base = float(params["Bc"]) + _period_component(params, dt[:, None])
    baseline_density = model.rho_ice + damping * base
    power_coefficient = damping * float(params["Bmem"]) * sign_x
    tanh_coefficient = damping * float(params["A"]) * sign_x / (abs_dh ** float(params["alpha"]))

    mean_density_at_nodes = (
        baseline_density + power_coefficient * mean_power[:, None] + tanh_coefficient * mean_tanh[:, None]
    )
    variance_density_at_nodes = (
        power_coefficient**2 * variance_power[:, None]
        + tanh_coefficient**2 * variance_tanh[:, None]
        + 2.0 * power_coefficient * tanh_coefficient * covariance[:, None]
    )

    # Total variance is mean conditional variance plus variance of conditional means
    conditional_mass = mean_density_at_nodes * x
    mean_mass_per_area = np.sum(weights_x * conditional_mass, axis=1)
    centered_mass = conditional_mass - mean_mass_per_area[:, None]
    variance_over_area2 = np.sum(weights_x * (variance_density_at_nodes * x**2 + centered_mass**2), axis=1)
    variance_over_area2 = np.maximum(variance_over_area2, 0.0)
    deterministic = (sigma_dh == 0) & (sigma_past_dhdt == 0)
    variance_over_area2[deterministic] = 0.0

    # Restore glacier area and the original broadcast input shape
    mean_mass = area * mean_mass_per_area
    variance_mass = area**2 * variance_over_area2
    return mean_mass.reshape(shape), variance_mass.reshape(shape)


def mean_density_volume_sigma_vectorized(
    model: RhoSurrogate,
    area_m2: np.ndarray,
    dh: np.ndarray,
    sigma_dh: np.ndarray,
    past_dhdt: np.ndarray,
    sigma_past_dhdt: np.ndarray,
    dt: np.ndarray,
) -> np.ndarray:
    """
    Calculate mass change uncertainty caused by uncertain current elevation change and past elevation change rate.

    This is the square root of the variance from mean_density_volume_moments_vectorized().
    It excludes the surrogate residual uncertainty.

    :param model: Effective density surrogate.
    :param area_m2: Glacier areas in square metres.
    :param dh: Measured current elevation changes in metres.
    :param sigma_dh: Current elevation change standard deviations in metres.
    :param past_dhdt: Past elevation change rate in m yr-1.
    :param sigma_past_dhdt: Past elevation change rate standard deviations in m yr-1.
    :param dt: Period lengths in years.
    :returns: Mass change standard deviations in kg, with the broadcast input shape.
    """

    _, variance = mean_density_volume_moments_vectorized(
        model,
        area_m2=area_m2,
        dh=dh,
        sigma_dh=sigma_dh,
        past_dhdt=past_dhdt,
        sigma_past_dhdt=sigma_past_dhdt,
        dt=dt,
    )
    return np.sqrt(np.maximum(variance, 0.0))


############################
# SHARED GRID RECONCILIATION
############################


def temporally_reconcile_periods(
    periods: pd.DataFrame,
    model: RhoSurrogate,
    *,
    id_col: str = "rgiid",
    start_col: str = "start_year",
    end_col: str = "end_year",
    dh_col: str = "dh_m",
    area_col: str = "area",
    dvol_col: str = "dV_m3",
    mean_col: str = "mu_rho_independent_kg_m3",
    sigma_mass_col: str = "sigma_dM_rho_independent_kg",
    mass_col: str | None = "dM_independent_kg",
) -> pd.DataFrame:
    """
    Fit additive mass change anomalies for glaciers sharing a complete observation grid.

    Consecutive unique boundaries define elementary intervals, which may have
    different lengths or fractional years. Every glacier must have measurements
    for those intervals and every other period in the table. Areas must be constant
    within a glacier and volume changes must add over its elementary intervals.

    The support matrix maps periods to elementary changes. Weighted least squares
    adjusts mass change relative to ice density, using residual variance accumulated along
    the absolute elevation change path. Elementary residuals are independent.

    :param periods: Glacier period table on a complete shared time grid.
    :param model: Effective density surrogate.
    :param id_col: Glacier identifier column.
    :param start_col: Period start column, in years.
    :param end_col: Period end column, in years.
    :param dh_col: Elevation change column, in metres.
    :param area_col: Glacier area column, in square metres.
    :param dvol_col: Volume change column, in cubic metres.
    :param mean_col: Independent mean density column, in kg m-3.
    :param sigma_mass_col: Compatibility argument; weights use the model's path variance.
    :param mass_col: Optional independent mass change column, used when present. Supply mass change
        directly when a period has zero volume change and undefined density.
    :returns: A copy with reconciled surrogate mass change, density and residual uncertainties.
    """
    out = periods.copy()
    result_columns = ["dM_surrogate_kg", "mu_rho_surrogate_kg_m3", "sigma_dM_rho_surrogate_kg", "sigma_rho_surrogate_kg_m3"]
    if out.empty:
        for column in result_columns:
            out[column] = pd.Series(index=out.index, dtype=float)
        return out

    # Reject incomplete records before any pivot can introduce missing observations
    for column in (start_col, end_col, dh_col, dvol_col):
        finite_array(out[column], column)
    finite_array(out[area_col], area_col, positive=True)
    finite_array(out[end_col] - out[start_col], "period length", positive=True)
    if out[id_col].isna().any():
        raise ValueError("Glacier identifiers must not be missing")
    keys = [id_col, start_col, end_col]
    if out.duplicated(keys).any():
        raise ValueError("Period table contains duplicate glacier periods")
    period_defs = sorted(out[[start_col, end_col]].drop_duplicates().itertuples(index=False, name=None))
    bounds = np.unique(out[[start_col, end_col]].to_numpy(float))
    elementary_defs = list(zip(bounds[:-1], bounds[1:]))
    if not set(elementary_defs).issubset(period_defs):
        raise ValueError("Every elementary interval must have an observation")
    ids = pd.Index(out[id_col].drop_duplicates(), name=id_col)
    counts = out.groupby(id_col).size()
    if not (counts == len(period_defs)).all():
        raise ValueError("Every glacier must contain the complete set of periods")

    # Arrange physical inputs on the same glacier by period grid
    aligned = out.set_index(keys).reindex(pd.MultiIndex.from_tuples(
        [(glacier, start, end) for glacier in ids for start, end in period_defs], names=keys,
    ))
    shape = (len(ids), len(period_defs))
    area = aligned[area_col].to_numpy(float).reshape(shape)
    volume = aligned[dvol_col].to_numpy(float).reshape(shape)
    changes = aligned[dh_col].to_numpy(float).reshape(shape)
    if not np.allclose(area, area[:, :1], rtol=1e-12, atol=0):
        raise ValueError("Reconciliation requires constant area within each glacier")
    support = np.array([
        [start <= a and b <= end for a, b in elementary_defs] for start, end in period_defs
    ], dtype=float)
    elementary_indices = [period_defs.index(period) for period in elementary_defs]
    expected_volume = volume[:, elementary_indices] @ support.T
    volume_scale = np.abs(volume[:, elementary_indices]) @ support.T
    tolerance = 1e-12 * np.maximum(volume_scale, 1.0)
    if np.any(np.abs(volume - expected_volume) > tolerance):
        raise ValueError("Period volume changes must add over elementary intervals")

    # Use mass change directly when zero volume change makes density undefined
    if mass_col is not None and mass_col in aligned:
        mass = finite_array(aligned[mass_col], mass_col).reshape(shape)
    else:
        density = finite_array(aligned[mean_col], mean_col).reshape(shape)
        mass = density * volume
    anomaly = mass - model.rho_ice * volume
    period_dt = np.array([end - start for start, end in period_defs])
    variance_per_area2 = (
        float(model.params["A0"]) ** 2 * (np.abs(changes[:, elementary_indices]) @ support.T)
        + float(model.params["A1"]) ** 2 * period_dt
    )
    sigma = area * np.sqrt(variance_per_area2)

    # Rescale each glacier's weights to avoid dependence on its area units
    floor = np.array([
        np.median(row[row > 0]) * 1e-3 if np.any(row > 0) else 1.0 for row in sigma
    ])
    fit_sigma = np.maximum(sigma, floor[:, None])
    weights = (np.min(fit_sigma, axis=1)[:, None] / fit_sigma) ** 2
    normal_matrix = np.einsum("pa,np,pb->nab", support, weights, support)
    right_hand_side = np.einsum("pa,np,np->na", support, weights, anomaly)
    elementary_anomaly = np.linalg.solve(normal_matrix, right_hand_side[..., None])[..., 0]
    closed_mass = model.rho_ice * volume + elementary_anomaly @ support.T

    # Divide only nonzero volume changes and restore the original observation order
    density = np.full(shape, np.nan)
    sigma_density = np.full(shape, np.inf)
    np.divide(closed_mass, volume, out=density, where=volume != 0)
    np.divide(sigma, np.abs(volume), out=sigma_density, where=volume != 0)
    result = pd.DataFrame(dict(zip(result_columns, [closed_mass.ravel(), density.ravel(), sigma.ravel(), sigma_density.ravel()])), index=aligned.index)
    for column in result_columns:
        out[column] = result[column].reindex(pd.MultiIndex.from_frame(out[keys])).to_numpy()
    return out
