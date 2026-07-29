"""Time-series helpers for applying the effective-density surrogate.

The :class:`~glacier_density_surrogate.surrogate.RhoSurrogate` class exposes
simple scalar methods for one glacier and one period. The functions here are
the same calculations written for large tables, where Python loops would be too
slow. They are used by the study scripts, but are generic enough for advanced
users who need to process many glaciers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from math import erf

from .surrogate import RhoSurrogate, _period_component, _signed_abs_power


def expected_abs_normal_array(mean: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Return ``E|X|`` for many normal variables.

    :param mean: Normal means
    :param sigma: Normal standard deviations
    """
    mean = np.asarray(mean, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    out = np.abs(mean).copy()

    valid = np.isfinite(mean) & np.isfinite(sigma) & (sigma > 0)
    z = np.zeros_like(mean, dtype=float)
    z[valid] = mean[valid] / (np.sqrt(2.0) * sigma[valid])
    erf_z = np.array([erf(float(value)) for value in z[valid]], dtype=float)
    out[valid] = sigma[valid] * np.sqrt(2.0 / np.pi) * np.exp(-z[valid] ** 2) + mean[valid] * erf_z
    return out


def integrated_mu_vectorized(
    model: RhoSurrogate,
    dh: np.ndarray,
    sigma_dh: np.ndarray,
    past_dh: np.ndarray,
    sigma_past_dh: np.ndarray,
    dt: np.ndarray,
) -> np.ndarray:
    """Evaluate integrated surrogate means for many glacier-period rows.

    The result is equivalent to calling ``model.integrated_mu`` row by row. It
    uses the model's Gauss-Hermite quadrature nodes directly to avoid a slow
    Python loop.

    :param model: Effective-density surrogate
    :param dh: Elevation changes in metres
    :param sigma_dh: Elevation-change uncertainties in metres
    :param past_dh: Past elevation-change predictors in metres
    :param sigma_past_dh: Past predictor uncertainties in metres
    :param dt: Period lengths in years
    """
    p = model.params
    dh = np.asarray(dh, dtype=float)
    sigma_dh = np.asarray(sigma_dh, dtype=float)
    past_dh = np.asarray(past_dh, dtype=float)
    sigma_past_dh = np.asarray(sigma_past_dh, dtype=float)
    dt = np.asarray(dt, dtype=float)

    # First integrate the two past-dh moments used by the mean function.
    y = past_dh[:, None] + np.sqrt(2.0) * sigma_past_dh[:, None] * model._gh_x_past[None, :]
    signed_power = np.sum(
        model._gh_w_past[None, :] * _signed_abs_power(y, float(model.params["P_p"])),
        axis=1,
    ) / np.sqrt(np.pi)
    tanh_moment = np.sum(model._gh_w_past[None, :] * np.tanh(y / float(p["H_q"])), axis=1) / np.sqrt(np.pi)

    # Then integrate the mean density times plausible current elevation change.
    x = dh[:, None] + np.sqrt(2.0) * sigma_dh[:, None] * model._gh_x_current[None, :]
    abs_safe = np.maximum(np.abs(x), 1.0e-12)
    sign_dh = np.where(x < 0, -1.0, 1.0)
    damping = np.exp(-np.clip((abs_safe / float(p["H_d"])) ** float(p["P_d"]), 0.0, 700.0))
    finite_past = float(p["B_p"]) * sign_dh * signed_power[:, None]
    singular_past = float(p["B_q"]) * sign_dh * tanh_moment[:, None] / (abs_safe ** float(p["P_q"]))
    bracket = float(p["B_h"]) + _period_component(p, dt[:, None]) + finite_past + singular_past
    mu_x = model.rho_ice + damping * bracket
    numerator = np.sum(model._gh_w_current[None, :] * mu_x * x, axis=1) / np.sqrt(np.pi)

    out = np.full_like(dh, np.nan, dtype=float)
    np.divide(numerator, dh, out=out, where=dh != 0)
    return out


def integrated_sigma_vectorized(model: RhoSurrogate, dh: np.ndarray, sigma_dh: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """Evaluate integrated surrogate uncertainties in density units.

    :param model: Effective-density surrogate
    :param dh: Elevation changes in metres
    :param sigma_dh: Elevation-change uncertainties in metres
    :param dt: Period lengths in years
    """
    variance_over_area2 = (
        float(model.params["U_h"]) ** 2 * expected_abs_normal_array(dh, sigma_dh)
        + float(model.params["U_t"]) ** 2 * np.maximum(dt, 0.0)
    )
    out = np.full_like(np.asarray(dh, dtype=float), np.inf, dtype=float)
    np.divide(np.sqrt(np.maximum(variance_over_area2, 0.0)), np.abs(dh), out=out, where=np.asarray(dh) != 0)
    return out


def integrated_sigma_mass_vectorized(
    model: RhoSurrogate,
    area_m2: np.ndarray,
    dh: np.ndarray,
    sigma_dh: np.ndarray,
    dt: np.ndarray,
) -> np.ndarray:
    """Evaluate density-related mass uncertainty for many rows.

    :param model: Effective-density surrogate
    :param area_m2: Glacier areas in square metres
    :param dh: Elevation changes in metres
    :param sigma_dh: Elevation-change uncertainties in metres
    :param dt: Period lengths in years
    """
    variance_over_area2 = (
        float(model.params["U_h"]) ** 2 * expected_abs_normal_array(dh, sigma_dh)
        + float(model.params["U_t"]) ** 2 * np.maximum(dt, 0.0)
    )
    return np.asarray(area_m2, dtype=float) * np.sqrt(np.maximum(variance_over_area2, 0.0))


def _past_mean_second_moments_vectorized(
    model: RhoSurrogate,
    past_dh: np.ndarray,
    sigma_past_dh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the past-change moments needed for ``E[mu]`` and ``E[mu^2]``."""
    y = past_dh[:, None] + np.sqrt(2.0) * sigma_past_dh[:, None] * model._gh_x_past[None, :]
    weights = model._gh_w_past[None, :] / np.sqrt(np.pi)

    # The mean function is linear in these two functions of past elevation
    # change. Their second moments are sufficient for the variance of the mean
    # mass contribution.
    signed_power = _signed_abs_power(y, float(model.params["P_p"]))
    tanh_term = np.tanh(y / float(model.params["H_q"]))
    mean_signed_power = np.sum(weights * signed_power, axis=1)
    mean_tanh = np.sum(weights * tanh_term, axis=1)
    mean_signed_power2 = np.sum(weights * signed_power**2, axis=1)
    mean_tanh2 = np.sum(weights * tanh_term**2, axis=1)
    mean_cross = np.sum(weights * signed_power * tanh_term, axis=1)
    return mean_signed_power, mean_tanh, mean_signed_power2, mean_tanh2, mean_cross


def mean_density_volume_moments_vectorized(
    model: RhoSurrogate,
    area_m2: np.ndarray,
    dh: np.ndarray,
    sigma_dh: np.ndarray,
    past_dh: np.ndarray,
    sigma_past_dh: np.ndarray,
    dt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate moments of ``mu_rho(X, Y) * DeltaV(X)``.

    ``X`` is the plausible true current elevation change and ``Y`` is the
    plausible true past elevation-change predictor. The first returned value is
    ``E[mu_rho(X, Y) * A * X]``. The second is the corresponding variance,
    ``Var[mu_rho(X, Y) * A * X]``.

    :param model: Effective-density surrogate
    :param area_m2: Glacier areas in square metres
    :param dh: Measured current elevation changes in metres
    :param sigma_dh: Current elevation-change uncertainties in metres
    :param past_dh: Measured past elevation-change predictors in metres
    :param sigma_past_dh: Past predictor uncertainties in metres
    :param dt: Period lengths in years
    """
    area_m2, dh, sigma_dh, past_dh, sigma_past_dh, dt = np.broadcast_arrays(
        np.asarray(area_m2, dtype=float),
        np.asarray(dh, dtype=float),
        np.asarray(sigma_dh, dtype=float),
        np.asarray(past_dh, dtype=float),
        np.asarray(sigma_past_dh, dtype=float),
        np.asarray(dt, dtype=float),
    )
    shape = dh.shape
    area = area_m2.ravel()
    dh = dh.ravel()
    sigma_dh = sigma_dh.ravel()
    past_dh = past_dh.ravel()
    sigma_past_dh = sigma_past_dh.ravel()
    dt = dt.ravel()

    p = model.params
    m_a, m_t, m_a2, m_t2, m_at = _past_mean_second_moments_vectorized(model, past_dh, sigma_past_dh)

    # Integrate over current elevation change. For fixed X, the Y dependence is
    # linear in two past-change functions, so E[mu^2 | X] follows from the five
    # past moments above without building an N x current_nodes x past_nodes cube.
    x = dh[:, None] + np.sqrt(2.0) * sigma_dh[:, None] * model._gh_x_current[None, :]
    weights_x = model._gh_w_current[None, :] / np.sqrt(np.pi)
    abs_safe = np.maximum(np.abs(x), 1.0e-12)
    sign_x = np.where(x < 0, -1.0, 1.0)
    damping = np.exp(-np.clip((abs_safe / float(p["H_d"])) ** float(p["P_d"]), 0.0, 700.0))

    base = float(p["B_h"]) + _period_component(p, dt[:, None])
    c = model.rho_ice + damping * base
    f = damping * float(p["B_p"]) * sign_x
    g = damping * float(p["B_q"]) * sign_x / (abs_safe ** float(p["P_q"]))

    mean_mu_given_x = c + f * m_a[:, None] + g * m_t[:, None]
    mean_mu2_given_x = (
        c**2
        + f**2 * m_a2[:, None]
        + g**2 * m_t2[:, None]
        + 2.0 * c * f * m_a[:, None]
        + 2.0 * c * g * m_t[:, None]
        + 2.0 * f * g * m_at[:, None]
    )

    first_over_area = np.sum(weights_x * mean_mu_given_x * x, axis=1)
    second_over_area2 = np.sum(weights_x * mean_mu2_given_x * x**2, axis=1)
    variance_over_area2 = np.maximum(second_over_area2 - first_over_area**2, 0.0)
    deterministic = (sigma_dh == 0) & (sigma_past_dh == 0)
    variance_over_area2[deterministic] = 0.0

    mean_mass = area * first_over_area
    variance_mass = area**2 * variance_over_area2
    return mean_mass.reshape(shape), variance_mass.reshape(shape)


def mean_density_volume_sigma_vectorized(
    model: RhoSurrogate,
    area_m2: np.ndarray,
    dh: np.ndarray,
    sigma_dh: np.ndarray,
    past_dh: np.ndarray,
    sigma_past_dh: np.ndarray,
    dt: np.ndarray,
) -> np.ndarray:
    """Evaluate ``sqrt(Var[mu_rho(X, Y) * DeltaV(X)])`` for many rows.

    :param model: Effective-density surrogate
    :param area_m2: Glacier areas in square metres
    :param dh: Measured current elevation changes in metres
    :param sigma_dh: Current elevation-change uncertainties in metres
    :param past_dh: Measured past elevation-change predictors in metres
    :param sigma_past_dh: Past predictor uncertainties in metres
    :param dt: Period lengths in years
    """
    _, variance = mean_density_volume_moments_vectorized(
        model,
        area_m2=area_m2,
        dh=dh,
        sigma_dh=sigma_dh,
        past_dh=past_dh,
        sigma_past_dh=sigma_past_dh,
        dt=dt,
    )
    return np.sqrt(np.maximum(variance, 0.0))


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
) -> pd.DataFrame:
    """Apply temporal reconciliation to a glacier-period table.

    The input must contain all observed periods to reconcile and the elementary
    periods from which they are built. The shortest period length in the table
    is treated as the elementary step.

    :param periods: Glacier-period table
    :param model: Effective-density surrogate
    :param id_col: Glacier identifier column
    :param start_col: Period start column
    :param end_col: Period end column
    :param dh_col: Elevation-change column
    :param area_col: Glacier area column in square metres
    :param dvol_col: Volume-change column in cubic metres
    :param mean_col: Independent surrogate mean-density column
    :param sigma_mass_col: Independent density-related mass uncertainty column
    """
    out = periods.copy()
    period_defs = sorted(out[[start_col, end_col]].drop_duplicates().itertuples(index=False, name=None))
    period_index = pd.MultiIndex.from_tuples(period_defs, names=[start_col, end_col])

    min_dt = np.min([float(end) - float(start) for start, end in period_defs])
    elem_defs = [(int(start), int(end)) for start, end in period_defs if np.isclose(float(end) - float(start), min_dt)]
    n_period = len(period_defs)
    n_elem = len(elem_defs)

    support_matrix = np.zeros((n_period, n_elem), dtype=float)
    for p_idx, (start, end) in enumerate(period_defs):
        for e_idx, (e_start, e_end) in enumerate(elem_defs):
            if e_start >= start and e_end <= end:
                support_matrix[p_idx, e_idx] = 1.0

    ids = out[[id_col]].drop_duplicates().reset_index(drop=True)
    out["period_key"] = pd.MultiIndex.from_frame(out[[start_col, end_col]])
    anomaly_series = ((out[mean_col] - model.rho_ice) * out[dvol_col]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    anomaly = (
        pd.DataFrame({id_col: out[id_col], "period_key": out["period_key"], "value": anomaly_series})
        .pivot(index=id_col, columns="period_key", values="value")
        .reindex(index=ids[id_col], columns=period_index)
        .fillna(0.0)
        .to_numpy(float)
    )

    elementary = (
        out.loc[np.isclose(out[end_col] - out[start_col], min_dt), [id_col, start_col, end_col, dh_col]]
        .assign(period_key=lambda d: pd.MultiIndex.from_frame(d[[start_col, end_col]]))
        .pivot(index=id_col, columns="period_key", values=dh_col)
        .reindex(index=ids[id_col], columns=pd.MultiIndex.from_tuples(elem_defs, names=[start_col, end_col]))
        .fillna(0.0)
        .abs()
        .to_numpy(float)
    )
    period_dt = np.array([float(end) - float(start) for start, end in period_defs], dtype=float)
    sigma_support = (
        float(model.params["U_h"]) ** 2 * (elementary @ support_matrix.T)
        + float(model.params["U_t"]) ** 2 * period_dt[None, :]
    )
    area = (
        pd.DataFrame({id_col: out[id_col], "period_key": out["period_key"], "value": out[area_col]})
        .pivot(index=id_col, columns="period_key", values="value")
        .reindex(index=ids[id_col], columns=period_index)
        .to_numpy(float)
    )
    finite_area = np.isfinite(area) & (area > 0)
    row_area = np.divide(
        np.sum(np.where(finite_area, area, 0.0), axis=1),
        np.sum(finite_area, axis=1),
        out=np.ones(area.shape[0], dtype=float),
        where=np.sum(finite_area, axis=1) > 0,
    )
    area = np.where(np.isfinite(area) & (area > 0), area, row_area[:, None])
    sigma = area * np.sqrt(np.maximum(sigma_support, 0.0))

    # The temporal fit is weighted by the same path-length variance used for
    # the closed uncertainty: composed periods accumulate elementary mass
    # variance along the absolute elevation-change path.
    finite = np.isfinite(sigma) & (sigma > 0)
    row_floor = np.nanmedian(np.where(finite, sigma, np.nan), axis=1)
    row_floor = np.where(np.isfinite(row_floor) & (row_floor > 0), row_floor * 1.0e-3, 1.0)
    sigma = np.where(finite, sigma, row_floor[:, None])
    sigma = np.maximum(sigma, row_floor[:, None])
    weights = 1.0 / sigma**2
    weight_scale = np.nanmax(weights, axis=1)
    weights = np.divide(
        weights,
        weight_scale[:, None],
        out=np.ones_like(weights),
        where=np.isfinite(weight_scale[:, None]) & (weight_scale[:, None] > 0),
    )

    normal = np.einsum("pa,np,pb->nab", support_matrix, weights, support_matrix)
    rhs = np.einsum("pa,np,np->na", support_matrix, weights, anomaly)
    normal += np.eye(n_elem)[None, :, :] * 1.0e-20
    elem_anomaly = np.linalg.solve(normal, rhs[..., None])[..., 0]
    closed = elem_anomaly @ support_matrix.T
    closed_df = (
        pd.DataFrame(closed, index=ids[id_col], columns=period_index)
        .stack([start_col, end_col])
        .rename("closed_anomaly_kg")
        .reset_index()
    )
    support_df = (
        pd.DataFrame(np.sqrt(np.maximum(sigma_support, 0.0)), index=ids[id_col], columns=period_index)
        .stack([start_col, end_col])
        .rename("sigma_support_per_area")
        .reset_index()
    )

    out = out.merge(closed_df, on=[id_col, start_col, end_col], how="left")
    out = out.merge(support_df, on=[id_col, start_col, end_col], how="left")
    out["dM_surrogate_kg"] = model.rho_ice * out[dvol_col].to_numpy(float) + out["closed_anomaly_kg"].to_numpy(float)
    out["mu_rho_surrogate_kg_m3"] = np.where(out[dvol_col] != 0, out["dM_surrogate_kg"] / out[dvol_col], np.nan)
    out["sigma_dM_rho_surrogate_kg"] = out[area_col].to_numpy(float) * out["sigma_support_per_area"].to_numpy(float)
    out["sigma_rho_surrogate_kg_m3"] = np.where(
        out[dvol_col] != 0,
        out["sigma_dM_rho_surrogate_kg"] / np.abs(out[dvol_col]),
        np.inf,
    )
    return out.drop(columns=["period_key", "closed_anomaly_kg", "sigma_support_per_area"])
