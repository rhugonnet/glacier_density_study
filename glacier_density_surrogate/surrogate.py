#!/usr/bin/env python3
"""Reusable effective-density surrogate model.

This module is the single downstream interface for the final surrogate:

* mean effective-density function;
* uncertainty function;
* spatial correlation function;
* temporal correlation function.

It accepts the parameter names written by the fitting scripts and
manuscript-style aliases used in parameter tables.
"""

from __future__ import annotations

import json
import math
import re
from importlib import resources
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss


DEFAULT_PARAMS: dict[str, float | str] = {
    "rho_ice": 900.0,
    "H_d": 16.7,
    "P_d": 0.69,
    "B_h": -139.0,
    "B_t": -314.0,
    "T_t": 2.74,
    "P0": 0.0,
    "P1": -314.0,
    "TP": 2.74,
    "period_form": "exp_no_offset",
    "B_p": 99.0,
    "P_p": 0.5,
    "B_q": 359.0,
    "H_q": 0.87,
    "P_q": 0.90,
    "T_p": 5.0,
    "tau_max": 20.0,
    "U_h": 95.0,
    "U_t": 105.0,
    "sigma_numeric_floor": 1.0,
    "spatial_corr_form": "exponential",
    "n_spatial_components": 3,
    "q0_nugget_fraction": 0.283,
    "q1_short_range_fraction": 0.353,
    "q2_long_range_fraction": 0.340,
    "q1_range_fraction": 0.353,
    "q2_range_fraction": 0.340,
    "q3_range_fraction": 0.024,
    "r1_km": 148.0,
    "r2_km": 752.0,
    "r3_km": 20000.0,
    "temporal_nugget": 0.70,
    "temporal_timescale_yr": 3.0,
    "temporal_exponent": 1.0,
    "temporal_model_form": "nugget_exponential",
}


def load_packaged_params() -> dict[str, float | bool | str]:
    """Load the bundled finalized surrogate parameters

    :returns: Final mean, uncertainty, spatial and temporal parameters
    """
    try:
        path = resources.files("glacier_density_surrogate").joinpath("parameters/final_parameters.json")
        with path.open() as f:
            payload = json.load(f)
        params = DEFAULT_PARAMS.copy()
        params.update(payload)
        return params
    except (FileNotFoundError, ModuleNotFoundError, json.JSONDecodeError):
        return DEFAULT_PARAMS.copy()


def write_packaged_params(
    parameter_path: Path | str | None = None,
    spatial_path: Path | str | None = None,
    temporal_path: Path | str | None = None,
    out_path: Path | str | None = None,
) -> Path:
    """Write the bundled finalized parameter JSON from fit outputs

    :param parameter_path: Optional mean and uncertainty parameter file
    :param spatial_path: Optional spatial correlation parameter file
    :param temporal_path: Optional temporal correlation parameter file
    :param out_path: Optional output JSON path
    """
    params = load_packaged_params()
    if parameter_path is not None:
        params.update(load_params(parameter_path, verbose=False))
    if spatial_path is not None or temporal_path is not None:
        params.update(load_correlation_params(spatial_path, temporal_path, verbose=False))
    if out_path is None:
        out_path = Path(__file__).resolve().parent / "parameters" / "final_parameters.json"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(params, f, indent=2, sort_keys=True)
    return out_path


def make_dh_error_correlation(form: str | None = None, range_years: float | None = None) -> Callable[[np.ndarray], np.ndarray]:
    """Build a temporal correlation function for elevation-change errors

    :param form: Correlation form, one of none, exponential, gaussian or spherical
    :param range_years: Correlation range in years
    """
    name = "none" if form is None else str(form).lower()
    if name in {"none", "uncorrelated", "independent"} or range_years is None or range_years <= 0:
        return lambda lag: np.where(np.asarray(lag, dtype=float) == 0, 1.0, 0.0)
    if name in {"exp", "exponential"}:
        return lambda lag: np.exp(-3.0 * np.maximum(np.asarray(lag, dtype=float), 0.0) / float(range_years))
    if name in {"gauss", "gaussian"}:
        return lambda lag: np.exp(-3.0 * (np.maximum(np.asarray(lag, dtype=float), 0.0) / float(range_years)) ** 2)
    if name == "spherical":
        def spherical(lag: np.ndarray) -> np.ndarray:
            h = np.maximum(np.asarray(lag, dtype=float), 0.0) / float(range_years)
            return np.where(h < 1.0, 1.0 - 1.5 * h + 0.5 * h**3, 0.0)
        return spherical
    raise ValueError(f"Unsupported elevation-change error correlation form: {form}")


def _spatial_correlation_component(distance_km: np.ndarray | float, range_km: float, form: str) -> np.ndarray:
    """Evaluate a standard spatial correlation component"""
    h = np.maximum(np.asarray(distance_km, dtype=float), 0.0) / max(float(range_km), 1.0e-12)
    name = str(form).lower()
    if name in {"exp", "exponential"}:
        return np.exp(-3.0 * h)
    if name in {"gauss", "gaussian"}:
        return np.exp(-3.0 * h**2)
    if name == "spherical":
        return np.where(h < 1.0, 1.0 - 1.5 * h + 0.5 * h**3, 0.0)
    raise ValueError(f"Unsupported spatial correlation form: {form}")


def _first_existing(columns: pd.Index, candidates: list[str], required: bool = True) -> str | None:
    for col in candidates:
        if col in columns:
            return col
    if required:
        raise KeyError(f"None of these columns exist: {candidates}")
    return None


def _normal_error_covariance(sigmas: np.ndarray, times: np.ndarray, corr: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    lag = np.abs(np.asarray(times, dtype=float)[:, None] - np.asarray(times, dtype=float)[None, :])
    return np.asarray(corr(lag), dtype=float) * np.outer(sigmas, sigmas)


ALIASES: dict[str, tuple[str, ...]] = {
    "rho_ice": ("rho_ice", "rhoice", "rho_i", "rhoi", "rho_ice_fixed"),
    "H_d": ("H_d", "Hd", "Hdef", "H_damp", "LD", "L_D", "H"),
    "P_d": ("P_d", "Pd", "Pdef", "P_damp", "nD", "n_D", "beta"),
    "B_h": ("B_h", "Bh", "D0", "B0", "Bc"),
    "B_t": ("B_t", "Bt", "C1"),
    "T_t": ("T_t", "Tt", "T0"),
    "P0": ("P0",),
    "P1": ("P1",),
    "TP": ("TP", "period_TP"),
    "period_form": ("period_form", "period", "spec.period"),
    "B_p": ("B_p", "Bp", "Bmem"),
    "P_p": ("P_p", "Pp", "etaMem", "memory_power"),
    "B_q": ("B_q", "Bq", "A", "Aq"),
    "H_q": ("H_q", "Hq", "LA", "L_A"),
    "P_q": ("P_q", "Pq", "alpha", "alpha_q"),
    "T_p": ("T_p", "Tp", "tau_p", "tau_mem", "T_mem", "memory_tau_years", "memory_window_years"),
    "tau_max": ("tau_max", "taumax", "lookback", "lookback_max"),
    "U_h": ("U_h", "Uh", "A0", "sigma_Uh"),
    "U_t": ("U_t", "Ut", "A1", "sigma_Ut"),
    "sigma_numeric_floor": ("sigma_numeric_floor",),
}


def _norm_name(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def expected_abs_normal(mean: float, sigma: float) -> float:
    """Return E|X| for X ~ N(mean, sigma^2)

    :param mean: Normal mean
    :param sigma: Normal standard deviation
    """
    if sigma == 0:
        return abs(mean)
    return (
        sigma * math.sqrt(2.0 / math.pi) * math.exp(-(mean**2) / (2.0 * sigma**2))
        + mean * math.erf(mean / (math.sqrt(2.0) * sigma))
    )


def _signed_abs_power(x: np.ndarray | float, power: float) -> np.ndarray:
    """Return ``sign(x) * |x|**power`` with stable zero handling."""
    x_arr = np.asarray(x, dtype=float)
    return np.sign(x_arr) * np.abs(x_arr) ** float(power)


def _period_component(params: dict[str, float | bool | str], dt: np.ndarray | float) -> np.ndarray:
    """Evaluate the retained or legacy period-length component."""
    dt_arr = np.asarray(dt, dtype=float)
    form = str(params.get("period_form", "exp_no_offset"))
    if form in {"exp_no_offset", "exp", "exponential"}:
        return float(params["B_t"]) * np.exp(-dt_arr / float(params["T_t"]))
    if form == "power_param":
        return float(params["P0"]) + float(params["P1"]) / (1.0 + dt_arr / float(params["TP"]))
    if form in {"none", "zero"}:
        return np.zeros_like(dt_arr, dtype=float)
    raise ValueError(f"Unsupported period form: {form}")


def _read_long_or_wide_csv(path: Path) -> tuple[dict[str, float], dict[str, str]]:
    df = pd.read_csv(path)
    numeric: dict[str, float] = {}
    text: dict[str, str] = {}
    if df.empty:
        return numeric, text

    cols_norm = {_norm_name(c): c for c in df.columns}
    name_col = next((cols_norm[c] for c in ("parameter", "param", "name", "term", "variable") if c in cols_norm), None)
    value_col = next(
        (cols_norm[c] for c in ("valuenumeric", "value", "estimate", "fit", "coef", "coefficient") if c in cols_norm),
        None,
    )
    text_col = next((cols_norm[c] for c in ("valuetext", "text") if c in cols_norm), None)

    if name_col is not None and (value_col is not None or text_col is not None):
        for _, row in df.iterrows():
            key = _norm_name(row[name_col])
            if not key:
                continue
            if value_col is not None:
                value = pd.to_numeric(row[value_col], errors="coerce")
                if np.isfinite(value):
                    numeric[key] = float(value)
            if text_col is not None and pd.notna(row[text_col]) and str(row[text_col]) != "":
                text[key] = str(row[text_col])
        return numeric, text

    if len(df.columns) >= 2 and len(df) > 1:
        first, second = df.columns[:2]
        first_values = pd.to_numeric(df[first], errors="coerce")
        values = pd.to_numeric(df[second], errors="coerce")
        if values.notna().any() and not first_values.notna().any():
            for name, value in zip(df[first], values):
                key = _norm_name(name)
                if key and np.isfinite(value):
                    numeric[key] = float(value)
            if numeric:
                return numeric, text

    first_row = df.iloc[0]
    for col, raw in first_row.items():
        key = _norm_name(col)
        value = pd.to_numeric(raw, errors="coerce")
        if np.isfinite(value):
            numeric[key] = float(value)
        elif pd.notna(raw):
            text[key] = str(raw)
    return numeric, text


def _read_json(path: Path) -> tuple[dict[str, float], dict[str, str]]:
    with path.open() as f:
        payload = json.load(f)
    numeric: dict[str, float] = {}
    text: dict[str, str] = {}

    def walk(obj: object, prefix: str = "") -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_prefix = f"{prefix}.{key}" if prefix else str(key)
                walk(value, new_prefix)
            return
        key = _norm_name(prefix.split(".")[-1])
        if isinstance(obj, (int, float)) and np.isfinite(float(obj)):
            numeric[key] = float(obj)
        elif isinstance(obj, str):
            text[key] = obj

    walk(payload)
    return numeric, text


def load_params(path: Path | str | None = None, *, verbose: bool = True) -> dict[str, float | str]:
    """Load final mean/std parameters with defaults for missing values

    :param path: Parameter CSV or JSON path
    :param verbose: Print loading diagnostics
    """
    params = load_packaged_params()
    if path is None:
        return params

    path = Path(path)
    if not path.exists():
        if verbose:
            print(f"[params] Parameter file not found; using bundled defaults: {path}")
        return params

    raw_num, raw_text = _read_json(path) if path.suffix.lower() == ".json" else _read_long_or_wide_csv(path)
    used: set[str] = set()

    # Parameter tables written by the fitting script carry model-form metadata
    # as text rows such as spec.period = power_param. Preserve it before
    # applying numeric aliases, because TP can otherwise be confused with the
    # historical memory-time alias Tp.
    for key in ("specperiod", "period", "periodform"):
        if key in raw_text:
            params["period_form"] = raw_text[key]
            used.add("period_form")
            break

    for canonical, aliases in ALIASES.items():
        for alias in aliases:
            key = _norm_name(alias)
            if canonical == "T_p" and key == "tp" and str(params.get("period_form")) == "power_param":
                continue
            if key in raw_num:
                params[canonical] = raw_num[key]
                used.add(canonical)
                break
            if key in raw_text:
                params[canonical] = raw_text[key]
                used.add(canonical)
                break

    if verbose:
        print(f"[params] Loaded {len(used)} mean/std parameters from {path}")
    return params


def load_correlation_params(
    spatial_path: Path | str | None = None,
    temporal_path: Path | str | None = None,
    *,
    verbose: bool = True,
) -> dict[str, float | bool | str]:
    """Load spatial and temporal correlation parameters

    :param spatial_path: Spatial parameter CSV or JSON path
    :param temporal_path: Temporal parameter CSV or JSON path
    :param verbose: Print loading diagnostics
    """
    params: dict[str, float | bool | str] = {}

    def update_from_table(path: Path, rename: dict[str, str]) -> None:
        if path.suffix.lower() == ".json":
            numeric, text = _read_json(path)
        else:
            numeric, text = _read_long_or_wide_csv(path)
        for key, value in {**numeric, **text}.items():
            canonical = rename.get(key, key)
            params[canonical] = value

    if spatial_path is not None and Path(spatial_path).exists():
        update_from_table(
            Path(spatial_path),
            {
                "alphan": "alpha_n",
                "betan": "beta_n",
                "alphas": "alpha_s",
                "betas": "beta_s",
                "q0nuggetfraction": "q0_nugget_fraction",
                "q1shortrangefraction": "q1_short_range_fraction",
                "q2longrangefraction": "q2_long_range_fraction",
                "q1rangefraction": "q1_range_fraction",
                "q2rangefraction": "q2_range_fraction",
                "q3rangefraction": "q3_range_fraction",
                "spatialcorrform": "spatial_corr_form",
                "nspatialcomponents": "n_spatial_components",
                "r1km": "r1_km",
                "r2km": "r2_km",
                "r3km": "r3_km",
            },
        )
        if verbose:
            print(f"[params] Loaded spatial correlation parameters from {spatial_path}")
    elif verbose and spatial_path is not None:
        print(f"[params] Spatial correlation file not found; using packaged defaults: {spatial_path}")

    if temporal_path is not None and Path(temporal_path).exists():
        update_from_table(
            Path(temporal_path),
            {
                "nugget": "temporal_nugget",
                "empiricalnugget": "temporal_nugget",
                "timescaleyr": "temporal_timescale_yr",
                "empiricaltimescaleyr": "temporal_timescale_yr",
                "empiricalrangeyr": "temporal_timescale_yr",
                "exponent": "temporal_exponent",
                "empiricalexponent": "temporal_exponent",
                "empiricalamplitudeafternugget": "amplitude_after_nugget",
                "temporalsill": "amplitude_after_nugget",
                "empiricalsill": "amplitude_after_nugget",
                "temporalmodelform": "temporal_model_form",
                "appliedtemporalcorrelationatpositivelag": "applied_temporal_correlation_at_positive_lag",
                "appliedtemporalcovarianceassumption": "applied_temporal_covariance_assumption",
                "temporalfinalsilllagyears": "temporal_final_sill_lag_years",
                "temporalforcefinalsill": "temporal_force_final_sill",
            },
        )
        if verbose:
            print(f"[params] Loaded temporal correlation parameters from {temporal_path}")
    elif verbose and temporal_path is not None:
        print(f"[params] Temporal correlation file not found; using packaged defaults: {temporal_path}")

    return params


@dataclass
class RhoSurrogate:
    """Final effective-density surrogate model."""

    params: dict[str, float | bool | str] = field(default_factory=load_packaged_params)
    gh_order_current: int = 64
    gh_order_past: int = 64

    def __post_init__(self) -> None:
        merged = load_packaged_params()
        merged.update(self.params)
        self.params = merged
        if self.gh_order_current % 2 == 1:
            self.gh_order_current += 1
        if self.gh_order_past % 2 == 1:
            self.gh_order_past += 1
        self._gh_x_current, self._gh_w_current = hermgauss(self.gh_order_current)
        self._gh_x_past, self._gh_w_past = hermgauss(self.gh_order_past)

    @classmethod
    def from_files(
        cls,
        parameter_path: Path | str | None = None,
        spatial_path: Path | str | None = None,
        temporal_path: Path | str | None = None,
        **kwargs,
    ) -> "RhoSurrogate":
        """Build a surrogate from parameter files

        :param parameter_path: Mean and uncertainty parameter file
        :param spatial_path: Spatial correlation parameter file
        :param temporal_path: Temporal correlation parameter file
        """
        params = load_params(parameter_path)
        params.update(load_correlation_params(spatial_path, temporal_path))
        return cls(params=params, **kwargs)

    @property
    def rho_ice(self) -> float:
        return float(self.params["rho_ice"])

    def integrate_normal(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        mean: float,
        sigma: float,
        *,
        use_past_nodes: bool = False,
    ) -> float:
        """Integrate a function against a normal distribution

        :param func: Vectorized function to integrate
        :param mean: Normal mean
        :param sigma: Normal standard deviation
        :param use_past_nodes: Use the past-dh quadrature order
        """
        if sigma < 0 or not np.isfinite(sigma):
            raise ValueError(f"Invalid sigma: {sigma}")
        if sigma == 0:
            return float(np.asarray(func(np.array([mean], dtype=float)))[0])
        gh_x = self._gh_x_past if use_past_nodes else self._gh_x_current
        gh_w = self._gh_w_past if use_past_nodes else self._gh_w_current
        x = mean + math.sqrt(2.0) * sigma * gh_x
        return float(np.sum(gh_w * np.asarray(func(x), dtype=float)) / math.sqrt(math.pi))

    def mu_rho(self, dh: np.ndarray | float, dh_p: np.ndarray | float = 0.0, dt: float = 1.0) -> np.ndarray:
        """Mean effective density for exact current and past elevation change

        :param dh: Current elevation change in metres
        :param dh_p: Past elevation change in metres
        :param dt: Observation period in years
        """
        p = self.params
        dh_arr = np.asarray(dh, dtype=float)
        dh_p_arr = np.asarray(dh_p, dtype=float)
        dt_arr = np.asarray(dt, dtype=float)

        # The damping term makes the mean converge toward ice density for large
        # absolute elevation changes, where firn-volume changes are secondary.
        abs_safe = np.maximum(np.abs(dh_arr), 1.0e-12)
        sign_dh = np.where(dh_arr < 0, -1.0, 1.0)
        damping = np.exp(-np.clip((abs_safe / float(p["H_d"])) ** float(p["P_d"]), 0.0, 700.0))

        # Past elevation change has one finite-memory term and one ratio-like
        # term that represents the divergence near zero current volume change.
        finite_past = float(p["B_p"]) * sign_dh * _signed_abs_power(dh_p_arr, float(p["P_p"]))
        singular_past = float(p["B_q"]) * np.tanh(sign_dh * dh_p_arr / float(p["H_q"])) / (abs_safe ** float(p["P_q"]))
        bracket = float(p["B_h"]) + _period_component(p, dt_arr) + finite_past + singular_past
        return self.rho_ice + damping * bracket

    def sigma_rho(self, dh: np.ndarray | float, dt: float = 1.0) -> np.ndarray:
        """Surrogate uncertainty in effective-density units

        :param dh: Current elevation change in metres
        :param dt: Observation period in years
        """
        p = self.params
        abs_safe = np.maximum(np.abs(np.asarray(dh, dtype=float)), 1.0e-12)
        dt_arr = np.maximum(np.asarray(dt, dtype=float), 0.0)

        # Final retained form. It is written as a density uncertainty, but it
        # corresponds to additive mass-change variance after multiplying by dV.
        out = np.sqrt(float(p["U_h"]) ** 2 / abs_safe + float(p["U_t"]) ** 2 * dt_arr / abs_safe**2)
        return np.maximum(out, float(p.get("sigma_numeric_floor", 1.0)))

    def _past_moments(self, dh_p: float, sigma_dh_p: float) -> tuple[float, float]:
        """Integrate past-change moments used by the mean model

        :param dh_p: Measured past elevation change
        :param sigma_dh_p: Past elevation-change uncertainty
        """
        signed_power = self.integrate_normal(
            lambda y: _signed_abs_power(y, float(self.params["P_p"])),
            dh_p,
            sigma_dh_p,
            use_past_nodes=True,
        )
        tanh_moment = self.integrate_normal(
            lambda y: np.tanh(y / float(self.params["H_q"])),
            dh_p,
            sigma_dh_p,
            use_past_nodes=True,
        )
        return signed_power, tanh_moment

    def mean_integrated_over_past(
        self,
        dh: np.ndarray | float,
        dh_p: float,
        sigma_dh_p: float,
        dt: float,
        *,
        past_moments: tuple[float, float] | None = None,
    ) -> np.ndarray:
        """Mean model integrated over uncertain past elevation change

        :param dh: Current elevation change values
        :param dh_p: Measured past elevation change
        :param sigma_dh_p: Past elevation-change uncertainty
        :param dt: Observation period in years
        :param past_moments: Precomputed past moments
        """
        p = self.params
        dh_arr = np.asarray(dh, dtype=float)
        abs_safe = np.maximum(np.abs(dh_arr), 1.0e-12)
        sign_dh = np.where(dh_arr < 0, -1.0, 1.0)
        damping = np.exp(-np.clip((abs_safe / float(p["H_d"])) ** float(p["P_d"]), 0.0, 700.0))
        signed_power, tanh_moment = past_moments if past_moments is not None else self._past_moments(dh_p, sigma_dh_p)
        finite_past = float(p["B_p"]) * sign_dh * signed_power
        singular_past = float(p["B_q"]) * sign_dh * tanh_moment / (abs_safe ** float(p["P_q"]))
        bracket = float(p["B_h"]) + _period_component(p, float(dt)) + finite_past + singular_past
        return self.rho_ice + damping * bracket

    def integrated_mu(
        self,
        dh: float,
        sigma_dh: float = 0.0,
        dh_p: float = 0.0,
        sigma_dh_p: float = 0.0,
        dt: float = 1.0,
    ) -> float:
        """Mass-weighted mean density after integrating over uncertain dh

        :param dh: Measured current elevation change
        :param sigma_dh: Current elevation-change uncertainty
        :param dh_p: Measured past elevation change
        :param sigma_dh_p: Past elevation-change uncertainty
        :param dt: Observation period in years
        """
        if dh == 0:
            return np.nan
        moments = self._past_moments(dh_p, sigma_dh_p)
        numerator = self.integrate_normal(
            lambda x: self.mean_integrated_over_past(x, dh_p, sigma_dh_p, dt, past_moments=moments) * x,
            dh,
            sigma_dh,
        )
        return numerator / dh

    def integrated_sigma(self, dh: float, sigma_dh: float = 0.0, dt: float = 1.0) -> float:
        """Integrated surrogate uncertainty for uncertain current elevation change

        :param dh: Measured current elevation change
        :param sigma_dh: Current elevation-change uncertainty
        :param dt: Observation period in years
        """
        if dh == 0:
            return np.inf
        p = self.params
        variance_over_area2 = float(p["U_h"]) ** 2 * expected_abs_normal(dh, sigma_dh) + float(p["U_t"]) ** 2 * max(float(dt), 0.0)
        return math.sqrt(max(variance_over_area2, 0.0)) / abs(dh)

    def integrated_sigma_equiv_density(
        self,
        dh: float,
        sigma_dh: float = 0.0,
        dt: float = 1.0,
        area_m2: float = 1.0,
    ) -> tuple[float, float]:
        """Return integrated density and mass uncertainty for a period

        :param dh: Measured current elevation change
        :param sigma_dh: Current elevation-change uncertainty
        :param dt: Observation period in years
        :param area_m2: Glacier area in square metres
        """
        sigma_rho = self.integrated_sigma(dh=dh, sigma_dh=sigma_dh, dt=dt)
        sigma_mass = sigma_rho * abs(area_m2 * dh) if np.isfinite(sigma_rho) else np.inf
        return sigma_rho, sigma_mass


    def predict(
        self,
        dh: float,
        sigma_dh: float = 0.0,
        dt: float = 1.0,
        past_dh: float | None = None,
        sigma_past_dh: float | None = None,
        area_m2: float | None = None,
        past_missing: str = "current",
        past_error_factor: float = 2.0,
    ) -> dict[str, float]:
        """Predict effective density for one observation period

        :param dh: Glacier-wide elevation change over the period in metres
        :param sigma_dh: One-sigma uncertainty of dh in metres
        :param dt: Period length in years
        :param past_dh: Past elevation change predictor in metres
        :param sigma_past_dh: One-sigma uncertainty of past_dh in metres
        :param area_m2: Optional glacier area for volume and mass outputs
        :param past_missing: Assumption if past_dh is missing, current or zero
        :param past_error_factor: Multiplier for default sigma_past_dh
        """
        dh = float(dh)
        sigma_dh = float(0.0 if sigma_dh is None else sigma_dh)
        dt = float(dt)
        if past_dh is None or not np.isfinite(past_dh):
            if past_missing == "zero":
                past_dh = 0.0
            elif past_missing == "current":
                past_dh = dh / dt
            else:
                raise ValueError("past_missing must be 'current' or 'zero'")
        if sigma_past_dh is None or not np.isfinite(sigma_past_dh):
            sigma_past_dh = past_error_factor * sigma_dh / dt
        mu = self.integrated_mu(dh=dh, sigma_dh=sigma_dh, dh_p=float(past_dh), sigma_dh_p=float(sigma_past_dh), dt=dt)
        sigma = self.integrated_sigma(dh=dh, sigma_dh=sigma_dh, dt=dt)
        out = {
            "dh_m": dh,
            "sigma_dh_m": sigma_dh,
            "period_years": dt,
            "past_dh_m": float(past_dh),
            "sigma_past_dh_m": float(sigma_past_dh),
            "mu_rho_kg_m3": float(mu),
            "sigma_rho_kg_m3": float(sigma),
            "temporally_closed": False,
        }
        if area_m2 is not None:
            dV = float(area_m2) * dh
            out.update(
                {
                    "area_m2": float(area_m2),
                    "dV_m3": dV,
                    "dM_kg": float(mu * dV),
                    "sigma_dM_rho_kg": float(sigma * abs(dV)),
                }
            )
        return out

    def predict_timeseries(
        self,
        data: pd.DataFrame,
        *,
        start_col: str | None = None,
        end_col: str | None = None,
        dh_col: str | None = None,
        sigma_dh_col: str | None = None,
        dt_col: str | None = None,
        past_dh_col: str | None = None,
        sigma_past_dh_col: str | None = None,
        area_col: str | None = None,
        area_m2: float | None = None,
        expand_periods: bool = True,
        past_missing: str = "current",
        past_error_factor: float = 2.0,
        dh_error_corr: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> pd.DataFrame:
        """Predict effective density for an elevation-change time series

        :param data: Input period table
        :param start_col: Period start column
        :param end_col: Period end column
        :param dh_col: Elevation-change column
        :param sigma_dh_col: Elevation-change uncertainty column
        :param dt_col: Period length column
        :param past_dh_col: Optional past elevation-change column
        :param sigma_past_dh_col: Optional past uncertainty column
        :param area_col: Optional area column
        :param area_m2: Constant area for all rows
        :param expand_periods: Expand consecutive rows to all contiguous periods
        :param past_missing: Assumption if no past period exists
        :param past_error_factor: Multiplier for default past uncertainty
        :param dh_error_corr: Correlation function for elevation-change errors
        """
        table = self._prepare_input_table(
            data,
            start_col=start_col,
            end_col=end_col,
            dh_col=dh_col,
            sigma_dh_col=sigma_dh_col,
            dt_col=dt_col,
            past_dh_col=past_dh_col,
            sigma_past_dh_col=sigma_past_dh_col,
            area_col=area_col,
            area_m2=area_m2,
        )
        corr = dh_error_corr or make_dh_error_correlation("none")
        periods = self._expand_period_table(table, corr) if expand_periods and self._is_elementary_series(table) else table.copy()
        periods = self._attach_past_predictor(periods, table, corr, past_missing, past_error_factor)
        rows = []
        for row in periods.itertuples(index=False):
            rows.append(
                self.predict(
                    dh=row.dh_m,
                    sigma_dh=row.sigma_dh_m,
                    dt=row.period_years,
                    past_dh=row.past_dh_m,
                    sigma_past_dh=row.sigma_past_dh_m,
                    area_m2=row.area_m2,
                    past_missing=past_missing,
                    past_error_factor=past_error_factor,
                )
            )
        out = pd.concat([periods.reset_index(drop=True), pd.DataFrame(rows).add_prefix("raw_")], axis=1)
        out["mu_rho_independent_kg_m3"] = out["raw_mu_rho_kg_m3"]
        out["sigma_rho_independent_kg_m3"] = out["raw_sigma_rho_kg_m3"]
        out["dV_m3"] = out["area_m2"] * out["dh_m"]
        out["dM_independent_kg"] = out["mu_rho_independent_kg_m3"] * out["dV_m3"]
        out["sigma_dM_rho_independent_kg"] = out["sigma_rho_independent_kg_m3"] * np.abs(out["dV_m3"])
        if len(out) > 1 and {"start", "end"}.issubset(out.columns):
            out = self._temporal_closure(out)
        else:
            out["mu_rho_closed_kg_m3"] = out["mu_rho_independent_kg_m3"]
            out["sigma_rho_closed_kg_m3"] = out["sigma_rho_independent_kg_m3"]
            out["dM_closed_kg"] = out["dM_independent_kg"]
            out["temporally_closed"] = False
        keep = [
            "start",
            "end",
            "period_years",
            "dh_m",
            "sigma_dh_m",
            "past_dh_m",
            "sigma_past_dh_m",
            "area_m2",
            "mu_rho_closed_kg_m3",
            "sigma_rho_closed_kg_m3",
            "dV_m3",
            "dM_closed_kg",
            "mu_rho_independent_kg_m3",
            "sigma_rho_independent_kg_m3",
            "dM_independent_kg",
            "temporally_closed",
        ]
        return out[[c for c in keep if c in out.columns]].copy()

    def _prepare_input_table(
        self,
        data: pd.DataFrame,
        *,
        start_col: str | None,
        end_col: str | None,
        dh_col: str | None,
        sigma_dh_col: str | None,
        dt_col: str | None,
        past_dh_col: str | None,
        sigma_past_dh_col: str | None,
        area_col: str | None,
        area_m2: float | None,
    ) -> pd.DataFrame:
        """Standardize user input columns to the internal period-table names."""
        cols = data.columns

        # Accept several common column names so the CLI can be used with
        # lightweight CSV files without forcing an exact internal schema.
        dh_col = dh_col or _first_existing(cols, ["dh_m", "dh", "elevation_change_m", "delta_h_m"])
        sigma_dh_col = sigma_dh_col or _first_existing(cols, ["sigma_dh_m", "sig_dh_m", "dh_uncertainty_m", "sigma_dh"], required=False)
        start_col = start_col or _first_existing(cols, ["start", "start_year", "year0", "period_start"], required=False)
        end_col = end_col or _first_existing(cols, ["end", "end_year", "year1", "period_end"], required=False)
        dt_col = dt_col or _first_existing(cols, ["dt", "dt_yr", "period_years", "duration_yr"], required=False)
        past_dh_col = past_dh_col or _first_existing(cols, ["past_dh_m", "past_dh", "dh_p_m", "dh_p"], required=False)
        sigma_past_dh_col = sigma_past_dh_col or _first_existing(cols, ["sigma_past_dh_m", "sig_past_dh_m", "sigma_dh_p_m"], required=False)
        area_col = area_col or _first_existing(cols, ["area_m2", "area", "glacier_area_m2"], required=False)
        out = pd.DataFrame({"dh_m": pd.to_numeric(data[dh_col], errors="coerce")})
        out["sigma_dh_m"] = pd.to_numeric(data[sigma_dh_col], errors="coerce") if sigma_dh_col else 0.0

        # Either explicit start/end years or a duration can define the period.
        # Synthetic start/end values are enough when temporal closure is unused.
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
        out["past_dh_m"] = pd.to_numeric(data[past_dh_col], errors="coerce") if past_dh_col else np.nan
        out["sigma_past_dh_m"] = pd.to_numeric(data[sigma_past_dh_col], errors="coerce") if sigma_past_dh_col else np.nan
        if area_col:
            out["area_m2"] = pd.to_numeric(data[area_col], errors="coerce")
        else:
            out["area_m2"] = 1.0 if area_m2 is None else float(area_m2)

        # Invalid rows are dropped here rather than failing late inside the
        # numerical routines, which makes CSV use easier to diagnose.
        ok = np.isfinite(out["dh_m"]) & np.isfinite(out["sigma_dh_m"]) & np.isfinite(out["period_years"]) & (out["period_years"] > 0)
        ok &= np.isfinite(out["area_m2"]) & (out["area_m2"] > 0)
        return out.loc[ok].sort_values(["start", "end"]).reset_index(drop=True)

    def _is_elementary_series(self, table: pd.DataFrame) -> bool:
        ordered = table.sort_values(["start", "end"]).reset_index(drop=True)
        if len(ordered) <= 1:
            return False
        return bool(np.allclose(ordered["start"].to_numpy(float)[1:], ordered["end"].to_numpy(float)[:-1]))

    def _expand_period_table(self, table: pd.DataFrame, corr: Callable[[np.ndarray], np.ndarray]) -> pd.DataFrame:
        """Expand consecutive elementary periods to all contiguous periods."""
        elementary = table.sort_values(["start", "end"]).reset_index(drop=True)
        mids = 0.5 * (elementary["start"].to_numpy(float) + elementary["end"].to_numpy(float))
        sigma = elementary["sigma_dh_m"].to_numpy(float)
        covariance = _normal_error_covariance(sigma, mids, corr)
        rows = []

        # Every pair of elementary bounds defines one contiguous observation
        # period. The uncertainty is propagated through the supplied dh-error
        # covariance, so correlated elevation errors can be represented.
        for i0 in range(len(elementary)):
            for i1 in range(i0 + 1, len(elementary) + 1):
                sl = slice(i0, i1)
                sub = elementary.iloc[sl]
                dh = float(sub["dh_m"].sum())
                ones = np.ones(i1 - i0)
                sig = float(np.sqrt(max(ones @ covariance[sl, sl] @ ones, 0.0)))
                rows.append(
                    {
                        "start": float(sub["start"].iloc[0]),
                        "end": float(sub["end"].iloc[-1]),
                        "period_years": float(sub["end"].iloc[-1] - sub["start"].iloc[0]),
                        "dh_m": dh,
                        "sigma_dh_m": sig,
                        "area_m2": float(sub["area_m2"].iloc[0]),
                    }
                )
        return pd.DataFrame(rows)

    def _attach_past_predictor(
        self,
        periods: pd.DataFrame,
        elementary: pd.DataFrame,
        corr: Callable[[np.ndarray], np.ndarray],
        past_missing: str,
        past_error_factor: float,
    ) -> pd.DataFrame:
        """Attach the exponentially weighted past elevation-change predictor."""
        out = periods.copy()
        tau = float(self.params.get("T_p", 5.0))
        tau_max = float(self.params.get("tau_max", 20.0))
        elem = elementary.sort_values(["start", "end"]).reset_index(drop=True)

        # The memory predictor is defined as an exponentially weighted annual
        # elevation-change rate. Multi-annual elementary inputs are therefore
        # represented by constant-rate annual substeps instead of one sample at
        # the period end. This keeps the predictor invariant to the chosen
        # elementary period length when the underlying rate is constant.
        annual_rows = []
        for elem_idx, row in elem.iterrows():
            start = float(row["start"])
            end = float(row["end"])
            dt = float(row["period_years"])
            if not np.isfinite(dt) or dt <= 0:
                continue
            n_sub = max(1, int(round(dt)))
            edges = np.linspace(start, end, n_sub + 1)
            rate = float(row["dh_m"]) / dt
            sigma_rate = float(row["sigma_dh_m"]) / dt
            for i in range(n_sub):
                annual_rows.append(
                    {
                        "elem_idx": int(elem_idx),
                        "start": float(edges[i]),
                        "end": float(edges[i + 1]),
                        "mid": float(0.5 * (edges[i] + edges[i + 1])),
                        "rate": rate,
                        "sigma_rate": sigma_rate,
                    }
                )
        annual = pd.DataFrame(annual_rows)
        if annual.empty:
            out["past_dh_m"] = out["past_dh_m"].fillna(out["dh_m"] / out["period_years"])
            out["sigma_past_dh_m"] = out["sigma_past_dh_m"].fillna(past_error_factor * out["sigma_dh_m"] / out["period_years"])
            return out

        mids = annual["mid"].to_numpy(float)
        elem_rate = annual["rate"].to_numpy(float)
        elem_sigma_rate = annual["sigma_rate"].to_numpy(float)
        elem_ids = annual["elem_idx"].to_numpy(int)
        base_covariance = _normal_error_covariance(elem_sigma_rate, mids, corr)

        # Annual substeps from the same coarse observation share the same
        # endpoint-derived error. Their correlation is therefore one, otherwise
        # the provided temporal correlation controls the covariance.
        same_input = elem_ids[:, None] == elem_ids[None, :]
        covariance = np.where(same_input, elem_sigma_rate[:, None] * elem_sigma_rate[None, :], base_covariance)
        for idx, row in out.iterrows():
            if np.isfinite(row.get("past_dh_m", np.nan)) and np.isfinite(row.get("sigma_past_dh_m", np.nan)):
                continue
            before = annual.loc[annual["end"] <= row["start"]].copy()

            # If the record has no antecedent period, use the requested default
            # assumption. The "current" option means current annual rate.
            if before.empty:
                if past_missing == "zero":
                    out.loc[idx, "past_dh_m"] = 0.0
                elif past_missing == "current":
                    out.loc[idx, "past_dh_m"] = row["dh_m"] / row["period_years"]
                else:
                    raise ValueError("past_missing must be 'current' or 'zero'")
                out.loc[idx, "sigma_past_dh_m"] = past_error_factor * row["sigma_dh_m"] / row["period_years"]
                continue
            lag = row["start"] - before["end"].to_numpy(float)
            keep = lag <= tau_max
            before = before.loc[keep]
            lag = lag[keep]
            if before.empty:
                out.loc[idx, "past_dh_m"] = row["dh_m"] / row["period_years"] if past_missing == "current" else 0.0
                out.loc[idx, "sigma_past_dh_m"] = past_error_factor * row["sigma_dh_m"] / row["period_years"]
                continue

            # Past elevation change is an annual-equivalent predictor, even if
            # the elementary input periods are multi-annual.
            weights = np.exp(-np.maximum(lag, 0.0) / tau)
            weights = weights / np.sum(weights)
            ids = before.index.to_numpy(int)
            out.loc[idx, "past_dh_m"] = float(np.sum(weights * elem_rate[ids]))
            out.loc[idx, "sigma_past_dh_m"] = float(np.sqrt(max(weights @ covariance[np.ix_(ids, ids)] @ weights, 0.0)))
        out["past_dh_m"] = out["past_dh_m"].fillna(out["dh_m"] / out["period_years"])
        out["sigma_past_dh_m"] = out["sigma_past_dh_m"].fillna(past_error_factor * out["sigma_dh_m"] / out["period_years"])
        return out

    def _temporal_closure(self, periods: pd.DataFrame) -> pd.DataFrame:
        """Apply temporal reconciliation to one glacier time series."""
        out = periods.copy().sort_values(["start", "end"]).reset_index(drop=True)
        bounds = np.array(sorted(set(out["start"].to_numpy(float)).union(set(out["end"].to_numpy(float)))))
        n_elem = len(bounds) - 1
        if n_elem <= 0:
            out["mu_rho_closed_kg_m3"] = out["mu_rho_independent_kg_m3"]
            out["sigma_rho_closed_kg_m3"] = out["sigma_rho_independent_kg_m3"]
            out["dM_closed_kg"] = out["dM_independent_kg"]
            out["temporally_closed"] = False
            return out
        S = np.zeros((len(out), n_elem), dtype=float)
        for i, row in out.iterrows():
            S[i, (bounds[:-1] >= row["start"]) & (bounds[1:] <= row["end"])] = 1.0

        # Reconcile only the anomaly relative to ice density. The ice-density
        # mass contribution is already additive because volume change is.
        anomaly = (out["mu_rho_independent_kg_m3"].to_numpy(float) - self.rho_ice) * out["dV_m3"].to_numpy(float)
        elem_dh = np.full(n_elem, np.nan)
        elem_dt = bounds[1:] - bounds[:-1]
        elem_area = np.full(n_elem, np.nan)

        # For closed uncertainty, elementary density-related mass variances add.
        # The numerator uses absolute elementary dh, while the density-equivalent
        # output divides by the net period volume change.
        for j in range(n_elem):
            exact = out.loc[np.isclose(out["start"], bounds[j]) & np.isclose(out["end"], bounds[j + 1])]
            if len(exact):
                elem_dh[j] = float(exact["dh_m"].iloc[0])
                elem_area[j] = float(exact["area_m2"].iloc[0])
        if np.any(~np.isfinite(elem_dh)):
            elem_dh[~np.isfinite(elem_dh)] = 0.0
        if np.any(~np.isfinite(elem_area)):
            elem_area[~np.isfinite(elem_area)] = float(np.nanmedian(out["area_m2"].to_numpy(float)))
        support = (
            float(self.params["U_h"]) ** 2 * (S @ np.abs(elem_dh))
            + float(self.params["U_t"]) ** 2 * (S @ elem_dt)
        )
        period_area = np.where(
            out["area_m2"].to_numpy(float) > 0,
            out["area_m2"].to_numpy(float),
            float(np.nanmedian(elem_area)),
        )
        sigma = period_area * np.sqrt(np.maximum(support, 0.0))
        floor = np.nanmedian(sigma[np.isfinite(sigma) & (sigma > 0)]) * 1.0e-3 if np.any(np.isfinite(sigma) & (sigma > 0)) else 1.0
        sigma = np.maximum(sigma, floor)

        # Use path-length variance for the WLS weights. This is consistent with
        # the closed uncertainty because elementary density-related mass
        # variances add along the period path.
        sw = np.sqrt(1.0 / sigma**2)
        elem_anomaly, *_ = np.linalg.lstsq(S * sw[:, None], anomaly * sw, rcond=None)
        closed_anomaly = S @ elem_anomaly
        out["dM_closed_kg"] = self.rho_ice * out["dV_m3"].to_numpy(float) + closed_anomaly
        out["mu_rho_closed_kg_m3"] = np.where(out["dV_m3"] != 0, out["dM_closed_kg"] / out["dV_m3"], np.nan)
        out["sigma_dM_rho_closed_kg"] = period_area * np.sqrt(np.maximum(support, 0.0))
        out["sigma_rho_closed_kg_m3"] = np.where(out["dV_m3"] != 0, out["sigma_dM_rho_closed_kg"] / np.abs(out["dV_m3"]), np.inf)
        out["temporally_closed"] = True
        return out

    def spatial_corr(self, distance_km: np.ndarray | float, period_years: np.ndarray | float = 1.0) -> np.ndarray:
        """Spatial correlation of standardized residuals

        :param distance_km: Spatial lag in kilometres
        :param period_years: Accepted for backward compatibility; ignored by the final model
        """
        p = self.params
        d = np.asarray(distance_km, dtype=float)
        form = str(p.get("spatial_corr_form", "exponential"))
        n_components = int(float(p.get("n_spatial_components", 2)))
        corr = np.zeros_like(d, dtype=float)
        for i in range(1, n_components + 1):
            fallback_q = "q1_short_range_fraction" if i == 1 else "q2_long_range_fraction"
            fallback_r = "r1_km" if i == 1 else "r2_km"
            q = float(p.get(f"q{i}_range_fraction", p.get(fallback_q, 0.0)))
            r = float(p.get(f"r{i}_km", p.get(fallback_r, np.nan)))
            if np.isfinite(r) and q > 0:
                corr = corr + q * _spatial_correlation_component(d, r, form)
        return np.clip(corr, 0.0, 1.0)

    def temporal_corr(self, lag_years: np.ndarray | float) -> np.ndarray:
        """Temporal correlation of standardized residuals

        :param lag_years: Temporal lag in years
        """
        lag = np.asarray(lag_years, dtype=float)
        nugget = float(np.clip(self.params["temporal_nugget"], 0.0, 1.0))
        range_yr = max(float(self.params["temporal_timescale_yr"]), 1.0e-12)
        corr = (1.0 - nugget) * np.exp(-3.0 * np.maximum(lag, 0.0) / range_yr)
        corr = np.clip(corr, 0.0, 1.0)
        return np.where(lag == 0, 1.0, corr)


SurrogateModel = RhoSurrogate
