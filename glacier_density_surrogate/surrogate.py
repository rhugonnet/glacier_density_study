"""Predict effective density from elevation change and past elevation change rate."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss


######################################
# READ CALIBRATED SURROGATE PARAMETERS
######################################

# Some of the loading/reading logic below is only relevant to test various model in study scripts.
# For use of the final surrogate, it simply loads final_parameters.json.


def load_packaged_params() -> dict[str, float | bool | str]:
    """
    Read the final fit distributed with the package.

    :returns: A new dictionary with the fitted mean, uncertainty and correlations.
    """
    resource = resources.files("glacier_density_surrogate").joinpath("final_parameters.json")
    with resource.open() as stream:
        return json.load(stream)


DEFAULT_PARAMS = load_packaged_params()
SPATIAL_KEYS = frozenset({
    "spatial_corr_form", "n_spatial_components", "q0_nugget_fraction",
    "q1_range_fraction", "q2_range_fraction", "q3_range_fraction", "r1_km", "r2_km", "r3_km",
})
TEMPORAL_KEYS = frozenset({
    "empirical_nugget", "empirical_sill", "empirical_range_yr", "empirical_exponent",
    "temporal_model_form", "applied_temporal_correlation_at_positive_lag",
    "applied_temporal_covariance_assumption",
})


def _read_parameter_file(path: Path, correlation_keys: frozenset[str] | None = None) -> dict[str, float | bool | str]:
    """Read the final study's long model CSV, one-row correlation CSV or JSON."""
    if path.suffix.lower() == ".csv":
        table = pd.read_csv(path)
        if correlation_keys is not None and "parameter" not in table:
            if len(table) != 1:
                raise ValueError("Final correlation CSV must contain exactly one row")
            values = table.to_dict(orient="records")[0]
            return {key: value for key, value in values.items() if key in correlation_keys}
        required = {"parameter", "value_numeric", "value_text"}
        if not required.issubset(table.columns):
            raise ValueError(f"Parameter CSV must contain {sorted(required)}")
        if table["parameter"].duplicated().any():
            raise ValueError("Parameter CSV contains duplicate parameter names")

        # Each fitting row has either a numeric coefficient or text metadata
        values = {}
        for row in table.itertuples(index=False):
            if pd.notna(row.value_numeric):
                values[row.parameter] = float(row.value_numeric)
            elif pd.notna(row.value_text):
                values[row.parameter] = str(row.value_text)
        if "spec.period" in values:
            values["period_form"] = values.pop("spec.period")
        return values

    if path.suffix.lower() != ".json":
        raise ValueError("Parameter files must be final study CSV or JSON outputs")
    with path.open() as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("Parameter JSON must contain an object")
    if "rho_mean" not in payload:
        return payload

    # Our fitting script write coefficients in these sections
    values = dict(payload["rho_mean"]["parameters"])
    values.update(payload["rho_std"]["parameters"])
    values.update(payload.get("memory", {}))
    values.update(payload.get("fixed", {}))
    values["period_form"] = payload["rho_mean"]["spec"]["period"]
    return values


def load_params(path: Path | str | None = None, *, verbose: bool = True) -> dict[str, float | bool | str]:
    """
    Read fitted parameters using the final study names.

    :param path: Final study parameter CSV or JSON, omitted uses the bundled JSON file.
    :param verbose: Print the source of a loaded fitting output.

    :returns: Bundled values updated with the recognized final parameters.
    """
    params = load_packaged_params()
    if path is None:
        return params
    values = _read_parameter_file(Path(path))
    params.update({key: value for key, value in values.items() if key in DEFAULT_PARAMS})
    if verbose:
        print(f"Loaded surrogate parameters from {path}")
    return params


def load_correlation_params(
    spatial_path: Path | str | None = None,
    temporal_path: Path | str | None = None,
    *,
    verbose: bool = True,
) -> dict[str, float | bool | str]:
    """
    Read correlation coefficients from the final study outputs.

    :param spatial_path: Optional spatial fit CSV or JSON.
    :param temporal_path: Optional temporal fit CSV or JSON.
    :param verbose: Print the source of loaded fitting outputs.

    :returns: Overrides for the supplied correlations, with their original names.
    """
    params = {}
    for path, keys in ((spatial_path, SPATIAL_KEYS), (temporal_path, TEMPORAL_KEYS)):
        if path is None:
            continue
        values = _read_parameter_file(Path(path), keys)
        params.update({key: value for key, value in values.items() if key in keys})
        if verbose:
            print(f"Loaded correlation parameters from {path}")
    return params


def write_packaged_params(
    parameter_path: Path | str | None = None,
    spatial_path: Path | str | None = None,
    temporal_path: Path | str | None = None,
    out_path: Path | str | None = None,
) -> Path:
    """
    Store final fitting outputs using the same names as the study scripts.

    :param parameter_path: Optional mean and uncertainty fitting output.
    :param spatial_path: Optional spatial correlation fitting output.
    :param temporal_path: Optional temporal correlation fitting output.
    :param out_path: Destination, omitted updates the bundled parameter JSON.

    :returns: Path of the written JSON file.
    """
    params = load_params(parameter_path, verbose=False)
    params.update(load_correlation_params(spatial_path, temporal_path, verbose=False))
    if out_path is None:
        out_path = Path(__file__).with_name("final_parameters.json")
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as stream:
        json.dump(params, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return destination


##################
# INPUT VALIDATION
##################


def finite_array(value, name: str, *, positive: bool = False, nonnegative: bool = False) -> np.ndarray:
    """Convert values to floats and reject missing or physically invalid inputs."""
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    if positive and np.any(array <= 0):
        raise ValueError(f"{name} must be positive")
    if nonnegative and np.any(array < 0):
        raise ValueError(f"{name} must be nonnegative")
    return array


def positive_integer(value, name: str) -> int:
    """Check counts before they are used for quadrature, sampling or blocks."""
    if isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(value) or value < 1 or int(value) != value:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _glacier_id_column(data: pd.DataFrame, id_col: str | None) -> str | None:
    """Resolve the glacier identifier column and reject missing identifiers before grouping."""
    if id_col is None:
        for candidate in ("glacier_id", "rgiid"):
            if candidate in data.columns:
                id_col = candidate
                break
    if id_col is not None:
        if id_col not in data.columns:
            raise ValueError(f"Glacier identifier column is missing: {id_col}")
        if data[id_col].isna().any():
            raise ValueError("Glacier identifiers must not be missing")
    return id_col


####################################
# NORMAL EXPECTATIONS AND MEAN MODEL
####################################


def expected_abs_normal(mean: float, sigma: float) -> float:
    """
    Return the expected absolute value of a normal variable.

    :param mean: Mean of the variable, in the same units as sigma.
    :param sigma: Standard deviation; zero gives the absolute value of mean.
    :returns: Expected absolute value, in the input units.
    """

    finite_array(mean, "mean")
    finite_array(sigma, "sigma", nonnegative=True)
    if sigma == 0:
        return abs(mean)
    return (
        sigma * math.sqrt(2.0 / math.pi) * math.exp(-(mean**2) / (2.0 * sigma**2))
        + mean * math.erf(mean / (math.sqrt(2.0) * sigma))
    )


def expected_abs_normal_array(mean: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Calculate expected absolute values for arrays of normal variables.

    :param mean: Normal means, as a scalar or array.
    :param sigma: Nonnegative standard deviations, broadcast against mean.
    :returns: Expected absolute values in the input units and broadcast shape.
    """

    mean, sigma = np.broadcast_arrays(
        finite_array(mean, "mean"), finite_array(sigma, "sigma", nonnegative=True),
    )
    out = np.array(np.abs(mean), copy=True)

    valid = sigma > 0
    z = np.zeros_like(mean, dtype=float)
    z[valid] = mean[valid] / (np.sqrt(2.0) * sigma[valid])
    erf_z = np.array([math.erf(float(value)) for value in z[valid]], dtype=float)
    out[valid] = sigma[valid] * np.sqrt(2.0 / np.pi) * np.exp(-z[valid] ** 2) + mean[valid] * erf_z
    return out


def _residual_variance(model, dh, sigma_dh, dt) -> np.ndarray:
    """
    Average the additive mass change variance per squared unit area.

    Both scalar and array predictions use A0² E|dh| + A1² dt. The scalar
    expectation avoids array allocation; arrays use the broadcast calculation.
    """
    dt_array = finite_array(dt, "dt", positive=True)
    if np.ndim(dh) == 0 and np.ndim(sigma_dh) == 0:
        absolute_change = expected_abs_normal(dh, sigma_dh)
    else:
        absolute_change = expected_abs_normal_array(dh, sigma_dh)
    return float(model.params["A0"]) ** 2 * absolute_change + float(model.params["A1"]) ** 2 * dt_array


def _signed_abs_power(x: np.ndarray | float, power: float) -> np.ndarray:
    """Raise the magnitude to a power while preserving the sign and zeros."""
    x_arr = np.asarray(x, dtype=float)
    return np.sign(x_arr) * np.abs(x_arr) ** float(power)


def _period_component(params: dict[str, float | bool | str], dt: np.ndarray | float) -> np.ndarray:
    """Calculate the density correction for the selected period length model."""
    dt_arr = np.asarray(dt, dtype=float)
    form = str(params["period_form"])
    if form == "power_param":
        return float(params["P0"]) + float(params["P1"]) / (1.0 + dt_arr / float(params["TP"]))
    if form == "none":
        return np.zeros_like(dt_arr, dtype=float)
    raise ValueError(f"Unsupported period form: {form}")


#######################
# CORRELATION FUNCTIONS
#######################


def make_dh_error_correlation(
    form: str | None = None, range_years: float | None = None
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Build a correlation function for elevation change errors at different times.

    None form gives independent errors.
    Exponential and Gaussian correlations fall to exp(-3) at the correlation range, while spherical reaches 0 there.

    :param form: Correlation form: none, exponential, gaussian or spherical.
    :param range_years: Correlation range in years.

    :returns: A function returning correlations with the same shape as its lags.
    """

    name = "none" if form is None else str(form).lower()
    if name not in {"none", "exponential", "gaussian", "spherical"}:
        raise ValueError(f"Unsupported elevation-change error correlation form: {form}")
    if name != "none":
        finite_array(range_years, "range_years", positive=True)

    def correlation(lag: np.ndarray) -> np.ndarray:
        lag = finite_array(lag, "lag", nonnegative=True)
        if name == "none":
            return np.where(lag == 0, 1.0, 0.0)
        return _correlation_component(lag, range_years, name)

    return correlation


def _correlation_component(separation: np.ndarray | float, correlation_range: float, form: str) -> np.ndarray:
    """Calculate a spatial or temporal correlation using a separation and range in the same units."""
    h = np.asarray(separation, dtype=float) / float(correlation_range)
    name = str(form).lower()
    if name == "exponential":
        return np.exp(-3.0 * h)
    if name == "gaussian":
        return np.exp(-3.0 * h**2)
    if name == "spherical":
        return np.where(h < 1.0, 1.0 - 1.5 * h + 0.5 * h**3, 0.0)
    raise ValueError(f"Unsupported correlation form: {form}")


###############################
# EFFECTIVE DENSITY PREDICTIONS
###############################


@dataclass
class RhoSurrogate:
    """
    Surrogate model to predict effective density (rho_dv) from glacier-wide elevation changes (dh), which can then be
    converted to mass changes.

    Use ``predict()`` for one observation period and ``predict_timeseries()`` for several sequential periods.

    The mean and uncertainty methods can also be called  directly when building a larger analysis.

    The params dictionary contains fitted mean, uncertainty and correlation parameters of the final study scripts.

    For example::

        model = RhoSurrogate()
        result = model.predict(dh=-1.0, sigma_dh=0.2, dt=5.0, area_m2=1_000_000)
        density = result["mu_rho_kg_m3"]
    """

    params: dict[str, float | bool | str] = field(default_factory=dict)
    # Integrate model predictions over normally distributed elevation changes (``sigma_dh`` inputs)
    # Use Gauss-Hermite quadrature for both current elevation change and past elevation change rate
    # The node counts set how many weighted points approximate each integral
    # More nodes can improve accuracy, but take longer to compute
    gh_order_current: int = 64
    gh_order_past: int = 64

    def __post_init__(self) -> None:
        """Merge fitted defaults and prepare quadrature nodes for both predictors."""

        merged = load_packaged_params()
        unknown = self.params.keys() - merged.keys()
        if unknown:
            raise ValueError(f"Unknown parameter names: {sorted(unknown)}")
        merged.update(self.params)
        self.params = merged

        # Check scales and coefficients before they enter powers or divisions
        for key, value in merged.items():
            if isinstance(value, (int, float)):
                finite_array(value, key)
        for key in ("rho_ice_fixed", "H", "beta", "LA", "etaMem", "alpha", "TP", "memory_tau_years", "tau_max"):
            finite_array(merged[key], key, positive=True)
        for key in ("A0", "A1", "sigma_numeric_floor"):
            finite_array(merged[key], key, nonnegative=True)
        if merged["period_form"] not in {"power_param", "none"}:
            raise ValueError("period_form must be 'power_param' or 'none'")

        # Correlation ranges and fractions must describe valid components
        count = positive_integer(merged["n_spatial_components"], "n_spatial_components")
        if count > 3:
            raise ValueError("n_spatial_components must be at most three")
        fractions = [merged["q0_nugget_fraction"]]
        for index in range(1, count + 1):
            finite_array(merged[f"r{index}_km"], f"r{index}_km", positive=True)
            fractions.append(merged[f"q{index}_range_fraction"])
        finite_array(fractions, "spatial fractions", nonnegative=True)
        if sum(fractions) > 1 + 1e-12:
            raise ValueError("Spatial correlation fractions must sum to at most one")
        if merged["spatial_corr_form"] not in {"exponential", "gaussian", "spherical"}:
            raise ValueError("Unsupported spatial_corr_form")
        finite_array(merged["empirical_nugget"], "empirical_nugget", nonnegative=True)
        if merged["empirical_nugget"] > 1:
            raise ValueError("empirical_nugget must lie between zero and one")
        finite_array(merged["empirical_range_yr"], "empirical_range_yr", positive=True)

        # Avoid a node at zero for a predictor distribution centered on zero
        self.gh_order_current = positive_integer(self.gh_order_current, "gh_order_current")
        self.gh_order_past = positive_integer(self.gh_order_past, "gh_order_past")
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
        """
        Build a model using fitted parameters from CSV or JSON files.

        Omitted files use the bundled parameters; requested files must exist.

        :param parameter_path: Mean and uncertainty parameter file.
        :param spatial_path: Spatial correlation parameter file.
        :param temporal_path: Temporal correlation parameter file.
        :param kwargs: Additional constructor options, such as quadrature orders.
        :returns: A model containing the loaded parameters.
        """

        params = load_params(parameter_path)
        params.update(load_correlation_params(spatial_path, temporal_path))
        return cls(params=params, **kwargs)

    @property
    def rho_ice(self) -> float:
        """Ice density in kg m-3, used as the large elevation change limit."""
        return float(self.params["rho_ice_fixed"])

    def integrate_normal(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        mean: float,
        sigma: float,
        *,
        use_past_nodes: bool = False,
    ) -> float:
        """
        Average a function over normally distributed values using quadrature.

        :param func: Function accepting and returning a vector of values.
        :param mean: Mean of the normal distribution.
        :param sigma: Standard deviation of the distribution.
        :param use_past_nodes: Use the quadrature order for the past elevation change rate.

        :returns: Expected value of the function under the normal distribution.
        """

        finite_array(mean, "mean")
        finite_array(sigma, "sigma", nonnegative=True)
        if sigma == 0:
            return float(np.asarray(func(np.array([mean], dtype=float)))[0])

        # Rescale Hermite nodes and weights to the requested normal distribution
        gh_x = self._gh_x_past if use_past_nodes else self._gh_x_current
        gh_w = self._gh_w_past if use_past_nodes else self._gh_w_current
        x = mean + math.sqrt(2.0) * sigma * gh_x
        return float(np.sum(gh_w * np.asarray(func(x), dtype=float)) / math.sqrt(math.pi))

    def mu_rho(self, dh: np.ndarray | float, past_dhdt: np.ndarray | float = 0.0, dt: float = 1.0) -> np.ndarray:
        """
        Surrogate mean function for effective density when the elevation changes are exact.

        Support array inputs (follows NumPy broadcasting).

        Effective density can diverge near zero net elevation change, where the ratio
        of mass change to volume change is poorly defined.

        :param dh: Current elevation change over the observation period, in metres.
        :param past_dhdt: Past elevation change rate in m yr-1.
        :param dt: Observation period length in years (may also be an array).

        :returns: Mean effective density in kg m-3, with the broadcast input shape.
        """

        params = self.params
        dh_arr = finite_array(dh, "dh")
        past_dhdt_arr = finite_array(past_dhdt, "past_dhdt")
        dt_arr = finite_array(dt, "dt", positive=True)

        # Approach ice density as elevation change grows and firn changes matter less
        abs_dh = np.maximum(np.abs(dh_arr), 1.0e-12)
        sign_dh = np.where(dh_arr < 0, -1.0, 1.0)
        damping = np.exp(-np.clip((abs_dh / float(params["H"])) ** float(params["beta"]), 0.0, 700.0))

        # Combine a finite memory term with a term that diverges near zero volume change
        finite_past = float(params["Bmem"]) * sign_dh * _signed_abs_power(past_dhdt_arr, float(params["etaMem"]))
        singular_past = (
            float(params["A"])
            * np.tanh(sign_dh * past_dhdt_arr / float(params["LA"]))
            / (abs_dh ** float(params["alpha"]))
        )
        density_correction = float(params["Bc"]) + _period_component(params, dt_arr) + finite_past + singular_past
        return self.rho_ice + damping * density_correction

    def sigma_rho(self, dh: np.ndarray | float, dt: float = 1.0) -> np.ndarray:
        """
        Surrogate uncertainty function for exact elevation change.

        Multiply by absolute volume change to convert this uncertainty to mass change units, where it becomes additive in
        variance (inherent from its form!).

        :param dh: Current elevation change over the period, in metres.
        :param dt: Observation period length in years; may also be an array.

        :returns: Uncertainty in effective density (1-sigma level) in kg m-3, with the broadcast input shape.
        """

        params = self.params
        abs_dh = np.maximum(np.abs(finite_array(dh, "dh")), 1.0e-12)
        dt_arr = finite_array(dt, "dt", positive=True)

        # Express the two additive mass change variance terms in density units
        sigma = np.sqrt(float(params["A0"]) ** 2 / abs_dh + float(params["A1"]) ** 2 * dt_arr / abs_dh**2)
        return np.maximum(sigma, float(params.get("sigma_numeric_floor", 1.0)))

    def _past_moments(self, past_dhdt: float, sigma_past_dhdt: float) -> tuple[float, float]:
        """Average the signed power and tanh terms over uncertain past elevation change rate."""
        signed_power = self.integrate_normal(
            lambda y: _signed_abs_power(y, float(self.params["etaMem"])),
            past_dhdt,
            sigma_past_dhdt,
            use_past_nodes=True,
        )
        tanh_moment = self.integrate_normal(
            lambda y: np.tanh(y / float(self.params["LA"])),
            past_dhdt,
            sigma_past_dhdt,
            use_past_nodes=True,
        )
        return signed_power, tanh_moment

    def _mean_integrated_over_past(
        self,
        dh: np.ndarray | float,
        past_dhdt: float,
        sigma_past_dhdt: float,
        dt: float,
    ) -> np.ndarray:
        """
        Mean model integrated over only uncertain past elevation change rate

        The current elevation changes remain fixed. This function is mostly used as an internal helper.

        :param dh: Current elevation changes in metres.
        :param past_dhdt: Past elevation change rate in m yr-1.
        :param sigma_past_dhdt: Standard deviation of the past elevation change rate in m yr-1.
        :param dt: Observation period length in years.

        :returns: Mean density in kg m-3, with the same shape as dh.
        """

        params = self.params
        dh_arr = finite_array(dh, "dh")
        finite_array(dt, "dt", positive=True)
        abs_dh = np.maximum(np.abs(dh_arr), 1.0e-12)
        sign_dh = np.where(dh_arr < 0, -1.0, 1.0)
        damping = np.exp(-np.clip((abs_dh / float(params["H"])) ** float(params["beta"]), 0.0, 700.0))

        # Average the past elevation change rate terms once for all current integration points
        signed_power, tanh_moment = self._past_moments(past_dhdt, sigma_past_dhdt)
        finite_past = float(params["Bmem"]) * sign_dh * signed_power
        singular_past = float(params["A"]) * sign_dh * tanh_moment / (abs_dh ** float(params["alpha"]))
        density_correction = float(params["Bc"]) + _period_component(params, float(dt)) + finite_past + singular_past
        return self.rho_ice + damping * density_correction

    def integrated_mu(
        self,
        dh: float,
        sigma_dh: float = 0.0,
        past_dhdt: float = 0.0,
        sigma_past_dhdt: float = 0.0,
        dt: float = 1.0,
    ) -> float:
        """
        Integrated surrogate mean model to predict effective density from uncertain predictors.

        We average mean density times current elevation change over independent normal
        distributions for current elevation change and past elevation change rate.
        We then divide by observed dh.

        This makes density times observed volume change equal the expected mass change
        (in other words, it preserves mass change additivity given volume change additivity).

        :param dh: Measured current elevation change over the period, in metres.
        :param sigma_dh: Standard deviation of current elevation change in metres.
        :param past_dhdt: Past elevation change rate in m yr-1.
        :param sigma_past_dhdt: Standard deviation of the past elevation change rate in m yr-1.
        :param dt: Observation period length in years.

        :returns: Mean effective density in kg m-3, or NaN when dh is zero.
        """

        numerator = self._integrated_mass_per_area(dh, sigma_dh, past_dhdt, sigma_past_dhdt, dt)
        return numerator / dh if dh != 0 else np.nan

    def _integrated_mass_per_area(self, dh, sigma_dh, past_dhdt, sigma_past_dhdt, dt) -> float:
        """Average density times change without dividing by observed volume change."""
        finite_array(dh, "dh")
        finite_array(dt, "dt", positive=True)

        # Average density times elevation change over current uncertainty
        numerator = self.integrate_normal(
            lambda x: self._mean_integrated_over_past(x, past_dhdt, sigma_past_dhdt, dt) * x,
            dh,
            sigma_dh,
        )
        return numerator

    def integrated_sigma(self, dh: float, sigma_dh: float = 0.0, dt: float = 1.0) -> float:
        """
        Integrated surrogate uncertainty model to predict effective density from uncertain predictors.

        The term based on absolute elevation change uses E|dh| rather than |E[dh]|. We divide the resulting
        mass change standard deviation per unit area by |dh| to express it as a density.
        This is the surrogate residual uncertainty: it excludes the additional variance of the mean mass change
        caused by uncertain current elevation change and past elevation change rate.

        :param dh: Measured current elevation change in metres.
        :param sigma_dh: Standard deviation of current elevation change in metres.
        :param dt: Observation period length in years.

        :returns: Standard deviation in kg m-3, or infinity when dh is zero.
        """

        variance_over_area2 = _residual_variance(self, dh, sigma_dh, dt)
        return math.sqrt(variance_over_area2) / abs(dh) if dh != 0 else np.inf

    def integrated_sigma_equiv_density(
        self,
        dh: float,
        sigma_dh: float = 0.0,
        dt: float = 1.0,
        area_m2: float = 1.0,
    ) -> tuple[float, float]:
        """
        Integrated surrogate uncertainty expressed in density and mass change units.

        :param dh: Measured current elevation change in metres.
        :param sigma_dh: Standard deviation of current elevation change in metres.
        :param dt: Observation period length in years.
        :param area_m2: Glacier area in square metres.
        :returns: Density standard deviation in kg m-3 and mass change standard deviation in kg.
            The mass change uncertainty is finite even when observed dh is zero.
        """

        finite_array(area_m2, "area_m2", positive=True)
        sigma_per_area = math.sqrt(_residual_variance(self, dh, sigma_dh, dt))
        sigma_rho = sigma_per_area / abs(dh) if dh != 0 else np.inf
        sigma_mass = area_m2 * sigma_per_area
        return sigma_rho, sigma_mass

    def predict(
        self,
        dh: float,
        sigma_dh: float = 0.0,
        *,
        dt: float,
        past_dhdt: float | None = None,
        sigma_past_dhdt: float | None = None,
        area_m2: float | None = None,
        past_missing: Literal["current", "zero"] = "current",
        past_error_factor: float = 2.0,
    ) -> dict[str, float]:
        """
        Predict effective density and its uncertainty for one glacier and observation period.

        When the past elevation change rate is missing, "current" uses dh / dt.
        The "zero" assumption uses zero. Missing uncertainty is estimated as
        past_error_factor * sigma_dh / dt, with a default factor of 2.

        :param dh: Mean elevation change over the glacier and observation period, in metres.
        :param sigma_dh: One-sigma uncertainty of dh in metres.
        :param dt: Required period length in years.
        :param past_dhdt: Past elevation change rate in m yr-1.
        :param sigma_past_dhdt: One-sigma uncertainty of past elevation change rate in m yr-1.
        :param area_m2: Optional glacier area in square metres for volume change and mass change outputs.
        :param past_missing: Assumption for missing past_dhdt: "current" or "zero".
        :param past_error_factor: Multiplier for the default past elevation change rate uncertainty.

        :returns: A dictionary containing the predictors, density in kg m-3 and its uncertainty (1-sigma). With area
          provided as input, it also contains volume change in m3, mass change and its uncertainty in kg.
          Mass change uncertainty includes separate input and surrogate residual components and their total.
        """

        dh = float(dh)
        sigma_dh = float(0.0 if sigma_dh is None else sigma_dh)
        dt = float(dt)
        finite_array(dh, "dh")
        finite_array(sigma_dh, "sigma_dh", nonnegative=True)
        finite_array(dt, "dt", positive=True)
        finite_array(past_error_factor, "past_error_factor", nonnegative=True)
        if past_missing not in {"current", "zero"}:
            raise ValueError("past_missing must be 'current' or 'zero'")

        # Supply a past elevation change rate and uncertainty when observations are missing
        if past_dhdt is None or np.isnan(past_dhdt):
            if past_missing == "zero":
                past_dhdt = 0.0
            else:
                past_dhdt = dh / dt
        if sigma_past_dhdt is None or np.isnan(sigma_past_dhdt):
            sigma_past_dhdt = past_error_factor * sigma_dh / dt

        # Average the model over predictor errors before converting to mass change
        mass_per_area = self._integrated_mass_per_area(
            dh=dh,
            sigma_dh=sigma_dh,
            past_dhdt=float(past_dhdt),
            sigma_past_dhdt=float(sigma_past_dhdt),
            dt=dt,
        )
        mean_density = mass_per_area / dh if dh != 0 else np.nan

        # Compute the residual variance once, in the output units requested by the caller
        if area_m2 is None:
            sigma_density = self.integrated_sigma(dh=dh, sigma_dh=sigma_dh, dt=dt)
        else:
            sigma_density, sigma_mass = self.integrated_sigma_equiv_density(dh, sigma_dh, dt, area_m2)
        out = {
            "dh_m": dh,
            "sigma_dh_m": sigma_dh,
            "period_years": dt,
            "past_dhdt_m_yr": float(past_dhdt),
            "sigma_past_dhdt_m_yr": float(sigma_past_dhdt),
            "mu_rho_kg_m3": float(mean_density),
            "sigma_rho_kg_m3": float(sigma_density),
        }

        # Add physical volume change and mass change outputs when the glacier area is known
        if area_m2 is not None:
            from .timeseries import mean_density_volume_sigma_vectorized

            volume_change = float(area_m2) * dh
            sigma_input = float(mean_density_volume_sigma_vectorized(
                self, area_m2, dh, sigma_dh, past_dhdt, sigma_past_dhdt, dt,
            ))
            out.update(
                {
                    "area_m2": float(area_m2),
                    "dV_m3": volume_change,
                    "dM_kg": float(mass_per_area * area_m2),
                    "sigma_dM_rho_kg": sigma_mass,
                    "sigma_dV_m3": float(area_m2) * sigma_dh,
                    "sigma_dM_dh_kg": sigma_input,
                    "sigma_dM_total_kg": math.hypot(sigma_mass, sigma_input),
                }
            )
        return out

    def predict_timeseries(
        self,
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
        Predict effective density and its uncertainty for one or more glacier time series.

        We prepare the observations and estimate missing past elevation change rates.
        We then call predict() for each period. For connected periods, temporally_reconcile_periods() adjusts
        the predictions so mass changes are additive when periods are combined. It also
        propagates their error variances.

        This function accepts a table with glacier_id or rgiid containing several glaciers time series.
        We predict each glacier independently and return their results together with the same IDs.
        Then, the function aggregate_regions() can be used to sum their glacier mass changes and propagate
        uncertainty in space.

        Input uncertainties are propagated automatically when glacier area is supplied.
        We reuse each observation error in all periods and estimated past elevation change rates
        that depend on it, then reconcile the sampled mass changes. Supplied past rates are
        independent of elevation observations; periods with the same start share their past rate error.
        Input errors are independent between glaciers.

        With expand_periods=True, consecutive observations also produce estimates
        for longer periods. For example, 2000-2005 and 2005-2010 also produce a
        2000-2010 estimate.

        If no past elevation change rates are passed, our default fallback is to estimate them
        from earlier annual rates, giving more weight to recent observations.
        Use the arguments past_missing and past_error_factor to control the fallback
        behaviour when observations are unavailable.

        Time reconciliation require start and end years.

        With durations alone, each row is predicted independently. Reconciliation
        also requires constant area and observations covering each step in time.

        Longer period elevation changes must equal the sum over shorter periods.

        :param data: Observation period table for one or more glaciers.
        :param id_col: Glacier identifier column; omitted detects glacier_id or rgiid.
        :param start_col: Start year column.
        :param end_col: End year column.
        :param dh_col: Elevation change column in metres.
        :param sigma_dh_col: Elevation change uncertainty column in metres.
        :param dt_col: Period length column in years, used when start/end are absent.
        :param past_dhdt_col: Optional past elevation change rate column in m yr-1.
        :param sigma_past_dhdt_col: Optional past elevation change rate uncertainty column in m yr-1.
        :param area_col: Optional glacier area column in square metres.
        :param area_m2: Constant area used when an area column is absent.
        :param expand_periods: Include all contiguous combinations of consecutive
            observations (default True). False returns only the supplied periods.
            Reconciliation applies in both cases when the observations allow it.
        :param past_missing: Fallback for a missing past elevation change rate
            without earlier observations.
            "current" uses the current annual rate, dh / dt (default); "zero" uses zero.
        :param past_error_factor: Multiplier for the current annual rate uncertainty
            when past elevation change rate uncertainty and earlier observations are missing (default 2).
        :param dh_error_corr: Correlation between elevation change errors as a function
            of lag in years. Used for combined periods and uncertainty in the
            past elevation change rate.
            Omitted assumes independent errors.
        :param return_components: Include estimates before and after reconciliation,
            plus the temporally_closed flag (default False).
            Mass change components use unit area if no glacier area is supplied.

        :returns: A DataFrame with period bounds, current elevation change,
            past elevation change rate, and density means and standard deviations.
            Supplying area adds volume change, mass change and its uncertainty from
            input and surrogate errors, including sigma_dM_dh_kg, sigma_dM_rho_kg
            and sigma_dM_total_kg. Exact inputs have zero input uncertainty.
            Invalid observations raise ValueError.
            Empty input returns an empty table.
        """

        from .timeseries import predict_timeseries

        return predict_timeseries(
            self,
            data,
            id_col=id_col,
            start_col=start_col,
            end_col=end_col,
            dh_col=dh_col,
            sigma_dh_col=sigma_dh_col,
            dt_col=dt_col,
            past_dhdt_col=past_dhdt_col,
            sigma_past_dhdt_col=sigma_past_dhdt_col,
            area_col=area_col,
            area_m2=area_m2,
            expand_periods=expand_periods,
            past_missing=past_missing,
            past_error_factor=past_error_factor,
            dh_error_corr=dh_error_corr,
            return_components=return_components,
        )

    def aggregate_regions(
        self,
        predictions: pd.DataFrame,
        *,
        id_col: str | None = None,
        group_col: str = "region_group",
        lat_col: str = "lat",
        lon_col: str = "lon",
    ) -> pd.DataFrame:
        """
        Sum glacier predictions and propagate their uncertainty accounting for spatial correlation.

        Pass the output of predict_timeseries() with glacier identifiers, glacier areas and coordinates.

        Glaciers in each region must cover the same consecutive elementary intervals, with constant area for each
        glacier.

        We propagate spatial covariance within each interval and sum independent elementary variances for
        longer periods. Regional effective density is total mass change divided by total volume change.

        Input errors from predict_timeseries() are combined across independent glaciers,
        and total mass change uncertainty includes both input and residual variance.

        :param predictions: Glacier predictions with area_m2, dV_m3, dM_kg and sigma_dM_rho_kg.
        :param id_col: Glacier identifier column; omitted detects glacier_id or rgiid.
        :param group_col: Region column; absent groups all glaciers into "All glaciers".
        :param lat_col: Glacier latitude column in decimal degrees.
        :param lon_col: Glacier longitude column in decimal degrees.

        :returns: One row per region and period shared by all its glaciers, with glacier
            count, total area, volume change, mass change and input, residual and total uncertainties.
        """
        from .aggregation import aggregate_regions

        return aggregate_regions(
            predictions, self.spatial_corr, id_col=id_col, group_col=group_col,
            lat_col=lat_col, lon_col=lon_col,
        )

    def spatial_corr(self, distance_km: np.ndarray | float) -> np.ndarray:
        """
        Surrogate model of spatial error correlation at a given distance.

        Correlation at zero distance between distinct glaciers is less than one (nugget term), but a glacier own
        uncertainty is perfectly correlated with itself.

        :param distance_km: Distance between glaciers in kilometres.

        :returns: Correlations with the same shape as distance_km.
        """

        params = self.params
        distance = finite_array(distance_km, "distance_km", nonnegative=True)
        form = str(params["spatial_corr_form"])
        n_components = int(params["n_spatial_components"])
        corr = np.zeros_like(distance, dtype=float)
        for i in range(1, n_components + 1):
            fraction = float(params[f"q{i}_range_fraction"])
            range_km = float(params[f"r{i}_km"])
            if np.isfinite(range_km) and fraction > 0:
                corr = corr + fraction * _correlation_component(distance, range_km, form)
        return np.clip(corr, 0.0, 1.0)

    def temporal_corr(self, lag_years: np.ndarray | float) -> np.ndarray:
        """
        Surrogate model of temporal error correlation at a given temporal lag.

        :param lag_years: Temporal separation in years.

        :returns: Correlations with the same shape as lag_years, equal to one at zero lag.
        """

        lag = finite_array(lag_years, "lag_years", nonnegative=True)
        nugget = float(self.params["empirical_nugget"])
        range_yr = float(self.params["empirical_range_yr"])
        corr = (1.0 - nugget) * np.exp(-3.0 * lag / range_yr)
        return np.where(lag == 0, 1.0, corr)
