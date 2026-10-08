"""Tests for the surrogate model definition and methods."""

from __future__ import annotations

import json
import math
from importlib import resources

import numpy as np
import pandas as pd
import pytest

from glacier_density_surrogate import (
    RhoSurrogate,
    expected_abs_normal,
    expected_abs_normal_array,
    integrated_sigma_mass_vectorized,
    load_correlation_params,
    load_packaged_params,
    load_params,
    make_dh_error_correlation,
    write_packaged_params,
)


def test_package_does_not_export_study_paths() -> None:
    """Checks that the public package does not export manuscript output paths."""
    import glacier_density_surrogate as gds

    assert not hasattr(gds, "DEFAULT_FIGURES_DIR")
    assert not hasattr(gds, "DEFAULT_PARAM_PATH")
    assert not hasattr(gds, "DEFAULT_RESULTS_DIR")


#################
# PARAMETER FILES
#################


class TestParameterFiles:
    """Test module for exact parameter names and final study file formats."""

    def test_load_packaged_params__package_root(self) -> None:
        """Checks that the model loads the fitted JSON directly from the package root."""

        # Read the distributed file independently of the parameter loader
        resource = resources.files("glacier_density_surrogate").joinpath("final_parameters.json")
        with resource.open() as stream:
            expected = json.load(stream)

        # Both the public loader and the default model use this same fit
        assert load_packaged_params() == expected
        assert RhoSurrogate().params == expected

    def test_write_packaged_params__default_destination(self, tmp_path, monkeypatch) -> None:
        """Checks that saving the fit uses the package root when no destination is supplied."""

        # Redirect the module location so the bundled fit cannot be overwritten
        from glacier_density_surrogate import surrogate

        expected = load_packaged_params()
        monkeypatch.setattr(surrogate, "__file__", str(tmp_path / "surrogate.py"))

        # The default destination is beside the module rather than in a subdirectory
        destination = write_packaged_params()
        assert destination == tmp_path / "final_parameters.json"
        assert json.loads(destination.read_text()) == expected

    def test_write_packaged_params__round_trip(self, tmp_path) -> None:
        """Checks that saving and loading the fit preserves every parameter and prediction."""

        # Save the published fit without changing the installed package
        original = RhoSurrogate()
        path = write_packaged_params(out_path=tmp_path / "fit.json")
        loaded = RhoSurrogate.from_files(parameter_path=path)

        # The period parameter TP and memory_tau_years describe separate time scales
        assert loaded.params == original.params
        assert "T_p" not in loaded.params
        assert loaded.params["TP"] != loaded.params["memory_tau_years"]
        assert loaded.predict(dh=-1, dt=5) == original.predict(dh=-1, dt=5)

    def test_load_params__final_study_csv(self, tmp_path) -> None:
        """Checks that the final study's numeric and text rows load under their exact names."""

        # Use the names and long table columns written by the final fitting script
        path = tmp_path / "fit.csv"
        pd.DataFrame({
            "parameter": ["spec.period", "TP", "memory_tau_years", "Bmem", "A0"],
            "value_numeric": [np.nan, 2.5, 4.9, 12.0, 80.0],
            "value_text": ["power_param", "", "", "", ""],
        }).to_csv(path, index=False)

        # Load period metadata separately from the unchanged fitted names
        params = load_params(path, verbose=False)
        assert params["period_form"] == "power_param"
        assert params["TP"] == 2.5
        assert params["memory_tau_years"] == 4.9
        assert params["Bmem"] == 12.0
        assert params["A0"] == 80.0

    def test_load_params__structured_study_json(self, tmp_path) -> None:
        """Checks that the final structured JSON uses the same names as its CSV."""

        # The study JSON separates fitted parameters from memory and fixed constants
        payload = {
            "rho_mean": {"spec": {"period": "power_param"}, "parameters": {"TP": 2.5, "Bmem": 12.0}},
            "rho_std": {"parameters": {"A0": 80.0, "A1": 110.0}},
            "memory": {"memory_tau_years": 4.9},
            "fixed": {"rho_ice_fixed": 900.0, "sigma_numeric_floor": 1.0},
        }
        path = tmp_path / "fit.json"
        path.write_text(json.dumps(payload))

        params = load_params(path, verbose=False)
        assert params["TP"] == 2.5
        assert params["memory_tau_years"] == 4.9
        assert params["Bmem"] == 12.0
        assert params["rho_ice_fixed"] == 900.0

    def test_load_correlation_params__exact_temporal_names(self, tmp_path) -> None:
        """Checks that the final temporal fit overrides the diagnostic curve."""

        path = tmp_path / "temporal.json"
        path.write_text(json.dumps({"empirical_nugget": 0.1, "empirical_range_yr": 9.0}))
        overrides = load_correlation_params(temporal_path=path, verbose=False)
        model = RhoSurrogate(params=overrides)

        # A nonzero lag follows the fitted range and nugget rather than bundled values
        np.testing.assert_allclose(model.temporal_corr(3.0), 0.9 * np.exp(-1.0))

    def test_load_correlation_params__final_study_csv(self, tmp_path):
        """Checks that the final correlation CSVs with one row load without renaming."""
        spatial = tmp_path / "spatial.csv"
        temporal = tmp_path / "temporal.csv"
        pd.DataFrame([{
            "r1_km": 140.0, "n_spatial_components": 3, "spatial_corr_form": "exponential", "rmse": 0.1,
        }]).to_csv(spatial, index=False)
        pd.DataFrame([{"empirical_nugget": 0.1, "empirical_range_yr": 9.0, "n_fit_bins": 14}]).to_csv(temporal, index=False)

        model = RhoSurrogate.from_files(spatial_path=spatial, temporal_path=temporal)

        assert model.params["r1_km"] == 140.0
        np.testing.assert_allclose(model.temporal_corr(3.0), 0.9 * np.exp(-1.0))
        assert "rmse" not in model.params
        saved = write_packaged_params(spatial_path=spatial, temporal_path=temporal, out_path=tmp_path / "combined.json")
        assert load_params(saved, verbose=False) == model.params

    def test_load_correlation_params__separate_json_round_trip(self, tmp_path):
        """Checks that storing and reloading correlation files preserves exact names."""
        original = RhoSurrogate()
        path = write_packaged_params(out_path=tmp_path / "fit.json")
        loaded = RhoSurrogate.from_files(spatial_path=path, temporal_path=path)
        assert loaded.params == original.params


#####################
# DENSITY PREDICTIONS
#####################


def test_expected_abs_normal_matches_limiting_cases() -> None:
    """Checks that zero variance and a centered normal give their known absolute means."""
    means = np.array([-2.0, 0.0, 3.0])
    sigmas = np.array([0.0, 2.0, 0.0])

    assert expected_abs_normal(-2.0, 0.0) == 2.0
    assert math.isclose(expected_abs_normal(0.0, 2.0), 2.0 * math.sqrt(2.0 / math.pi))
    np.testing.assert_allclose(expected_abs_normal_array(means, sigmas), [2.0, 2.0 * math.sqrt(2.0 / math.pi), 3.0])


class TestDensityPredictions:
    """Test module for the mean, uncertainty and assumed past elevation change rate."""

    def test_mean_converges_toward_ice_density_for_large_elevation_change(self) -> None:
        """Checks that very large gains and losses both approach ice density."""
        model = RhoSurrogate()
        rho = model.mu_rho(np.array([-10000.0, 10000.0]), dh_p=0.0, dt=20.0)

        np.testing.assert_allclose(rho, model.rho_ice, atol=1.0e-3)

    def test_period_power_mean_form_matches_manual_expression(self) -> None:
        """Checks that the power period model agrees with its direct formula."""

        # Remove the past terms to isolate the elevation and period terms
        params = {
            "period_form": "power_param",
            "rho_ice_fixed": 900.0,
            "H": 9.0,
            "beta": 0.7,
            "Bc": -20.0,
            "P0": 150.0,
            "P1": -420.0,
            "TP": 2.0,
            "Bmem": 0.0,
            "A": 0.0,
        }
        model = RhoSurrogate(params=params)
        dh = np.array([-5.0, -5.0])
        dt = np.array([1.0, 20.0])

        # Calculate the expected density directly from the mean equation
        damping = np.exp(-((np.abs(dh) / params["H"]) ** params["beta"]))
        period = params["P0"] + params["P1"] / (1.0 + dt / params["TP"])
        expected = params["rho_ice_fixed"] + damping * (params["Bc"] + period)

        # The public prediction must agree for short and long periods
        np.testing.assert_allclose(model.mu_rho(dh, dh_p=0.0, dt=dt), expected)

    def test_sigma_follows_final_variance_additive_form(self) -> None:
        """Checks that density uncertainty grows for smaller changes and longer periods."""

        # Use simple coefficients for the elevation and time contributions
        model = RhoSurrogate(params={"A0": 2.0, "A1": 3.0, "sigma_numeric_floor": 1.0})

        sigma_large = float(model.sigma_rho(-20.0, dt=5.0))
        sigma_small = float(model.sigma_rho(-1.0, dt=5.0))
        sigma_long = float(model.sigma_rho(-1.0, dt=20.0))

        # Convert the known mass change variances to density standard deviations
        expected = [math.sqrt(125) / 20, 7.0, math.sqrt(184)]

        # The largest elevation change reaches the configured uncertainty floor
        expected = np.maximum(expected, 1.0)
        np.testing.assert_allclose([sigma_large, sigma_small, sigma_long], expected)
        assert sigma_small > sigma_large
        assert sigma_long > sigma_small
        assert "sigma_form" not in model.params

    def test_missing_past_current_uses_current_annual_rate(self) -> None:
        """Checks that the default past elevation change rate divides the current change and error by duration."""
        model = RhoSurrogate()
        out = model.predict(dh=-10.0, sigma_dh=1.0, dt=20.0, past_missing="current", past_error_factor=2.0)

        assert out["past_dh_m"] == -0.5
        assert out["sigma_past_dh_m"] == 0.1


class TestIntegratedPredictions:
    """Test module for density uncertainty with exact or uncertain elevation change."""

    def test_integrated_sigma_matches_exact_sigma_without_dh_uncertainty(self) -> None:
        """Checks that averaging an exact current change leaves its uncertainty unchanged."""
        model = RhoSurrogate()

        exact = float(model.sigma_rho(-4.0, dt=10.0))
        integrated = model.integrated_sigma(dh=-4.0, sigma_dh=0.0, dt=10.0)

        assert math.isclose(integrated, exact, rel_tol=1.0e-12)

    @pytest.mark.parametrize("dh,sigma_dh,absolute_change", [
        (-2.0, 0.0, 2.0), (2.0, 0.0, 2.0), (0.0, 0.0, 0.0),
        (0.0, 0.2, 0.2 * math.sqrt(2.0 / math.pi)),
    ])
    def test_integrated_sigma_equiv_density__known_variance(self, dh, sigma_dh, absolute_change):
        """Checks that scalar and array mass change errors follow the known residual variance for exact or centered inputs."""

        # Exact changes have E|dh| = |dh|; a centered normal has E|dh| = sigma sqrt(2/pi)
        model = RhoSurrogate(params={"A0": 3.0, "A1": 4.0})
        area = 1e6
        duration = 2.0
        expected_per_area = math.sqrt(9.0 * absolute_change + 16.0 * duration)
        expected_density = expected_per_area / abs(dh) if dh != 0 else np.inf
        expected_mass = area * expected_per_area

        # Check both units and the uncertainty returned with physical predictions
        density, mass = model.integrated_sigma_equiv_density(dh, sigma_dh, duration, area)
        prediction = model.predict(dh, sigma_dh, dt=duration, area_m2=area)
        array_mass = integrated_sigma_mass_vectorized(model, area, np.array([dh]), sigma_dh, duration)

        np.testing.assert_allclose(density, expected_density)
        np.testing.assert_allclose(mass, expected_mass)
        np.testing.assert_allclose(prediction["sigma_rho_kg_m3"], expected_density)
        np.testing.assert_allclose(prediction["sigma_dM_rho_kg"], expected_mass)
        np.testing.assert_allclose(array_mass, expected_mass)

    def test_integrated_sigma_equiv_density__zero_volume(self):
        """Checks that zero volume change leaves finite mass change uncertainty and infinite density error."""
        model = RhoSurrogate()
        density, mass = model.integrated_sigma_equiv_density(0.0, 0.2, 5.0, 1e6)

        # A centered normal has mean absolute change equal to sigma times sqrt(2 / pi)
        mean_absolute_change = 0.2 * math.sqrt(2 / math.pi)
        variance = float(model.params["A0"]) ** 2 * mean_absolute_change + float(model.params["A1"]) ** 2 * 5.0
        expected = 1e6 * math.sqrt(variance)

        # Density is undefined, while the mass change error follows the known variance
        assert np.isinf(density)
        np.testing.assert_allclose(mass, expected)


####################
# ERROR CORRELATIONS
####################


class TestCorrelationFunctions:
    """Test module for fitted correlations and elevation error correlation options."""

    def test_correlation_functions_have_expected_bounds(self) -> None:
        """Checks that fitted correlations stay between zero and one and decrease with lag."""
        model = RhoSurrogate()

        spatial = model.spatial_corr(np.array([0.0, 100.0, 1000.0, 20000.0]))
        temporal = model.temporal_corr(np.array([0.0, 1.0, 5.0, 14.0]))

        assert np.all((spatial >= 0.0) & (spatial <= 1.0))
        assert spatial[0] > spatial[-1]
        assert temporal[0] == 1.0
        assert np.all((temporal >= 0.0) & (temporal <= 1.0))
        assert temporal[1] > temporal[2] >= temporal[3]

    def test_elevation_error_correlation_builders(self) -> None:
        """Checks that independent and exponential error models have their expected correlations."""
        independent = make_dh_error_correlation("none")
        exponential = make_dh_error_correlation("exponential", 5.0)

        np.testing.assert_allclose(independent(np.array([0.0, 1.0])), [1.0, 0.0])
        assert exponential(np.array([0.0]))[0] == 1.0
        assert 0.0 < exponential(np.array([5.0]))[0] < 1.0

    @pytest.mark.parametrize("form,expected", [
        ("exponential", [1.0, math.exp(-1.5), math.exp(-3.0), math.exp(-6.0)]),
        ("gaussian", [1.0, math.exp(-0.75), math.exp(-3.0), math.exp(-12.0)]),
        ("spherical", [1.0, 0.3125, 0.0, 0.0]),
    ])
    def test_correlation_functions__shared_forms(self, form, expected):
        """Checks that spatial and elevation error correlations follow their formulas at known fractions of the range."""

        # A single spatial component isolates the curve from the nugget and other ranges
        separation = np.array([0.0, 1.0, 2.0, 4.0])
        model = RhoSurrogate(params={
            "spatial_corr_form": form, "n_spatial_components": 1,
            "q0_nugget_fraction": 0.2, "q1_range_fraction": 0.8, "r1_km": 2.0,
        })
        input_correlation = make_dh_error_correlation(form, range_years=2.0)

        # Input correlation uses years, while the spatial component uses kilometres and its fitted fraction
        np.testing.assert_allclose(input_correlation(separation), expected)
        np.testing.assert_allclose(model.spatial_corr(separation), 0.8 * np.asarray(expected))

    def test_make_dh_error_correlation__error_unknown_form(self):
        """Checks an error is raised for an unknown form even without a range."""
        with pytest.raises(ValueError, match="Unsupported"):
            make_dh_error_correlation("typo")

    @pytest.mark.parametrize("range_years", [None, 0, -1, np.inf])
    def test_make_dh_error_correlation__error_invalid_range(self, range_years):
        """Checks an error is raised for a missing or invalid error correlation range."""
        with pytest.raises(ValueError, match="range_years"):
            make_dh_error_correlation("exponential", range_years)


########################
# MASS INPUT UNCERTAINTY
########################


class TestMassInputUncertainty:
    """Test module for automatic input and total mass change uncertainty from single-period predictions."""

    @pytest.mark.parametrize("change", [-1.0, 0.0])
    @pytest.mark.parametrize("sigma", [0.0, 0.2])
    def test_predict__constant_density_input_error(self, change, sigma):
        """Checks that input mass change uncertainty equals density times volume change uncertainty."""

        # A constant density gives an exact result, including zero net volume change
        model = RhoSurrogate(params={"period_form": "none", "Bc": 0.0, "Bmem": 0.0, "A": 0.0})
        result = model.predict(dh=change, sigma_dh=sigma, dt=1.0, area_m2=1e6)
        expected_input = model.rho_ice * 1e6 * sigma
        expected_total = np.hypot(result["sigma_dM_rho_kg"], expected_input)

        np.testing.assert_allclose(result["sigma_dV_m3"], 1e6 * sigma)
        np.testing.assert_allclose(result["sigma_dM_dh_kg"], expected_input, rtol=1e-12)
        np.testing.assert_allclose(result["sigma_dM_total_kg"], expected_total, rtol=1e-12)

    def test_predict__past_rate_error_without_current_error(self):
        """Checks that uncertainty in a supplied past rate propagates despite exact current elevation change."""

        # With only a linear past rate term, its derivative gives the exact mass change error
        model = RhoSurrogate(params={"period_form": "none", "Bc": 0.0, "Bmem": 2.0, "A": 0.0, "etaMem": 1.0})
        result = model.predict(dh=-1.0, sigma_dh=0.0, dt=1.0, past_dh=-1.0, sigma_past_dh=0.2, area_m2=1e6)
        damping = np.exp(-(1.0 / model.params["H"]) ** model.params["beta"])
        expected_input = 1e6 * damping * 2.0 * 0.2

        assert result["sigma_dV_m3"] == 0.0
        np.testing.assert_allclose(result["sigma_dM_dh_kg"], expected_input, rtol=1e-12)
        assert result["sigma_dM_total_kg"] > result["sigma_dM_rho_kg"]


##############
# INPUT ERRORS
##############


class TestScalarInputErrors:
    """Test module for validation shared by scalar and integrated predictions."""

    def test_predict__error_missing_dt(self):
        """Checks an error is raised when the observation period length is omitted from predict()."""

        # A measured elevation change cannot identify its observation period by itself
        with pytest.raises(TypeError, match="dt"):
            RhoSurrogate().predict(dh=-1.0)

    @pytest.mark.parametrize("argument,value", [
        ("dh", np.nan), ("sigma_dh", -0.1), ("sigma_dh", np.inf),
        ("dt", 0.0), ("dt", -1.0), ("dt", np.inf),
        ("area_m2", -1.0), ("area_m2", 0.0),
        ("past_dh", np.inf), ("sigma_past_dh", -0.1), ("past_error_factor", -1.0),
    ])
    def test_predict__error_invalid_input(self, argument, value):
        """Checks an error is raised before invalid physical inputs enter predictions."""
        arguments = {"dh": -1.0, "dt": 1.0, argument: value}
        with pytest.raises(ValueError):
            RhoSurrogate().predict(**arguments)

    @pytest.mark.parametrize("order", [0, -2, 1.5, np.nan])
    def test_constructor__error_quadrature_order(self, order):
        """Checks an error is raised for invalid quadrature node counts."""
        with pytest.raises(ValueError, match="gh_order_current"):
            RhoSurrogate(gh_order_current=order)


class TestParameterErrors:
    """Test module for unavailable fits and obsolete parameter spellings."""

    def test_load_packaged_params__error_missing_fit(self, tmp_path, monkeypatch) -> None:
        """Checks an error is raised when the published fit is missing."""

        from glacier_density_surrogate import surrogate

        monkeypatch.setattr(surrogate.resources, "files", lambda package: tmp_path)
        with pytest.raises(FileNotFoundError):
            load_packaged_params()

    @pytest.mark.parametrize("name", ["Hd", "H_d", "rhoice", "T_p", "temporal_nugget"])
    def test_constructor__error_obsolete_name(self, name) -> None:
        """Checks an error is raised for parameter names the final study does not use."""

        with pytest.raises(ValueError, match="parameter"):
            RhoSurrogate(params={name: 1.0})

    @pytest.mark.parametrize("name,value", [
        ("H", 0), ("A0", -1), ("TP", np.inf), ("r1_km", -1),
        ("n_spatial_components", 4), ("empirical_nugget", 1.1), ("empirical_range_yr", 0),
    ])
    def test_constructor__error_invalid_parameter(self, name, value):
        """Checks an error is raised for invalid scales, fractions and component counts."""
        with pytest.raises(ValueError):
            RhoSurrogate(params={name: value})

    def test_load_params__error_missing_file(self, tmp_path) -> None:
        """Checks an error is raised rather than silently replacing a requested fit."""

        with pytest.raises(FileNotFoundError):
            load_params(tmp_path / "missing.json", verbose=False)
