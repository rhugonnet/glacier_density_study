"""Tests for time series reconciliation/manipulation."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from glacier_density_surrogate import (
    RhoSurrogate,
    expected_abs_normal_array,
    integrated_mu_vectorized,
    integrated_sigma_mass_vectorized,
    integrated_sigma_vectorized,
    make_dh_error_correlation,
    mean_density_volume_moments_vectorized,
    mean_density_volume_sigma_vectorized,
    temporally_reconcile_periods,
)


@pytest.fixture
def observations() -> pd.DataFrame:
    """Provide two consecutive annual observations on a constant glacier area."""
    return pd.DataFrame({
        "start_year": [2000.0, 2001.0], "end_year": [2001.0, 2002.0],
        "dh_m": [-1.0, -1.0], "sigma_dh_m": [0.1, 0.1], "area_m2": [1e6, 1e6],
    })


@pytest.fixture
def irregular_periods():
    """Provide two observation periods of different lengths and their combined period."""
    return pd.DataFrame({
        "rgiid": ["g1"] * 3,
        "start_year": [2000.25, 2001.25, 2000.25],
        "end_year": [2001.25, 2003.25, 2003.25],
        "dh_m": [1.0, -2.0, -1.0],
        "area": [1.0] * 3,
        "dV_m3": [1.0, -2.0, -1.0],
        "mu_rho_independent_kg_m3": [901.0, 899.0, 902.0],
    })


####################
# PERIOD PREDICTIONS
####################


class TestPeriodPredictions:
    """Test module for supplied past elevation change rates and mass changes that sum across consecutive periods."""

    def test_timeseries_without_area_hides_mass_outputs(self) -> None:
        """Checks that a time series without area returns density estimates alone."""

        # Supply consecutive elevation changes without a glacier area
        model = RhoSurrogate()
        data = pd.DataFrame(
            {
                "start": [2000, 2001],
                "end": [2001, 2002],
                "dh_m": [-0.5, -0.4],
                "sigma_dh_m": [0.1, 0.1],
            }
        )

        # Predict the observations and their combined period
        out = model.predict_timeseries(data)

        # Area, volume change and mass change cannot be reported without area
        assert "mu_rho_kg_m3" in out.columns
        assert "sigma_rho_kg_m3" in out.columns
        assert "area_m2" not in out.columns
        assert "dV_m3" not in out.columns
        assert "dM_kg" not in out.columns
        assert "sigma_dM_rho_kg" not in out.columns

    @pytest.mark.parametrize("expand", [False, True])
    @pytest.mark.parametrize("past_column,sigma_column", [
        ("past_dhdt_m_yr", "sigma_past_dhdt_m_yr"),
        ("past_dhdt", "sigma_past_dhdt"),
        ("past_dh_m", "sigma_past_dh_m"),
        ("past_dh", "sig_past_dh_m"),
        ("dh_p_m", "sigma_dh_p_m"),
        ("dh_p", "sigma_dh_p_m"),
    ])
    def test_predict_timeseries__manual_past(self, observations, expand, past_column, sigma_column) -> None:
        """Checks that expansion preserves supplied past elevation change rates and their uncertainties."""

        # Explicit rate columns and older aliases must all preserve the supplied values
        data = observations.assign(**{past_column: [-7.0, -8.0], sigma_column: [0.3, 0.4]})
        predictions = RhoSurrogate().predict_timeseries(data, expand_periods=expand)

        # Original observations preserve their predictors; longer periods use history from the same start date
        annual = predictions.loc[predictions.period_years.eq(1)]
        np.testing.assert_array_equal(annual.past_dhdt_m_yr, [-7.0, -8.0])
        np.testing.assert_array_equal(annual.sigma_past_dhdt_m_yr, [0.3, 0.4])
        if expand:
            full = predictions.loc[predictions.period_years.eq(2)].iloc[0]
            assert full.past_dhdt_m_yr == -7.0
            assert full.sigma_past_dhdt_m_yr == 0.3

    def test_predict_timeseries__past_rate_column_precedence(self, observations) -> None:
        """Checks that explicit rate columns take precedence over older column names."""

        # Conflicting older values reveal which columns the input reader selects
        data = observations.assign(
            past_dhdt_m_yr=[-0.3, -0.4], sigma_past_dhdt_m_yr=[0.1, 0.2],
            past_dh_m=[-7.0, -8.0], sigma_past_dh_m=[0.3, 0.4],
        )

        # Output rates must use the columns whose names include m yr-1
        result = RhoSurrogate().predict_timeseries(data, expand_periods=False)
        np.testing.assert_array_equal(result["past_dhdt_m_yr"], [-0.3, -0.4])
        np.testing.assert_array_equal(result["sigma_past_dhdt_m_yr"], [0.1, 0.2])

    def test_predict_timeseries__partial_manual_past(self, observations) -> None:
        """Checks that a supplied past elevation change rate or its uncertainty survives filling the other field."""

        data = observations.assign(past_dhdt_m_yr=[-7.0, np.nan], sigma_past_dhdt_m_yr=[np.nan, 0.4])
        result = RhoSurrogate().predict_timeseries(data, expand_periods=False)

        assert result.past_dhdt_m_yr.iloc[0] == -7.0
        assert result.sigma_past_dhdt_m_yr.iloc[0] == 0.2
        assert result.past_dhdt_m_yr.iloc[1] == -1.0
        assert result.sigma_past_dhdt_m_yr.iloc[1] == 0.4

    def test_timeseries_past_predictor_uses_annual_rate_for_multiannual_steps(self) -> None:
        """Checks that a five-year observation supplies a past elevation change rate and uncertainty."""

        # The first observation has a known annual rate and uncertainty
        model = RhoSurrogate()
        data = pd.DataFrame(
            {
                "start": [2000, 2005],
                "end": [2005, 2010],
                "dh_m": [-5.0, -10.0],
                "sigma_dh_m": [0.5, 0.5],
                "area_m2": [1.0e6, 1.0e6],
            }
        )

        # Predict the later observation using the earlier elevation change
        out = model.predict_timeseries(data, expand_periods=True)
        second_elementary = out.loc[(out["start"] == 2005) & (out["end"] == 2010)].iloc[0]

        # Divide both the five-year change and its error by five years
        assert math.isclose(second_elementary["past_dhdt_m_yr"], -1.0)
        assert math.isclose(second_elementary["sigma_past_dhdt_m_yr"], 0.1)

    def test_timeseries_past_predictor_redistributes_multiannual_steps_before_weighting(self) -> None:
        """Checks that memory weights apply to annual steps sharing each observation's error."""

        # Different annual rates make the contribution of each observation visible
        model = RhoSurrogate()
        data = pd.DataFrame(
            {
                "start": [2000, 2005, 2010],
                "end": [2005, 2010, 2015],
                "dh_m": [-5.0, -10.0, -5.0],
                "sigma_dh_m": [0.5, 0.5, 0.5],
                "area_m2": [1.0e6, 1.0e6, 1.0e6],
            }
        )

        # The third observation uses both earlier five-year observations
        out = model.predict_timeseries(data, expand_periods=True)
        third_elementary = out.loc[(out["start"] == 2010) & (out["end"] == 2015)].iloc[0]

        # Weight annual rates, then combine weights that share an observed error
        annual_ends = np.arange(2001, 2011, dtype=float)
        annual_rates = np.array([-1.0] * 5 + [-2.0] * 5)
        raw_weights = np.exp(-(2010.0 - annual_ends) / float(model.params["memory_tau_years"]))
        weights = raw_weights / raw_weights.sum()
        block_weights = np.array([weights[:5].sum(), weights[5:].sum()])
        expected_sigma = math.sqrt(np.sum((block_weights * 0.1) ** 2))

        # The filled rate and uncertainty must match the independent calculation
        assert math.isclose(third_elementary["past_dhdt_m_yr"], float(np.sum(weights * annual_rates)))
        assert math.isclose(third_elementary["sigma_past_dhdt_m_yr"], expected_sigma)

    @pytest.mark.parametrize("changes", [[0.0, -1.0], [1.0, -1.0], [0.0, 0.0]])
    def test_predict_timeseries__zero_change_mass(self, observations, changes) -> None:
        """Checks that zero change in an observation or their sum leaves finite mass changes that add across periods."""

        data = observations.assign(dh_m=changes)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = RhoSurrogate().predict_timeseries(data)

        # Density is undefined at zero volume change, but mass change and its error remain usable
        assert np.isfinite(result.dM_kg).all()
        assert np.isfinite(result.sigma_dM_rho_kg).all()
        assert result.loc[result.dV_m3.eq(0), "mu_rho_kg_m3"].isna().all()
        full_mass = result.loc[result.period_years.eq(2), "dM_kg"].iloc[0]
        annual_mass = result.loc[result.period_years.eq(1), "dM_kg"].sum()
        np.testing.assert_allclose(full_mass, annual_mass, atol=1e-6)

    def test_predict_timeseries__duration_only(self, observations) -> None:
        """Checks that durations without dates do not imply an overlapping temporal history."""

        data = observations.drop(columns=["start_year", "end_year"]).assign(period_years=[5, 5])
        result = RhoSurrogate().predict_timeseries(data, return_components=True)

        assert len(result) == 2
        assert not result.temporally_closed.any()
        np.testing.assert_array_equal(result.past_dhdt_m_yr, [-0.2, -0.2])

    def test_predict_timeseries__incomplete_elementary_grid(self, observations) -> None:
        """Checks that overlapping periods are predicted separately when the shorter observations are missing."""

        data = observations.assign(start_year=[2000, 2001], end_year=[2002, 2003])
        result = RhoSurrogate().predict_timeseries(data, expand_periods=False, return_components=True)

        assert not result.temporally_closed.any()
        np.testing.assert_array_equal(result.dM_kg, result.dM_independent_kg)

    @pytest.mark.parametrize("gap,duration", [(0.01, 1.0), (5.0, 5.0)])
    def test_predict_timeseries__small_gap(self, observations, gap, duration) -> None:
        """Checks that a real gap is not erased by a tolerance proportional to the calendar year."""

        # Separate annual or multiannual observations by a small or large gap
        data = observations.assign(
            start_year=[2000.0, 2000.0 + duration + gap],
            end_year=[2000.0 + duration, 2000.0 + 2 * duration + gap],
        )

        # Include independent predictions so we can compare them with the outputs
        result = RhoSurrogate().predict_timeseries(data, return_components=True)

        # A gap prevents reconciliation across the two observations
        assert len(result) == 2
        assert not result.temporally_closed.any()
        np.testing.assert_allclose(result["mu_rho_kg_m3"], result["mu_rho_independent_kg_m3"])
        np.testing.assert_allclose(result["dM_kg"], result["dM_independent_kg"])

    def test_predict_timeseries__empty_input(self, observations) -> None:
        """Checks that an empty observation table returns an empty table with the documented columns."""

        result = RhoSurrogate().predict_timeseries(observations.iloc[:0])
        assert result.empty
        assert {"start", "end", "mu_rho_kg_m3", "dM_kg"}.issubset(result.columns)

    def test_predict_timeseries__correlated_elevation_errors(self, observations):
        """Checks that expansion includes covariance between neighboring elevation errors."""
        correlation = make_dh_error_correlation("exponential", 5.0)
        result = RhoSurrogate().predict_timeseries(observations, dh_error_corr=correlation)
        combined = result.loc[result.period_years.eq(2), "sigma_dh_m"].iloc[0]

        # Two periods a year apart contribute twice their covariance to the summed variance
        expected_variance = 2 * 0.1**2 * (1 + np.exp(-3 / 5))
        np.testing.assert_allclose(combined**2, expected_variance)

    def test_predict_timeseries__anticorrelated_elevation_errors(self, observations):
        """Checks that valid opposing errors cancel when two period changes are summed."""

        # Cosine correlation is -1 for observations a year apart
        correlation = lambda lag: np.cos(np.pi * lag)
        result = RhoSurrogate().predict_timeseries(observations, dh_error_corr=correlation)
        combined = result.loc[result.period_years.eq(2), "sigma_dh_m"].iloc[0]

        # Equal and opposite errors have zero variance in their sum
        assert combined == 0.0


class TestMultipleGlacierTimeseries:
    """Test module for independent glacier histories, identifiers and regional metadata in shared tables."""

    @pytest.mark.parametrize("column,id_col", [("glacier_id", None), ("rgiid", None), ("name", "name")])
    @pytest.mark.parametrize("expand", [False, True])
    def test_predict_timeseries__multiple_glaciers(self, observations, column, id_col, expand):
        """Checks that shared table predictions match separate glacier calls, including past rates and reconciliation."""

        # Distinct elevation histories and areas make accidental mixing visible
        first = observations.assign(**{column: "g2"}, region_group="A", lat=28.0, lon=87.0)
        second = observations.assign(
            **{column: "g1"}, dh_m=[0.6, -1.8], area_m2=2e6, region_group="B", lat=29.0, lon=88.0,
        )
        shared = pd.concat([first, second], ignore_index=True).iloc[[0, 2, 1, 3]]
        original = shared.copy()
        model = RhoSurrogate()
        options = {"expand_periods": expand, "return_components": True, "past_error_factor": 3.0}

        # Predict both glaciers in one call, forwarding the same options to each
        result = model.predict_timeseries(shared, id_col=id_col, **options)

        # Identifiers and static metadata accompany every original or expanded period
        assert result[column].drop_duplicates().tolist() == ["g2", "g1"]
        metadata = [column, "region_group", "lat", "lon"]
        for glacier, table in (("g2", first), ("g1", second)):
            expected = model.predict_timeseries(table.drop(columns=metadata), **options)
            actual = result.loc[result[column] == glacier].reset_index(drop=True)
            pd.testing.assert_frame_equal(actual.drop(columns=metadata), expected)
            for name in metadata[1:]:
                assert actual[name].eq(table[name].iloc[0]).all()
        pd.testing.assert_frame_equal(shared, original)

    def test_predict_timeseries__multiple_glaciers_without_area(self, observations):
        """Checks that glacier IDs are returned with density estimates when no area is supplied."""

        # Neither glacier has a physical area for volume or mass change outputs
        table = observations.drop(columns="area_m2")
        shared = pd.concat([table.assign(glacier_id="g1"), table.assign(glacier_id="g2")], ignore_index=True)
        result = RhoSurrogate().predict_timeseries(shared)

        # Both independent time series expand, without inventing physical mass changes
        assert len(result) == 6
        assert result["glacier_id"].nunique() == 2
        assert "dM_kg" not in result

    def test_predict_timeseries__empty_glacier_table(self, observations):
        """Checks that an empty shared table preserves identifier and metadata columns."""

        table = observations.assign(glacier_id="g1", lat=28.0, lon=87.0).iloc[:0]
        result = RhoSurrogate().predict_timeseries(table)

        assert result.empty
        assert {"glacier_id", "lat", "lon", "mu_rho_kg_m3"}.issubset(result.columns)

    @pytest.mark.parametrize("column", ["lat", "lon", "region_group"])
    def test_predict_timeseries__error_changing_metadata(self, observations, column):
        """Checks an error is raised for coordinates or region membership that change within a glacier."""

        table = observations.assign(glacier_id="g1", lat=28.0, lon=87.0, region_group="A")
        table[column] = [1, 2]
        with pytest.raises(ValueError, match=f"consistent {column}"):
            RhoSurrogate().predict_timeseries(table)

    def test_predict_timeseries__error_missing_identifier(self, observations):
        """Checks an error is raised for missing glacier IDs instead of dropping their observations."""

        table = observations.assign(glacier_id=["g1", None])
        with pytest.raises(ValueError, match="identifiers must not be missing"):
            RhoSurrogate().predict_timeseries(table)

    def test_predict_timeseries__error_unknown_identifier_column(self, observations):
        """Checks an error is raised when an explicitly selected glacier ID column is absent."""

        with pytest.raises(ValueError, match="identifier column is missing"):
            RhoSurrogate().predict_timeseries(observations, id_col="name")


##########################
# SHARED INPUT UNCERTAINTY
##########################


class TestInputUncertainty:
    """Test module for automatic input propagation, shared histories and reproducible mass change errors."""

    @pytest.mark.parametrize("id_col", [None, "glacier_id", "rgiid", "inventory_id"])
    def test_predict_timeseries__constant_density_input_error(self, observations, id_col):
        """Checks that automatic input mass change error equals ice density times volume change error."""

        # Constant density gives an exact reference independent of sampling convergence
        model = RhoSurrogate(params={"period_form": "none", "Bc": 0.0, "Bmem": 0.0, "A": 0.0})
        if id_col is not None:
            first = observations.assign(**{id_col: "g1"})
            second = observations.assign(area_m2=2e6, **{id_col: "g2"})
            observations = pd.concat([first, second], ignore_index=True)
        original = observations.copy()

        # Prediction includes the input component without any additional public call
        result = model.predict_timeseries(observations, id_col=id_col)
        expected_volume_error = result["area_m2"] * result["sigma_dh_m"]
        expected_input_error = model.rho_ice * expected_volume_error
        expected_total_error = np.hypot(expected_input_error, result["sigma_dM_rho_kg"])

        np.testing.assert_allclose(result["sigma_dV_m3"], expected_volume_error)
        np.testing.assert_allclose(result["sigma_dM_dh_kg"], expected_input_error, rtol=1e-12)
        np.testing.assert_allclose(result["sigma_dM_total_kg"], expected_total_error, rtol=1e-12)
        pd.testing.assert_frame_equal(observations, original)

    def test_predict_timeseries__reproducible_input_errors(self, observations):
        """Checks that input errors are reproducible and unaffected by the order of observation rows."""

        # Sorting the input must preserve both predictions and sampled error estimates
        model = RhoSurrogate()
        first = model.predict_timeseries(observations)
        second = model.predict_timeseries(observations.iloc[::-1])

        pd.testing.assert_frame_equal(first, second)
        assert np.isfinite(first["sigma_dM_dh_kg"]).all()

    def test_predict_timeseries__empty_uncertainty_schema(self, observations):
        """Checks that empty observations return the same automatic uncertainty columns as populated tables."""

        result = RhoSurrogate().predict_timeseries(observations.iloc[:0])

        assert result.empty
        assert {"sigma_dV_m3", "sigma_dM_dh_kg", "sigma_dM_total_kg"}.issubset(result.columns)

    @pytest.mark.parametrize("omit_errors", [False, True])
    def test_predict_timeseries__exact_inputs_skip_sampling(self, observations, omit_errors, monkeypatch):
        """Checks that omitted or zero input errors skip sampling and leave only surrogate residual uncertainty."""
        import glacier_density_surrogate.timeseries as timeseries

        def reject_sampling(*args, **kwargs):
            """Reject sampling when every input is exact."""
            raise AssertionError("Exact observations must not be sampled")

        # Both an omitted error column and explicit zeros represent exact observations
        data = observations.drop(columns="sigma_dh_m") if omit_errors else observations.assign(sigma_dh_m=0.0)
        monkeypatch.setattr(timeseries, "_sample_reconciled_mass", reject_sampling)
        result = RhoSurrogate().predict_timeseries(data)

        np.testing.assert_array_equal(result["sigma_dV_m3"], 0.0)
        np.testing.assert_array_equal(result["sigma_dM_dh_kg"], 0.0)
        np.testing.assert_array_equal(result["sigma_dM_total_kg"], result["sigma_dM_rho_kg"])

    def test_predict_timeseries__supplied_past_rate_error(self, observations):
        """Checks that uncertain supplied past rates propagate into mass even when current elevations are exact."""

        # With etaMem=1, only the linear past rate term varies when current change is exact
        model = RhoSurrogate(params={"period_form": "none", "Bc": 0.0, "Bmem": 2.0, "A": 0.0, "etaMem": 1.0})
        data = observations.assign(sigma_dh_m=0.0, past_dhdt_m_yr=-1.0, sigma_past_dhdt_m_yr=[0.2, 0.3])
        result = model.predict_timeseries(data, expand_periods=False)

        # The derivative of mass with respect to the past rate gives the exact input error
        damping = np.exp(-(np.abs(result["dh_m"]) / model.params["H"]) ** model.params["beta"])
        expected = result["area_m2"] * np.abs(result["dh_m"]) * damping * 2.0 * result["sigma_past_dhdt_m_yr"]
        np.testing.assert_allclose(result["sigma_dV_m3"], 0.0)
        np.testing.assert_allclose(result["sigma_dM_dh_kg"], expected, rtol=1e-11)
        assert (result["sigma_dM_total_kg"] > result["sigma_dM_rho_kg"]).all()

    def test_predict_timeseries__earlier_observation_error(self, observations):
        """Checks that an uncertain earlier observation changes later mass uncertainty despite exact current change."""

        # The later period has exact volume change and a past rate inferred from the earlier observation
        model = RhoSurrogate(params={"period_form": "none", "Bc": 0.0, "Bmem": 2.0, "A": 0.0, "etaMem": 1.0})
        data = observations.assign(sigma_dh_m=[0.1, 0.0])
        result = model.predict_timeseries(data, expand_periods=False)
        later = result.iloc[1]

        damping = np.exp(-(abs(later.dh_m) / model.params["H"]) ** model.params["beta"])
        expected = later.area_m2 * abs(later.dh_m) * damping * 2.0 * 0.1
        assert later.sigma_dV_m3 == 0.0
        np.testing.assert_allclose(later.sigma_dM_dh_kg, expected, rtol=1e-11)

    @pytest.mark.parametrize("correlation", ["exponential", "opposing"])
    def test_predict_timeseries__correlated_input_mass_error(self, observations, correlation):
        """Checks that constant-density input mass errors include positive covariance or perfect cancellation."""

        model = RhoSurrogate(params={"period_form": "none", "Bc": 0.0, "Bmem": 0.0, "A": 0.0})
        if correlation == "exponential":
            corr = make_dh_error_correlation("exponential", 5.0)
            combined_sigma = 0.1 * np.sqrt(2 * (1 + np.exp(-3 / 5)))
        else:
            corr = lambda lag: np.cos(np.pi * lag)
            combined_sigma = 0.0

        # The full-period input variance is known from the two-observation covariance matrix
        result = model.predict_timeseries(observations, dh_error_corr=corr)
        full = result.loc[result.period_years.eq(2)].iloc[0]
        expected = model.rho_ice * 1e6 * combined_sigma
        np.testing.assert_allclose(full.sigma_dM_dh_kg, expected, rtol=1e-11, atol=1e-6)
        np.testing.assert_allclose(full.sigma_dV_m3, 1e6 * combined_sigma, atol=1e-9)

    @pytest.mark.parametrize("options", [
        {"past_missing": "zero"}, {"past_error_factor": 0.0}, {"past_error_factor": 0.5},
        {"past_error_factor": 3.0}, {"expand_periods": False},
    ])
    def test_predict_timeseries__history_options_propagate(self, observations, options):
        """Checks that existing history and expansion options also produce automatic input uncertainty."""

        result = RhoSurrogate().predict_timeseries(observations, **options)

        assert np.isfinite(result["sigma_dM_dh_kg"]).all()
        assert (result["sigma_dM_dh_kg"] > 0).all()
        np.testing.assert_allclose(
            result["sigma_dM_total_kg"]**2,
            result["sigma_dM_rho_kg"]**2 + result["sigma_dM_dh_kg"]**2,
        )

    def test_predict_timeseries__custom_columns_propagate(self, observations):
        """Checks that custom elevation and history column names produce the same input errors as canonical names."""

        model = RhoSurrogate()
        original = observations.assign(past_dhdt_m_yr=[-0.3, np.nan], sigma_past_dhdt_m_yr=[0.2, 0.3])
        renamed = original.rename(columns={
            "start_year": "begin", "end_year": "finish", "dh_m": "change", "sigma_dh_m": "error",
            "area_m2": "size", "past_dhdt_m_yr": "history", "sigma_past_dhdt_m_yr": "history_error",
        })
        result = model.predict_timeseries(
            renamed, start_col="begin", end_col="finish", dh_col="change", sigma_dh_col="error",
            area_col="size", past_dhdt_col="history", sigma_past_dhdt_col="history_error",
        )

        pd.testing.assert_frame_equal(result, model.predict_timeseries(original))

    def test_predict_timeseries__supplied_overlapping_observations(self, observations):
        """Checks that independent overlapping observations propagate through the additive mass change fit."""

        # An independently measured two-year period supplies an additional estimate on the same grid
        model = RhoSurrogate(params={"period_form": "none", "Bc": 0.0, "Bmem": 0.0, "A": 0.0, "A0": 3.0, "A1": 4.0})
        full_period = observations.iloc[:1].assign(end_year=2002.0, dh_m=-2.0, sigma_dh_m=np.sqrt(0.02))
        data = pd.concat([observations, full_period], ignore_index=True)
        result = model.predict_timeseries(data)

        # At constant density the fit is linear, so its covariance has an exact matrix solution
        support = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        weights = np.diag([1.0, 0.5, 1.0])
        projection = support @ np.linalg.solve(support.T @ weights @ support, support.T @ weights)
        input_covariance = np.diag((model.rho_ice * 1e6 * result["sigma_dh_m"])**2)
        expected = np.sqrt(np.diag(projection @ input_covariance @ projection.T))

        np.testing.assert_allclose(result["sigma_dM_dh_kg"], expected, rtol=1e-11)
        np.testing.assert_allclose(result.iloc[1].dM_kg, result.iloc[[0, 2]].dM_kg.sum())

    def test_predict_timeseries__shared_supplied_past_rate_error(self, observations):
        """Checks that periods with the same start reuse a supplied past rate error before mass reconciliation."""

        # Exact current changes and a linear past rate term make the input covariance explicit
        model = RhoSurrogate(params={"period_form": "none", "Bc": 0.0, "Bmem": 2.0, "A": 0.0, "etaMem": 1.0, "A0": 3.0, "A1": 4.0})
        data = observations.assign(sigma_dh_m=0.0, past_dhdt_m_yr=-1.0, sigma_past_dhdt_m_yr=0.2)
        result = model.predict_timeseries(data)
        damping = np.exp(-(np.abs(result["dh_m"]) / model.params["H"]) ** model.params["beta"])
        raw_sigma = result["area_m2"] * np.abs(result["dh_m"]) * damping * 2.0 * 0.2

        # The first and combined periods share one history error; the later period has another
        same_start = result["start"].to_numpy()[:, None] == result["start"].to_numpy()
        input_covariance = np.outer(raw_sigma, raw_sigma) * same_start
        support = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        weights = np.diag([1.0, 0.5, 1.0])
        projection = support @ np.linalg.solve(support.T @ weights @ support, support.T @ weights)
        expected = np.sqrt(np.diag(projection @ input_covariance @ projection.T))

        np.testing.assert_allclose(result["sigma_dM_dh_kg"], expected, rtol=1e-11)

    @pytest.mark.parametrize("dated", [False, True])
    def test_predict_timeseries__single_period_matches_predict(self, observations, dated):
        """Checks that a single observation returns the same mass uncertainty components as predict()."""

        model = RhoSurrogate()
        data = observations.iloc[:1]
        if not dated:
            data = data.drop(columns=["start_year", "end_year"]).assign(period_years=1.0)
        result = model.predict_timeseries(data).iloc[0]
        expected = model.predict(dh=-1.0, sigma_dh=0.1, dt=1.0, area_m2=1e6)

        for column in ("dM_kg", "sigma_dM_rho_kg", "sigma_dM_dh_kg", "sigma_dM_total_kg", "sigma_dV_m3"):
            np.testing.assert_allclose(result[column], expected[column], rtol=1e-12)

    def test_predict_timeseries__duration_only_input_errors(self, observations):
        """Checks that duration-only rows automatically propagate their independent predictor uncertainties."""

        model = RhoSurrogate()
        data = observations.drop(columns=["start_year", "end_year"]).assign(period_years=5.0)
        result = model.predict_timeseries(data)

        # Durations imply neither shared history nor reconciliation between rows
        expected = model.predict(dh=-1.0, sigma_dh=0.1, dt=5.0, area_m2=1e6)
        np.testing.assert_allclose(result["sigma_dM_dh_kg"], expected["sigma_dM_dh_kg"])
        np.testing.assert_allclose(result["sigma_dM_total_kg"], expected["sigma_dM_total_kg"])

    def test_predict_timeseries__error_indefinite_covariance(self, observations):
        """Checks an error is raised for elevation correlations that cannot represent a joint error distribution."""

        # Three errors with pairwise correlation -0.75 have a negative covariance eigenvalue
        third = observations.iloc[:1].assign(start_year=2002.0, end_year=2003.0)
        data = pd.concat([observations, third], ignore_index=True)
        correlation = lambda lag: np.where(lag == 0, 1.0, -0.75)

        with pytest.raises(ValueError, match="positive semidefinite"):
            RhoSurrogate().predict_timeseries(data, expand_periods=False, dh_error_corr=correlation)


###################
# BATCH INTEGRATION
###################


class TestBatchIntegration:
    """Test module for scalar, broadcast and invalid predictor inputs."""

    @pytest.mark.parametrize("shape", [(), (3,), (2, 3)])
    def test_integrated_mu_vectorized__broadcast(self, shape):
        """Checks that broadcasting scalar errors agrees with individual predictions."""
        model = RhoSurrogate()
        dh = np.full(shape, -2.0)
        expected = model.integrated_mu(dh=-2.0, sigma_dh=0.2, past_dhdt=-0.4, sigma_past_dhdt=0.1, dt=5.0)

        result = integrated_mu_vectorized(model, dh, 0.2, -0.4, 0.1, 5.0)

        assert result.shape == shape
        np.testing.assert_allclose(result, expected)

    def test_integrated_sigma_vectorized__broadcast(self):
        """Checks that density and mass change uncertainties broadcast periods and areas."""
        model = RhoSurrogate()
        durations = np.array([1.0, 5.0])
        expected = [model.integrated_sigma(-2.0, 0.2, dt) for dt in durations]

        density = integrated_sigma_vectorized(model, -2.0, 0.2, durations)
        mass = integrated_sigma_mass_vectorized(model, 3.0, -2.0, 0.2, durations)

        np.testing.assert_allclose(density, expected)
        np.testing.assert_allclose(mass, 6.0 * np.asarray(expected))

    def test_vectorized_integration_matches_scalar_api(self) -> None:
        """Checks that batch mean and uncertainty calculations agree with individual calls."""

        # Cover losses, gains, uncertain predictors and different period lengths
        model = RhoSurrogate(gh_order_current=32, gh_order_past=32)
        dh = np.array([-10.0, -1.0, 0.5, 8.0])
        sigma_dh = np.array([0.2, 0.4, 0.1, 0.5])
        past = np.array([-0.5, 0.0, -0.5, 0.25])
        sigma_past = np.array([0.1, 0.0, 0.2, 0.1])
        dt = np.array([20.0, 1.0, 5.0, 10.0])

        # Use individual public predictions as the reference for each observation
        mu_loop = np.array([model.integrated_mu(a, b, c, d, e) for a, b, c, d, e in zip(dh, sigma_dh, past, sigma_past, dt)])
        sig_loop = np.array([model.integrated_sigma(a, b, e) for a, b, e in zip(dh, sigma_dh, dt)])

        # Batch calculations must give the same mean and uncertainty
        np.testing.assert_allclose(integrated_mu_vectorized(model, dh, sigma_dh, past, sigma_past, dt), mu_loop)
        np.testing.assert_allclose(integrated_sigma_vectorized(model, dh, sigma_dh, dt), sig_loop)

    def test_expected_abs_normal_array__scalar_sigma(self):
        """Checks that a scalar standard deviation broadcasts over normal means."""
        means = np.array([-1.0, 0.0, 1.0])
        result = expected_abs_normal_array(means, 0.0)
        np.testing.assert_equal(result, np.abs(means))

    def test_mean_density_volume_variance_vanishes_without_predictor_uncertainty(self) -> None:
        """Checks that exact predictors give zero variance in the mean mass change."""

        # Exact predictors provide a known zero-variance reference
        model = RhoSurrogate()

        # Calculate input uncertainty for two different glacier areas
        sigma = mean_density_volume_sigma_vectorized(
            model,
            area_m2=np.array([1.0e6, 2.0e6]),
            dh=np.array([-4.0, 3.0]),
            sigma_dh=np.array([0.0, 0.0]),
            past_dhdt=np.array([-0.5, 0.25]),
            sigma_past_dhdt=np.array([0.0, 0.0]),
            dt=np.array([5.0, 10.0]),
        )

        # Neither prediction can vary when both inputs are exact
        np.testing.assert_allclose(sigma, 0.0, atol=1.0e-6)

    def test_mean_density_volume_variance_matches_constant_mean_limit(self) -> None:
        """Checks that constant density gives the usual area times density conversion of error."""

        # Remove all predictor dependence to give a constant density of 850
        params = {
            "rho_ice_fixed": 850.0,
            "period_form": "none",
            "Bc": 0.0,
            "P0": 0.0,
            "P1": 0.0,
            "Bmem": 0.0,
            "A": 0.0,
        }
        model = RhoSurrogate(params=params)
        area = np.array([1.0e6, 3.0e6])
        dh = np.array([-5.0, 2.0])
        sigma_dh = np.array([0.4, 1.5])

        # Integrate the mass change over the two uncertain predictors
        mean, variance = mean_density_volume_moments_vectorized(
            model,
            area_m2=area,
            dh=dh,
            sigma_dh=sigma_dh,
            past_dhdt=np.array([0.0, -1.0]),
            sigma_past_dhdt=np.array([0.0, 0.2]),
            dt=np.array([5.0, 1.0]),
        )

        # Constant density scales elevation change and its error by density times area
        np.testing.assert_allclose(mean, model.rho_ice * area * dh, rtol=1.0e-12)
        np.testing.assert_allclose(variance, (model.rho_ice * area * sigma_dh) ** 2, rtol=1.0e-12)

    @pytest.mark.parametrize("sigma", [1e-5, 1e-7, 1e-9])
    def test_mean_density_volume_moments_vectorized__small_error(self, sigma):
        """Checks that small real errors are preserved without subtracting large moments."""
        model = RhoSurrogate(params={"period_form": "none", "Bc": 0.0, "Bmem": 0.0, "A": 0.0})
        area = 1e6
        mean, variance = mean_density_volume_moments_vectorized(model, area, 20.0, sigma, 0.0, 0.0, 5.0)

        # Constant density gives an independent reference even at tiny uncertainty
        np.testing.assert_allclose(mean, area * model.rho_ice * 20.0)
        np.testing.assert_allclose(variance, (area * model.rho_ice * sigma) ** 2, rtol=1e-5)

    def test_mean_density_volume_moments_vectorized__random_reference(self):
        """Checks that mass change means and variances match direct integration over random predictors."""
        model = RhoSurrogate(gh_order_current=20, gh_order_past=20)
        rng = np.random.default_rng(46)
        changes = rng.uniform(-15, 15, 24)
        past = rng.uniform(-2, 2, 24)
        sigma = rng.uniform(0.01, 1, 24)
        sigma_past = rng.uniform(0.01, 0.5, 24)
        duration = rng.uniform(0.5, 20, 24)
        mean, variance = mean_density_volume_moments_vectorized(
            model, 1e6, changes, sigma, past, sigma_past, duration,
        )

        # Calculate variance from the mass changes at the integration nodes and their distances from the mean
        nodes, node_weights = np.polynomial.hermite.hermgauss(20)
        weights = np.outer(node_weights, node_weights) / np.pi
        for index in range(len(changes)):
            current = changes[index] + np.sqrt(2) * sigma[index] * nodes
            history = past[index] + np.sqrt(2) * sigma_past[index] * nodes
            density = model.mu_rho(current[:, None], history[None, :], duration[index])
            mass = 1e6 * current[:, None] * density
            expected_mean = np.sum(weights * mass)
            expected_variance = np.sum(weights * (mass - expected_mean) ** 2)
            np.testing.assert_allclose(mean[index], expected_mean, rtol=1e-12)
            np.testing.assert_allclose(variance[index], expected_variance, rtol=1e-11)

    @pytest.mark.parametrize("sigma", [-0.1, np.nan, np.inf])
    def test_expected_abs_normal_array__error_invalid_sigma(self, sigma):
        """Checks an error is raised for invalid normal standard deviations."""
        with pytest.raises(ValueError, match="sigma"):
            expected_abs_normal_array(np.array([0.0]), sigma)


#########################
# TEMPORAL RECONCILIATION
#########################


class TestTemporalReconciliation:
    """Test module for fractional dates, missing observations and volume changes that sum across periods."""

    def test_temporal_closure_preserves_additive_mass_change(self) -> None:
        """Checks that reconciled annual mass changes sum to the mass change over the full period."""

        # Include a gain between two losses to check changes of sign
        model = RhoSurrogate()
        data = pd.DataFrame(
            {
                "start": [2000, 2001, 2002],
                "end": [2001, 2002, 2003],
                "dh_m": [-0.5, 0.2, -0.7],
                "sigma_dh_m": [0.1, 0.1, 0.1],
                "area_m2": [1.0e6, 1.0e6, 1.0e6],
            }
        )

        # Compare the three annual predictions with their combined period
        out = model.predict_timeseries(data, expand_periods=True)
        annual = out.loc[np.isclose(out["period_years"], 1.0), "dM_kg"].sum()
        full = out.loc[(out["start"] == 2000) & (out["end"] == 2003), "dM_kg"].iloc[0]

        # Mass changes must add, and intermediate results are hidden by default
        assert math.isclose(annual, full, rel_tol=1.0e-12)
        assert "mu_rho_independent_kg_m3" not in out.columns
        assert "mu_rho_closed_kg_m3" not in out.columns
        assert "temporally_closed" not in out.columns

        # Requesting components exposes the independent and reconciled estimates
        components = model.predict_timeseries(data, expand_periods=True, return_components=True)
        np.testing.assert_allclose(components["mu_rho_kg_m3"], components["mu_rho_closed_kg_m3"])
        assert "mu_rho_independent_kg_m3" in components.columns

    def test_temporally_reconcile_periods__irregular_grid(self, irregular_periods):
        """Checks that mass changes from periods of different lengths sum to the total mass change."""
        model = RhoSurrogate(params={"A0": 1.0, "A1": 0.0})
        result = temporally_reconcile_periods(irregular_periods, model)

        np.testing.assert_allclose(result["dM_surrogate_kg"].iloc[2], result["dM_surrogate_kg"].iloc[:2].sum())
        np.testing.assert_allclose(result["sigma_dM_rho_surrogate_kg"], [1.0, np.sqrt(2), np.sqrt(3)])
        np.testing.assert_equal(result["start_year"].to_numpy(), irregular_periods["start_year"].to_numpy())

    def test_temporal_closure_uses_path_length_weights(self) -> None:
        """Checks that reconciliation weights use the variance from accumulated elevation changes."""

        # With A1 zero, residual variance equals the accumulated elevation change
        model = RhoSurrogate(params={"A0": 1.0, "A1": 0.0})
        periods = pd.DataFrame(
            {
                "rgiid": ["g1", "g1", "g1"],
                "start_year": [2000, 2001, 2000],
                "end_year": [2001, 2002, 2002],
                "dh_m": [1.0, 1.0, 2.0],
                "area": [1.0, 1.0, 1.0],
                "dV_m3": [1.0, 1.0, 2.0],
                "mu_rho_independent_kg_m3": [901.0, 901.0, 950.0],
                "sigma_dM_rho_independent_kg": [1.0, 1.0, 1.0e-6],
            }
        )

        # Reconcile the supplied estimates and subtract the ice density contribution
        out = temporally_reconcile_periods(periods, model)
        annual_anomaly = (
            out.loc[(out["start_year"] == 2000) & (out["end_year"] == 2001), "dM_surrogate_kg"].iloc[0]
            - model.rho_ice
        )
        combined_anomaly = (
            out.loc[(out["start_year"] == 2000) & (out["end_year"] == 2002), "dM_surrogate_kg"].iloc[0]
            - 2.0 * model.rho_ice
        )

        # The combined anomaly of 51 is divided equally between the two annual steps
        assert math.isclose(annual_anomaly, 25.5, rel_tol=1.0e-12)
        assert math.isclose(combined_anomaly, 51.0, rel_tol=1.0e-12)

    def test_temporal_closure_is_invariant_to_area_units(self) -> None:
        """Checks that rescaling area and volume change together leaves reconciled densities unchanged."""

        # Supply two glacier histories with matching individual and combined changes
        model = RhoSurrogate()
        base = pd.DataFrame(
            {
                "rgiid": ["g1", "g1", "g1", "g2", "g2", "g2"],
                "start_year": [2000, 2005, 2000, 2000, 2005, 2000],
                "end_year": [2005, 2010, 2010, 2005, 2010, 2010],
                "dh_m": [-5.0, -3.0, -8.0, 2.0, -6.0, -4.0],
                "area": [1.2, 1.2, 1.2, 0.8, 0.8, 0.8],
                "dV_m3": [-6.0, -3.6, -9.6, 1.6, -4.8, -3.2],
                "mu_rho_independent_kg_m3": [860.0, 880.0, 870.0, 740.0, 850.0, 820.0],
                "sigma_dM_rho_independent_kg": [10.0, 8.0, 12.0, 7.0, 9.0, 11.0],
            }
        )
        # Rescale all area-dependent quantities together
        scaled = base.copy()
        scaled["area"] *= 1.0e6
        scaled["dV_m3"] *= 1.0e6
        scaled["sigma_dM_rho_independent_kg"] *= 1.0e6

        # Reconcile both tables and align their rows for comparison
        out_base = temporally_reconcile_periods(base, model).sort_values(["rgiid", "start_year", "end_year"])
        out_scaled = temporally_reconcile_periods(scaled, model).sort_values(["rgiid", "start_year", "end_year"])

        # Effective density is a ratio and must not change with the area scale
        np.testing.assert_allclose(
            out_scaled["mu_rho_surrogate_kg_m3"].to_numpy(float),
            out_base["mu_rho_surrogate_kg_m3"].to_numpy(float),
            rtol=1.0e-12,
        )

    def test_temporally_reconcile_periods__zero_volume_with_mass(self, irregular_periods):
        """Checks that explicit mass change handles undefined density at zero volume change."""
        table = irregular_periods.copy()
        table["dh_m"] = [1.0, -1.0, 0.0]
        table["dV_m3"] = [1.0, -1.0, 0.0]
        table["dM_independent_kg"] = [901.0, -899.0, 2.0]
        table.loc[2, "mu_rho_independent_kg_m3"] = np.nan

        result = temporally_reconcile_periods(table, RhoSurrogate())

        np.testing.assert_allclose(result["dM_surrogate_kg"], table["dM_independent_kg"])
        assert np.isnan(result["mu_rho_surrogate_kg_m3"].iloc[2])

    def test_temporally_reconcile_periods__empty(self, irregular_periods):
        """Checks that empty period tables return an empty table with the expected prediction columns."""
        result = temporally_reconcile_periods(irregular_periods.iloc[:0], RhoSurrogate())
        assert result.empty
        assert "dM_surrogate_kg" in result

    def test_temporally_reconcile_periods__error_missing_observation(self, irregular_periods):
        """Checks an error is raised instead of fitting absent observations as zero."""
        partial = irregular_periods.iloc[:2].copy()
        partial["rgiid"] = "g2"
        table = pd.concat([irregular_periods, partial], ignore_index=True)

        with pytest.raises(ValueError, match="complete"):
            temporally_reconcile_periods(table, RhoSurrogate())

    def test_temporally_reconcile_periods__error_missing_elementary(self, irregular_periods):
        """Checks an error is raised when the first observation period has no measurement."""
        with pytest.raises(ValueError, match="elementary"):
            temporally_reconcile_periods(irregular_periods.iloc[1:], RhoSurrogate())

    def test_temporally_reconcile_periods__error_inconsistent_volume(self, irregular_periods):
        """Checks an error is raised when individual volume changes do not sum to the combined change."""
        irregular_periods.loc[2, "dV_m3"] = -10.0
        with pytest.raises(ValueError, match="volume"):
            temporally_reconcile_periods(irregular_periods, RhoSurrogate())


####################
# OBSERVATION ERRORS
####################


class TestPeriodErrors:
    """Test module for invalid observations and incompatible volume change histories."""

    @pytest.mark.parametrize("column,value", [
        ("dh_m", np.nan), ("sigma_dh_m", -0.1), ("sigma_dh_m", np.inf),
        ("area_m2", 0), ("area_m2", -1), ("start_year", np.inf),
        ("end_year", 1999), ("sigma_past_dhdt_m_yr", -0.1),
    ])
    def test_predict_timeseries__error_invalid_observation(self, observations, column, value) -> None:
        """Checks an error is raised for invalid observations rather than silently dropping rows."""

        data = observations.assign(**{column: value})
        with pytest.raises(ValueError):
            RhoSurrogate().predict_timeseries(data)

    def test_predict_timeseries__error_varying_area_expansion(self, observations) -> None:
        """Checks an error is raised when changing areas would prevent volume changes from adding across periods."""

        with pytest.raises(ValueError, match="area"):
            RhoSurrogate().predict_timeseries(observations.assign(area_m2=[1e6, 2e6]))

    @pytest.mark.parametrize("case", ["shape", "missing", "range", "diagonal", "asymmetric"])
    def test_predict_timeseries__error_invalid_correlation(self, observations, case):
        """Checks an error is raised for malformed temporal error correlation matrices."""

        # Each callback violates one requirement of a correlation matrix
        def correlation(lag):
            """Return a malformed matrix for the selected validation case."""
            if case == "shape":
                return np.ones(1)
            if case == "missing":
                return np.full_like(lag, np.nan)
            if case == "range":
                return np.where(lag == 0, 1.0, 1.1)
            if case == "diagonal":
                return np.where(lag == 0, 0.5, 0.0)
            return np.eye(len(lag)) + 0.2 * np.triu(np.ones_like(lag), k=1)

        with pytest.raises(ValueError, match="correlation"):
            RhoSurrogate().predict_timeseries(observations, dh_error_corr=correlation)

    @pytest.mark.parametrize("expand", [False, True])
    def test_predict_timeseries__error_negative_correlated_variance(self, expand):
        """Checks an error is raised instead of clipping impossible error variance to zero."""

        # Pairwise correlation -0.9 is valid for two errors but impossible for three
        data = pd.DataFrame({
            "start_year": np.arange(2000, 2004), "end_year": np.arange(2001, 2005),
            "dh_m": -0.5, "sigma_dh_m": 0.1,
        })
        correlation = lambda lag: np.where(lag == 0, 1.0, -0.9)

        # Both summed changes and weighted past elevation change rates must reject negative variance
        with pytest.raises(ValueError, match="negative variance"):
            RhoSurrogate().predict_timeseries(data, expand_periods=expand, dh_error_corr=correlation)
