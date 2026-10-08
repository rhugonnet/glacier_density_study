"""Tests for spatial propagation accounting for correlated errors."""

import numpy as np
import pandas as pd
import pytest

from glacier_density_surrogate import (
    RhoSurrogate,
    aggregate_regions,
    haversine_distance_matrix,
    spherical_correlation,
    spatially_correlated_sigma_by_group_period,
    spatially_correlated_component_sigma_by_group_period,
    summarize_global_period_conversions,
    summarize_region_period_conversions,
)


@pytest.fixture
def glaciers():
    """Synthetic dataframe with two glacier errors with an independent STD."""
    return pd.DataFrame({
        "region_group": ["A", "A"], "rgiid": ["g1", "g2"],
        "start_year": [2000.25] * 2, "end_year": [2001.25] * 2,
        "lat": [0.0, 1.0], "lon": [0.0, 1.0], "sigma": [3.0, 4.0], "area": [1.0, 1.0],
    })


def independent(distance):
    """Return zero correlation between glaciers so their variances add independently."""
    return np.zeros_like(distance)


class TestSpatialPropagation:

    def test_spatially_correlated_sigma_by_group_period__fractional_years(self, glaciers):
        """Checks that fractional period bounds preserve both errors and keys."""
        result = spatially_correlated_sigma_by_group_period(glaciers, independent, sigma_col="sigma")
        assert result == {("A", 2000.25, 2001.25): 5.0}

    @pytest.mark.parametrize("fraction", [0.0, 0.5, 1.0])
    def test_spatially_correlated_sigma_by_group_period__sample_all(self, glaciers, fraction):
        """Checks that sampling the full group agrees with exact covariance."""
        correlation = lambda distance: np.full_like(distance, fraction)
        exact = spatially_correlated_sigma_by_group_period(glaciers, correlation, sigma_col="sigma", block_size=1)
        sampled = spatially_correlated_sigma_by_group_period(
            glaciers, correlation, sigma_col="sigma", exact_max_items=1, subsample_max_items=2, block_size=1,
        )
        np.testing.assert_allclose(list(exact.values()), np.sqrt(25.0 + 24.0 * fraction))
        np.testing.assert_allclose(list(sampled.values()), list(exact.values()))

    def test_spatially_correlated_component_sigma_by_group_period__period_area(self, glaciers):
        """Checks that each observation uses its own area when scaling errors."""
        second = glaciers.copy()
        second["start_year"] += 1
        second["end_year"] += 1
        second["area"] *= 2
        table = pd.concat([glaciers, second], ignore_index=True)

        result = spatially_correlated_component_sigma_by_group_period(
            table, component_cols=("sigma",), corr_ranges_km=(1.0,),
        )
        assert result == {("A", 2000.25, 2001.25): 5.0, ("A", 2001.25, 2002.25): 10.0}

    def test_spatially_correlated_component_sigma_by_group_period__independent_components(self, glaciers):
        """Checks that independent component variances add after spatial propagation."""
        table = glaciers.assign(second=glaciers["sigma"] / 10)
        result = spatially_correlated_component_sigma_by_group_period(
            table, component_cols=("sigma", "second"), corr_ranges_km=(1.0, 1.0),
        )
        np.testing.assert_allclose(result[("A", 2000.25, 2001.25)], np.sqrt(5.0**2 + 0.5**2))

    @pytest.mark.parametrize("column,value", [("lat", np.nan), ("lon", np.inf), ("sigma", -1.0)])
    def test_spatially_correlated_sigma_by_group_period__error_invalid_input(self, glaciers, column, value):
        """Checks an error is raised instead of discarding glacier uncertainty."""
        glaciers.loc[0, column] = value
        with pytest.raises(ValueError):
            spatially_correlated_sigma_by_group_period(glaciers, independent, sigma_col="sigma")

    def test_spatially_correlated_sigma_by_group_period__error_duplicate(self, glaciers):
        """Checks an error is raised for repeated glacier observations in a period."""
        table = pd.concat([glaciers, glaciers.iloc[:1]], ignore_index=True)
        with pytest.raises(ValueError, match="duplicate"):
            spatially_correlated_sigma_by_group_period(table, independent, sigma_col="sigma")

    def test_spatially_correlated_sigma_by_group_period__error_block_size(self, glaciers):
        """Checks an error is raised for a negative distance matrix block size."""
        with pytest.raises(ValueError, match="block_size"):
            spatially_correlated_sigma_by_group_period(glaciers, independent, sigma_col="sigma", block_size=-1)


def test_summarize_global_period_conversions__empty():
    """Checks that an empty regional summary returns an empty global summary."""
    regional = pd.DataFrame(columns=["start_year", "end_year", "period_years"])
    assert summarize_global_period_conversions(regional).empty


class TestRegionAggregation:

    @pytest.fixture
    def predictions(self):
        """Provide additive glacier changes with elementary errors of 3 and 4 in reversed order."""
        return pd.DataFrame({
            "glacier_id": ["g1"] * 3 + ["g2"] * 3,
            "region_group": ["A"] * 6,
            "start": [2000.25, 2001.25, 2000.25] * 2,
            "end": [2001.25, 2002.25, 2002.25] * 2,
            "lat": [28.0] * 3 + [29.0] * 3, "lon": [87.0] * 6,
            "area_m2": [1.0] * 3 + [2.0] * 3,
            "dV_m3": [-1.0, -2.0, -3.0, -4.0, -2.0, -6.0],
            "dM_kg": [-30.0, -60.0, -90.0, -80.0, -40.0, -120.0],
            "sigma_dM_rho_kg": [3.0, 4.0, 5.0, 4.0, 3.0, 5.0],
        })

    @pytest.mark.parametrize("fraction", [0.0, 0.5, 1.0])
    def test_aggregate_regions__elementary_covariance(self, predictions, fraction):
        """Checks that spatial covariance adds within intervals and their variances add for longer periods."""

        # Each interval has variance 3² + 4² + 2 r 3 4, and the two intervals are independent
        correlation = lambda distance: np.full_like(distance, fraction)
        original = predictions.copy()
        result = aggregate_regions(predictions, correlation)
        full = result.loc[result["period_years"] == 2].iloc[0]
        annual = result.loc[result["period_years"] == 1]

        # Mass and volume changes add once, while density uses their ratio
        assert full["n_glaciers"] == 2
        assert full["area_m2"] == 3
        assert full["start"] == 2000.25
        assert full["dM_kg"] == -210
        assert full["dV_m3"] == -9
        np.testing.assert_allclose(full["mu_rho_kg_m3"], 210 / 9)
        np.testing.assert_allclose(full["dM_kg"], annual["dM_kg"].sum())
        np.testing.assert_allclose(full["sigma_dM_rho_kg"] ** 2, 50 + 48 * fraction)
        np.testing.assert_allclose(full["sigma_dM_rho_kg"] ** 2, np.sum(annual["sigma_dM_rho_kg"] ** 2))
        np.testing.assert_allclose(full["sigma_rho_kg_m3"], full["sigma_dM_rho_kg"] / 9)
        pd.testing.assert_frame_equal(predictions, original)

    def test_aggregate_regions__shared_input_errors(self, predictions):
        """Checks that whole-period input errors combine across glaciers while residual variance still adds in time."""

        # Full-period input errors include shared observations, so their variance need not add over intervals
        predictions["sigma_dM_dh_kg"] = [1.0, 2.0, 7.0, 3.0, 4.0, 8.0]
        predictions["sigma_dV_m3"] = [0.1, 0.2, 0.5, 0.3, 0.4, 1.2]
        result = aggregate_regions(predictions, independent)
        full = result.loc[result["period_years"] == 2].iloc[0]

        # Two independent glacier input variances add to 7² + 8²; residual variance is 50
        np.testing.assert_allclose(full["sigma_dM_dh_kg"], np.sqrt(113))
        np.testing.assert_allclose(full["sigma_dM_total_kg"], np.sqrt(163))
        np.testing.assert_allclose(full["sigma_dV_m3"], 1.3)
        np.testing.assert_allclose(full["dh_m"], -3.0)
        np.testing.assert_allclose(full["sigma_dh_m"], 1.3 / 3)

    @pytest.mark.parametrize("column", ["sigma_dM_dh_kg", "sigma_dV_m3"])
    def test_aggregate_regions__error_invalid_input_error(self, predictions, column):
        """Checks an error is raised for a negative mass or volume change input uncertainty."""
        predictions[column] = -1.0

        with pytest.raises(ValueError, match=column):
            aggregate_regions(predictions, independent)

    def test_aggregate_regions__model_and_default_group(self, predictions):
        """Checks that the model groups all glaciers together when region_group is absent."""

        # Omitting region labels still defines one spatial group with known glacier coordinates
        table = predictions.drop(columns="region_group").rename(columns={"glacier_id": "rgiid"})
        model = RhoSurrogate()
        result = model.aggregate_regions(table)
        expected = aggregate_regions(table, model.spatial_corr)

        pd.testing.assert_frame_equal(result, expected)
        assert result["region_group"].eq("All glaciers").all()
        assert result["n_glaciers"].eq(2).all()

    def test_aggregate_regions__separate_regions(self, predictions):
        """Checks that glaciers in different regions contribute only to their own regional totals."""

        # Each region contains one glacier, so its full-period residual standard deviation is five
        predictions.loc[predictions["glacier_id"] == "g2", "region_group"] = "B"
        result = aggregate_regions(predictions, independent)
        full = result.loc[result["period_years"] == 2].set_index("region_group")

        assert len(result) == 6
        np.testing.assert_allclose(full.loc[["A", "B"], "dM_kg"], [-90, -120])
        np.testing.assert_allclose(full["sigma_dM_rho_kg"], 5)

    def test_aggregate_regions__zero_volume(self, predictions):
        """Checks that cancelling volume changes leave finite mass change and residual uncertainty."""

        # Opposite volume changes sum to zero, while the additive mass changes remain available
        predictions.loc[predictions["glacier_id"] == "g2", "dV_m3"] = [1.0, 2.0, 3.0]
        result = aggregate_regions(predictions, independent)

        assert result["dV_m3"].eq(0).all()
        assert result["mu_rho_kg_m3"].isna().all()
        assert np.isfinite(result["dM_kg"]).all()
        assert np.isfinite(result["sigma_dM_rho_kg"]).all()

    def test_aggregate_regions__empty(self):
        """Checks that empty predictions return an empty regional table with the public output columns."""

        result = aggregate_regions(pd.DataFrame(), independent)
        assert result.empty
        assert {"region_group", "n_glaciers", "dM_kg", "sigma_dM_rho_kg"}.issubset(result.columns)

    @pytest.mark.parametrize("column", ["glacier_id", "area_m2", "lat", "lon"])
    def test_aggregate_regions__error_missing_input(self, predictions, column):
        """Checks an error is raised when a glacier ID, physical area or coordinate is unavailable."""

        with pytest.raises(ValueError, match="Regional aggregation requires"):
            aggregate_regions(predictions.drop(columns=column), independent)

    def test_aggregate_regions__error_incomplete_coverage(self, predictions):
        """Checks an error is raised when a glacier lacks an elementary observation interval."""

        with pytest.raises(ValueError, match="same consecutive elementary periods"):
            aggregate_regions(predictions.drop(index=3), independent)

    def test_aggregate_regions__error_nonadditive_mass(self, predictions):
        """Checks an error is raised when a longer-period mass change differs from its elementary sum."""

        predictions.loc[2, "dM_kg"] += 1
        with pytest.raises(ValueError, match="additive dM_kg"):
            aggregate_regions(predictions, independent)

    def test_aggregate_regions__error_changing_area(self, predictions):
        """Checks an error is raised when a glacier's area changes across its observation periods."""

        predictions.loc[0, "area_m2"] = 3
        with pytest.raises(ValueError, match="constant area"):
            aggregate_regions(predictions, independent)

    def test_aggregate_regions__error_duplicate_period(self, predictions):
        """Checks an error is raised instead of counting a repeated glacier period twice."""

        table = pd.concat([predictions, predictions.iloc[:1]], ignore_index=True)
        with pytest.raises(ValueError, match="duplicate"):
            aggregate_regions(table, independent)

    def test_aggregate_regions__error_durations_only(self, predictions):
        """Checks an error is raised when synthetic bounds from durations cannot locate glaciers in time."""

        predictions.attrs["has_time_bounds"] = False
        with pytest.raises(ValueError, match="start and end years"):
            aggregate_regions(predictions, independent)


class TestRegionalSummaries:

    @pytest.fixture
    def conversions(self):
        """Provide a half-year period with two known mass change and uncertainty totals."""
        return pd.DataFrame({
            "region_group": ["A", "A"], "region_label": ["Region A"] * 2,
            "rgiid": ["g1", "g2"], "start_year": [2000.25] * 2, "end_year": [2000.75] * 2,
            "period_years": [0.5] * 2, "area": [1e6, 2e6], "dV_m3": [-1e9, -2e9],
            "dM_old_kg": [-850e9, -1700e9], "dM_surrogate_kg": [-800e9, -1600e9],
            "sigma_dM_volume_old_kg": [3e9, 4e9], "sigma_dM_volume_surrogate_kg": [6e9, 8e9],
            "filled_from_region_period_mean": [False, True],
        })

    def test_summarize_region_period_conversions__units(self, conversions):
        """Checks that half-year rates and independent uncertainties use the right units."""
        regional = summarize_region_period_conversions(
            conversions, spatial_density_sigma={("A", 2000.25, 2000.75): 24e9},
        )
        global_summary = summarize_global_period_conversions(regional)

        np.testing.assert_allclose(regional["surrogate_mass_gt"], -2.4)
        np.testing.assert_allclose(regional["surrogate_mass_rate_gt_yr"], -4.8)
        np.testing.assert_allclose(regional["surrogate_sigma_mass_gt"], 0.026)
        np.testing.assert_allclose(global_summary["surrogate_sigma_mass_gt"], 0.026)
        np.testing.assert_allclose(global_summary["surrogate_rho_mean_kg_m3"], 800.0)
        assert regional["start_year"].iloc[0] == 2000.25
        assert regional["period_years"].iloc[0] == 0.5

    def test_summarize_region_period_conversions__missing_uncertainty(self, conversions):
        """Checks that an unknown uncertainty stays unknown instead of becoming zero."""
        conversions.loc[0, "sigma_dM_volume_surrogate_kg"] = np.nan
        regional = summarize_region_period_conversions(
            conversions, spatial_density_sigma={("A", 2000.25, 2000.75): 24e9},
        )
        assert np.isnan(regional["surrogate_sigma_mass_gt"].iloc[0])
        assert np.isnan(summarize_global_period_conversions(regional)["surrogate_sigma_mass_gt"].iloc[0])

    def test_summarize_global_period_conversions__independent_regions(self, conversions):
        """Checks that regional mass changes add while independent regional variances add."""
        second = conversions.copy()
        second["region_group"] = "B"
        physical_columns = [
            "area", "dV_m3", "dM_old_kg", "dM_surrogate_kg",
            "sigma_dM_volume_old_kg", "sigma_dM_volume_surrogate_kg",
        ]
        second[physical_columns] *= 2
        regions = summarize_region_period_conversions(
            pd.concat([conversions, second], ignore_index=True),
            spatial_density_sigma={("A", 2000.25, 2000.75): 24e9, ("B", 2000.25, 2000.75): 48e9},
        )
        result = summarize_global_period_conversions(regions)

        np.testing.assert_allclose(result["surrogate_mass_gt"], -7.2)
        np.testing.assert_allclose(result["surrogate_sigma_mass_gt"], np.hypot(0.026, 0.052))

    @pytest.mark.parametrize("column,value", [("period_years", 0), ("dV_m3", np.nan), ("area", -1)])
    def test_summarize_region_period_conversions__error_invalid_input(self, conversions, column, value):
        """Checks an error is raised before invalid periods, volume changes or areas are summed."""
        conversions[column] = value
        with pytest.raises(ValueError):
            summarize_region_period_conversions(
                conversions, spatial_density_sigma={("A", 2000.25, 2000.75): 24e9},
            )


def test_haversine_distance_matrix__quarter_circumference():
    """Checks that ninety degrees on the equator gives a quarter Earth circumference."""
    distance = haversine_distance_matrix(np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([90.0]))
    np.testing.assert_allclose(distance, np.pi * 6371.0 / 2)


@pytest.mark.parametrize("radius", [0, -1, np.nan, np.inf])
def test_spherical_correlation__error_invalid_range(radius):
    """Checks an error is raised for a nonpositive or missing spatial range."""
    with pytest.raises(ValueError, match="range_km"):
        spherical_correlation(np.array([1.0]), radius)
