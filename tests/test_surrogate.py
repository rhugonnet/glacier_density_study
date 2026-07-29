from __future__ import annotations

import math

import numpy as np
import pandas as pd

from glacier_density_surrogate import (
    RhoSurrogate,
    expected_abs_normal,
    expected_abs_normal_array,
    integrated_mu_vectorized,
    integrated_sigma_vectorized,
    load_params,
    make_dh_error_correlation,
    mean_density_volume_moments_vectorized,
    mean_density_volume_sigma_vectorized,
    spatially_correlated_sigma_by_group_period,
    temporally_reconcile_periods,
)


def test_package_does_not_export_study_paths() -> None:
    import glacier_density_surrogate as gds

    assert not hasattr(gds, "DEFAULT_FIGURES_DIR")
    assert not hasattr(gds, "DEFAULT_PARAM_PATH")
    assert not hasattr(gds, "DEFAULT_RESULTS_DIR")


def test_expected_abs_normal_matches_limiting_cases() -> None:
    means = np.array([-2.0, 0.0, 3.0])
    sigmas = np.array([0.0, 2.0, 0.0])

    assert expected_abs_normal(-2.0, 0.0) == 2.0
    assert math.isclose(expected_abs_normal(0.0, 2.0), 2.0 * math.sqrt(2.0 / math.pi))
    np.testing.assert_allclose(expected_abs_normal_array(means, sigmas), [2.0, 2.0 * math.sqrt(2.0 / math.pi), 3.0])


def test_mean_converges_toward_ice_density_for_large_elevation_change() -> None:
    model = RhoSurrogate()
    rho = model.mu_rho(np.array([-10000.0, 10000.0]), dh_p=0.0, dt=20.0)

    np.testing.assert_allclose(rho, model.rho_ice, atol=1.0e-3)


def test_sigma_follows_final_variance_additive_form() -> None:
    model = RhoSurrogate()

    sigma_large = float(model.sigma_rho(-20.0, dt=5.0))
    sigma_small = float(model.sigma_rho(-1.0, dt=5.0))
    sigma_long = float(model.sigma_rho(-1.0, dt=20.0))

    assert sigma_small > sigma_large
    assert sigma_long > sigma_small
    assert "sigma_form" not in model.params


def test_period_power_mean_form_matches_manual_expression() -> None:
    params = {
        "period_form": "power_param",
        "rho_ice": 900.0,
        "H_d": 9.0,
        "P_d": 0.7,
        "B_h": -20.0,
        "P0": 150.0,
        "P1": -420.0,
        "TP": 2.0,
        "B_p": 0.0,
        "B_q": 0.0,
    }
    model = RhoSurrogate(params=params)
    dh = np.array([-5.0, -5.0])
    dt = np.array([1.0, 20.0])

    damping = np.exp(-((np.abs(dh) / params["H_d"]) ** params["P_d"]))
    period = params["P0"] + params["P1"] / (1.0 + dt / params["TP"])
    expected = params["rho_ice"] + damping * (params["B_h"] + period)

    np.testing.assert_allclose(model.mu_rho(dh, dh_p=0.0, dt=dt), expected)


def test_loader_keeps_period_power_tp_separate_from_memory_tau(tmp_path) -> None:
    path = tmp_path / "params.csv"
    pd.DataFrame(
        {
            "parameter": ["spec.period", "TP", "memory_tau_years", "P0", "P1"],
            "value_numeric": [np.nan, 2.5, 4.9, 100.0, -300.0],
            "value_text": ["power_param", "", "", "", ""],
        }
    ).to_csv(path, index=False)

    params = load_params(path, verbose=False)

    assert params["period_form"] == "power_param"
    assert math.isclose(float(params["TP"]), 2.5)
    assert math.isclose(float(params["T_p"]), 4.9)


def test_integrated_sigma_matches_exact_sigma_without_dh_uncertainty() -> None:
    model = RhoSurrogate()

    exact = float(model.sigma_rho(-4.0, dt=10.0))
    integrated = model.integrated_sigma(dh=-4.0, sigma_dh=0.0, dt=10.0)

    assert math.isclose(integrated, exact, rel_tol=1.0e-12)


def test_mean_density_volume_variance_vanishes_without_predictor_uncertainty() -> None:
    model = RhoSurrogate()

    sigma = mean_density_volume_sigma_vectorized(
        model,
        area_m2=np.array([1.0e6, 2.0e6]),
        dh=np.array([-4.0, 3.0]),
        sigma_dh=np.array([0.0, 0.0]),
        past_dh=np.array([-0.5, 0.25]),
        sigma_past_dh=np.array([0.0, 0.0]),
        dt=np.array([5.0, 10.0]),
    )

    np.testing.assert_allclose(sigma, 0.0, atol=1.0e-6)


def test_mean_density_volume_variance_matches_constant_mean_limit() -> None:
    params = {
        "rho_ice": 850.0,
        "period_form": "none",
        "B_h": 0.0,
        "B_t": 0.0,
        "P0": 0.0,
        "P1": 0.0,
        "B_p": 0.0,
        "B_q": 0.0,
    }
    model = RhoSurrogate(params=params)
    area = np.array([1.0e6, 3.0e6])
    dh = np.array([-5.0, 2.0])
    sigma_dh = np.array([0.4, 1.5])

    mean, variance = mean_density_volume_moments_vectorized(
        model,
        area_m2=area,
        dh=dh,
        sigma_dh=sigma_dh,
        past_dh=np.array([0.0, -1.0]),
        sigma_past_dh=np.array([0.0, 0.2]),
        dt=np.array([5.0, 1.0]),
    )

    np.testing.assert_allclose(mean, model.rho_ice * area * dh, rtol=1.0e-12)
    np.testing.assert_allclose(variance, (model.rho_ice * area * sigma_dh) ** 2, rtol=1.0e-12)


def test_mean_density_volume_moments_match_brute_force_quadrature() -> None:
    model = RhoSurrogate(gh_order_current=20, gh_order_past=20)
    area = 2.0e6
    dh = -1.2
    sigma_dh = 0.4
    past_dh = 0.35
    sigma_past_dh = 0.2
    dt = 5.0

    mean, variance = mean_density_volume_moments_vectorized(
        model,
        area_m2=np.array([area]),
        dh=np.array([dh]),
        sigma_dh=np.array([sigma_dh]),
        past_dh=np.array([past_dh]),
        sigma_past_dh=np.array([sigma_past_dh]),
        dt=np.array([dt]),
    )

    x = dh + math.sqrt(2.0) * sigma_dh * model._gh_x_current
    y = past_dh + math.sqrt(2.0) * sigma_past_dh * model._gh_x_past
    weight = np.outer(model._gh_w_current, model._gh_w_past) / math.pi
    mass = area * x[:, None] * model.mu_rho(x[:, None], dh_p=y[None, :], dt=dt)
    brute_mean = np.sum(weight * mass)
    brute_variance = np.sum(weight * mass**2) - brute_mean**2

    np.testing.assert_allclose(mean[0], brute_mean, rtol=1.0e-12)
    np.testing.assert_allclose(variance[0], brute_variance, rtol=1.0e-12)


def test_vectorized_integration_matches_scalar_api() -> None:
    model = RhoSurrogate(gh_order_current=32, gh_order_past=32)
    dh = np.array([-10.0, -1.0, 0.5, 8.0])
    sigma_dh = np.array([0.2, 0.4, 0.1, 0.5])
    past = np.array([-0.5, 0.0, -0.5, 0.25])
    sigma_past = np.array([0.1, 0.0, 0.2, 0.1])
    dt = np.array([20.0, 1.0, 5.0, 10.0])

    mu_loop = np.array([model.integrated_mu(a, b, c, d, e) for a, b, c, d, e in zip(dh, sigma_dh, past, sigma_past, dt)])
    sig_loop = np.array([model.integrated_sigma(a, b, e) for a, b, e in zip(dh, sigma_dh, dt)])

    np.testing.assert_allclose(integrated_mu_vectorized(model, dh, sigma_dh, past, sigma_past, dt), mu_loop)
    np.testing.assert_allclose(integrated_sigma_vectorized(model, dh, sigma_dh, dt), sig_loop)


def test_missing_past_current_uses_current_annual_rate() -> None:
    model = RhoSurrogate()
    out = model.predict(dh=-10.0, sigma_dh=1.0, dt=20.0, past_missing="current", past_error_factor=2.0)

    assert out["past_dh_m"] == -0.5
    assert out["sigma_past_dh_m"] == 0.1


def test_timeseries_past_predictor_uses_annual_rate_for_multiannual_steps() -> None:
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

    out = model.predict_timeseries(data, expand_periods=True)
    second_elementary = out.loc[(out["start"] == 2005) & (out["end"] == 2010)].iloc[0]

    assert math.isclose(second_elementary["past_dh_m"], -1.0)
    assert math.isclose(second_elementary["sigma_past_dh_m"], 0.1)


def test_timeseries_past_predictor_redistributes_multiannual_steps_before_weighting() -> None:
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

    out = model.predict_timeseries(data, expand_periods=True)
    third_elementary = out.loc[(out["start"] == 2010) & (out["end"] == 2015)].iloc[0]

    annual_ends = np.arange(2001, 2011, dtype=float)
    annual_rates = np.array([-1.0] * 5 + [-2.0] * 5)
    raw_weights = np.exp(-(2010.0 - annual_ends) / float(model.params["T_p"]))
    weights = raw_weights / raw_weights.sum()
    block_weights = np.array([weights[:5].sum(), weights[5:].sum()])
    expected_sigma = math.sqrt(np.sum((block_weights * 0.1) ** 2))

    assert math.isclose(third_elementary["past_dh_m"], float(np.sum(weights * annual_rates)))
    assert math.isclose(third_elementary["sigma_past_dh_m"], expected_sigma)


def test_temporal_closure_preserves_additive_mass_change() -> None:
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

    out = model.predict_timeseries(data, expand_periods=True)
    annual = out.loc[np.isclose(out["period_years"], 1.0), "dM_closed_kg"].sum()
    full = out.loc[(out["start"] == 2000) & (out["end"] == 2003), "dM_closed_kg"].iloc[0]

    assert math.isclose(annual, full, rel_tol=1.0e-12)


def test_temporal_closure_uses_path_length_weights() -> None:
    model = RhoSurrogate(params={"U_h": 1.0, "U_t": 0.0})
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

    out = temporally_reconcile_periods(periods, model)
    annual_anomaly = (
        out.loc[(out["start_year"] == 2000) & (out["end_year"] == 2001), "dM_surrogate_kg"].iloc[0]
        - model.rho_ice
    )
    combined_anomaly = (
        out.loc[(out["start_year"] == 2000) & (out["end_year"] == 2002), "dM_surrogate_kg"].iloc[0]
        - 2.0 * model.rho_ice
    )

    assert math.isclose(annual_anomaly, 25.5, rel_tol=1.0e-12)
    assert math.isclose(combined_anomaly, 51.0, rel_tol=1.0e-12)


def test_temporal_closure_is_invariant_to_area_units() -> None:
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
    scaled = base.copy()
    scaled["area"] *= 1.0e6
    scaled["dV_m3"] *= 1.0e6
    scaled["sigma_dM_rho_independent_kg"] *= 1.0e6

    out_base = temporally_reconcile_periods(base, model).sort_values(["rgiid", "start_year", "end_year"])
    out_scaled = temporally_reconcile_periods(scaled, model).sort_values(["rgiid", "start_year", "end_year"])

    np.testing.assert_allclose(
        out_scaled["mu_rho_surrogate_kg_m3"].to_numpy(float),
        out_base["mu_rho_surrogate_kg_m3"].to_numpy(float),
        rtol=1.0e-12,
    )


def test_correlation_functions_have_expected_bounds() -> None:
    model = RhoSurrogate()

    spatial = model.spatial_corr(np.array([0.0, 100.0, 1000.0, 20000.0]))
    temporal = model.temporal_corr(np.array([0.0, 1.0, 5.0, 14.0]))

    assert np.all((spatial >= 0.0) & (spatial <= 1.0))
    assert spatial[0] > spatial[-1]
    assert temporal[0] == 1.0
    assert np.all((temporal >= 0.0) & (temporal <= 1.0))
    assert temporal[1] > temporal[2] >= temporal[3]


def test_spatial_group_propagation_matches_independent_limit() -> None:
    table = pd.DataFrame(
        {
            "region_group": ["A", "A", "A"],
            "start_year": [2000, 2000, 2000],
            "end_year": [2005, 2005, 2005],
            "rgiid": ["g1", "g2", "g3"],
            "lat": [0.0, 1.0, 2.0],
            "lon": [0.0, 1.0, 2.0],
            "sigma": [3.0, 4.0, 12.0],
        }
    )
    zero_offdiag = lambda distance: np.zeros_like(distance, dtype=float)

    out = spatially_correlated_sigma_by_group_period(table, zero_offdiag, sigma_col="sigma")

    assert math.isclose(out[("A", 2000, 2005)], 13.0)


def test_elevation_error_correlation_builders() -> None:
    independent = make_dh_error_correlation("none")
    exponential = make_dh_error_correlation("exponential", 5.0)

    np.testing.assert_allclose(independent(np.array([0.0, 1.0])), [1.0, 0.0])
    assert exponential(np.array([0.0]))[0] == 1.0
    assert 0.0 < exponential(np.array([5.0]))[0] < 1.0
