#!/usr/bin/env python3
"""
Estimate variance that we can attribute to firn parametrization sensitivity tests.

This script refits the retained mean/sigma surrogate separately for the three firn density variants,
then evaluates agreement between the final surrogate fitted to all variants at once, or a separate surrogate
fitted only to a given variant.

Use --uncertainty-only to reproduce the regional residual uncertainty share from
existing fits. This writes each region's variance components and their median share.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from glacier_density_surrogate import RhoSurrogate, haversine_distance_matrix, integrated_sigma_mass_vectorized


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

STUDY_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = STUDY_DIR.parent
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from study_paths import (  # noqa: E402
    INPUT_CSV,
    PARAM_PATH,
    SPATIAL_PARAM_PATH,
    TEMPORAL_PARAM_PATH,
    FIRN_PARAMETRIZATION_DIAGNOSTICS_DIR,
)


VARIANTS = ["iteration9", "sensmin", "sensmax"]
POOLED_MODEL_SOURCE = "pooled_all_variants"
INDIVIDUAL_MODEL_SOURCE = "fit_to_variant"

# Keep this False for the main diagnostic. It isolates the mean/sigma parameter
# effect while using the same final exponential-memory definition as the pooled
# model. Set True only for a slower follow-up where memory tau is also refit.
REFIT_JOINT_MEMORY_PER_VARIANT = False

WRITE_REGIONAL_DETAIL = True
WRITE_GLACIER_DETAIL = False
GH_ORDER = 64

OUT_DIR = FIRN_PARAMETRIZATION_DIAGNOSTICS_DIR
FIT_DIR = OUT_DIR / "individual_variant_fits"

OUT_REGIONAL_DETAIL_CSV = OUT_DIR / "agreement_regional_period_detail_by_model_variant.csv"
OUT_REGIONAL_SUMMARY_CSV = OUT_DIR / "agreement_regional_period_summary_by_model_variant.csv"
OUT_GLACIER_DETAIL_CSV = OUT_DIR / "agreement_glacier_period_detail_by_model_variant.csv"
OUT_GLACIER_SUMMARY_CSV = OUT_DIR / "agreement_glacier_period_summary_by_model_variant.csv"
OUT_COMPARISON_CSV = OUT_DIR / "pooled_vs_individual_unexplained_variance_comparison.csv"
OUT_PRIMARY_CSV = OUT_DIR / "firn_parametrization_unexplained_variance_percent.csv"
OUT_FIT_SUMMARY_CSV = OUT_DIR / "individual_variant_fit_summary.csv"
OUT_FIT_METADATA_CSV = OUT_DIR / "individual_variant_fit_metadata.csv"
OUT_UNCERTAINTY_DETAIL_CSV = OUT_DIR / "firn_parametrization_regional_uncertainty_variance.csv"
OUT_UNCERTAINTY_SUMMARY_CSV = OUT_DIR / "firn_parametrization_uncertainty_summary.csv"


def load_script_module(path: Path, name: str):
    """Load an analysis script as a module without running its main block."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finite_or_none(value) -> float | None:
    """Return a finite float or None."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def read_final_memory_config(fitmod) -> dict[str, float | str | None]:
    """Read the final pooled memory definition, falling back to script defaults."""
    config: dict[str, float | str | None] = {
        "mode": fitmod.MODEL_MEMORY_MODE,
        "window": finite_or_none(fitmod.MODEL_MEMORY_WINDOW_YEARS),
        "tau": finite_or_none(fitmod.MODEL_MEMORY_TAU_YEARS),
        "source": "fit_script_defaults",
    }
    if not PARAM_PATH.exists():
        return config

    table = pd.read_csv(PARAM_PATH)
    if not {"parameter", "value_numeric", "value_text"}.issubset(table.columns):
        return config

    lookup = {str(row.parameter): row for row in table.itertuples(index=False)}

    if "memory_mode" in lookup:
        text = str(lookup["memory_mode"].value_text)
        if text and text.lower() != "nan":
            config["mode"] = text
            config["source"] = str(PARAM_PATH)
    if "memory_window_years" in lookup:
        config["window"] = finite_or_none(lookup["memory_window_years"].value_numeric)
    if "memory_tau_years" in lookup:
        config["tau"] = finite_or_none(lookup["memory_tau_years"].value_numeric)

    return config


def configure_fit_module_for_diagnostic(fitmod, variant: str, variant_dir: Path) -> None:
    """Redirect fit-script globals to variant-local diagnostic outputs."""
    variant_dir.mkdir(parents=True, exist_ok=True)
    fitmod.VARIANTS_TO_USE = [variant]
    fitmod.RUN_PAPER_FIGURES = False
    fitmod.WRITE_FIT_DIAGNOSTIC_PLOTS = False
    fitmod.RUN_MEMORY_PROFILE = False
    fitmod.RUN_NO_PERIOD_DIAGNOSTIC = False
    fitmod.RUN_RESIDUAL_MEMORY_SCAN_DIAGNOSTICS = False
    fitmod.RUN_PATH_LENGTH_SIGMA_DIAGNOSTIC = False
    fitmod.RUN_JOINT_KERNEL_MEMORY_FIT = bool(REFIT_JOINT_MEMORY_PER_VARIANT)
    fitmod.CANDIDATE_MODELS = [dict(fitmod.FINAL_MEAN_MODEL_SPEC)]
    fitmod.SELECTED_MODEL_FOR_DIAGNOSTICS = None

    fitmod.out_dir = variant_dir
    fitmod.out_plot_dir = variant_dir
    fitmod.run_label = "rho_surrogate"
    fitmod.main_suffix = variant
    stem = f"{fitmod.run_label}_{fitmod.main_suffix}"

    fitmod.out_fit_summary_csv = variant_dir / f"{stem}_fit_summary.csv"
    fitmod.out_params_csv = variant_dir / f"{stem}_parameters.csv"
    fitmod.out_target_csv = variant_dir / f"{stem}_target_predictions.csv"
    fitmod.out_target_dh_category_csv = variant_dir / f"{stem}_target_prediction_metrics_by_dh_category.csv"
    fitmod.out_target_period_absdh_csv = variant_dir / f"{stem}_target_prediction_metrics_by_period_absdh.csv"
    fitmod.out_target_period_signeddh_csv = variant_dir / f"{stem}_target_prediction_metrics_by_period_signeddh.csv"
    fitmod.out_row_sample_csv = variant_dir / f"{stem}_row_predictions_sample.csv"
    fitmod.out_standardized_residuals_csv = variant_dir / f"{stem}_standardized_residuals.csv"
    fitmod.out_diag_csv = variant_dir / f"{stem}_factor_diagnostics.csv"
    fitmod.out_sigma_abs_table_csv = variant_dir / f"{stem}_sigma_absdh_residual_std_table.csv"
    fitmod.out_sigma_params_csv = variant_dir / f"{stem}_sigma_params_mean_removed_residual.csv"
    fitmod.out_sigma_model_selection_csv = variant_dir / f"{stem}_sigma_model_selection.csv"
    fitmod.out_final_joint_model_csv = variant_dir / "mu_sigma_model_parameters.csv"
    fitmod.out_final_joint_model_json = variant_dir / "mu_sigma_model_parameters.json"

    fitmod.out_joint_kernel_memory_summary_csv = variant_dir / f"{stem}_joint_kernel_memory_summary.csv"
    fitmod.out_joint_kernel_memory_params_csv = variant_dir / f"{stem}_joint_kernel_memory_parameters.csv"
    fitmod.out_joint_kernel_memory_weights_csv = variant_dir / f"{stem}_joint_kernel_memory_weights.csv"
    fitmod.out_joint_kernel_memory_weights_png = variant_dir / f"{stem}_joint_kernel_memory_weights.png"
    fitmod.out_joint_kernel_memory_residual_png = variant_dir / f"{stem}_joint_kernel_memory_residuals.png"
    fitmod.out_joint_kernel_memory_scatter_png = variant_dir / f"{stem}_joint_kernel_memory_observed_predicted.png"

    for name in ["SIGMA_SELECTED_FORM", "SIGMA_SELECTED_PARAMS", "SIGMA_SELECTED_THETA"]:
        fitmod.__dict__.pop(name, None)
    fitmod.SIGMA_MODEL_SELECTION_TABLE = pd.DataFrame()


def choose_memory_for_variant(fitmod, all_rows: pd.DataFrame, fallback: dict[str, float | str | None]):
    """Return memory definition for a single-variant fit."""
    if not REFIT_JOINT_MEMORY_PER_VARIANT:
        return dict(fallback)

    base_spec = dict(fitmod.FINAL_MEAN_MODEL_SPEC)
    base_spec["name"] = str(base_spec.get("name", "base_spec"))
    joint_results = fitmod.maybe_run_joint_kernel_memory_fit(all_rows, base_spec)
    chosen = fitmod.choose_diagnostic_memory_from_joint_results(joint_results, preferred_kernel="exponential")
    if chosen is None:
        return dict(fallback)
    chosen["source"] = "refit_joint_memory"
    return chosen


def write_target_fit_outputs(fitmod, variant: str, summary: pd.DataFrame, target: pd.DataFrame, fits: dict) -> None:
    """Write lightweight target and model-fit diagnostics for one variant."""
    summary.assign(rho_variant=variant).to_csv(fitmod.out_fit_summary_csv, index=False)
    target.assign(rho_variant=variant).to_csv(fitmod.out_target_csv, index=False)
    rows = []
    for model, obj in fits.items():
        rows.append({"rho_variant": variant, "model": model, **fitmod.unpack(obj["theta"], obj["param_names"])})
    pd.DataFrame(rows).to_csv(fitmod.out_params_csv, index=False)


def fit_one_variant(
    fitmod,
    variant: str,
    memory_config: dict[str, float | str | None],
) -> dict[str, object]:
    """Fit retained mean and sigma forms to one firn density variant."""
    variant_dir = FIT_DIR / variant
    configure_fit_module_for_diagnostic(fitmod, variant, variant_dir)

    print(f"[fit] Preparing rows for {variant}", flush=True)
    all_rows = fitmod.prepare_dataframe()
    this_memory = choose_memory_for_variant(fitmod, all_rows, memory_config)

    print(
        f"[fit] Fitting {variant} with memory mode={this_memory['mode']}, "
        f"window={this_memory.get('window')}, tau={this_memory.get('tau')}",
        flush=True,
    )
    d, memory_col, target, target_fit, summary, fits = fitmod.run_one_memory_fit(
        all_rows,
        this_memory["mode"],
        window=this_memory.get("window"),
        tau=this_memory.get("tau"),
        candidate_models=[dict(fitmod.FINAL_MEAN_MODEL_SPEC)],
    )
    write_target_fit_outputs(fitmod, variant, summary, target, fits)

    selected_model, fit_obj = fitmod.select_fit(summary, fits)
    d_pred = fitmod.compute_row_predictions(d, fit_obj, memory_col)
    del d
    gc.collect()

    theta_sig, sigma_abs_table = fitmod.fit_sigma_model_from_rows(d_pred, resid_col="rho_mean_removed")
    sigma_abs_table.assign(rho_variant=variant).to_csv(fitmod.out_sigma_abs_table_csv, index=False)
    pd.DataFrame(
        [
            {
                "rho_variant": variant,
                "sigma_form": fitmod.__dict__.get("SIGMA_SELECTED_FORM", fitmod.SIGMA_MODEL_FOR_DIAGNOSTICS),
                **{
                    name: float(value)
                    for name, value in zip(
                        fitmod.__dict__.get("SIGMA_SELECTED_PARAMS", fitmod.SIGMA_PARAMS),
                        np.asarray(theta_sig, dtype=float),
                    )
                },
            }
        ]
    ).to_csv(fitmod.out_sigma_params_csv, index=False)
    fitmod.SIGMA_MODEL_SELECTION_TABLE.assign(rho_variant=variant).to_csv(
        fitmod.out_sigma_model_selection_csv,
        index=False,
    )

    fitmod.write_final_joint_rho_model_files(
        selected_model=selected_model,
        fit_obj=fit_obj,
        theta_sig=theta_sig,
        memory_mode=this_memory["mode"],
        memory_window=this_memory.get("window"),
        memory_tau=this_memory.get("tau"),
        memory_col=memory_col,
        sigma_resid_col="rho_mean_removed",
    )

    sample_cols = [
        "rgiid",
        fitmod.variant_col,
        "start_date",
        "end_date",
        "period_years",
        fitmod.rho_col,
        "rho_pred",
        "rho_mean_removed",
        "signed_dh",
        memory_col,
        fitmod.weight_col,
    ]
    d_pred[[c for c in sample_cols if c in d_pred.columns]].head(fitmod.MAX_ROW_SAMPLE_OUTPUT).to_csv(
        fitmod.out_row_sample_csv,
        index=False,
    )
    del d_pred, target, target_fit, fits, all_rows
    gc.collect()

    return {
        "rho_variant": variant,
        "parameter_path": fitmod.out_final_joint_model_csv,
        "selected_model": selected_model,
        "memory_mode": this_memory["mode"],
        "memory_window_years": this_memory.get("window"),
        "memory_tau_years": this_memory.get("tau"),
        "memory_source": this_memory.get("source"),
        "n_fit_summary_rows": int(len(summary)),
    }


def run_agreement_case(
    agreement,
    parameter_path: Path,
    model_source: str,
    evaluation_variant: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate one parameter file against one full model variant."""
    print(f"[agreement] {model_source} on {evaluation_variant}", flush=True)
    model = RhoSurrogate.from_files(
        parameter_path,
        SPATIAL_PARAM_PATH,
        TEMPORAL_PARAM_PATH,
        gh_order_current=GH_ORDER,
        gh_order_past=GH_ORDER,
    )
    df = agreement.read_full_model_input(INPUT_CSV, variants=[evaluation_variant])
    periods = agreement.prepare_periods(
        df,
        model,
        agreement.START_YEAR,
        agreement.END_YEAR,
        evaluation_variant,
    )
    closed = agreement.add_temporal_closure(periods, model, agreement.START_YEAR, agreement.END_YEAR)
    regional_detail = pd.concat(
        [
            agreement.aggregate_region_period(closed, "independent", model),
            agreement.aggregate_region_period(closed, "closed", model),
        ],
        ignore_index=True,
    )
    regional_detail = agreement.add_agreement_categories(regional_detail, "regional_abs_dh_m")
    regional_detail.insert(0, "evaluation_variant", evaluation_variant)
    regional_detail.insert(0, "model_source", model_source)

    if WRITE_GLACIER_DETAIL:
        glacier_detail = agreement.add_agreement_categories(
            agreement.glacier_period_agreement(closed),
            "abs_dh_m",
        )
        glacier_detail.insert(0, "evaluation_variant", evaluation_variant)
        glacier_detail.insert(0, "model_source", model_source)
    else:
        glacier_detail = pd.DataFrame()

    del df, periods, closed, model
    gc.collect()
    return regional_detail, glacier_detail


def summarize_regional_detail(agreement, detail: pd.DataFrame) -> pd.DataFrame:
    """Summarize regional agreement by model source and evaluation variant."""
    by_variant = agreement.summarize_grouped_agreement(
        detail,
        ["model_source", "evaluation_variant"],
        abs_dh_col="regional_abs_dh_m",
        count_col="n_region_periods",
    )
    overall = agreement.summarize_grouped_agreement(
        detail,
        ["model_source"],
        abs_dh_col="regional_abs_dh_m",
        count_col="n_region_periods",
    )
    overall["evaluation_variant"] = "all_variants"
    out = pd.concat([by_variant, overall], ignore_index=True, sort=False)
    out.insert(0, "level", "regional_period")
    return out


def summarize_glacier_detail(agreement, detail: pd.DataFrame) -> pd.DataFrame:
    """Summarize glacier and period agreement when glacier detail is enabled."""
    if detail.empty:
        return pd.DataFrame()
    by_variant = agreement.summarize_grouped_agreement(
        detail,
        ["model_source", "evaluation_variant"],
        abs_dh_col="abs_dh_m",
        count_col="n_glacier_periods",
    )
    overall = agreement.summarize_grouped_agreement(
        detail,
        ["model_source"],
        abs_dh_col="abs_dh_m",
        count_col="n_glacier_periods",
    )
    overall["evaluation_variant"] = "all_variants"
    out = pd.concat([by_variant, overall], ignore_index=True, sort=False)
    out.insert(0, "level", "glacier_period")
    return out


def build_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    """Build paired pooled-minus-individual metric differences."""
    if summary.empty:
        return pd.DataFrame()
    metrics = [
        "weighted_residual_variance",
        "weighted_residual_mse",
        "weighted_rmse",
        "weighted_mae",
        "weighted_bias",
        "weighted_r2",
    ]
    id_cols = ["level", "mode", "evaluation_variant"]
    pooled = summary.loc[summary["model_source"].eq(POOLED_MODEL_SOURCE)].copy()
    individual = summary.loc[summary["model_source"].eq(INDIVIDUAL_MODEL_SOURCE)].copy()
    rows = []
    for metric in metrics:
        keep = id_cols + ["volume_weight_sum", metric]
        if metric not in pooled.columns or metric not in individual.columns:
            continue
        merged = pooled[keep].merge(
            individual[keep],
            on=id_cols,
            suffixes=("_pooled", "_individual"),
            how="inner",
        )
        for row in merged.itertuples(index=False):
            pooled_value = getattr(row, f"{metric}_pooled")
            individual_value = getattr(row, f"{metric}_individual")
            diff = pooled_value - individual_value
            pct = 100.0 * diff / pooled_value if np.isfinite(pooled_value) and pooled_value != 0 else np.nan
            rows.append(
                {
                    "level": row.level,
                    "mode": row.mode,
                    "evaluation_variant": row.evaluation_variant,
                    "metric": metric,
                    "pooled_all_variants": pooled_value,
                    "individual_variant_fit": individual_value,
                    "pooled_minus_individual": diff,
                    "percent_of_pooled_metric": pct,
                    "volume_weight_sum_pooled": row.volume_weight_sum_pooled,
                    "volume_weight_sum_individual": row.volume_weight_sum_individual,
                }
            )
    return pd.DataFrame(rows)


def primary_variance_table(comparison: pd.DataFrame) -> pd.DataFrame:
    """Return the compact unexplained-variance percentage table."""
    if comparison.empty:
        return comparison
    primary = comparison.loc[
        comparison["level"].eq("regional_period")
        & comparison["evaluation_variant"].eq("all_variants")
        & comparison["metric"].isin(["weighted_residual_variance", "weighted_residual_mse"])
    ].copy()
    primary = primary.rename(
        columns={
            "pooled_all_variants": "pooled_unexplained",
            "individual_variant_fit": "individual_fit_unexplained",
            "pooled_minus_individual": "unexplained_removed_by_individual_fits",
            "percent_of_pooled_metric": "firn_parametrization_share_percent",
        }
    )
    return primary.sort_values(["mode", "metric"]).reset_index(drop=True)


###############################
# REGIONAL RESIDUAL UNCERTAINTY
###############################


def read_common_uncertainty_inputs(agreement, model: RhoSurrogate, chunk_size: int = 500_000) -> pd.DataFrame:
    """
    Read a common glacier sample for the full observation period and its past elevation change rate.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    start_year = agreement.START_YEAR
    end_year = agreement.END_YEAR
    reference_variant = VARIANTS[0]
    history_years = int(round(float(model.params["tau_max"])))
    columns = ["rgiid", "rho_variant", "start_date", "end_date", "rho", "b", "area", "lat", "lon"]
    period_parts = []
    annual_parts = []

    # Select the full-period records and the reference history before filtering values
    for chunk in pd.read_csv(INPUT_CSV, usecols=columns, chunksize=chunk_size):
        full_period = chunk.start_date.eq(start_year) & chunk.end_date.eq(end_year)
        annual_history = (
            chunk.rho_variant.eq(reference_variant)
            & (chunk.end_date - chunk.start_date).eq(1)
            & chunk.start_date.lt(start_year)
            & chunk.start_date.ge(start_year - history_years)
        )
        selected = chunk.loc[(full_period | annual_history) & chunk.rho_variant.isin(VARIANTS)].copy()
        if selected.empty:
            continue

        # Apply the same density sentinels and finite-input rules as the agreement diagnostic
        valid = (
            np.isfinite(selected[["rho", "b", "area"]]).all(axis=1)
            & selected.area.gt(0)
            & selected.rho.abs().gt(1.0e-6)
            & ~selected.rho.isin(agreement.RHO_SENTINELS)
        )
        selected = selected.loc[valid].copy()
        selected["period_years"] = selected.end_date - selected.start_date
        selected["signed_dh"] = selected.period_years * selected.b / selected.rho
        selected = selected.loc[np.isfinite(selected.signed_dh)]
        period_parts.append(selected.loc[selected.start_date.eq(start_year) & selected.end_date.eq(end_year)])
        annual_parts.append(selected.loc[selected.end_date.lt(start_year + 1)])

    # Match identifiers before selecting predictors, so every variant has the same glaciers
    if not period_parts or not annual_parts:
        raise ValueError("No full-period records or annual history are available")
    periods = pd.concat(period_parts, ignore_index=True)
    by_variant = {}
    for variant in VARIANTS:
        table = periods.loc[periods.rho_variant.eq(variant)].copy()
        if table.empty or table.rgiid.isna().any() or table.rgiid.duplicated().any():
            raise ValueError(f"Full-period glacier records must be present and unique for {variant}")
        by_variant[variant] = table.set_index("rgiid")
    common_ids = by_variant[reference_variant].index
    for variant in VARIANTS[1:]:
        common_ids = common_ids.intersection(by_variant[variant].index, sort=False)
    if common_ids.empty:
        raise ValueError("Firn variants have no common full-period glaciers")

    # Areas must describe the same glacier support in every full-model variant
    reference = by_variant[reference_variant].loc[common_ids].reset_index()
    for variant in VARIANTS[1:]:
        areas = by_variant[variant].loc[common_ids, "area"].to_numpy(float)
        if not np.allclose(areas, reference.area.to_numpy(float), rtol=1.0e-12, atol=1.0e-12):
            raise ValueError("Glacier areas differ between firn variants")
    annual = pd.concat(annual_parts, ignore_index=True)
    annual = annual.loc[annual.rgiid.isin(common_ids)]
    if annual.duplicated(["rgiid", "start_date"]).any():
        raise ValueError("Annual reference records must be unique for each glacier and year")

    # A single start year is sufficient because we evaluate one full observation period
    reference = agreement.add_past_change(reference, annual, model, start_year, start_year + 1)
    reference["rgi_region"] = pd.to_numeric(reference.rgiid.str.extract(r"RGI60-(\d+)")[0], errors="raise")
    if reference.rgi_region.isna().any():
        raise ValueError("Glacier identifiers must specify an RGI region")
    return reference


def regional_uncertainty_variance(
    data: pd.DataFrame,
    pooled: RhoSurrogate,
    variant_models: dict[str, RhoSurrogate],
    block_size: int = 400,
) -> pd.DataFrame:
    """
    Separate residual uncertainty within firn variants from differences between their regional means.
    """
    required = {"rgiid", "rgi_region", "area", "lat", "lon", "signed_dh", "past_dh", "period_years"}
    if not required.issubset(data.columns) or data.empty:
        raise ValueError("Common glacier inputs are empty or missing required columns")
    if block_size < 1 or len(variant_models) < 2:
        raise ValueError("Use a positive block size and at least two firn variants")
    numeric_columns = sorted(required - {"rgiid"})
    if not np.isfinite(data[numeric_columns]).all().all():
        raise ValueError("Common glacier inputs must be finite")
    if data.rgiid.isna().any() or data.rgiid.duplicated().any():
        raise ValueError("Common glacier identifiers must be present and unique")
    if not data.area.gt(0).all() or not data.period_years.gt(0).all() or data.period_years.nunique() != 1:
        raise ValueError("Glacier areas must be positive and all rows must share one positive period length")

    # Store all model means and standard deviations on the same input support
    model_names = ["pooled", *variant_models]
    models = [pooled, *variant_models.values()]
    records = []
    for region, group in data.groupby("rgi_region", sort=True):
        dh = group.signed_dh.to_numpy(float)
        past = group.past_dh.to_numpy(float)
        dt = group.period_years.to_numpy(float)
        area_m2 = group.area.to_numpy(float) * 1.0e6
        lat = group.lat.to_numpy(float)
        lon = group.lon.to_numpy(float)
        regional_means = []
        sigma_columns = []
        for model in models:
            density_mean = model.mu_rho(dh, past_dhdt=past, dt=dt)
            regional_means.append(float(np.sum(density_mean * dh * area_m2)))
            sigma_mass = integrated_sigma_mass_vectorized(model, area_m2, dh, np.zeros_like(dh), dt)
            density_floor_mass = float(model.params["sigma_numeric_floor"]) * np.abs(dh) * area_m2
            sigma_columns.append(np.maximum(sigma_mass, density_floor_mass))

        # Include each glacier's own variance once and covariance between different glaciers twice
        support = np.column_stack(sigma_columns)
        conditional_variances = np.zeros(len(models))
        for first in range(0, len(group), block_size):
            last = min(first + block_size, len(group))
            distances = haversine_distance_matrix(lat[first:last], lon[first:last], lat, lon)
            correlation = pooled.spatial_corr(distances)
            correlation[np.arange(last - first), np.arange(first, last)] = 1.0
            conditional_variances += np.sum(support[first:last] * (correlation @ support), axis=0)

        # Equal scenario weights give the population variance of the alternative regional means
        within_variance = float(np.mean(conditional_variances[1:]))
        between_variance = float(np.var(regional_means[1:], ddof=0))
        total_variance = within_variance + between_variance
        share_percent = 100.0 * between_variance / total_variance if total_variance > 0 else np.nan
        record = {
            "rgi_region": int(region),
            "n_glaciers": len(group),
            "area_km2": float(group.area.sum()),
            "period_years": float(dt[0]),
            "n_variants": len(variant_models),
            "within_variant_residual_variance_kg2": within_variance,
            "between_variant_mean_variance_kg2": between_variance,
            "total_residual_uncertainty_variance_kg2": total_variance,
            "firn_share_of_residual_uncertainty_percent": share_percent,
            "pooled_residual_variance_kg2": float(conditional_variances[0]),
        }
        for index, name in enumerate(model_names):
            record[f"{name}_mean_mass_change_kg"] = regional_means[index]
            record[f"{name}_conditional_variance_kg2"] = float(conditional_variances[index])
        records.append(record)
    return pd.DataFrame(records)


def uncertainty_variance_summary(detail: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize the firn share of the regional residual uncertainty budget for the paper.
    """
    shares = detail["firn_share_of_residual_uncertainty_percent"]
    if detail.empty or not np.isfinite(shares).all() or detail.rgi_region.duplicated().any():
        raise ValueError("Each region must have one finite firn uncertainty share")
    return pd.DataFrame([{
        "statistic": "median_firn_share_of_regional_residual_uncertainty_percent",
        "value_percent": float(shares.median()),
        "minimum_percent": float(shares.min()),
        "maximum_percent": float(shares.max()),
        "n_regions": len(detail),
        "n_glaciers": int(detail.n_glaciers.sum()),
        "denominator": "mean_within_variant_residual_variance_plus_between_variant_mean_variance",
        "full_model_variance_denominator": False,
        "variant_weights": "equal",
        "spatial_correlations": "pooled",
        "temporal_reconciliation": False,
        "elevation_measurement_uncertainty": False,
    }])


def run_uncertainty_analysis(output_dir: Path = OUT_DIR) -> dict[str, Path]:
    """
    Reproduce the paper's regional firn uncertainty share using existing fitted parameters.
    """
    agreement = load_script_module(STUDY_DIR / "analysis" / "surrogate_full_model_agreement.py", "agreement_uncertainty")
    pooled = RhoSurrogate.from_files(PARAM_PATH, SPATIAL_PARAM_PATH, TEMPORAL_PARAM_PATH)
    variant_models = {}
    for variant in VARIANTS:
        parameter_path = FIT_DIR / variant / "mu_sigma_model_parameters.csv"
        model = RhoSurrogate.from_files(parameter_path)
        for name in ["memory_tau_years", "tau_max", "rho_ice_fixed", "sigma_numeric_floor"]:
            if model.params[name] != pooled.params[name]:
                raise ValueError(f"The regional uncertainty comparison requires a shared {name}")
        variant_models[variant] = model

    # Rebuild the same sample and predictors from the source data rather than saved row samples
    data = read_common_uncertainty_inputs(agreement, pooled)
    print(f"[uncertainty] Matched {len(data):,} glaciers for {agreement.START_YEAR:g}-{agreement.END_YEAR:g}", flush=True)
    detail = regional_uncertainty_variance(data, pooled, variant_models)
    summary = uncertainty_variance_summary(detail)
    summary["start_year"] = agreement.START_YEAR
    summary["end_year"] = agreement.END_YEAR
    summary["reference_variant"] = VARIANTS[0]
    summary["input_csv"] = str(INPUT_CSV)
    summary["pooled_parameter_path"] = str(PARAM_PATH)
    summary["spatial_parameter_path"] = str(SPATIAL_PARAM_PATH)
    for variant in VARIANTS:
        summary[f"{variant}_parameter_path"] = str(FIT_DIR / variant / "mu_sigma_model_parameters.csv")

    # Save the numerator and denominator for every region alongside the paper's aggregate
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / OUT_UNCERTAINTY_DETAIL_CSV.name
    summary_path = output_dir / OUT_UNCERTAINTY_SUMMARY_CSV.name
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    value = float(summary.loc[0, "value_percent"])
    print(f"[uncertainty] Median firn share of regional residual uncertainty variance: {value:.6f}%", flush=True)
    return {"uncertainty_detail": detail_path, "uncertainty_summary": summary_path}


def run() -> dict[str, Path]:
    """Run per-variant fits, agreement cases, and comparison summaries."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIT_DIR.mkdir(parents=True, exist_ok=True)

    fitmod = load_script_module(STUDY_DIR / "analysis" / "fit_mean_uncertainty_rho.py", "fit_mean_uncertainty_rho_diag")
    agreement = load_script_module(STUDY_DIR / "analysis" / "surrogate_full_model_agreement.py", "agreement_diag")
    memory_config = read_final_memory_config(fitmod)

    fit_records = []
    for variant in VARIANTS:
        fit_records.append(fit_one_variant(fitmod, variant, memory_config))
    fit_metadata = pd.DataFrame(fit_records)
    fit_metadata.to_csv(OUT_FIT_METADATA_CSV, index=False)

    fit_summaries = []
    for record in fit_records:
        path = FIT_DIR / str(record["rho_variant"]) / f"rho_surrogate_{record['rho_variant']}_fit_summary.csv"
        if path.exists():
            fit_summaries.append(pd.read_csv(path))
    if fit_summaries:
        pd.concat(fit_summaries, ignore_index=True, sort=False).to_csv(OUT_FIT_SUMMARY_CSV, index=False)

    regional_parts = []
    glacier_parts = []
    individual_params = {str(row["rho_variant"]): Path(row["parameter_path"]) for row in fit_records}
    for variant in VARIANTS:
        for model_source, parameter_path in [
            (POOLED_MODEL_SOURCE, PARAM_PATH),
            (INDIVIDUAL_MODEL_SOURCE, individual_params[variant]),
        ]:
            regional, glacier = run_agreement_case(agreement, parameter_path, model_source, variant)
            regional_parts.append(regional)
            if not glacier.empty:
                glacier_parts.append(glacier)

    regional_detail = pd.concat(regional_parts, ignore_index=True, sort=False)
    if WRITE_REGIONAL_DETAIL:
        regional_detail.to_csv(OUT_REGIONAL_DETAIL_CSV, index=False)
    regional_summary = summarize_regional_detail(agreement, regional_detail)
    regional_summary.to_csv(OUT_REGIONAL_SUMMARY_CSV, index=False)

    if glacier_parts:
        glacier_detail = pd.concat(glacier_parts, ignore_index=True, sort=False)
        if WRITE_GLACIER_DETAIL:
            glacier_detail.to_csv(OUT_GLACIER_DETAIL_CSV, index=False)
        glacier_summary = summarize_glacier_detail(agreement, glacier_detail)
        glacier_summary.to_csv(OUT_GLACIER_SUMMARY_CSV, index=False)
        combined_summary = pd.concat([regional_summary, glacier_summary], ignore_index=True, sort=False)
    else:
        combined_summary = regional_summary

    comparison = build_comparison(combined_summary)
    comparison.to_csv(OUT_COMPARISON_CSV, index=False)
    primary = primary_variance_table(comparison)
    primary.to_csv(OUT_PRIMARY_CSV, index=False)

    return {
        "primary_percent": OUT_PRIMARY_CSV,
        "comparison": OUT_COMPARISON_CSV,
        "regional_summary": OUT_REGIONAL_SUMMARY_CSV,
        "regional_detail": OUT_REGIONAL_DETAIL_CSV,
        "fit_metadata": OUT_FIT_METADATA_CSV,
        "fit_summary": OUT_FIT_SUMMARY_CSV,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare pooled and variant-specific firn surrogate fits.")
    parser.add_argument(
        "--uncertainty-only", action="store_true",
        help="Reproduce the regional residual uncertainty share using existing fits, without refitting.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for --uncertainty-only; defaults to the firn diagnostic directory.",
    )
    args = parser.parse_args()
    if args.output_dir is not None and not args.uncertainty_only:
        parser.error("--output-dir requires --uncertainty-only")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if args.uncertainty_only:
            outputs = run_uncertainty_analysis(args.output_dir if args.output_dir is not None else OUT_DIR)
        else:
            outputs = run()
    for name, path in outputs.items():
        print(f"[done] {name}: {path}", flush=True)
