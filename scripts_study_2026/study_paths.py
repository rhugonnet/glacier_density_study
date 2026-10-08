"""Project paths for the 2026 manuscript scripts."""

from __future__ import annotations

from pathlib import Path


# Define final study workspace
PROJECT_DIR = Path("/home/atom/ongoing/own/glacier_density_study/final")
FIGURES_DIR = PROJECT_DIR / "figures"
RESULTS_DIR = PROJECT_DIR / "results"
DATA_DIR = PROJECT_DIR / "data"

# Define clean final outputs, one primary file per analysis step
MU_SIGMA_MAIN_PATH = RESULTS_DIR / "mu_sigma_model_parameters.csv"
CORRELATION_MAIN_PATH = RESULTS_DIR / "correlation_model_parameters.csv"
AGREEMENT_MAIN_PATH = RESULTS_DIR / "surrogate_full_model_agreement.csv"
HUGONNET_MAIN_PATH = RESULTS_DIR / "hugonnet2021_surrogate_application.csv"
MANUSCRIPT_VALUES_MAIN_PATH = RESULTS_DIR / "manuscript_values.csv"

# Define diagnostic folders for supporting tables and temporary products
DIAGNOSTICS_DIR = RESULTS_DIR / "diagnostics"
MU_SIGMA_DIAGNOSTICS_DIR = DIAGNOSTICS_DIR / "mu_sigma"
CORRELATION_DIAGNOSTICS_DIR = DIAGNOSTICS_DIR / "correlations"
AGREEMENT_DIAGNOSTICS_DIR = DIAGNOSTICS_DIR / "surrogate_full_model_agreement"
FIRN_PARAMETRIZATION_DIAGNOSTICS_DIR = DIAGNOSTICS_DIR / "firn_parametrization_variance"
HUGONNET_DIAGNOSTICS_DIR = DIAGNOSTICS_DIR / "hugonnet2021_application"
MANUSCRIPT_VALUES_DIAGNOSTICS_DIR = DIAGNOSTICS_DIR / "manuscript_values"

# Define parameter and diagnostic files written by the 2026 fitting scripts
PARAM_PATH = MU_SIGMA_MAIN_PATH
STANDARDIZED_RESIDUALS_PATH = MU_SIGMA_DIAGNOSTICS_DIR / "standardized_residuals.csv"
FIT_SUMMARY_PATH = MU_SIGMA_DIAGNOSTICS_DIR / "rho_surrogate__fit_summary.csv"
TARGET_PREDICTIONS_PATH = MU_SIGMA_DIAGNOSTICS_DIR / "rho_surrogate__target_predictions.csv"
SIGMA_ABSDH_TABLE_PATH = MU_SIGMA_DIAGNOSTICS_DIR / "rho_surrogate__sigma_absdh_residual_std_table.csv"
FACTOR_DIAGNOSTICS_PATH = MU_SIGMA_DIAGNOSTICS_DIR / "rho_surrogate__factor_diagnostics.csv"
SPATIAL_PARAM_PATH = (
    CORRELATION_DIAGNOSTICS_DIR
    / "rho_error_correlation_standardized_residuals_directcorr_constantspace_temporaldiagnostic_spatial_fit_parameters.csv"
)
TEMPORAL_PARAM_PATH = (
    CORRELATION_DIAGNOSTICS_DIR
    / "rho_error_correlation_standardized_residuals_directcorr_constantspace_temporaldiagnostic_temporal_fit_parameters.csv"
)

# Define the full model table used by manuscript diagnostics
INPUT_CSV = DATA_DIR / "rho_dV_may26_final_iteration9_sensmin_sensmax.csv"
