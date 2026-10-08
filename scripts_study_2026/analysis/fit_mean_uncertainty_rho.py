#!/usr/bin/env python3
"""
Main fitting script for the effective density surrogate mean and uncertainty model.

The final setup is activated, but many other parametrizations were explored and tested.

Notably, we tested many functional forms for the five mean components, all archived here (which is why the script is
massive...). We kept only the final best performing form with the least parameters (most parsimonious).

For the final fit, we run the optimization directly on with the final "past elevation change rate" definition of an exponentially weighted
annual rate, which requires some computation time (previously called "memory" model below).
"""

from __future__ import annotations

import gc
import json
import re
import sys
import warnings
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import least_squares
from scipy import stats

STUDY_DIR = Path(__file__).resolve().parents[1]
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

from study_paths import (
    INPUT_CSV,
    PROJECT_DIR,
    RESULTS_DIR,
    MU_SIGMA_DIAGNOSTICS_DIR,
    MU_SIGMA_MAIN_PATH,
    STANDARDIZED_RESIDUALS_PATH,
)

pd.options.mode.copy_on_write = True

# =============================================================================
# Paths
# =============================================================================

input_csv = str(INPUT_CSV)

project_dir = PROJECT_DIR
results_dir = RESULTS_DIR
results_dir.mkdir(parents=True, exist_ok=True)
out_dir = MU_SIGMA_DIAGNOSTICS_DIR
out_plot_dir = MU_SIGMA_DIAGNOSTICS_DIR
out_plot_dir.mkdir(parents=True, exist_ok=True)
run_label = "rho_surrogate"

# =============================================================================
# Data controls
# =============================================================================

rho_col = "rho"
b_col = "b"
area_col = "area"
variant_col = "rho_variant"
weight_col = "abs_dV_weight"
mean_fit_weight_col = "mean_fit_weight_sum"
OPTIONAL_OUTPUT_METADATA_COLS = ["lat", "lon", "region", "cenlat", "cenlon", "rgi_region", "O1Region", "O2Region"]

VARIANTS_TO_USE = ["iteration9", "sensmin", "sensmax"]
LOW_MEMORY_MODE = True
RUN_RESIDUAL_MEMORY_SCAN_DIAGNOSTICS = False
RUN_NO_PERIOD_DIAGNOSTIC = False
SIGNED_DH_MODE = "b_over_rho"
WEIGHT_MODE = "volume_from_row_rho_no_dt"
DIVIDE_WEIGHT_BY_N_VARIANTS_PER_PERIOD = True

# Mean model fitting uses binned volume change weighted effective densities as targets.
# This option changes the weight of each target cell in the least-squares fit to do sensitivity checks
# We use the natural weight in the end (inverse variance)
MEAN_FIT_WEIGHT_MODE = "volume_over_sigma2"
MEAN_FIT_SIGMA_U_H = 94.97725542634642
MEAN_FIT_SIGMA_U_T = 104.6354337888041
MEAN_FIT_SIGMA_FLOOR = 1.0
DH_CATEGORY_EDGES_M = np.array([0.0, 2.0, 10.0, np.inf], dtype=float)
DH_CATEGORY_LABELS = ["low_absdh_lt2m", "mid_absdh_2to10m", "high_absdh_ge10m"]
TARGET_ABS_DH_METRIC_EDGES_M = np.array(
    [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, np.inf],
    dtype=float,
)
TARGET_SIGNED_DH_METRIC_EDGES_M = np.array(
    [
        -np.inf,
        -50.0,
        -20.0,
        -10.0,
        -5.0,
        -2.0,
        -1.0,
        -0.5,
        -0.2,
        -0.1,
        0.1,
        0.2,
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
        20.0,
        50.0,
        np.inf,
    ],
    dtype=float,
)

RHO_SENTINELS = {-99999.0, 99999.0}
MIN_ABS_RHO_FOR_DH_AND_WEIGHT = 1e-6

# =============================================================================
# Memory controls
# =============================================================================

MODEL_MEMORY_MODE = "fixed_window"  # "fixed_window" or "exp_cumulative"
MODEL_MEMORY_WINDOW_YEARS = 5
MODEL_MEMORY_TAU_YEARS = 4.0
MODEL_MEMORY_KMAX_YEARS = 15
REQUIRE_FULL_EXP_MEMORY = False
EXP_MEMORY_MIN_VALID_LAGS = 1
NORMALIZE_EXP_MEMORY_WEIGHTS = True

RUN_MEMORY_PROFILE = False
MEMORY_PROFILE_FIXED_WINDOWS = list(range(1, 16))
RUN_EXP_MEMORY_PROFILE = False
MEMORY_PROFILE_EXP_TAUS = [0.5, 1, 1.5, 2, 3, 4, 5, 7, 10, 15]

# Configure joint comparison of past elevation change rate kernels
# Optimize one timescale per kernel with the same annual lag matrix
RUN_JOINT_KERNEL_MEMORY_FIT = True
USE_JOINT_EXPONENTIAL_MEMORY_FOR_DIAGNOSTICS = True
JOINT_KERNEL_MEMORY_FORMS = ["exponential"]
JOINT_KERNEL_MEMORY_MAX_ROWS = None  # Fit joint memory on all eligible rows
JOINT_KERNEL_MEMORY_KMAX_YEARS = MODEL_MEMORY_KMAX_YEARS
JOINT_KERNEL_MEMORY_TAU_BOUNDS = (0.5, 30.0)
JOINT_KERNEL_MEMORY_TAU_INITIALS = {"exponential": 5.0, "gaussian": 5.0, "spherical": 5.0}
JOINT_KERNEL_MEMORY_NORMALIZE_WEIGHTS = True
JOINT_KERNEL_MEMORY_MAX_STARTS = 1
JOINT_KERNEL_MEMORY_RANDOM_SEED = 743
JOINT_KERNEL_MEMORY_VERBOSE_EVERY_N_EVAL = 25
# Joint kernel fits use relaxed tolerances because the objective is flat near the optimum
JOINT_KERNEL_MEMORY_MAX_NFEV = 80
# Residual-call limits stop nearly flat optimizations at a stable solution
JOINT_KERNEL_MEMORY_MAX_RESIDUAL_EVALS = 1200
JOINT_KERNEL_MEMORY_EARLY_STOP_PATIENCE_EVALS = 250
JOINT_KERNEL_MEMORY_EARLY_STOP_REL_IMPROVEMENT = 1e-7
JOINT_KERNEL_MEMORY_FTOL = 1e-5
JOINT_KERNEL_MEMORY_XTOL = 1e-5
JOINT_KERNEL_MEMORY_GTOL = 1e-5

RESIDUAL_MEMORY_WINDOWS_TO_SCAN = list(range(1, 16))

# =============================================================================
# Binning / fitting controls
# =============================================================================

CURRENT_DH_MIN_ABS_FOR_FIT = 0.05
DH_MIN_POSITIVE_EDGE = 0.05
DH_LOG_PATTERN = np.array([0.05, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500], dtype=float)
USE_ACTUAL_MINMAX_AS_OUTER_EDGES = True
MEMORY_N_QUANTILE_BINS_FOR_FIT = 8
AREA_N_QUANTILE_BINS = 10

# Residual-memory diagnostic binning
MEMORY_N_QUANTILE_BINS = 12
MEMORY_MIN_COUNT_BIN = 50
MEMORY_MIN_NEFF_BIN = 10

MIN_COUNT_3D_CELL = 40
MIN_NEFF_3D_CELL = 10
MIN_COUNT_DIAG_BIN = 40
MIN_NEFF_DIAG_BIN = 10
MIN_VALID_CELLS_PER_PERIOD_LINE = 3

# Stricter support requirement for binned residual STD used to fit sigma_rho
# STD estimates are much more sensitive to sparse bins/outliers than mean bins
MIN_COUNT_SIGMA_BIN = 120
MIN_NEFF_SIGMA_BIN = 30

RHO_ICE_FIXED = 900.0
ETA_D_FIXED = 0.5

# =============================================================================
# Candidate models
# =============================================================================

# Configure candidate model generation
# The component grid replaces the manual candidate list when enabled
GENERATE_COMPONENT_GRID = True
GRID_MEMORY_FORMS = ["logistic_sqrt_centered", "signed_sqrt_contrast", "signed_power_contrast"]
GRID_RATIO_FORMS = ["tanh_power", "tanh_fixed_alpha1", "signed_power_fixed_alpha1"]
GRID_DAMPING_FORMS = ["exp_stretched"]
GRID_PERIOD_FORMS = ["exp_no_offset", "exp_no_offset_fixed_T0_3"]
GRID_CURRENT_FORMS = ["constant"]
MAX_GRID_MODELS = None  # Use an integer for a shorter model search

# Fit only the selected parsimonious mean architecture for final diagnostics
SKIP_FULL_MEAN_MODEL_SEARCH = True
FINAL_MEAN_MODEL_SPEC = dict(
    name="final_signed_power_tanh_power_period_power_constant_current_volume_over_sigma2",
    memory="signed_power_contrast",
    ratio="tanh_power",
    damping="exp_stretched",
    period="power_param",
    current="constant",
)

# Old candidate models
CANDIDATE_MODELS = [
    dict(name="m34_param_period", memory="logistic_sqrt", ratio="tanh_power", damping="exp_stretched", period="exp_param", current="constant"),
    dict(name="no_period", memory="logistic_sqrt", ratio="tanh_power", damping="exp_stretched", period="none", current="rational_power"),
    dict(name="no_current_curvature", memory="logistic_sqrt", ratio="tanh_power", damping="exp_stretched", period="exp_param", current="none"),
    dict(name="period_power", memory="logistic_sqrt", ratio="tanh_power", damping="exp_stretched", period="power_param", current="rational_power"),
    dict(name="current_band", memory="logistic_sqrt", ratio="tanh_power", damping="exp_stretched", period="exp_param", current="rational_band"),
    dict(name="damping_rational", memory="logistic_sqrt", ratio="tanh_power", damping="rational_power", period="exp_param", current="rational_power"),
    dict(name="memory_free_eta", memory="logistic_free_eta", ratio="tanh_power", damping="exp_stretched", period="exp_param", current="rational_power"),
    dict(name="centered_full_reference", memory="logistic_sqrt_centered", ratio="tanh_power", damping="exp_stretched", period="exp_param", current="rational_power"),
    dict(name="centered_parsimonious_no_offset", memory="logistic_sqrt_centered", ratio="tanh_power", damping="exp_stretched", period="exp_no_offset", current="rational_power"),
    dict(name="centered_parsimonious_no_offset_constant_current", memory="logistic_sqrt_centered", ratio="tanh_power", damping="exp_stretched", period="exp_no_offset", current="constant"),
    dict(name="sqrtmem_tanh_alpha1_Tfree_constant", memory="signed_sqrt_contrast", ratio="tanh_fixed_alpha1", damping="exp_stretched", period="exp_no_offset", current="constant"),
    dict(name="sqrtmem_tanh_alpha1_T3_constant", memory="signed_sqrt_contrast", ratio="tanh_fixed_alpha1", damping="exp_stretched", period="exp_no_offset_fixed_T0_3", current="constant"),
    dict(name="sqrtmem_tanh_power_Tfree_constant", memory="signed_sqrt_contrast", ratio="tanh_power", damping="exp_stretched", period="exp_no_offset", current="constant"),
    dict(name="powermem_tanh_power_Tfree_constant", memory="signed_power_contrast", ratio="tanh_power", damping="exp_stretched", period="exp_no_offset", current="constant"),
    dict(name="centeredmem_tanh_alpha1_T3_constant", memory="logistic_sqrt_centered", ratio="tanh_fixed_alpha1", damping="exp_stretched", period="exp_no_offset_fixed_T0_3", current="constant"),
]

def build_component_grid_candidates():
    candidates = []
    for mem in GRID_MEMORY_FORMS:
        for ratio in GRID_RATIO_FORMS:
            for damping in GRID_DAMPING_FORMS:
                for period in GRID_PERIOD_FORMS:
                    for current in GRID_CURRENT_FORMS:
                        name = f"mem-{mem}__ratio-{ratio}__damp-{damping}__period-{period}__cur-{current}"
                        candidates.append(dict(name=name, memory=mem, ratio=ratio, damping=damping, period=period, current=current))
    if MAX_GRID_MODELS is not None:
        candidates = candidates[:int(MAX_GRID_MODELS)]
    return candidates

if GENERATE_COMPONENT_GRID:
    CANDIDATE_MODELS = build_component_grid_candidates()

if SKIP_FULL_MEAN_MODEL_SEARCH:
    CANDIDATE_MODELS = [dict(FINAL_MEAN_MODEL_SPEC)]

SELECTED_MODEL_FOR_DIAGNOSTICS = None  # None = best WRMSE; with SKIP_FULL_MEAN_MODEL_SEARCH=True, this is FINAL_MEAN_MODEL_SPEC

# Final default mean model:
#   signed-power memory + tanh ratio + stretched-exponential damping
#   + period-power correction + constant damped current-dh offset.
# This is used as the reference architecture for the joint memory-kernel fit.

BOUNDS = {
    "D0": (0.0, 5000.0), "LD": (0.1, 200.0), "etaD": (0.05, 2.0), "xD": (0.05, 500.0), "qD": (0.05, 8.0),
    "A": (-10000.0, 10000.0), "Aneg": (-10000.0, 10000.0), "Apos": (-10000.0, 10000.0),
    "LA": (0.1, 200.0), "alpha": (0.05, 0.95), "alphaH": (0.05, 1.5),
    "alphaNeg": (0.05, 1.5), "alphaPos": (0.05, 1.5), "etaQ": (0.1, 3.0),
    "alphaBase": (0.05, 0.95), "alphaAmp": (0.0, 0.8), "LP": (0.05, 10.0), "etaAlpha": (0.1, 3.0),
    "H": (0.1, 500.0), "beta": (0.05, 3.0),
    "xR": (0.1, 500.0), "qR": (0.05, 8.0),
    "C0": (-2000.0, 2000.0), "C1": (-2000.0, 2000.0), "T0": (0.25, 50.0),
    "P0": (-2000.0, 2000.0), "P1": (-2000.0, 2000.0), "TP": (0.25, 50.0),
    "Bc": (-2000.0, 2000.0), "xS": (0.1, 500.0), "qS": (0.05, 8.0),
    "xSL": (0.1, 50.0), "qSL": (0.05, 8.0),
    "xC": (0.02, 2.0), "qC": (0.05, 2.0),
    "x1": (0.1, 500.0), "x2": (0.2, 1000.0),
    "x1L": (0.1, 10.0), "x2L": (0.2, 100.0),
    "Bmem": (-1000.0, 1000.0),
    "etaMem": (0.1, 3.0),
}

INITIAL = {
    "D0": 700.0, "LD": 1.5, "etaD": 0.5, "xD": 8.7, "qD": 0.5,
    "A": 200.0, "Aneg": 200.0, "Apos": 200.0,
    "LA": 6.0, "alpha": 0.9, "alphaH": 1.0,
    "alphaNeg": 0.9, "alphaPos": 0.9, "etaQ": 0.5,
    "alphaBase": 0.6, "alphaAmp": 0.25, "LP": 0.8, "etaAlpha": 0.8,
    "H": 8.7, "beta": 0.47,
    "xR": 8.7, "qR": 0.5,
    "C0": 0.0, "C1": 100.0, "T0": 5.0,
    "P0": 0.0, "P1": 100.0, "TP": 5.0,
    "Bc": 100.0, "xS": 8.7, "qS": 0.5,
    "xSL": 8.7, "qSL": 0.5,
    "xC": 0.2, "qC": 0.7,
    "x1": 2.0, "x2": 50.0,
    "x1L": 1.0, "x2L": 20.0,
    "Bmem": 100.0,
    "etaMem": 0.5,
}

N_RANDOM_STARTS = 25
MAX_NFEV = 30000
RANDOM_SEED = 42

# =============================================================================
# Plot controls and outputs
# =============================================================================

DPI = 250
PLOT_CMAP = "viridis"
PLOT_CMAP_PERIOD = "viridis"
PLOT_CMAP_MEMORY = "PuOr"
MAIN_USE_LEGENDS = True
MAIN_MEMORY_N_CURVES = 5
MAIN_MEMORY_TARGET_LABELS = np.array([-4.0, -1.0, -0.25, 0.25, 1.0, 4.0], dtype=float)
MAIN_DENSE_X_MIN = 0.05
MAIN_DENSE_X_MAX = 50.0
MAIN_DENSE_X_N = 400
SIGMA_MAX_ABS_DH_FOR_FIT = 100.0
MAIN_SMALLBIN_DISPLAY_CENTER = 0.1
MAIN_SMALLBIN_EDGE_LOW = 0.05
MAIN_SMALLBIN_EDGE_HIGH = 0.2
MAIN_RIGHTPANEL_MIN_COUNT = 1
MAIN_RIGHTPANEL_MIN_NEFF = 1.0
SIGNED_DH_SYMLOG_LINTHRESH = CURRENT_DH_MIN_ABS_FOR_FIT
PERIOD_LENGTHS_TO_PLOT = [1, 2, 4, 7, 10, 15]
DISPLAY_SKIP_CURRENT_ABS_BELOW = 0.11
RESIDUAL_YLIM = None
MAX_ROW_SAMPLE_OUTPUT = 100000
RUN_PAPER_FIGURES = False
WRITE_FIT_DIAGNOSTIC_PLOTS = False
MAIN_MEAN_LEFT_MIN_ABS_DH_DISPLAY = 1.0
MAIN_MEAN_DYNAMIC_Y_MIN_PAD = 25.0

# Optional temporal closure diagnostic for sigma_rho
# This tests whether replacing net |dh| by the cumulative absolute path length
# Use L_A = sum_i |dh_i| for nested-period uncertainty consistency
RUN_PATH_LENGTH_SIGMA_DIAGNOSTIC = False
PATH_LENGTH_SOURCE_PERIOD_YEARS = 1.0
PATH_LENGTH_REQUIRE_FULL_COVERAGE = True
PATH_RATIO_N_BINS = 4
PATH_SIGMA_MAX_ABS_DH_FOR_FIT = SIGMA_MAX_ABS_DH_FOR_FIT

PATH_LENGTH_PERIOD_MARKERS = {
    1.0: "o",
    2.0: "s",
    4.0: "^",
    7.0: "D",
    10.0: "P",
    15.0: "X",
    20.0: "v",
}
PATH_LENGTH_DEFAULT_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "8"]

# Metrics and standardization
# The floor avoids exploding normalized metrics in bins with artificially tiny
# Within-bin variability
NORMALIZED_METRIC_SIGMA_FLOOR = 50.0
SIGMA_NUMERIC_FLOOR = 1.0

# Small fixed offset used only by log-linear sigma candidates to avoid log(0)
# Keep this fixed to avoid an unnecessary extra parameter absorbing the slope
SIGMA_LOGLINEAR_X0 = CURRENT_DH_MIN_ABS_FOR_FIT
SIGMA_TWOSLOPE_ETA_FIXED = 4.0
# Fixed large-|dh| exponent for sigma candidates constrained to alpha/sqrt(|dh|)
SIGMA_SQRT_TAIL_EXPONENT_FIXED = 0.5
SIGMA_Q0_FLAT_BASE_FIXED = 0.0
SIGMA_PERIOD_DECAY_SQRT_FIXED = 0.5
SIGMA_PERIOD_DECAY_LINEAR_FIXED = 1.0
SIGMA_FIXED_X0_FOR_SQRT_BASE = CURRENT_DH_MIN_ABS_FOR_FIT

# Candidate residual-spread models. The final choice is selected by weighted
# RMSE on binned mean-removed residual STD. The centered-period anomaly model
# Tests an intermediate-period common slope with short-period negative and
# Long-period positive departures localized to small |dh|
#
# Rational_floor:
#     S_floor + s_amp / [1 + (|dh|/x_sig)^q_sig]
#
# Inv_power:
#     A / (|dh| + x0)^q
#
# Inv_power_period_amp:
#     [A0 + A1 (1 - exp(-dt/T_sig))] / (|dh| + x0)^q
#
# Inv_power_period_damped:
#     A0/(|dh|+x0)^q0
#     + A1[1-exp(-dt/T_sig)] / [(|dh|+x0)^q1 (1+(|dh|/x_sig)^q_sig)]
#
# Inv_power_period_damped_sqrt_tail:
#     Same as inv_power_period_damped, but the common baseline bends toward
#     Alpha/sqrt(|dh|) at large |dh|
#
# Inv_power_period_damped_sqrt_tail_fixed_x0:
#     Smooth additive sqrt-tail version with x0 fixed
#
# Inv_power_period_damped_sqrt_tail_shared_q:
#     Smooth additive sqrt-tail version with the period exponent q1 tied to q0
#
# Inv_power_period_damped_sqrt_tail_shared_q_fixed_x0:
#     Seven-parameter nested version with fixed x0 and q1 tied to q0
#
# Inv_power_period_damped_sqrt_tail_flatbase_fixed_x0:
#     Six-parameter nested additive version with q0=0 and fixed x0
#
# Inv_power_period_damped_sqrt_tail_flatbase_psig05_fixed_x0:
#     Five-parameter nested additive version with q0=0, fixed x0 and q_sig=1/2
#
# Inv_power_period_damped_sqrt_tail_flatbase_psig1_fixed_x0:
#     Five-parameter nested additive version with q0=0, fixed x0 and q_sig=1
#
# Sqrt_base_period_addition:
#     Pure A0/sqrt(|dh|) baseline plus additive positive period anomaly
#
# Sqrt_base_period_addition_psig05:
#     Same, with q_sig fixed to 1/2
#
# Sqrt_base_period_addition_psig1:
#     Same, with q_sig fixed to 1
#
# Variance_additive_sqrt_time:
#     Variance-additive model with sigma^2 = A0^2/|dh| + A1^2 dt/|dh|^2,
#     Respecting independent sub-period uncertainty aggregation
#
# Variance_additive_sqrt_time_localized:
#     Same, with the time-accumulating variance term localized to small |dh|
#
# Sqrt_tail_multiplicative_shared_bend:
#     Multiplicative sqrt-tail model where the period anomaly uses the same
#     Transition scale as the baseline bend, avoiding two slope-change stages
#
# Sqrt_tail_multiplicative_shared_bend_fixed_x0:
#     Five-parameter fixed-x0 version
#
# Sqrt_zero_base_period_addition_fixed_x0:
#     Five-parameter zero-period alpha/sqrt(|dh|+fixed_x0) baseline plus a
#     Positive period anomaly localized to small |dh|
SIGMA_MODEL_FOR_DIAGNOSTICS = "variance_additive_sqrt_time"
SIGMA_CANDIDATE_FORMS = [
    "rational_floor",
    "inv_power",
    "inv_power_period_amp",
    "inv_power_period_damped",
    "inv_power_period_damped_sqrt_tail",
    "inv_power_period_damped_sqrt_tail_fixed_x0",
    "inv_power_period_damped_sqrt_tail_shared_q",
    "inv_power_period_damped_sqrt_tail_shared_q_fixed_x0",
    "inv_power_period_damped_sqrt_tail_flatbase_fixed_x0",
    "inv_power_period_damped_sqrt_tail_flatbase_psig05_fixed_x0",
    "inv_power_period_damped_sqrt_tail_flatbase_psig1_fixed_x0",
    "sqrt_base_period_addition",
    "sqrt_base_period_addition_psig05",
    "sqrt_base_period_addition_psig1",
    "variance_additive_sqrt_time",
    "variance_additive_sqrt_time_localized",
    "sqrt_tail_multiplicative_shared_bend",
    "sqrt_tail_multiplicative_shared_bend_fixed_x0",
    "sqrt_zero_base_period_addition",
    "sqrt_zero_base_period_addition_fixed_x0",
    "inv_power_short_reduction",
    "inv_power_centered_period_anomaly",
    "inv_power_period_damped_twoslope",
    "inv_power_short_reduction_twoslope",
    "sqrt_zero_base_period_addition",
    "loglinear_centered_period_anomaly",
    "loglinear_damped_centered_period_anomaly",
    "logsaturating_centered_period_anomaly",
]
SIGMA_PARAMS_BY_FORM = {
    "rational_floor": ["s_floor", "s_amp", "x_sig", "q_sig"],
    "inv_power": ["A", "x0", "q"],
    "inv_power_period_amp": ["A0", "A1", "T_sig", "x0", "q"],
    "inv_power_period_damped": ["A0", "q0", "A1", "T_sig", "x0", "q1", "x_sig", "q_sig"],
    "inv_power_period_damped_sqrt_tail": ["A0", "x0", "q0", "xb_sig", "A1", "T_sig", "q1", "x_sig", "q_sig"],
    "inv_power_period_damped_sqrt_tail_fixed_x0": ["A0", "q0", "xb_sig", "A1", "T_sig", "q1", "x_sig", "q_sig"],
    "inv_power_period_damped_sqrt_tail_shared_q": ["A0", "x0", "q0", "xb_sig", "A1", "T_sig", "x_sig", "q_sig"],
    "inv_power_period_damped_sqrt_tail_shared_q_fixed_x0": ["A0", "q0", "xb_sig", "A1", "T_sig", "x_sig", "q_sig"],
    "inv_power_period_damped_sqrt_tail_flatbase_fixed_x0": ["A0", "xb_sig", "A1", "T_sig", "x_sig", "q_sig"],
    "inv_power_period_damped_sqrt_tail_flatbase_psig05_fixed_x0": ["A0", "xb_sig", "A1", "T_sig", "x_sig"],
    "inv_power_period_damped_sqrt_tail_flatbase_psig1_fixed_x0": ["A0", "xb_sig", "A1", "T_sig", "x_sig"],
    "sqrt_base_period_addition": ["A0", "A1", "T_sig", "x_sig", "q_sig"],
    "sqrt_base_period_addition_psig05": ["A0", "A1", "T_sig", "x_sig"],
    "sqrt_base_period_addition_psig1": ["A0", "A1", "T_sig", "x_sig"],
    "variance_additive_sqrt_time": ["A0", "A1"],
    "variance_additive_sqrt_time_localized": ["A0", "A1", "x_sig", "q_sig"],
    "sqrt_tail_multiplicative_shared_bend": ["A", "x0", "q0", "xb_sig", "lam_add", "T_sig"],
    "sqrt_tail_multiplicative_shared_bend_fixed_x0": ["A", "q0", "xb_sig", "lam_add", "T_sig"],
    "sqrt_zero_base_period_addition_fixed_x0": ["A", "lam_add", "T_sig", "x_sig", "q_sig"],
    "sqrt_tail_fractional_period_addition": ["A", "x0", "q0", "xb_sig", "lam_add", "T_sig", "x_sig", "q_sig"],
    "sqrt_tail_fractional_period_addition_fixed_x0": ["A", "q0", "xb_sig", "lam_add", "T_sig", "x_sig", "q_sig"],
    "inv_power_short_reduction": ["A", "x0", "q", "lam", "T_sig", "x_sig", "q_sig"],
    "inv_power_centered_period_anomaly": ["A", "x0", "q", "lam", "T_sig", "F0", "x_sig", "q_sig"],
    "inv_power_period_damped_twoslope": ["A0", "x0", "q0", "xb_sig", "qinf", "A1", "T_sig", "q1", "x_sig", "q_sig"],
    "inv_power_short_reduction_twoslope": ["A", "x0", "q", "xb_sig", "qinf", "lam", "T_sig", "x_sig", "q_sig"],
    "sqrt_zero_base_period_addition": ["A", "x0", "lam_add", "T_sig", "x_sig", "q_sig"],
    "loglinear_centered_period_anomaly": ["B0", "B1", "Aper", "T_sig", "F0", "x_sig", "q_sig"],
    "loglinear_damped_centered_period_anomaly": ["B0", "B1", "Aper", "T_sig", "F0", "x_sig", "q_sig", "xD_sig", "qD_sig"],
    "logsaturating_centered_period_anomaly": ["B0", "B1", "xL_sig", "Aper", "T_sig", "F0", "x_sig", "q_sig"],
}
SIGMA_BOUNDS = {
    "s_floor": (0.0, 1000.0),
    "s_amp": (0.0, 10000.0),
    "x_sig": (0.001, 500.0),
    "q_sig": (0.05, 8.0),
    "A": (1.0, 50000.0),
    "x0": (0.001, 5.0),
    "q": (0.05, 3.0),
    "A0": (1.0, 50000.0),
    "A1": (0.0, 50000.0),
    "T_sig": (0.25, 50.0),
    "q0": (0.05, 3.0),
    "q1": (0.05, 3.0),
    "lam": (-0.95, 0.95),
    "lam_add": (0.0, 5.0),
    "F0": (0.0, 1.0),
    "B0": (1.0, 5000.0),
    "B1": (-2000.0, 2000.0),
    "Aper": (-5000.0, 5000.0),
    "xD_sig": (0.05, 1000.0),
    "qD_sig": (0.05, 8.0),
    "xL_sig": (0.01, 500.0),
    "xb_sig": (0.1, 500.0),
    "qinf": (0.01, 4.0),
}
SIGMA_INITIAL = {
    "s_floor": 10.0,
    "s_amp": 300.0,
    "x_sig": 5.0,
    "q_sig": 0.8,
    "A": 300.0,
    "x0": 0.2,
    "q": 0.8,
    "A0": 200.0,
    "A1": 200.0,
    "T_sig": 5.0,
    "q0": 0.8,
    "q1": 0.8,
    "lam": 0.5,
    "lam_add": 0.5,
    "F0": 0.5,
    "B0": 500.0,
    "B1": 80.0,
    "Aper": 150.0,
    "xD_sig": 100.0,
    "qD_sig": 1.0,
    "xL_sig": 1.0,
    "xb_sig": 10.0,
    "qinf": 0.5,
}

# Standard default names; overwritten by the selected sigma form
SIGMA_PARAMS = SIGMA_PARAMS_BY_FORM["rational_floor"]

def _mem_suffix(mode, window=None, tau=None):
    if mode == "fixed_window":
        return f"mem_fixed{window:g}yr"
    if mode == "exp_cumulative":
        return f"mem_exp_tau{tau:g}yr"
    return f"mem_{mode}"

main_suffix = ""

out_fit_summary_csv = out_dir / f"{run_label}_{main_suffix}_fit_summary.csv"
out_params_csv = out_dir / f"{run_label}_{main_suffix}_parameters.csv"
out_final_joint_model_csv = MU_SIGMA_MAIN_PATH
out_final_joint_model_json = out_dir / f"{run_label}_{main_suffix}_final_joint_rho_model_parameters.json"
out_target_csv = out_plot_dir / f"{run_label}_{main_suffix}_target_predictions.csv"
out_target_dh_category_csv = out_plot_dir / f"{run_label}_{main_suffix}_target_prediction_metrics_by_dh_category.csv"
out_target_period_absdh_csv = out_plot_dir / f"{run_label}_{main_suffix}_target_prediction_metrics_by_period_absdh.csv"
out_target_period_signeddh_csv = out_plot_dir / f"{run_label}_{main_suffix}_target_prediction_metrics_by_period_signeddh.csv"
out_row_sample_csv = out_plot_dir / f"{run_label}_{main_suffix}_row_predictions_sample.csv"
out_standardized_residuals_csv = STANDARDIZED_RESIDUALS_PATH
out_diag_csv = out_plot_dir / f"{run_label}_{main_suffix}_factor_diagnostics.csv"
out_memory_scan_csv = out_plot_dir / f"{run_label}_{main_suffix}_residual_memory_scan.csv"
out_memory_rate_scan_csv = out_plot_dir / f"{run_label}_{main_suffix}_residual_memory_rate_scan.csv"
out_memory_score_csv = out_plot_dir / f"{run_label}_{main_suffix}_residual_memory_scores.csv"
out_memory_rate_score_csv = out_plot_dir / f"{run_label}_{main_suffix}_residual_memory_rate_scores.csv"
out_memory_profile_csv = out_plot_dir / f"{run_label}_memory_profile_fit_summary.csv"

out_joint_kernel_memory_summary_csv = out_plot_dir / f"{run_label}_{main_suffix}_joint_kernel_memory_summary.csv"
out_joint_kernel_memory_params_csv = out_plot_dir / f"{run_label}_{main_suffix}_joint_kernel_memory_parameters.csv"
out_joint_kernel_memory_weights_csv = out_plot_dir / f"{run_label}_{main_suffix}_joint_kernel_memory_weights.csv"
out_joint_kernel_memory_weights_png = out_plot_dir / f"{run_label}_{main_suffix}_joint_kernel_memory_weights.png"
out_joint_kernel_memory_residual_png = out_plot_dir / f"{run_label}_{main_suffix}_joint_kernel_memory_residuals.png"
out_joint_kernel_memory_scatter_png = out_plot_dir / f"{run_label}_{main_suffix}_joint_kernel_memory_observed_predicted.png"

out_metric_png = out_plot_dir / f"{run_label}_{main_suffix}_model_metric_comparison.png"
out_current_png = out_plot_dir / f"{run_label}_{main_suffix}_before_after_currentdh_by_period.png"
out_abs_current_png = out_plot_dir / f"{run_label}_{main_suffix}_before_after_abs_currentdh_by_period.png"
out_memory_png = out_plot_dir / f"{run_label}_{main_suffix}_before_after_memorydh_by_period.png"
out_area_png = out_plot_dir / f"{run_label}_{main_suffix}_before_after_area_by_period.png"
out_period_png = out_plot_dir / f"{run_label}_{main_suffix}_before_after_period.png"
out_period_component_png = out_plot_dir / f"{run_label}_{main_suffix}_period_component.png"
out_current_component_png = out_plot_dir / f"{run_label}_{main_suffix}_current_component.png"
out_memory_scan_png = out_plot_dir / f"{run_label}_{main_suffix}_residuals_vs_memory_windows.png"
out_memory_rate_scan_png = out_plot_dir / f"{run_label}_{main_suffix}_residuals_vs_memory_rate_windows.png"
out_memory_score_png = out_plot_dir / f"{run_label}_{main_suffix}_memory_score_curve.png"
out_memory_rate_score_png = out_plot_dir / f"{run_label}_{main_suffix}_memory_rate_score_curve.png"
out_memory_profile_png = out_plot_dir / f"{run_label}_memory_profile_score.png"

out_memory_profile_nwrmse_png = out_plot_dir / f"{run_label}_memory_profile_nwrmse.png"
out_memory_profile_resid_score_png = out_plot_dir / f"{run_label}_memory_profile_residual_memory_score.png"
out_memory_heatmap_cumulative_png = out_plot_dir / f"{run_label}_memory_residual_score_heatmap_cumulative.png"
out_memory_heatmap_rate_png = out_plot_dir / f"{run_label}_memory_residual_score_heatmap_rate.png"
out_memory_profile_diagnostic_long_csv = out_plot_dir / f"{run_label}_memory_profile_diagnostic_long.csv"
out_memory_profile_summary_csv = out_plot_dir / f"{run_label}_memory_profile_summary.csv"
out_memory_profile_best_cumulative_png = out_plot_dir / f"{run_label}_best_memory_residuals_vs_pastdh.png"
out_memory_profile_best_rate_png = out_plot_dir / f"{run_label}_best_memory_residuals_vs_pastdh_rate.png"
out_no_period_current_png = out_plot_dir / f"{run_label}_{main_suffix}_no_period_residuals_currentdh.png"
out_period_correction_performance_png = out_plot_dir / f"{run_label}_{main_suffix}_period_correction_performance.png"
out_no_period_diag_csv = out_plot_dir / f"{run_label}_{main_suffix}_no_period_residual_diagnostics.csv"

out_std_current_png = out_plot_dir / f"{run_label}_{main_suffix}_std_before_after_currentdh_by_period.png"
out_std_abs_current_png = out_plot_dir / f"{run_label}_{main_suffix}_std_before_after_abs_currentdh_by_period.png"
out_std_memory_png = out_plot_dir / f"{run_label}_{main_suffix}_std_before_after_memorydh_by_period.png"
out_std_area_png = out_plot_dir / f"{run_label}_{main_suffix}_std_before_after_area_by_period.png"
out_std_period_png = out_plot_dir / f"{run_label}_{main_suffix}_std_before_after_period.png"

out_z_current_png = out_plot_dir / f"{run_label}_{main_suffix}_standardized_after_currentdh_by_period.png"
out_z_abs_current_png = out_plot_dir / f"{run_label}_{main_suffix}_standardized_after_abs_currentdh_by_period.png"
out_z_memory_png = out_plot_dir / f"{run_label}_{main_suffix}_standardized_after_memorydh_by_period.png"
out_z_area_png = out_plot_dir / f"{run_label}_{main_suffix}_standardized_after_area_by_period.png"
out_z_period_png = out_plot_dir / f"{run_label}_{main_suffix}_standardized_after_period.png"
out_sigma_abs_table_csv = out_plot_dir / f"{run_label}_{main_suffix}_sigma_absdh_residual_std_table.csv"
out_sigma_params_csv = out_plot_dir / f"{run_label}_{main_suffix}_sigma_params_mean_removed_residual.csv"
out_sigma_model_selection_csv = out_plot_dir / f"{run_label}_{main_suffix}_sigma_model_selection.csv"
out_sigma_fit_by_period_png = out_plot_dir / f"{run_label}_{main_suffix}_sigma_fit_by_period_absdh.png"
out_sigma_fit_by_model_dir = out_plot_dir / f"{run_label}_{main_suffix}_sigma_fit_by_model"
out_sigma_path_table_csv = out_plot_dir / f"{run_label}_{main_suffix}_sigma_pathlength_residual_std_table.csv"
out_sigma_path_model_selection_csv = out_plot_dir / f"{run_label}_{main_suffix}_sigma_pathlength_model_selection.csv"
out_sigma_path_observed_predicted_png = out_plot_dir / f"{run_label}_{main_suffix}_sigma_pathlength_observed_predicted.png"
out_sigma_path_observed_predicted_net_png = out_plot_dir / f"{run_label}_{main_suffix}_sigma_pathlength_observed_predicted_net_formula.png"
out_sigma_path_observed_predicted_path_png = out_plot_dir / f"{run_label}_{main_suffix}_sigma_pathlength_observed_predicted_path_formula.png"
out_sigma_path_observed_predicted_absbin_png = out_plot_dir / f"{run_label}_{main_suffix}_sigma_pathlength_observed_predicted_absbin_markers.png"
out_sigma_path_ratio_diagnostic_png = out_plot_dir / f"{run_label}_{main_suffix}_sigma_pathlength_ratio_diagnostic.png"
out_sigma_path_ratio_diagnostic_net_png = out_plot_dir / f"{run_label}_{main_suffix}_sigma_pathlength_ratio_diagnostic_net_formula.png"
out_sigma_path_ratio_diagnostic_path_png = out_plot_dir / f"{run_label}_{main_suffix}_sigma_pathlength_ratio_diagnostic_path_formula.png"
out_sigma_path_ratio_diagnostic_absbin_png = out_plot_dir / f"{run_label}_{main_suffix}_sigma_pathlength_ratio_diagnostic_absbin_markers.png"

out_mean_fit_dir = out_plot_dir / f"{run_label}_{main_suffix}_mean_fit_by_model"
out_sigma_fit_signed_png = out_plot_dir / f"{run_label}_{main_suffix}_sigma_fit_signed_currentdh.png"

out_main_mean_figure_png = out_dir / f"{run_label}_{main_suffix}_FIG_main_rho_mean.png"
out_main_std_figure_png = out_dir / f"{run_label}_{main_suffix}_FIG_main_rho_std.png"
out_main_mean_figure_linear_png = out_dir / f"{run_label}_{main_suffix}_FIG_main_rho_mean_linear_x.png"
out_main_std_figure_linear_png = out_dir / f"{run_label}_{main_suffix}_FIG_main_rho_std_linear_x.png"
out_supp_mean_figure_png = out_dir / f"{run_label}_{main_suffix}_FIG_supp_rho_mean_diagnostics.png"
out_supp_std_figure_png = out_dir / f"{run_label}_{main_suffix}_FIG_supp_rho_std_diagnostics.png"
out_supp_normality_figure_png = out_dir / f"{run_label}_{main_suffix}_FIG_supp_00_rho_normality.png"

# =============================================================================
# Utilities
# =============================================================================

def log(msg, t0=None):
    if t0 is None:
        print(f"[progress] {msg}", flush=True)
    else:
        print(f"[progress] {msg} | elapsed {perf_counter() - t0:.1f} s", flush=True)


def _valid_xy(x, w):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return x[ok], w[ok]


def weighted_mean(values, weights):
    values, weights = _valid_xy(values, weights)
    if values.size == 0:
        return np.nan
    return np.average(values, weights=weights)


def weighted_std(values, weights):
    values, weights = _valid_xy(values, weights)
    if values.size < 2:
        return np.nan
    weights = weights / np.sum(weights)
    mu = np.average(values, weights=weights)
    return np.sqrt(np.average((values - mu) ** 2, weights=weights))


def weighted_skewness(values, weights):
    values, weights = _valid_xy(values, weights)
    if values.size < 3:
        return np.nan
    weights = weights / np.sum(weights)
    mu = np.average(values, weights=weights)
    sigma = np.sqrt(np.average((values - mu) ** 2, weights=weights))
    if not np.isfinite(sigma) or sigma <= 0:
        return np.nan
    m3 = np.average((values - mu) ** 3, weights=weights)
    return m3 / sigma ** 3


def weighted_excess_kurtosis(values, weights):
    values, weights = _valid_xy(values, weights)
    if values.size < 4:
        return np.nan
    weights = weights / np.sum(weights)
    mu = np.average(values, weights=weights)
    sigma = np.sqrt(np.average((values - mu) ** 2, weights=weights))
    if not np.isfinite(sigma) or sigma <= 0:
        return np.nan
    m4 = np.average((values - mu) ** 4, weights=weights)
    return m4 / sigma ** 4 - 3.0


def weighted_quantile_skewness(values, weights):
    """Bowley quantile skewness, robust to extreme tails."""
    q25, q50, q75 = weighted_quantile(values, [0.25, 0.50, 0.75], weights)
    denom = q75 - q25
    if not np.isfinite(denom) or denom <= 0:
        return np.nan
    return (q75 + q25 - 2.0 * q50) / denom


def weighted_effective_n(weights):
    weights = np.asarray(weights, dtype=float)
    weights = weights[np.isfinite(weights) & (weights > 0)]
    if weights.size == 0:
        return np.nan
    return (np.sum(weights) ** 2) / np.sum(weights ** 2)


def weighted_quantile(values, quantiles, sample_weight=None):
    values = np.asarray(values, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    sample_weight = np.ones(values.size, dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    values, sample_weight = _valid_xy(values, sample_weight)
    if values.size == 0:
        return np.full_like(quantiles, np.nan, dtype=float)
    order = np.argsort(values)
    values, sample_weight = values[order], sample_weight[order]
    weighted_q = np.cumsum(sample_weight) - 0.5 * sample_weight
    weighted_q /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_q, values)


def weighted_metrics(y, yhat, w, n_params):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    w = np.asarray(w, dtype=float)
    ok = np.isfinite(y) & np.isfinite(yhat) & np.isfinite(w) & (w > 0)
    y, yhat, w = y[ok], yhat[ok], w[ok]
    if y.size == 0:
        return {}
    wsum = np.sum(w)
    resid = yhat - y
    wmse = np.sum(w * resid**2) / wsum
    wrmse = np.sqrt(wmse)
    wmae = np.sum(w * np.abs(resid)) / wsum
    bias = np.sum(w * resid) / wsum
    ybar = np.sum(w * y) / wsum
    tss = np.sum(w * (y - ybar)**2)
    rss = np.sum(w * resid**2)
    r2 = 1.0 - rss / tss if tss > 0 else np.nan
    n_eff = weighted_effective_n(w)
    sigma2 = max(wmse, 1e-12)
    nll = 0.5 * n_eff * (np.log(2 * np.pi * sigma2) + 1.0)
    return {
        "n": int(y.size), "n_eff": n_eff, "weight_sum": wsum,
        "wrmse": wrmse, "wmae": wmae, "wbias_pred_minus_obs": bias,
        "weighted_r2": r2, "pseudo_aic": 2*n_params + 2*nll,
        "pseudo_bic": np.log(max(n_eff, 1.0))*n_params + 2*nll,
    }


def make_quantile_edges(values, weights, n_bins):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[ok], weights[ok]
    if values.size < 2:
        return None
    edges = weighted_quantile(values, np.linspace(0, 1, n_bins+1), weights)
    edges = np.unique(edges[np.isfinite(edges)])
    if edges.size < 2:
        return None
    eps = np.finfo(float).eps
    span = max(abs(edges[-1] - edges[0]), 1.0)
    edges[0] -= eps * span * 10
    edges[-1] += eps * span * 10
    return edges


def robust_symmetric_ylim(values, q=0.995, min_halfspan=50.0, pad=0.1):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (-min_halfspan, min_halfspan)
    half = np.nanquantile(np.abs(vals), q)
    if not np.isfinite(half):
        half = np.nanmax(np.abs(vals))
    half = max(float(half), min_halfspan) * (1+pad)
    return (-half, half)

# =============================================================================
# Binning helpers
# =============================================================================



def _sigma_params_for_form(form):
    return list(SIGMA_PARAMS_BY_FORM[form])


def _unpack_sigma(theta_sig, param_names):
    return {n: float(v) for n, v in zip(param_names, np.asarray(theta_sig, dtype=float))}


def sigma_model_form(theta_sig, absx, period_years=None, form="rational_floor", param_names=None):
    """
    Positive residual spread model.

    Some candidate forms intentionally diverge as |dh| approaches zero. This is
    consistent with the ratio nature of rho_dV; the singularity is handled later
    when propagating over the elevation change uncertainty distribution.
    """
    absx = np.asarray(absx, dtype=float)
    if param_names is None:
        param_names = _sigma_params_for_form(form)
    p = _unpack_sigma(theta_sig, param_names)

    if form == "rational_floor":
        out = p["s_floor"] + p["s_amp"] / (1.0 + (absx / p["x_sig"]) ** p["q_sig"])

    elif form == "inv_power":
        out = p["A"] / ((absx + p["x0"]) ** p["q"])

    elif form == "inv_power_period_amp":
        if period_years is None:
            T = np.zeros_like(absx, dtype=float)
        else:
            T = np.asarray(period_years, dtype=float)
        amp = p["A0"] + p["A1"] * (1.0 - np.exp(-T / p["T_sig"]))
        out = amp / ((absx + p["x0"]) ** p["q"])

    elif form == "inv_power_period_damped":
        if period_years is None:
            T = np.zeros_like(absx, dtype=float)
        else:
            T = np.asarray(period_years, dtype=float)
        base = p["A0"] / ((absx + p["x0"]) ** p["q0"])
        per = p["A1"] * (1.0 - np.exp(-T / p["T_sig"]))
        per = per / ((absx + p["x0"]) ** p["q1"])
        per = per / (1.0 + (absx / p["x_sig"]) ** p["q_sig"])
        out = base + per

    elif form == "inv_power_period_damped_sqrt_tail":
        if period_years is None:
            T = np.zeros_like(absx, dtype=float)
        else:
            T = np.asarray(period_years, dtype=float)

        # Same structure as inv_power_period_damped, but with the common
        # Baseline bent so the large-|dh| tail converges to alpha/sqrt(|dh|)
        eta = SIGMA_TWOSLOPE_ETA_FIXED
        qinf = SIGMA_SQRT_TAIL_EXPONENT_FIXED
        base = p["A0"] / ((absx + p["x0"]) ** p["q0"])
        bend = (1.0 + (absx / p["xb_sig"]) ** eta) ** (-(qinf - p["q0"]) / eta)
        base = base * bend

        per = p["A1"] * (1.0 - np.exp(-T / p["T_sig"]))
        per = per / ((absx + p["x0"]) ** p["q1"])
        per = per / (1.0 + (absx / p["x_sig"]) ** p["q_sig"])
        out = base + per

    elif form == "inv_power_short_reduction":
        if period_years is None:
            T = np.zeros_like(absx, dtype=float)
        else:
            T = np.asarray(period_years, dtype=float)

        # Long-period/common baseline, with a short-period negative departure
        # Localized to small |dh|. This is the sign-inverted analogue of the
        # Period-damped form and preserves convergence across periods for large |dh|
        base = p["A"] / ((absx + p["x0"]) ** p["q"])
        local = 1.0 / (1.0 + (absx / p["x_sig"]) ** p["q_sig"])
        reduction = p["lam"] * np.exp(-T / p["T_sig"]) * local
        out = base * (1.0 - reduction)

    elif form == "inv_power_centered_period_anomaly":
        if period_years is None:
            T = np.zeros_like(absx, dtype=float)
        else:
            T = np.asarray(period_years, dtype=float)

        # Intermediate/common baseline, with a centered period anomaly localized
        # To small |dh|. Short periods can fall below the baseline and long
        # Periods above it, while all periods converge to the same large-|dh|
        # Slope as local -> 0
        base = p["A"] / ((absx + p["x0"]) ** p["q"])
        local = 1.0 / (1.0 + (absx / p["x_sig"]) ** p["q_sig"])
        period_anom = 1.0 - np.exp(-T / p["T_sig"]) - p["F0"]
        out = base * (1.0 + p["lam"] * period_anom * local)

    elif form == "inv_power_period_damped_twoslope":
        if period_years is None:
            T = np.zeros_like(absx, dtype=float)
        else:
            T = np.asarray(period_years, dtype=float)

        # Same successful structure as inv_power_period_damped, but the common
        # Inverse-power baseline has a separate large-|dh| slope qinf
        eta = SIGMA_TWOSLOPE_ETA_FIXED
        base = p["A0"] / ((absx + p["x0"]) ** p["q0"])
        bend = (1.0 + (absx / p["xb_sig"]) ** eta) ** (-(p["qinf"] - p["q0"]) / eta)
        base = base * bend

        per = p["A1"] * (1.0 - np.exp(-T / p["T_sig"]))
        per = per / ((absx + p["x0"]) ** p["q1"])
        per = per / (1.0 + (absx / p["x_sig"]) ** p["q_sig"])
        out = base + per

    elif form == "inv_power_short_reduction_twoslope":
        if period_years is None:
            T = np.zeros_like(absx, dtype=float)
        else:
            T = np.asarray(period_years, dtype=float)

        # Same successful structure as inv_power_short_reduction, but the
        # Common inverse-power baseline has a separate large-|dh| slope qinf
        eta = SIGMA_TWOSLOPE_ETA_FIXED
        base = p["A"] / ((absx + p["x0"]) ** p["q"])
        bend = (1.0 + (absx / p["xb_sig"]) ** eta) ** (-(p["qinf"] - p["q"]) / eta)
        base = base * bend

        local = 1.0 / (1.0 + (absx / p["x_sig"]) ** p["q_sig"])
        reduction = p["lam"] * np.exp(-T / p["T_sig"]) * local
        out = base * (1.0 - reduction)

    elif form == "inv_power_period_damped_sqrt_tail_fixed_x0":
        if period_years is None:
            T = np.zeros_like(absx, dtype=float)
        else:
            T = np.asarray(period_years, dtype=float)

        # Smooth additive period-damped model with sqrt tail and fixed x0
        # This preserves the successful 9-parameter shape but removes one weakly
        # Identified near-zero offset parameter
        eta = SIGMA_TWOSLOPE_ETA_FIXED
        qinf = SIGMA_SQRT_TAIL_EXPONENT_FIXED
        x0 = SIGMA_FIXED_X0_FOR_SQRT_BASE

        base = p["A0"] / ((absx + x0) ** p["q0"])
        bend = (1.0 + (absx / p["xb_sig"]) ** eta) ** (-(qinf - p["q0"]) / eta)
        base = base * bend

        per = p["A1"] * (1.0 - np.exp(-T / p["T_sig"]))
        per = per / ((absx + x0) ** p["q1"])
        per = per / (1.0 + (absx / p["x_sig"]) ** p["q_sig"])
        out = base + per

    elif form == "inv_power_period_damped_sqrt_tail_shared_q":
        if period_years is None:
            T = np.zeros_like(absx, dtype=float)
        else:
            T = np.asarray(period_years, dtype=float)

        # Smooth additive period-damped model with q1 tied to q0
        # This removes the extra period-branch inverse-power exponent while
        # Preserving the additive structure that behaved smoothly
        eta = SIGMA_TWOSLOPE_ETA_FIXED
        qinf = SIGMA_SQRT_TAIL_EXPONENT_FIXED

        base = p["A0"] / ((absx + p["x0"]) ** p["q0"])
        bend = (1.0 + (absx / p["xb_sig"]) ** eta) ** (-(qinf - p["q0"]) / eta)
        base = base * bend

        per = p["A1"] * (1.0 - np.exp(-T / p["T_sig"]))
        per = per / ((absx + p["x0"]) ** p["q0"])
        per = per / (1.0 + (absx / p["x_sig"]) ** p["q_sig"])
        out = base + per

    elif form == "inv_power_period_damped_sqrt_tail_shared_q_fixed_x0":
        if period_years is None:
            T = np.zeros_like(absx, dtype=float)
        else:
            T = np.asarray(period_years, dtype=float)

        # Seven-parameter smooth additive version: fixed x0 and q1 tied to q0
        # This is the most parsimonious nested form that keeps the successful
        # Smooth additive period-damped architecture
        eta = SIGMA_TWOSLOPE_ETA_FIXED
        qinf = SIGMA_SQRT_TAIL_EXPONENT_FIXED
        x0 = SIGMA_FIXED_X0_FOR_SQRT_BASE

        base = p["A0"] / ((absx + x0) ** p["q0"])
        bend = (1.0 + (absx / p["xb_sig"]) ** eta) ** (-(qinf - p["q0"]) / eta)
        base = base * bend

        per = p["A1"] * (1.0 - np.exp(-T / p["T_sig"]))
        per = per / ((absx + x0) ** p["q0"])
        per = per / (1.0 + (absx / p["x_sig"]) ** p["q_sig"])
        out = base + per

    elif form == "inv_power_period_damped_sqrt_tail_flatbase_fixed_x0":
        if period_years is None:
            T = np.zeros_like(absx, dtype=float)
        else:
            T = np.asarray(period_years, dtype=float)

        # Six-parameter nested smooth additive version. q0 is fixed to 0,
        # X0 is fixed, and the baseline is flat at small |dh| but tends to
        # Alpha/sqrt(|dh|) at large |dh|
        eta = SIGMA_TWOSLOPE_ETA_FIXED
        qinf = SIGMA_SQRT_TAIL_EXPONENT_FIXED
        q0 = SIGMA_Q0_FLAT_BASE_FIXED
        x0 = SIGMA_FIXED_X0_FOR_SQRT_BASE

        base = p["A0"] / ((absx + x0) ** q0)
        bend = (1.0 + (absx / p["xb_sig"]) ** eta) ** (-(qinf - q0) / eta)
        base = base * bend

        per = p["A1"] * (1.0 - np.exp(-T / p["T_sig"]))
        per = per / ((absx + x0) ** q0)
        per = per / (1.0 + (absx / p["x_sig"]) ** p["q_sig"])
        out = base + per

    elif form == "inv_power_period_damped_sqrt_tail_flatbase_psig05_fixed_x0":
        if period_years is None:
            T = np.zeros_like(absx, dtype=float)
        else:
            T = np.asarray(period_years, dtype=float)

        # Five-parameter nested version: q0=0, x0 fixed, and q_sig=1/2
        eta = SIGMA_TWOSLOPE_ETA_FIXED
        qinf = SIGMA_SQRT_TAIL_EXPONENT_FIXED
        q0 = SIGMA_Q0_FLAT_BASE_FIXED
        q_sig = SIGMA_PERIOD_DECAY_SQRT_FIXED
        x0 = SIGMA_FIXED_X0_FOR_SQRT_BASE

        base = p["A0"] / ((absx + x0) ** q0)
        bend = (1.0 + (absx / p["xb_sig"]) ** eta) ** (-(qinf - q0) / eta)
        base = base * bend

        per = p["A1"] * (1.0 - np.exp(-T / p["T_sig"]))
        per = per / ((absx + x0) ** q0)
        per = per / (1.0 + (absx / p["x_sig"]) ** q_sig)
        out = base + per

    elif form == "inv_power_period_damped_sqrt_tail_flatbase_psig1_fixed_x0":
        if period_years is None:
            T = np.zeros_like(absx, dtype=float)
        else:
            T = np.asarray(period_years, dtype=float)

        # Five-parameter nested version: q0=0, x0 fixed, and q_sig=1
        eta = SIGMA_TWOSLOPE_ETA_FIXED
        qinf = SIGMA_SQRT_TAIL_EXPONENT_FIXED
        q0 = SIGMA_Q0_FLAT_BASE_FIXED
        q_sig = SIGMA_PERIOD_DECAY_LINEAR_FIXED
        x0 = SIGMA_FIXED_X0_FOR_SQRT_BASE

        base = p["A0"] / ((absx + x0) ** q0)
        bend = (1.0 + (absx / p["xb_sig"]) ** eta) ** (-(qinf - q0) / eta)
        base = base * bend

        per = p["A1"] * (1.0 - np.exp(-T / p["T_sig"]))
        per = per / ((absx + x0) ** q0)
        per = per / (1.0 + (absx / p["x_sig"]) ** q_sig)
        out = base + per

    elif form == "sqrt_base_period_addition":
        if period_years is None:
            T = np.zeros_like(absx, dtype=float)
        else:
            T = np.asarray(period_years, dtype=float)

        # Pure square-root baseline. Singular at |dh|=0 by construction:
        # A0/sqrt(|dh|). Periods add a positive uncertainty anomaly localized
        # To small |dh|
        base = p["A0"] / np.sqrt(absx)
        per = p["A1"] * (1.0 - np.exp(-T / p["T_sig"]))
        per = per / (1.0 + (absx / p["x_sig"]) ** p["q_sig"])
        out = base + per

    elif form == "sqrt_base_period_addition_psig05":
        if period_years is None:
            T = np.zeros_like(absx, dtype=float)
        else:
            T = np.asarray(period_years, dtype=float)

        # Pure square-root baseline with q_sig fixed to 1/2
        q_sig = SIGMA_PERIOD_DECAY_SQRT_FIXED
        base = p["A0"] / np.sqrt(absx)
        per = p["A1"] * (1.0 - np.exp(-T / p["T_sig"]))
        per = per / (1.0 + (absx / p["x_sig"]) ** q_sig)
        out = base + per

    elif form == "sqrt_base_period_addition_psig1":
        if period_years is None:
            T = np.zeros_like(absx, dtype=float)
        else:
            T = np.asarray(period_years, dtype=float)

        # Pure square-root baseline with q_sig fixed to 1
        q_sig = SIGMA_PERIOD_DECAY_LINEAR_FIXED
        base = p["A0"] / np.sqrt(absx)
        per = p["A1"] * (1.0 - np.exp(-T / p["T_sig"]))
        per = per / (1.0 + (absx / p["x_sig"]) ** q_sig)
        out = base + per

    elif form == "variance_additive_sqrt_time":
        if period_years is None:
            T = np.zeros_like(absx, dtype=float)
        else:
            T = np.asarray(period_years, dtype=float)

        # Variance-additive model that respects independent sub-period
        # Uncertainty aggregation for persistent same-sign elevation change:
        # Sigma^2 = A0^2/|dh| + A1^2 dt / |dh|^2
        out = np.sqrt(
            (p["A0"] ** 2) / absx
            + (p["A1"] ** 2) * np.maximum(T, 0.0) / (absx ** 2)
        )

    elif form == "variance_additive_sqrt_time_localized":
        if period_years is None:
            T = np.zeros_like(absx, dtype=float)
        else:
            T = np.asarray(period_years, dtype=float)

        # Localized variant: the time-accumulating variance term is damped at
        # Large |dh|. This relaxes strict aggregation closure for the second
        # Component but tests whether the data require localization
        local = 1.0 / (1.0 + (absx / p["x_sig"]) ** p["q_sig"])
        out = np.sqrt(
            (p["A0"] ** 2) / absx
            + (p["A1"] ** 2) * np.maximum(T, 0.0) * local / (absx ** 2)
        )

    elif form == "sqrt_tail_multiplicative_shared_bend":
        if period_years is None:
            T = np.zeros_like(absx, dtype=float)
        else:
            T = np.asarray(period_years, dtype=float)

        # Multiplicative version with one shared transition scale. The baseline
        # Bends toward alpha/sqrt(|dh|), and the positive period anomaly is
        # Localized by the same bend. This avoids two independent slope changes
        eta = SIGMA_TWOSLOPE_ETA_FIXED
        qinf = SIGMA_SQRT_TAIL_EXPONENT_FIXED

        bend_raw = 1.0 + (absx / p["xb_sig"]) ** eta
        base = p["A"] / ((absx + p["x0"]) ** p["q0"])
        base = base * bend_raw ** (-(qinf - p["q0"]) / eta)

        local = 1.0 / bend_raw
        age = np.maximum(T, 0.0)
        addition = p["lam_add"] * (1.0 - np.exp(-age / p["T_sig"])) * local
        out = base * (1.0 + addition)

    elif form == "sqrt_tail_multiplicative_shared_bend_fixed_x0":
        if period_years is None:
            T = np.zeros_like(absx, dtype=float)
        else:
            T = np.asarray(period_years, dtype=float)

        # Five-parameter version with fixed x0 and one shared transition scale
        eta = SIGMA_TWOSLOPE_ETA_FIXED
        qinf = SIGMA_SQRT_TAIL_EXPONENT_FIXED
        x0 = SIGMA_FIXED_X0_FOR_SQRT_BASE

        bend_raw = 1.0 + (absx / p["xb_sig"]) ** eta
        base = p["A"] / ((absx + x0) ** p["q0"])
        base = base * bend_raw ** (-(qinf - p["q0"]) / eta)

        local = 1.0 / bend_raw
        age = np.maximum(T, 0.0)
        addition = p["lam_add"] * (1.0 - np.exp(-age / p["T_sig"])) * local
        out = base * (1.0 + addition)

    elif form == "sqrt_zero_base_period_addition":
        if period_years is None:
            T = np.zeros_like(absx, dtype=float)
        else:
            T = np.asarray(period_years, dtype=float)

        # Zero-period baseline with fixed inverse-square-root dependence
        # The 1-year period already carries a positive offset, and longer
        # Periods add progressively more uncertainty. The anomaly is localized
        # Mostly to small |dh| and tends to lam_add * local for long periods
        base = p["A"] / np.sqrt(absx + p["x0"])
        local = 1.0 / (1.0 + (absx / p["x_sig"]) ** p["q_sig"])
        age = np.maximum(T, 0.0)
        addition = p["lam_add"] * (1.0 - np.exp(-age / p["T_sig"])) * local
        out = base * (1.0 + addition)

    elif form == "sqrt_zero_base_period_addition_fixed_x0":
        if period_years is None:
            T = np.zeros_like(absx, dtype=float)
        else:
            T = np.asarray(period_years, dtype=float)

        # Five-parameter variant of sqrt_zero_base_period_addition with fixed
        # X0, so the zero-period baseline is exactly alpha/sqrt(|dh|+fixed_x0)
        x0 = SIGMA_FIXED_X0_FOR_SQRT_BASE
        base = p["A"] / np.sqrt(absx + x0)
        local = 1.0 / (1.0 + (absx / p["x_sig"]) ** p["q_sig"])
        age = np.maximum(T, 0.0)
        addition = p["lam_add"] * (1.0 - np.exp(-age / p["T_sig"])) * local
        out = base * (1.0 + addition)

    elif form == "sqrt_tail_fractional_period_addition":
        if period_years is None:
            T = np.zeros_like(absx, dtype=float)
        else:
            T = np.asarray(period_years, dtype=float)

        # Parsimonious fractional version of inv_power_period_damped_sqrt_tail
        # The common baseline is flexible at small/intermediate |dh| but tends
        # To alpha/sqrt(|dh|) at large |dh|. Period effects are a fractional
        # Positive anomaly localized to small |dh|
        eta = SIGMA_TWOSLOPE_ETA_FIXED
        qinf = SIGMA_SQRT_TAIL_EXPONENT_FIXED
        base = p["A"] / ((absx + p["x0"]) ** p["q0"])
        bend = (1.0 + (absx / p["xb_sig"]) ** eta) ** (-(qinf - p["q0"]) / eta)
        base = base * bend
        local = 1.0 / (1.0 + (absx / p["x_sig"]) ** p["q_sig"])
        age = np.maximum(T, 0.0)
        addition = p["lam_add"] * (1.0 - np.exp(-age / p["T_sig"])) * local
        out = base * (1.0 + addition)

    elif form == "sqrt_tail_fractional_period_addition_fixed_x0":
        if period_years is None:
            T = np.zeros_like(absx, dtype=float)
        else:
            T = np.asarray(period_years, dtype=float)

        # Seven-parameter variant with fixed x0. This keeps the bent baseline
        # And sqrt tail but avoids fitting a small-offset parameter
        eta = SIGMA_TWOSLOPE_ETA_FIXED
        qinf = SIGMA_SQRT_TAIL_EXPONENT_FIXED
        x0 = SIGMA_FIXED_X0_FOR_SQRT_BASE
        base = p["A"] / ((absx + x0) ** p["q0"])
        bend = (1.0 + (absx / p["xb_sig"]) ** eta) ** (-(qinf - p["q0"]) / eta)
        base = base * bend
        local = 1.0 / (1.0 + (absx / p["x_sig"]) ** p["q_sig"])
        age = np.maximum(T, 0.0)
        addition = p["lam_add"] * (1.0 - np.exp(-age / p["T_sig"])) * local
        out = base * (1.0 + addition)

    elif form == "loglinear_centered_period_anomaly":
        if period_years is None:
            T = np.zeros_like(absx, dtype=float)
        else:
            T = np.asarray(period_years, dtype=float)

        # Common branch is a straight line against log(|dh|). Period effects are
        # Centered around F0 and localized to small |dh|
        xlog = np.log(absx + SIGMA_LOGLINEAR_X0)
        base = p["B0"] - p["B1"] * xlog
        local = 1.0 / (1.0 + (absx / p["x_sig"]) ** p["q_sig"])
        period_anom = 1.0 - np.exp(-T / p["T_sig"]) - p["F0"]
        out = base + p["Aper"] * period_anom * local

    elif form == "loglinear_damped_centered_period_anomaly":
        if period_years is None:
            T = np.zeros_like(absx, dtype=float)
        else:
            T = np.asarray(period_years, dtype=float)

        # Same log-linear middle branch, but weakly damped at very large |dh| to
        # Preserve asymptotic decline while retaining the near-linear log-x slope
        xlog = np.log(absx + SIGMA_LOGLINEAR_X0)
        base = p["B0"] - p["B1"] * xlog
        damping = 1.0 / (1.0 + (absx / p["xD_sig"]) ** p["qD_sig"])
        local = 1.0 / (1.0 + (absx / p["x_sig"]) ** p["q_sig"])
        period_anom = 1.0 - np.exp(-T / p["T_sig"]) - p["F0"]
        out = base * damping + p["Aper"] * period_anom * local

    elif form == "logsaturating_centered_period_anomaly":
        if period_years is None:
            T = np.zeros_like(absx, dtype=float)
        else:
            T = np.asarray(period_years, dtype=float)

        # Smooth alternative to a straight log-linear branch: similar curvature
        # Over the calibration range, but remains positive and decays more gently
        base = p["B0"] / (1.0 + p["B1"] * np.log1p(absx / p["xL_sig"]))
        local = 1.0 / (1.0 + (absx / p["x_sig"]) ** p["q_sig"])
        period_anom = 1.0 - np.exp(-T / p["T_sig"]) - p["F0"]
        out = base + p["Aper"] * period_anom * local

    else:
        raise ValueError(f"Unknown sigma model form: {form}")

    out = np.asarray(out, dtype=float)
    out[~np.isfinite(out)] = np.nan
    return np.maximum(out, SIGMA_NUMERIC_FLOOR)



def _resolve_sigma_form_and_params(theta_sig, preferred_form=None):
    """
    Resolve a sigma form and parameter list from the current theta length.

    This prevents stale interactive globals from mixing a selected form with a
    theta vector from another form. If preferred_form is inconsistent with theta
    length, choose the successfully fitted candidate with matching length and
    best model-selection metric.
    """
    theta_len = len(np.asarray(theta_sig, dtype=float))

    if preferred_form is None:
        preferred_form = globals().get("SIGMA_SELECTED_FORM", SIGMA_MODEL_FOR_DIAGNOSTICS)
    if preferred_form == "best":
        preferred_form = globals().get("SIGMA_SELECTED_FORM", "rational_floor")
        if preferred_form == "best":
            preferred_form = "rational_floor"

    if preferred_form in SIGMA_PARAMS_BY_FORM and len(SIGMA_PARAMS_BY_FORM[preferred_form]) == theta_len:
        return preferred_form, SIGMA_PARAMS_BY_FORM[preferred_form]

    sel = globals().get("SIGMA_MODEL_SELECTION_TABLE", None)
    if isinstance(sel, pd.DataFrame) and len(sel) and "sigma_form" in sel.columns:
        rows = sel.copy()
        if "wrmse" in rows.columns:
            rows = rows.loc[np.isfinite(rows["wrmse"])].copy()
        candidates = []
        for _, row in rows.iterrows():
            form = str(row["sigma_form"])
            if form in SIGMA_PARAMS_BY_FORM and len(SIGMA_PARAMS_BY_FORM[form]) == theta_len:
                candidates.append(row)
        if candidates:
            cand = pd.DataFrame(candidates)
            metric = "pseudo_bic" if "pseudo_bic" in cand.columns and np.isfinite(cand["pseudo_bic"]).any() else "wrmse"
            form = str(cand.sort_values(metric).iloc[0]["sigma_form"])
            return form, SIGMA_PARAMS_BY_FORM[form]

    matches = [form for form, names in SIGMA_PARAMS_BY_FORM.items() if len(names) == theta_len]
    if len(matches) == 1:
        form = matches[0]
        return form, SIGMA_PARAMS_BY_FORM[form]

    raise RuntimeError(
        f"Could not resolve sigma model form for theta length {theta_len}. "
        f"Preferred form was {preferred_form!r}; matching forms are {matches}."
    )


def sigma_model(theta_sig, absx, period_years=None):
    """
    Selected positive residual spread model.

    The form is resolved from theta length if necessary, so downstream plotting
    cannot mix a stale SIGMA_SELECTED_FORM with a theta vector from another form.
    """
    form, param_names = _resolve_sigma_form_and_params(theta_sig)
    return sigma_model_form(theta_sig, absx, period_years=period_years, form=form, param_names=param_names)


def sigma_model_safe(theta_sig, absx, period_years=None):
    """Selected sigma model with numerical floor for divisions."""
    return np.maximum(sigma_model(theta_sig, absx, period_years=period_years), SIGMA_NUMERIC_FLOOR)



def summarize_absdh_residual_std_for_sigma(df, resid_col):
    """
    Build binned residual STD table used to fit sigma_model.

    Binning is by display period group and |current dh|.
    """
    period_col = "period_plot_years" if "period_plot_years" in df.columns else "period_years"
    d = df.loc[
        np.isfinite(df[resid_col])
        & np.isfinite(df["abs_signed_dh"])
        & np.isfinite(df[period_col])
        & np.isfinite(df[weight_col])
        & (df[weight_col] > 0)
    ].copy()
    if d.empty:
        return pd.DataFrame()

    abs_edges = make_abs_dh_edges(d["abs_signed_dh"].to_numpy(float))
    d["abs_bin"] = pd.cut(d["abs_signed_dh"], bins=abs_edges, labels=False, include_lowest=True)
    d = d.dropna(subset=["abs_bin"]).copy()
    d["abs_bin"] = d["abs_bin"].astype(int)

    rows = []
    for (T, ib), g in d.groupby([period_col, "abs_bin"], sort=True):
        w = g[weight_col].to_numpy(float)
        n_eff = weighted_effective_n(w)
        if len(g) < MIN_COUNT_SIGMA_BIN or not np.isfinite(n_eff) or n_eff < MIN_NEFF_SIGMA_BIN:
            continue
        rows.append({
            "period_years": float(T),
            "abs_bin": int(ib),
            "abs_center_w": weighted_mean(g["abs_signed_dh"].to_numpy(float), w),
            "n": len(g),
            "n_eff": n_eff,
            "weight_sum": np.nansum(w),
            "resid_std_w": weighted_std(g[resid_col].to_numpy(float), w),
            "resid_mean_w": weighted_mean(g[resid_col].to_numpy(float), w),
        })
    return pd.DataFrame(rows)

def _fit_one_sigma_form(tab, form):
    param_names = _sigma_params_for_form(form)
    x = tab["abs_center_w"].to_numpy(float)
    T = tab["period_years"].to_numpy(float)
    y = tab["resid_std_w"].to_numpy(float)
    w = tab["weight_sum"].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(T) & np.isfinite(y) & np.isfinite(w) & (w > 0) & (y > 0)
    x, T, y, w = x[ok], T[ok], y[ok], w[ok]
    if len(y) < len(param_names) + 1:
        raise ValueError(f"Too few sigma bins for {form}")

    lo = np.asarray([SIGMA_BOUNDS[p][0] for p in param_names], dtype=float)
    hi = np.asarray([SIGMA_BOUNDS[p][1] for p in param_names], dtype=float)
    theta0 = np.asarray([SIGMA_INITIAL[p] for p in param_names], dtype=float)
    sqrtw = np.sqrt(w / np.nanmean(w))

    def residual(theta):
        pred = sigma_model_form(theta, x, period_years=T, form=form, param_names=param_names)
        rr = (pred - y) * sqrtw
        rr[~np.isfinite(rr)] = 1e12
        return rr

    res = least_squares(
        residual,
        theta0,
        bounds=(lo, hi),
        loss="soft_l1",
        f_scale=50.0,
        max_nfev=20000,
        x_scale="jac",
    )
    pred = sigma_model_form(res.x, x, period_years=T, form=form, param_names=param_names)
    metrics = weighted_metrics(y, pred, w, len(res.x))
    row = {
        "sigma_form": form,
        "n_params": len(param_names),
        "cost": float(res.cost),
        "success": bool(res.success),
        **metrics,
    }
    for name, val in zip(param_names, res.x):
        row[f"param_{name}"] = float(val)
    return row, res.x, param_names




def attach_path_length_from_subperiods(current_rows, all_rows, out_col="path_abs_dh"):
    """
    Attach cumulative absolute elevation change path length L_A for each row.

    L_A is computed from nested source-period rows, by default annual rows:
        L_A = sum_i |dh_i|
    over the current row's [start_date, end_date] interval.

    If full annual coverage is unavailable and PATH_LENGTH_REQUIRE_FULL_COVERAGE
    is True, L_A is set to NaN. This keeps the diagnostic conservative.
    """
    out = current_rows.copy()
    if out_col in out.columns:
        return out

    src_period = float(PATH_LENGTH_SOURCE_PERIOD_YEARS)
    finite_periods = out["period_years"].to_numpy(float)
    finite_periods = finite_periods[np.isfinite(finite_periods)]
    if finite_periods.size == 0:
        out[out_col] = np.nan
        return out

    max_period = int(np.ceil(np.nanmax(finite_periods) / src_period))
    max_period = max(max_period, 1)

    out["_path_row_id"] = np.arange(len(out))
    base = out[["rgiid", variant_col, "start_date", "end_date", "period_years", "_path_row_id"]].copy()

    sub = all_rows.loc[
        np.isclose(all_rows["period_years"].to_numpy(float), src_period),
        ["rgiid", variant_col, "start_date", "end_date", "signed_dh"],
    ].copy()
    sub = sub.drop_duplicates(subset=["rgiid", variant_col, "start_date", "end_date"])

    accum = np.zeros(len(out), dtype=float)
    valid = np.zeros(len(out), dtype=int)

    expected_float = out["period_years"].to_numpy(float) / src_period
    expected = np.rint(expected_float).astype(int)
    expected = np.where(np.isfinite(expected_float) & (expected > 0), expected, 0)

    period_ok = np.isclose(
        out["period_years"].to_numpy(float),
        expected.astype(float) * src_period,
        atol=1e-6,
        rtol=0.0,
    )

    for k in range(max_period):
        need = base.copy()
        need["_sub_start"] = need["start_date"] + k * src_period
        need["_sub_end"] = need["_sub_start"] + src_period
        active = need["_sub_end"].to_numpy(float) <= need["end_date"].to_numpy(float) + 1e-9
        if not np.any(active):
            continue
        need = need.loc[active].copy()

        lookup = sub.rename(columns={
            "start_date": "_sub_start",
            "end_date": "_sub_end",
            "signed_dh": f"_sub_dh_{k}",
        })
        m = need.merge(
            lookup[["rgiid", variant_col, "_sub_start", "_sub_end", f"_sub_dh_{k}"]],
            on=["rgiid", variant_col, "_sub_start", "_sub_end"],
            how="left",
        )
        vals = m[f"_sub_dh_{k}"].to_numpy(float)
        ids = m["_path_row_id"].to_numpy(int)
        ok = np.isfinite(vals)
        if np.any(ok):
            accum[ids[ok]] += np.abs(vals[ok])
            valid[ids[ok]] += 1

    if PATH_LENGTH_REQUIRE_FULL_COVERAGE:
        ok_final = period_ok & (valid == expected) & (expected > 0)
    else:
        ok_final = period_ok & (valid > 0)

    out[out_col] = np.nan
    out.loc[ok_final, out_col] = accum[ok_final]
    out[f"{out_col}_n_subperiods"] = valid
    out[f"{out_col}_expected_subperiods"] = expected
    out[f"{out_col}_full_coverage"] = ok_final
    out[f"{out_col}_ratio_to_net_abs_dh"] = np.nan
    absdh = out["abs_signed_dh"].to_numpy(float) if "abs_signed_dh" in out.columns else np.abs(out["signed_dh"].to_numpy(float))
    ok_ratio = ok_final & np.isfinite(absdh) & (absdh > 0)
    out.loc[ok_ratio, f"{out_col}_ratio_to_net_abs_dh"] = out.loc[ok_ratio, out_col].to_numpy(float) / absdh[ok_ratio]

    return out.drop(columns=["_path_row_id"])


def summarize_pathlength_residual_std_for_sigma(df, resid_col, path_col="path_abs_dh"):
    """
    Binned residual-STD table for testing the L_A-based uncertainty model.

    Binning is by display period, net |dh|, and a path-ratio quantile bin. This
    tests whether L_A improves calibration for sign-reversal/path-length cases,
    instead of only fitting marginal sigma(|dh|, dt).
    """
    ratio_col = f"{path_col}_ratio_to_net_abs_dh"
    period_col = "period_plot_years" if "period_plot_years" in df.columns else "period_years"
    required = [resid_col, "abs_signed_dh", path_col, ratio_col, period_col, weight_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"summarize_pathlength_residual_std_for_sigma missing columns: {missing}")

    d = df.loc[
        np.isfinite(df[resid_col])
        & np.isfinite(df["abs_signed_dh"])
        & (df["abs_signed_dh"] > 0)
        & np.isfinite(df[path_col])
        & (df[path_col] > 0)
        & np.isfinite(df[ratio_col])
        & np.isfinite(df[period_col])
        & np.isfinite(df[weight_col])
        & (df[weight_col] > 0)
    ].copy()
    if d.empty:
        return pd.DataFrame()

    abs_edges = make_abs_dh_edges(d["abs_signed_dh"].to_numpy(float))
    d["abs_bin"] = pd.cut(d["abs_signed_dh"], bins=abs_edges, labels=False, include_lowest=True)
    d = d.dropna(subset=["abs_bin"]).copy()
    d["abs_bin"] = d["abs_bin"].astype(int)

    ratio_edges = make_quantile_edges(
        d[ratio_col].to_numpy(float),
        d[weight_col].to_numpy(float),
        PATH_RATIO_N_BINS,
    )
    if ratio_edges is None or len(ratio_edges) < 2:
        d["path_ratio_bin"] = 0
    else:
        d["path_ratio_bin"] = pd.cut(d[ratio_col], bins=ratio_edges, labels=False, include_lowest=True)
        d = d.dropna(subset=["path_ratio_bin"]).copy()
        d["path_ratio_bin"] = d["path_ratio_bin"].astype(int)

    rows = []
    for (T, ib, rb), g in d.groupby([period_col, "abs_bin", "path_ratio_bin"], sort=True):
        w = g[weight_col].to_numpy(float)
        n_eff = weighted_effective_n(w)
        if len(g) < MIN_COUNT_SIGMA_BIN or not np.isfinite(n_eff) or n_eff < MIN_NEFF_SIGMA_BIN:
            continue

        rows.append({
            "period_years": float(T),
            "abs_bin": int(ib),
            "path_ratio_bin": int(rb),
            "abs_center_w": weighted_mean(g["abs_signed_dh"].to_numpy(float), w),
            "path_abs_center_w": weighted_mean(g[path_col].to_numpy(float), w),
            "path_ratio_center_w": weighted_mean(g[ratio_col].to_numpy(float), w),
            "n": len(g),
            "n_eff": n_eff,
            "weight_sum": np.nansum(w),
            "resid_std_w": weighted_std(g[resid_col].to_numpy(float), w),
            "resid_mean_w": weighted_mean(g[resid_col].to_numpy(float), w),
        })
    return pd.DataFrame(rows)


PATH_SIGMA_PARAMS_BY_FORM = {
    "net_variance_additive": ["A0", "A1"],
    "path_variance_additive": ["A0", "A1"],
    "path_variance_baseline_only": ["A0"],
}


def path_sigma_model_form(theta, absx, path_abs, period_years, form, param_names=None):
    absx = np.asarray(absx, dtype=float)
    path_abs = np.asarray(path_abs, dtype=float)
    T = np.asarray(period_years, dtype=float)
    p = {n: float(v) for n, v in zip(param_names or PATH_SIGMA_PARAMS_BY_FORM[form], np.asarray(theta, dtype=float))}

    if form == "net_variance_additive":
        return np.sqrt((p["A0"] ** 2) / absx + (p["A1"] ** 2) * np.maximum(T, 0.0) / (absx ** 2))

    if form == "path_variance_additive":
        return np.sqrt(((p["A0"] ** 2) * path_abs + (p["A1"] ** 2) * np.maximum(T, 0.0)) / (absx ** 2))

    if form == "path_variance_baseline_only":
        return np.sqrt(((p["A0"] ** 2) * path_abs) / (absx ** 2))

    raise ValueError(f"Unknown path sigma form: {form}")


def _fit_one_path_sigma_form(tab, form):
    param_names = PATH_SIGMA_PARAMS_BY_FORM[form]
    x = tab["abs_center_w"].to_numpy(float)
    L = tab["path_abs_center_w"].to_numpy(float)
    T = tab["period_years"].to_numpy(float)
    y = tab["resid_std_w"].to_numpy(float)
    w = tab["weight_sum"].to_numpy(float)
    ok = (
        np.isfinite(x) & (x > 0)
        & np.isfinite(L) & (L > 0)
        & np.isfinite(T)
        & np.isfinite(y) & (y > 0)
        & np.isfinite(w) & (w > 0)
        & (x <= PATH_SIGMA_MAX_ABS_DH_FOR_FIT)
    )
    x, L, T, y, w = x[ok], L[ok], T[ok], y[ok], w[ok]
    if len(y) < len(param_names) + 1:
        raise ValueError(f"Too few path-sigma bins for {form}")

    lo = np.asarray([SIGMA_BOUNDS[pn][0] for pn in param_names], dtype=float)
    hi = np.asarray([SIGMA_BOUNDS[pn][1] for pn in param_names], dtype=float)
    theta0 = np.asarray([SIGMA_INITIAL[pn] for pn in param_names], dtype=float)
    sqrtw = np.sqrt(w / np.nanmean(w))

    def residual(theta):
        pred = path_sigma_model_form(theta, x, L, T, form, param_names)
        rr = (pred - y) * sqrtw
        rr[~np.isfinite(rr)] = 1e12
        return rr

    res = least_squares(
        residual,
        theta0,
        bounds=(lo, hi),
        loss="soft_l1",
        f_scale=50.0,
        max_nfev=20000,
        x_scale="jac",
    )
    pred = path_sigma_model_form(res.x, x, L, T, form, param_names)
    metrics = weighted_metrics(y, pred, w, len(res.x))
    row = {
        "path_sigma_form": form,
        "n_params": len(param_names),
        "cost": float(res.cost),
        "success": bool(res.success),
        **metrics,
    }
    for name, val in zip(param_names, res.x):
        row[f"param_{name}"] = float(val)
    return row, res.x, param_names


def fit_pathlength_sigma_models(tab):
    rows = []
    fitted = {}
    for form in PATH_SIGMA_PARAMS_BY_FORM:
        try:
            row, theta, param_names = _fit_one_path_sigma_form(tab, form)
            rows.append(row)
            fitted[form] = (theta, param_names)
        except Exception as exc:
            rows.append({"path_sigma_form": form, "error": repr(exc), "wrmse": np.nan, "n_params": np.nan})
    sel = pd.DataFrame(rows)
    return sel, fitted




def _pathlength_marker_map(period_values):
    uniq = [float(v) for v in np.unique(np.asarray(period_values, dtype=float)) if np.isfinite(v)]
    uniq = sorted(uniq)
    marker_map = {}
    marker_i = 0
    for p in uniq:
        if p in PATH_LENGTH_PERIOD_MARKERS:
            marker_map[p] = PATH_LENGTH_PERIOD_MARKERS[p]
        else:
            marker_map[p] = PATH_LENGTH_DEFAULT_MARKERS[marker_i % len(PATH_LENGTH_DEFAULT_MARKERS)]
            marker_i += 1
    return marker_map


def _add_pathlength_period_legend(ax, marker_map):
    from matplotlib.lines import Line2D
    handles = []
    labels = []
    for p in sorted(marker_map):
        handles.append(
            Line2D(
                [0], [0],
                linestyle="none",
                marker=marker_map[p],
                markersize=7,
                markerfacecolor="0.6",
                markeredgecolor="0.25",
                alpha=0.9,
            )
        )
        if abs(p - round(p)) < 1e-9:
            labels.append(f"{int(round(p))} yr")
        else:
            labels.append(f"{p:g} yr")
    if handles:
        ax.legend(
            handles,
            labels,
            title="Period length",
            loc="best",
            frameon=False,
            fontsize=8,
            title_fontsize=8,
        )


def _pathlength_absbin_marker_map(abs_bin_values):
    uniq = [int(v) for v in np.unique(np.asarray(abs_bin_values, dtype=int))]
    marker_map = {}
    for i, b in enumerate(sorted(uniq)):
        marker_map[b] = PATH_LENGTH_DEFAULT_MARKERS[i % len(PATH_LENGTH_DEFAULT_MARKERS)]
    return marker_map


def _format_absbin_label(center_value):
    if not np.isfinite(center_value):
        return "NA"
    val = float(center_value)
    if val < 1:
        return f"{val:.2g} m"
    if val < 10:
        return f"{val:.2f} m".rstrip("0").rstrip(".")
    return f"{val:.0f} m"


def _add_pathlength_absbin_legend(ax, marker_map, center_map):
    from matplotlib.lines import Line2D
    handles = []
    labels = []
    for b in sorted(marker_map):
        handles.append(
            Line2D(
                [0], [0],
                linestyle="none",
                marker=marker_map[b],
                markersize=7,
                markerfacecolor="0.6",
                markeredgecolor="0.25",
                alpha=0.9,
            )
        )
        labels.append(_format_absbin_label(center_map.get(b, np.nan)))
    if handles:
        ax.legend(
            handles,
            labels,
            title=r"$|\Delta h_A|$ bin",
            loc="best",
            frameon=False,
            fontsize=8,
            title_fontsize=8,
        )


def _plot_one_pathlength_sigma_observed_predicted(tab, fitted, form, out_png, ratio_png, title_suffix=None):
    """
    Plot observed versus predicted binned residual STD for one path-sigma formula.

    Color represents path ratio L_A / |dh_A|. Marker shape represents the
    parent-period length used by each binned point.
    """
    if tab is None or tab.empty or form not in fitted:
        return

    theta, param_names = fitted[form]
    d = tab.loc[
        np.isfinite(tab["abs_center_w"]) & (tab["abs_center_w"] > 0)
        & np.isfinite(tab["path_abs_center_w"]) & (tab["path_abs_center_w"] > 0)
        & np.isfinite(tab["period_years"])
        & np.isfinite(tab["resid_std_w"]) & (tab["resid_std_w"] > 0)
        & np.isfinite(tab["weight_sum"]) & (tab["weight_sum"] > 0)
    ].copy()
    if d.empty:
        return

    d["pred_std"] = path_sigma_model_form(
        theta,
        d["abs_center_w"].to_numpy(float),
        d["path_abs_center_w"].to_numpy(float),
        d["period_years"].to_numpy(float),
        form,
        param_names,
    )
    d = d.loc[np.isfinite(d["pred_std"]) & (d["pred_std"] > 0)].copy()
    if d.empty:
        return

    label = title_suffix if title_suffix is not None else form
    marker_map = _pathlength_marker_map(d["period_years"].to_numpy(float))

    fig, ax = plt.subplots(figsize=(6.6, 5.8), constrained_layout=True)
    vmin = np.nanmin(d["path_ratio_center_w"].to_numpy(float))
    vmax = np.nanmax(d["path_ratio_center_w"].to_numpy(float))
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    for p, sub in d.groupby("period_years", sort=True):
        ax.scatter(
            sub["pred_std"],
            sub["resid_std_w"],
            c=sub["path_ratio_center_w"],
            cmap=cmap,
            norm=norm,
            marker=marker_map.get(float(p), "o"),
            s=46,
            alpha=0.82,
            edgecolors="0.25",
            linewidths=0.35,
        )

    lo = np.nanmin([d["pred_std"].min(), d["resid_std_w"].min()])
    hi = np.nanmax([d["pred_std"].max(), d["resid_std_w"].max()])
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        ax.plot([lo, hi], [lo, hi], color="black", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Predicted residual STD")
    ax.set_ylabel("Observed binned residual STD")
    ax.set_title(f"Path-length sigma diagnostic: {label}")
    ax.grid(alpha=0.25)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(r"$L_A / |\Delta h_A|$")

    _add_pathlength_period_legend(ax, marker_map)
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    d["std_ratio_obs_pred"] = d["resid_std_w"] / d["pred_std"]
    fig, ax = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
    vmin2 = np.nanmin(d["resid_std_w"].to_numpy(float))
    vmax2 = np.nanmax(d["resid_std_w"].to_numpy(float))
    cmap2 = plt.cm.plasma
    norm2 = plt.Normalize(vmin=vmin2, vmax=vmax2)

    for p, sub in d.groupby("period_years", sort=True):
        ax.scatter(
            sub["path_ratio_center_w"],
            sub["std_ratio_obs_pred"],
            c=sub["resid_std_w"],
            cmap=cmap2,
            norm=norm2,
            marker=marker_map.get(float(p), "o"),
            s=46,
            alpha=0.8,
            edgecolors="0.25",
            linewidths=0.35,
        )

    ax.axhline(1.0, color="black", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"Path ratio $L_A / |\Delta h_A|$")
    ax.set_ylabel("Observed / predicted STD")
    ax.set_title(f"Path-length sigma calibration: {label}")
    ax.grid(alpha=0.25)

    sm2 = plt.cm.ScalarMappable(norm=norm2, cmap=cmap2)
    sm2.set_array([])
    cbar = fig.colorbar(sm2, ax=ax)
    cbar.set_label("Observed binned STD")

    _add_pathlength_period_legend(ax, marker_map)
    fig.savefig(ratio_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_one_pathlength_sigma_observed_predicted_absbin(tab, fitted, form, out_png, ratio_png, title_suffix=None):
    """
    Alternate observed/predicted path-length sigma diagnostic.

    Color represents path ratio L_A / |dh_A|. Marker shape represents the
    net-|dh_A| bin of each binned point.
    """
    if tab is None or tab.empty or form not in fitted:
        return

    theta, param_names = fitted[form]
    d = tab.loc[
        np.isfinite(tab["abs_center_w"]) & (tab["abs_center_w"] > 0)
        & np.isfinite(tab["path_abs_center_w"]) & (tab["path_abs_center_w"] > 0)
        & np.isfinite(tab["period_years"])
        & np.isfinite(tab["resid_std_w"]) & (tab["resid_std_w"] > 0)
        & np.isfinite(tab["weight_sum"]) & (tab["weight_sum"] > 0)
        & np.isfinite(tab["abs_bin"])
    ].copy()
    if d.empty:
        return

    d["pred_std"] = path_sigma_model_form(
        theta,
        d["abs_center_w"].to_numpy(float),
        d["path_abs_center_w"].to_numpy(float),
        d["period_years"].to_numpy(float),
        form,
        param_names,
    )
    d = d.loc[np.isfinite(d["pred_std"]) & (d["pred_std"] > 0)].copy()
    if d.empty:
        return

    label = title_suffix if title_suffix is not None else form
    marker_map = _pathlength_absbin_marker_map(d["abs_bin"].to_numpy(int))
    center_map = {
        int(b): float(np.nanmean(sub["abs_center_w"].to_numpy(float)))
        for b, sub in d.groupby("abs_bin", sort=True)
    }

    fig, ax = plt.subplots(figsize=(6.8, 5.8), constrained_layout=True)
    vmin = np.nanmin(d["path_ratio_center_w"].to_numpy(float))
    vmax = np.nanmax(d["path_ratio_center_w"].to_numpy(float))
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    for b, sub in d.groupby("abs_bin", sort=True):
        ax.scatter(
            sub["pred_std"],
            sub["resid_std_w"],
            c=sub["path_ratio_center_w"],
            cmap=cmap,
            norm=norm,
            marker=marker_map.get(int(b), "o"),
            s=46,
            alpha=0.82,
            edgecolors="0.25",
            linewidths=0.35,
        )

    lo = np.nanmin([d["pred_std"].min(), d["resid_std_w"].min()])
    hi = np.nanmax([d["pred_std"].max(), d["resid_std_w"].max()])
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        ax.plot([lo, hi], [lo, hi], color="black", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Predicted residual STD")
    ax.set_ylabel("Observed binned residual STD")
    ax.set_title(f"Path-length sigma diagnostic: {label}")
    ax.grid(alpha=0.25)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(r"$L_A / |\Delta h_A|$")

    _add_pathlength_absbin_legend(ax, marker_map, center_map)
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    d["std_ratio_obs_pred"] = d["resid_std_w"] / d["pred_std"]
    fig, ax = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
    vmin2 = np.nanmin(d["resid_std_w"].to_numpy(float))
    vmax2 = np.nanmax(d["resid_std_w"].to_numpy(float))
    cmap2 = plt.cm.plasma
    norm2 = plt.Normalize(vmin=vmin2, vmax=vmax2)

    for b, sub in d.groupby("abs_bin", sort=True):
        ax.scatter(
            sub["path_ratio_center_w"],
            sub["std_ratio_obs_pred"],
            c=sub["resid_std_w"],
            cmap=cmap2,
            norm=norm2,
            marker=marker_map.get(int(b), "o"),
            s=46,
            alpha=0.8,
            edgecolors="0.25",
            linewidths=0.35,
        )

    ax.axhline(1.0, color="black", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"Path ratio $L_A / |\Delta h_A|$")
    ax.set_ylabel("Observed / predicted STD")
    ax.set_title(f"Path-length sigma calibration: {label}")
    ax.grid(alpha=0.25)

    sm2 = plt.cm.ScalarMappable(norm=norm2, cmap=cmap2)
    sm2.set_array([])
    cbar = fig.colorbar(sm2, ax=ax)
    cbar.set_label("Observed binned STD")

    _add_pathlength_absbin_legend(ax, marker_map, center_map)
    fig.savefig(ratio_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_pathlength_sigma_observed_predicted(tab, selection, fitted, out_png, ratio_png):
    """
    Plot path-length sigma diagnostics.

    The default output shows the best formula. Two
    additional outputs force the comparison of:
      - net_variance_additive: assumes L_A = |dh_A|;
      - path_variance_additive: uses measured L_A.
    """
    if tab is None or tab.empty or selection is None or selection.empty:
        return

    ok_models = selection.loc[np.isfinite(selection.get("wrmse", np.nan))].copy()
    if ok_models.empty:
        return

    best_form = str(ok_models.sort_values("wrmse").iloc[0]["path_sigma_form"])
    _plot_one_pathlength_sigma_observed_predicted(
        tab,
        fitted,
        best_form,
        out_png,
        ratio_png,
        title_suffix=f"best = {best_form}",
    )

    _plot_one_pathlength_sigma_observed_predicted(
        tab,
        fitted,
        "net_variance_additive",
        out_sigma_path_observed_predicted_net_png,
        out_sigma_path_ratio_diagnostic_net_png,
        title_suffix=r"net $|\Delta h_A|$ formula",
    )

    _plot_one_pathlength_sigma_observed_predicted(
        tab,
        fitted,
        "path_variance_additive",
        out_sigma_path_observed_predicted_path_png,
        out_sigma_path_ratio_diagnostic_path_png,
        title_suffix=r"path-length $L_A$ formula",
    )

    _plot_one_pathlength_sigma_observed_predicted_absbin(
        tab,
        fitted,
        "path_variance_additive",
        out_sigma_path_observed_predicted_absbin_png,
        out_sigma_path_ratio_diagnostic_absbin_png,
        title_suffix=r"path-length $L_A$ formula (marker = $|\Delta h_A|$ bin)",
    )

def run_pathlength_sigma_diagnostic(d_pred, all_rows, resid_col="rho_mean_removed"):
    """
    Fit and plot L_A-based sigma diagnostics in addition to the standard
    sigma(|dh|, dt) routine.
    """
    if not RUN_PATH_LENGTH_SIGMA_DIAGNOSTIC:
        return None, None

    log("Running path-length sigma diagnostic")
    d_path = attach_path_length_from_subperiods(d_pred, all_rows, out_col="path_abs_dh")
    tab = summarize_pathlength_residual_std_for_sigma(d_path, resid_col=resid_col, path_col="path_abs_dh")
    tab.to_csv(out_sigma_path_table_csv, index=False)

    if tab.empty:
        pd.DataFrame().to_csv(out_sigma_path_model_selection_csv, index=False)
        return tab, pd.DataFrame()

    selection, fitted = fit_pathlength_sigma_models(tab)
    selection.to_csv(out_sigma_path_model_selection_csv, index=False)
    plot_pathlength_sigma_observed_predicted(
        tab,
        selection,
        fitted,
        out_sigma_path_observed_predicted_png,
        out_sigma_path_ratio_diagnostic_png,
    )
    return tab, selection

def fit_sigma_model_from_rows(df, resid_col="rho_resid"):
    """
    Fit candidate sigma_rho(|dh|, period) models to binned row-level residual
    STD versus |current dh| and period length.

    To avoid a few extremely large-|dh| bins dominating the fit, bins with
    abs_center_w > SIGMA_MAX_ABS_DH_FOR_FIT are excluded from the sigma fit.
    """
    tab = summarize_absdh_residual_std_for_sigma(df, resid_col)
    if len(tab):
        tab = tab.loc[np.isfinite(tab["abs_center_w"]) & (tab["abs_center_w"] <= SIGMA_MAX_ABS_DH_FOR_FIT)].copy()

    if len(tab) < 4:
        r = df[resid_col].to_numpy(float)
        w = df[weight_col].to_numpy(float)
        sig = weighted_std(r, w)
        if not np.isfinite(sig) or sig <= 0:
            sig = NORMALIZED_METRIC_SIGMA_FLOOR
        globals()["SIGMA_SELECTED_FORM"] = "rational_floor"
        globals()["SIGMA_SELECTED_PARAMS"] = SIGMA_PARAMS_BY_FORM["rational_floor"]
        globals()["SIGMA_MODEL_SELECTION_TABLE"] = pd.DataFrame()
        theta = np.array([sig, 0.0, 1.0, 1.0], dtype=float)
        return theta, tab

    rows = []
    fitted = {}
    for form in SIGMA_CANDIDATE_FORMS:
        try:
            row, theta, param_names = _fit_one_sigma_form(tab, form)
            rows.append(row)
            fitted[form] = (theta, param_names)
        except Exception as exc:
            rows.append({"sigma_form": form, "error": repr(exc), "wrmse": np.nan, "n_params": np.nan})

    sel = pd.DataFrame(rows)
    globals()["SIGMA_MODEL_SELECTION_TABLE"] = sel.copy()

    if SIGMA_MODEL_FOR_DIAGNOSTICS == "best":
        ok = sel.loc[np.isfinite(sel.get("wrmse", np.nan))]
        if ok.empty:
            chosen = "rational_floor"
        else:
            metric = "pseudo_bic" if "pseudo_bic" in ok.columns and np.isfinite(ok["pseudo_bic"]).any() else "wrmse"
            chosen = str(ok.sort_values(metric).iloc[0]["sigma_form"])
    else:
        chosen = SIGMA_MODEL_FOR_DIAGNOSTICS

    if chosen not in fitted:
        raise RuntimeError(f"Selected sigma model {chosen!r} was not fitted successfully.")

    theta, param_names = fitted[chosen]
    # Store form, names and theta together so downstream plotting cannot mix a
    # Parameter list with the selected form
    globals()["SIGMA_SELECTED_FORM"] = chosen
    globals()["SIGMA_SELECTED_PARAMS"] = list(param_names)
    globals()["SIGMA_SELECTED_THETA"] = np.asarray(theta, dtype=float)

    resolved_form, resolved_params = _resolve_sigma_form_and_params(theta, preferred_form=chosen)
    if resolved_form != chosen or list(resolved_params) != list(param_names):
        raise RuntimeError(
            "Internal sigma selection mismatch after fitting: "
            f"chosen={chosen!r}, resolved={resolved_form!r}, "
            f"param_names={param_names}, resolved_params={resolved_params}."
        )

    return theta, tab

def plot_sigma_fit_all_models(tab, selection_table, out_dir_sigma):
    """
    Plot the fitted rho-STD model for each sigma candidate.

    Output is one PNG per sigma form, with:
      x = absolute current dh (log scale)
      y = binned STD of mean-removed rho residuals
      color = period length
      line = fitted sigma model for that period

    This mirrors the mean_fit_by_model diagnostics, but for sigma_rho.
    """
    if tab is None or tab.empty or selection_table is None or selection_table.empty:
        return

    out_dir_sigma.mkdir(parents=True, exist_ok=True)

    if "period_years" not in tab.columns:
        return

    periods = np.sort(tab["period_years"].unique())
    if PERIOD_LENGTHS_TO_PLOT is not None:
        periods = np.array([p for p in periods if p in PERIOD_LENGTHS_TO_PLOT], dtype=float)
    if len(periods) == 0:
        return

    cmap = plt.get_cmap(PLOT_CMAP)
    norm = mpl.colors.Normalize(vmin=np.nanmin(periods), vmax=np.nanmax(periods))
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    xvals = tab["abs_center_w"].to_numpy(float)
    yvals = tab["resid_std_w"].to_numpy(float)
    ok_xy = np.isfinite(xvals) & (xvals > 0) & np.isfinite(yvals) & (yvals > 0)
    if not np.any(ok_xy):
        return

    xmin = max(CURRENT_DH_MIN_ABS_FOR_FIT, np.nanmin(xvals[ok_xy]) * 0.8, 1e-4)
    xmax = max(np.nanmax(xvals[ok_xy]) * 1.2, xmin * 10)
    xline = np.geomspace(xmin, xmax, 600)

    # Only plot successfully fitted candidates
    rows = selection_table.copy()
    if "sigma_form" not in rows.columns:
        return
    if "wrmse" in rows.columns:
        rows = rows.loc[np.isfinite(rows["wrmse"])].copy()
    if rows.empty:
        return

    rows = rows.sort_values("wrmse" if "wrmse" in rows.columns else "sigma_form")

    for _, row in rows.iterrows():
        form = str(row["sigma_form"])
        if form not in SIGMA_PARAMS_BY_FORM:
            continue

        param_names = _sigma_params_for_form(form)
        theta = []
        missing = False
        for name in param_names:
            col = f"param_{name}"
            if col not in row.index or not np.isfinite(row[col]):
                missing = True
                break
            theta.append(float(row[col]))
        if missing:
            continue
        theta = np.asarray(theta, dtype=float)

        fig, ax = plt.subplots(figsize=(8.0, 5.5), constrained_layout=True)

        for T in periods:
            sub = tab.loc[tab["period_years"] == T].sort_values("abs_center_w")
            if sub.empty:
                continue
            col = cmap(norm(T))
            ax.scatter(
                sub["abs_center_w"],
                sub["resid_std_w"],
                s=22,
                color=col,
                alpha=0.78,
                edgecolors="none",
            )
            yline = sigma_model_form(
                theta,
                xline,
                period_years=np.full_like(xline, float(T), dtype=float),
                form=form,
                param_names=param_names,
            )
            ok = np.isfinite(yline) & (yline > 0)
            if np.any(ok):
                ax.plot(xline[ok], yline[ok], lw=1.9, color=col)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$|\Delta h|$ (m)")
        ax.set_ylabel(r"STD of mean-removed residual (kg m$^{-3}$)")
        title = f"Sigma fit: {form}"
        if "wrmse" in row.index and np.isfinite(row["wrmse"]):
            title += f"  |  WRMSE={row['wrmse']:.3g}"
        if "pseudo_bic" in row.index and np.isfinite(row["pseudo_bic"]):
            title += f"  |  BIC={row['pseudo_bic']:.3g}"
        ax.set_title(title)
        ax.grid(alpha=0.25)
        fig.colorbar(sm, ax=ax, pad=0.02).set_label("Period length (yr)")

        out_png = out_dir_sigma / f"{_safe_name(form)}_sigma_fit_abs_currentdh_loglog.png"
        fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
        plt.close(fig)


def build_positive_log_edges(max_abs_dh):
    if not np.isfinite(max_abs_dh) or max_abs_dh <= 0:
        raise ValueError("max_abs_dh must be positive and finite")
    edges = []
    decade = 1.0
    while True:
        vals = DH_LOG_PATTERN * decade
        edges.extend(float(v) for v in vals if v >= DH_MIN_POSITIVE_EDGE)
        if np.nanmax(vals) >= max_abs_dh:
            break
        decade *= 10.0
        if decade > 1e12:
            raise RuntimeError("Unreasonable signed-dh range")
    edges = np.unique(np.asarray(edges, dtype=float))
    edges = edges[edges >= DH_MIN_POSITIVE_EDGE]
    if edges.size == 0:
        edges = np.array([DH_MIN_POSITIVE_EDGE], dtype=float)
    if edges[-1] < max_abs_dh:
        edges = np.append(edges, max_abs_dh)
    return edges


def make_signed_log_dh_edges(dh):
    dh = np.asarray(dh, dtype=float)
    dh = dh[np.isfinite(dh)]
    if dh.size < 2:
        raise ValueError("Not enough finite dh values")
    dh_min, dh_max = float(np.nanmin(dh)), float(np.nanmax(dh))
    max_abs = float(np.nanmax(np.abs(dh)))
    pos = build_positive_log_edges(max_abs)
    edges = np.unique(np.concatenate([-pos[::-1], [0.0], pos]))
    if USE_ACTUAL_MINMAX_AS_OUTER_EDGES:
        if dh_min < edges[0]: edges[0] = dh_min
        if dh_max > edges[-1]: edges[-1] = dh_max
    return np.unique(edges)


def make_abs_dh_edges(abs_dh):
    abs_dh = np.asarray(abs_dh, dtype=float)
    abs_dh = abs_dh[np.isfinite(abs_dh)]
    edges = build_positive_log_edges(float(np.nanmax(abs_dh)))
    if edges[0] > CURRENT_DH_MIN_ABS_FOR_FIT:
        edges = np.insert(edges, 0, CURRENT_DH_MIN_ABS_FOR_FIT)
    return np.unique(edges)



def set_signed_log_xaxis(ax, values=None):
    ax.set_xscale("symlog", linthresh=SIGNED_DH_SYMLOG_LINTHRESH, linscale=0.06, base=10)
    if values is None:
        return
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return
    max_abs = np.nanmax(np.abs(vals))
    ticks_pos = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500], dtype=float)
    ticks_pos = ticks_pos[ticks_pos <= max_abs * 1.05]
    ticks = np.concatenate([-ticks_pos[::-1], [0], ticks_pos])
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])

def compute_signed_dh(df):
    if SIGNED_DH_MODE == "b_over_rho":
        return df["period_years"].astype(float) * df[b_col].astype(float) / df[rho_col].astype(float)
    raise ValueError(SIGNED_DH_MODE)


def compute_weight(df):
    if WEIGHT_MODE == "volume_from_row_rho_no_dt":
        return df[area_col].astype(float) * np.abs(df[b_col].astype(float) / df[rho_col].astype(float))
    if WEIGHT_MODE == "volume_from_row_rho_with_dt":
        return df[area_col].astype(float) * np.abs(df["period_years"].astype(float) * df[b_col].astype(float) / df[rho_col].astype(float))
    if WEIGHT_MODE == "area": return df[area_col].astype(float)
    if WEIGHT_MODE == "mass_change_no_dt": return df[area_col].astype(float) * np.abs(df[b_col].astype(float))
    if WEIGHT_MODE == "mass_change_with_dt": return df[area_col].astype(float) * np.abs(df["period_years"].astype(float) * df[b_col].astype(float))
    raise ValueError(WEIGHT_MODE)


def prepare_dataframe():
    header = pd.read_csv(input_csv, nrows=0).columns
    usecols = ["rgiid", "start_date", "end_date", rho_col, b_col, area_col]
    if variant_col in header:
        usecols.append(variant_col)
    # Keep spatial/region metadata when available so downstream correlation
    # Scripts can use the standardized residual output directly
    for c in OPTIONAL_OUTPUT_METADATA_COLS:
        if c in header and c not in usecols:
            usecols.append(c)
    df = pd.read_csv(input_csv, usecols=usecols, low_memory=True, memory_map=True)
    if variant_col not in df.columns: df[variant_col] = "single"
    if VARIANTS_TO_USE is not None:
        before = len(df)
        df = df.loc[df[variant_col].isin(VARIANTS_TO_USE)].copy()
        log(f"Variant filter kept {len(df):,}/{before:,} rows: {VARIANTS_TO_USE}")

    if LOW_MEMORY_MODE:
        for c in ["start_date", "end_date", rho_col, b_col, area_col, "lat", "lon", "cenlat", "cenlon"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce", downcast="float")
        for c in ["region", "rgi_region", "O1Region", "O2Region"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce", downcast="integer")
        if variant_col in df.columns:
            df[variant_col] = df[variant_col].astype("category")
        if "rgiid" in df.columns:
            df["rgiid"] = df["rgiid"].astype("category")

    df = df.replace([np.inf, -np.inf], np.nan)
    df["start_date"] = df["start_date"].astype("float32" if LOW_MEMORY_MODE else float)
    df["end_date"] = df["end_date"].astype("float32" if LOW_MEMORY_MODE else float)
    df["period_years"] = (df["end_date"] - df["start_date"]).astype("float32" if LOW_MEMORY_MODE else float)
    ok = (np.isfinite(df[rho_col]) & np.isfinite(df[b_col]) & np.isfinite(df[area_col]) &
          np.isfinite(df["period_years"]) & (df[area_col] > 0) & (df["period_years"] > 0) &
          (~df[rho_col].astype(float).isin(RHO_SENTINELS)))
    if SIGNED_DH_MODE == "b_over_rho" or WEIGHT_MODE in {"volume_from_row_rho_no_dt", "volume_from_row_rho_with_dt"}:
        ok &= np.abs(df[rho_col].astype(float)) > MIN_ABS_RHO_FOR_DH_AND_WEIGHT
    df = df.loc[ok].copy()
    df["signed_dh"] = compute_signed_dh(df)
    df[weight_col] = compute_weight(df)
    if DIVIDE_WEIGHT_BY_N_VARIANTS_PER_PERIOD:
        nvar = df.groupby(["rgiid", "start_date", "end_date"], observed=True)[variant_col].transform("nunique")
        df[weight_col] = df[weight_col] / nvar.clip(lower=1)
    ok = np.isfinite(df["signed_dh"]) & np.isfinite(df[weight_col]) & (df[weight_col] > 0)
    out = df.loc[ok].copy()
    if LOW_MEMORY_MODE:
        for c in ["signed_dh", weight_col, "period_years", rho_col, b_col, area_col, "start_date", "end_date", "lat", "lon", "cenlat", "cenlon"]:
            if c in out.columns:
                out[c] = out[c].astype("float32")
    del df
    gc.collect()
    return out


def attach_exact_past_window(current_rows, all_rows, window_years, out_col=None):
    if out_col is None: out_col = f"past{window_years}_signed_dh"
    start_col = f"{out_col}_start_date"
    has_col = f"has_{out_col}"
    w_col = f"{out_col}_weight"
    out0 = current_rows.copy()
    if out_col in out0.columns:
        out0[has_col] = np.isfinite(out0[out_col])
        return out0
    past = all_rows.loc[np.isclose(all_rows["period_years"], float(window_years)),
                        ["rgiid", variant_col, "start_date", "end_date", "signed_dh", weight_col]].copy()
    past = past.rename(columns={"start_date": start_col, "end_date": "start_date", "signed_dh": out_col, weight_col: w_col})
    past = past.sort_values(w_col, ascending=False).drop_duplicates(subset=["rgiid", variant_col, "start_date"], keep="first")
    out = out0.merge(past[["rgiid", variant_col, "start_date", start_col, out_col]],
                     on=["rgiid", variant_col, "start_date"], how="left", validate="many_to_one")
    out[has_col] = np.isfinite(out[out_col])
    return out



def attach_exp_cumulative_memory(current_rows, all_rows, tau, kmax, out_col):
    """
    Attach exponentially weighted past elevation change rate.

    The joint-kernel fit uses normalized kernel weights. The diagnostic past elevation change rate
    column uses the same normalization and accepts partial annual lag histories
    so that all period lengths retain sufficient support.
    """
    out = current_rows.copy()
    accum = np.zeros(len(out), dtype=float)
    total_weight = np.zeros(len(out), dtype=float)
    valid_count = np.zeros(len(out), dtype=int)

    base = out[["rgiid", variant_col, "start_date"]].copy()
    base["_row_id"] = np.arange(len(out))

    annual = all_rows.loc[
        np.isclose(all_rows["period_years"], 1.0),
        ["rgiid", variant_col, "start_date", "end_date", "signed_dh"],
    ].copy()
    annual = annual.drop_duplicates(subset=["rgiid", variant_col, "start_date", "end_date"])

    for k in range(1, int(kmax) + 1):
        wk = np.exp(-float(k) / float(tau))
        lookup = annual.rename(columns={
            "start_date": "_annual_start",
            "end_date": "_annual_end",
            "signed_dh": f"_dh{k}",
        })
        need = base.copy()
        need["_annual_start"] = need["start_date"] - float(k)
        need["_annual_end"] = need["start_date"] - float(k - 1)
        m = need.merge(
            lookup[["rgiid", variant_col, "_annual_start", "_annual_end", f"_dh{k}"]],
            on=["rgiid", variant_col, "_annual_start", "_annual_end"],
            how="left",
        ).sort_values("_row_id")
        vals = m[f"_dh{k}"].to_numpy(dtype=float)
        ok = np.isfinite(vals)
        accum[ok] += wk * vals[ok]
        total_weight[ok] += wk
        valid_count[ok] += 1

    if REQUIRE_FULL_EXP_MEMORY:
        ok_final = valid_count == int(kmax)
    else:
        ok_final = valid_count >= int(EXP_MEMORY_MIN_VALID_LAGS)
    ok_final = ok_final & np.isfinite(total_weight) & (total_weight > 0)

    out[out_col] = np.nan
    if NORMALIZE_EXP_MEMORY_WEIGHTS:
        out.loc[ok_final, out_col] = accum[ok_final] / total_weight[ok_final]
    else:
        out.loc[ok_final, out_col] = accum[ok_final]

    out[f"{out_col}_valid_lags"] = valid_count
    out[f"{out_col}_weight_sum"] = total_weight
    out[f"has_{out_col}"] = np.isfinite(out[out_col])
    return out

def attach_model_memory(df, all_rows, mode, window=None, tau=None):
    if mode == "fixed_window":
        col = f"memory_fixed{window:g}_dh"
        return attach_exact_past_window(df, all_rows, window, out_col=col), col
    if mode == "exp_cumulative":
        col = f"memory_exp_tau{tau:g}_dh"
        return attach_exp_cumulative_memory(df, all_rows, tau=tau, kmax=MODEL_MEMORY_KMAX_YEARS, out_col=col), col
    raise ValueError(mode)

# =============================================================================
# Model components
# =============================================================================

def params_for_forms(spec):
    names = []
    def add(seq):
        for p in seq:
            if p not in names: names.append(p)
    if spec["memory"] == "logistic_sqrt":
        add(["D0", "LD"])
    elif spec["memory"] == "logistic_sqrt_centered":
        add(["D0", "LD"])
    elif spec["memory"] == "signed_sqrt_contrast":
        add(["Bmem"])
    elif spec["memory"] == "signed_power_contrast":
        add(["Bmem", "etaMem"])
    elif spec["memory"] == "logistic_free_eta":
        add(["D0", "LD", "etaD"])
    elif spec["memory"] == "exp_deficit":
        add(["D0", "LD"])
    elif spec["memory"] == "current_damped_logistic":
        add(["D0", "LD", "xD", "qD"])
    else:
        raise ValueError(spec["memory"])
    if spec["ratio"] == "tanh_power": add(["A", "LA", "alpha"])
    elif spec["ratio"] == "tanh_power_high_alpha": add(["A", "LA", "alphaH"])
    elif spec["ratio"] == "tanh_signed_alpha": add(["Aneg", "Apos", "LA", "alphaNeg", "alphaPos"])
    elif spec["ratio"] == "tanh_signed_amp": add(["Aneg", "Apos", "LA", "alpha"])
    elif spec["ratio"] == "tanh_signed_alpha_common_amp": add(["A", "LA", "alphaNeg", "alphaPos"])
    elif spec["ratio"] == "tanh_past_alpha_fast_neutral": add(["A", "LA", "alphaBase", "alphaAmp", "LP", "etaAlpha"])
    elif spec["ratio"] == "tanh_past_alpha_fast_strong": add(["A", "LA", "alphaBase", "alphaAmp", "LP", "etaAlpha"])
    elif spec["ratio"] == "tanh_memory_power": add(["A", "LA", "alpha", "etaQ"])
    elif spec["ratio"] == "tanh_memory_power_high_alpha": add(["A", "LA", "alphaH", "etaQ"])
    elif spec["ratio"] == "tanh_memory_past_alpha_fast_neutral": add(["A", "LA", "alphaBase", "alphaAmp", "LP", "etaAlpha", "etaQ"])
    elif spec["ratio"] == "tanh_memory_past_alpha_fast_strong": add(["A", "LA", "alphaBase", "alphaAmp", "LP", "etaAlpha", "etaQ"])
    elif spec["ratio"] == "tanh_memory_power_fixed_alpha1": add(["A", "LA", "etaQ"])
    elif spec["ratio"] == "tanh_fixed_alpha1": add(["A", "LA"])
    elif spec["ratio"] == "signed_power_fixed_alpha1": add(["A", "LA"])
    elif spec["ratio"] != "none": raise ValueError(spec["ratio"])
    if spec["damping"] == "exp_stretched": add(["H", "beta"])
    elif spec["damping"] == "rational_power": add(["xR", "qR"])
    else: raise ValueError(spec["damping"])
    if spec["period"] == "exp_param": add(["C0", "C1", "T0"])
    elif spec["period"] == "exp_no_offset": add(["C1", "T0"])
    elif spec["period"] == "exp_no_offset_fixed_T0_3": add(["C1"])
    elif spec["period"] == "power_param": add(["P0", "P1", "TP"])
    elif spec["period"] != "none": raise ValueError(spec["period"])
    if spec["current"] == "constant": add(["Bc"])
    elif spec["current"] == "rational_power": add(["Bc", "xS", "qS"])
    elif spec["current"] == "rational_power_lowdh": add(["Bc", "xSL", "qSL"])
    elif spec["current"] == "rational_band": add(["Bc", "x1", "x2"])
    elif spec["current"] == "rational_band_lowdh": add(["Bc", "x1L", "x2L"])
    elif spec["current"] == "inverse_power": add(["Bc", "xC", "qC"])
    elif spec["current"] != "none": raise ValueError(spec["current"])
    return names


def unpack(theta, param_names): return {n: float(v) for n, v in zip(param_names, theta)}


def memory_component(pars, form, x, p):
    """
    Finite signed-memory deficit component.

    All alternatives are paired with the outer damping R(|dh|), so the full
    model still tends to rho_ice for large |current dh|.

    Forms:
      logistic_sqrt:
        -D0 / [1 + exp(sign(a) (|a|/LD)^0.5)]

      logistic_sqrt_centered:
        same as logistic_sqrt but centered as S(a)-1/2, so D0 controls only
        the signed-memory contrast and not the intercept.

      logistic_free_eta:
        same, with fitted etaD.

      exp_deficit:
        -D0 exp(-a/LD), clipped numerically. This preserves the asymptotic
        behaviour in |current dh| through R, but is less bounded in memory and
        should reveal whether the bounded logistic form is needed.

      current_damped_logistic:
        logistic_sqrt deficit additionally damped by |current dh| inside the
        bracket. This tests the older "current-damped deficit" alternative.
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)
    absx = np.abs(x)
    a = np.sign(x) * p

    if form == "logistic_sqrt":
        eta = ETA_D_FIXED
        z = np.sign(a) * (np.abs(a) / pars["LD"]) ** eta
        S = 1.0 / (1.0 + np.exp(np.clip(z, -60, 60)))
        return -pars["D0"] * S

    if form == "logistic_sqrt_centered":
        eta = ETA_D_FIXED
        z = np.sign(a) * (np.abs(a) / pars["LD"]) ** eta
        S = 1.0 / (1.0 + np.exp(np.clip(z, -60, 60)))
        return -pars["D0"] * (S - 0.5)

    if form == "signed_sqrt_contrast":
        # Parsimonious limit of the centered logistic memory term when LD is
        # Much larger than the observed memory-dh range:
        #   Logistic(z)-1/2 ~ -z/4, z = sign(dh*dh_mem)*sqrt(|dh_mem|/LD)
        return pars["Bmem"] * np.sign(a) * np.sqrt(np.abs(a))

    if form == "signed_power_contrast":
        # Same parsimonious signed-memory contrast, but with fitted sublinear
        # Memory exponent etaMem. The fixed sqrt form is etaMem = 0.5
        return pars["Bmem"] * np.sign(a) * (np.abs(a) ** pars["etaMem"])

    if form == "logistic_free_eta":
        eta = pars["etaD"]
        z = np.sign(a) * (np.abs(a) / pars["LD"]) ** eta
        S = 1.0 / (1.0 + np.exp(np.clip(z, -60, 60)))
        return -pars["D0"] * S

    if form == "exp_deficit":
        z = np.clip(-a / pars["LD"], -60, 60)
        return -pars["D0"] * np.exp(z)

    if form == "current_damped_logistic":
        z = np.sign(a) * (np.abs(a) / pars["LD"]) ** ETA_D_FIXED
        S = 1.0 / (1.0 + np.exp(np.clip(z, -60, 60)))
        D = 1.0 / (1.0 + (absx / pars["xD"]) ** pars["qD"])
        return -pars["D0"] * S * D

    raise ValueError(form)


def ratio_component(pars, form, x, p):
    if form == "none": return np.zeros_like(np.asarray(x, dtype=float))
    a = np.sign(x) * p
    absx = np.abs(x)
    absp = np.abs(p)

    def past_alpha(mode):
        z = (absp / pars["LP"]) ** pars["etaAlpha"]
        z = np.clip(z, 0, 700)
        if mode == "fast_neutral":
            return pars["alphaBase"] + pars["alphaAmp"] * np.exp(-z)
        if mode == "fast_strong":
            return pars["alphaBase"] + pars["alphaAmp"] * (1.0 - np.exp(-z))
        raise ValueError(mode)

    if form == "tanh_power":
        return pars["A"] * np.tanh(a / pars["LA"]) / (absx ** pars["alpha"])
    if form == "tanh_power_high_alpha":
        return pars["A"] * np.tanh(a / pars["LA"]) / (absx ** pars["alphaH"])
    if form == "tanh_signed_alpha":
        amp = np.where(np.asarray(x, dtype=float) < 0, pars["Aneg"], pars["Apos"])
        alpha = np.where(np.asarray(x, dtype=float) < 0, pars["alphaNeg"], pars["alphaPos"])
        return amp * np.tanh(a / pars["LA"]) / (absx ** alpha)
    if form == "tanh_signed_amp":
        amp = np.where(np.asarray(x, dtype=float) < 0, pars["Aneg"], pars["Apos"])
        return amp * np.tanh(a / pars["LA"]) / (absx ** pars["alpha"])
    if form == "tanh_signed_alpha_common_amp":
        alpha = np.where(np.asarray(x, dtype=float) < 0, pars["alphaNeg"], pars["alphaPos"])
        return pars["A"] * np.tanh(a / pars["LA"]) / (absx ** alpha)
    if form == "tanh_past_alpha_fast_neutral":
        return pars["A"] * np.tanh(a / pars["LA"]) / (absx ** past_alpha("fast_neutral"))
    if form == "tanh_past_alpha_fast_strong":
        return pars["A"] * np.tanh(a / pars["LA"]) / (absx ** past_alpha("fast_strong"))
    if form == "tanh_memory_power":
        z = np.sign(a) * (np.abs(a) / pars["LA"]) ** pars["etaQ"]
        return pars["A"] * np.tanh(z) / (absx ** pars["alpha"])
    if form == "tanh_memory_power_high_alpha":
        z = np.sign(a) * (np.abs(a) / pars["LA"]) ** pars["etaQ"]
        return pars["A"] * np.tanh(z) / (absx ** pars["alphaH"])
    if form == "tanh_memory_past_alpha_fast_neutral":
        z = np.sign(a) * (np.abs(a) / pars["LA"]) ** pars["etaQ"]
        return pars["A"] * np.tanh(z) / (absx ** past_alpha("fast_neutral"))
    if form == "tanh_memory_past_alpha_fast_strong":
        z = np.sign(a) * (np.abs(a) / pars["LA"]) ** pars["etaQ"]
        return pars["A"] * np.tanh(z) / (absx ** past_alpha("fast_strong"))
    if form == "tanh_memory_power_fixed_alpha1":
        z = np.sign(a) * (np.abs(a) / pars["LA"]) ** pars["etaQ"]
        return pars["A"] * np.tanh(z) / absx
    if form == "tanh_fixed_alpha1":
        return pars["A"] * np.tanh(a / pars["LA"]) / absx
    if form == "signed_power_fixed_alpha1":
        # Smooth sign approximation controlled by LA; for |a| >> LA this is
        # Approximately A * sign(a) / |dh|
        return pars["A"] * np.tanh(a / pars["LA"]) / absx
    raise ValueError(form)


def period_component_form(pars, form, T):
    T = np.asarray(T, dtype=float)
    if form == "none": return np.zeros_like(T)
    if form == "exp_param": return pars["C0"] + pars["C1"] * np.exp(-T / pars["T0"])
    if form == "exp_no_offset": return pars["C1"] * np.exp(-T / pars["T0"])
    if form == "exp_no_offset_fixed_T0_3": return pars["C1"] * np.exp(-T / 3.0)
    if form == "power_param": return pars["P0"] + pars["P1"] / (1.0 + T / pars["TP"])
    raise ValueError(form)


def current_component(pars, form, x):
    absx = np.abs(x)
    if form == "none": return np.zeros_like(absx)
    if form == "constant": return np.full_like(absx, pars["Bc"], dtype=float)
    if form == "rational_power": return pars["Bc"] / (1.0 + (absx / pars["xS"]) ** pars["qS"])
    if form == "rational_power_lowdh": return pars["Bc"] / (1.0 + (absx / pars["xSL"]) ** pars["qSL"])
    if form == "rational_band":
        x1, x2 = min(pars["x1"], pars["x2"]), max(pars["x1"], pars["x2"])
        return pars["Bc"] * (1/(1+absx/x1) - 1/(1+absx/x2))
    if form == "rational_band_lowdh":
        x1, x2 = min(pars["x1L"], pars["x2L"]), max(pars["x1L"], pars["x2L"])
        return pars["Bc"] * (1/(1+absx/x1) - 1/(1+absx/x2))
    if form == "inverse_power":
        return pars["Bc"] / ((absx + pars["xC"]) ** pars["qC"])
    raise ValueError(form)


def damping_component(pars, form, x):
    absx = np.abs(x)
    if form == "exp_stretched": return np.exp(-np.clip((absx / pars["H"]) ** pars["beta"], 0, 700))
    if form == "rational_power": return 1.0 / (1.0 + (absx / pars["xR"]) ** pars["qR"])
    raise ValueError(form)


def rho_model(theta, spec, param_names, x, p, T):
    pars = unpack(theta, param_names)
    x, p, T = np.asarray(x, dtype=float), np.asarray(p, dtype=float), np.asarray(T, dtype=float)
    absx = np.abs(x)
    out = np.full_like(absx, np.nan, dtype=float)
    ok = np.isfinite(absx) & (absx > 0) & np.isfinite(p) & np.isfinite(T)
    if not np.any(ok): return out
    bracket = (memory_component(pars, spec["memory"], x[ok], p[ok]) +
               ratio_component(pars, spec["ratio"], x[ok], p[ok]) +
               period_component_form(pars, spec["period"], T[ok]) +
               current_component(pars, spec["current"], x[ok]))
    R = damping_component(pars, spec["damping"], x[ok])
    out[ok] = RHO_ICE_FIXED + R * bracket
    return out


def component_values(theta, spec, param_names, x, p, T):
    pars = unpack(theta, param_names)
    R = damping_component(pars, spec["damping"], x)
    return {
        "memory_component": R * memory_component(pars, spec["memory"], x, p),
        "ratio_component": R * ratio_component(pars, spec["ratio"], x, p),
        "period_component": R * period_component_form(pars, spec["period"], T),
        "current_component": R * current_component(pars, spec["current"], x),
        "damping": R,
    }

# =============================================================================
# Fit functions
# =============================================================================

def initial_for_params(param_names): return np.asarray([INITIAL[p] for p in param_names], dtype=float)
def bounds_for_params(param_names): return np.asarray([BOUNDS[p][0] for p in param_names], dtype=float), np.asarray([BOUNDS[p][1] for p in param_names], dtype=float)


def random_initials(rng, param_names, n_random):
    starts = [initial_for_params(param_names)]
    lo, hi = bounds_for_params(param_names)
    logscale = {"LD", "LA", "etaD", "etaMem", "etaQ", "alpha", "alphaH", "alphaNeg", "alphaPos", "alphaBase", "alphaAmp", "LP", "etaAlpha", "H", "beta", "T0", "TP", "xS", "qS", "xSL", "qSL", "x1", "x2", "x1L", "x2L", "xC", "qC", "xR", "qR"}
    for _ in range(n_random):
        vals = []
        for name, l, h in zip(param_names, lo, hi):
            if l > 0 and (h/l > 100 or name in logscale): vals.append(np.exp(rng.uniform(np.log(l), np.log(h))))
            else: vals.append(rng.uniform(l, h))
        starts.append(np.asarray(vals, dtype=float))
    return starts


def prepare_binned_rows(df, current_edges, memory_edges, memory_col):
    d = df.loc[np.isfinite(df[memory_col])].copy()
    d["current_bin"] = pd.cut(d["signed_dh"], bins=current_edges, labels=False, include_lowest=True)
    d["memory_bin"] = pd.cut(d[memory_col], bins=memory_edges, labels=False, include_lowest=True)
    d = d.dropna(subset=["current_bin", "memory_bin"]).copy()
    d["current_bin"], d["memory_bin"] = d["current_bin"].astype(int), d["memory_bin"].astype(int)
    return d


def summarize_target_3d(row_df, memory_col):
    rows = []
    for (T, ib, jb), g in row_df.groupby(["period_years", "current_bin", "memory_bin"], sort=True):
        w = g[weight_col].to_numpy(dtype=float)
        n_eff = weighted_effective_n(w)
        rows.append({
            "period_years": float(T), "current_bin": int(ib), "memory_bin": int(jb),
            "n": len(g), "n_eff": n_eff, "weight_sum": np.nansum(w),
            "current_center_w": weighted_mean(g["signed_dh"].to_numpy(dtype=float), w),
            "memory_center_w": weighted_mean(g[memory_col].to_numpy(dtype=float), w),
            "rho_mean_w": weighted_mean(g[rho_col].to_numpy(dtype=float), w),
            "support_ok": bool(len(g) >= MIN_COUNT_3D_CELL and np.isfinite(n_eff) and n_eff >= MIN_NEFF_3D_CELL),
        })
    out = pd.DataFrame(rows)
    if len(out): out = out.sort_values(["period_years", "current_bin", "memory_bin"]).reset_index(drop=True)
    return out


def mean_fit_sigma_from_target(target):
    """Evaluate the retained uncertainty form at binned target centres."""
    absx = np.maximum(np.abs(target["current_center_w"].to_numpy(float)), 1.0e-12)
    dt = np.maximum(target["period_years"].to_numpy(float), 0.0)
    sigma = np.sqrt(
        MEAN_FIT_SIGMA_U_H**2 / absx
        + MEAN_FIT_SIGMA_U_T**2 * dt / absx**2
    )
    return np.maximum(sigma, MEAN_FIT_SIGMA_FLOOR)


def attach_mean_fit_weights(target):
    """Add the least-squares weights used to fit the mean function.

    ``weight_sum`` remains the physical volume change support used to estimate
    each binned mean. ``mean_fit_weight_sum`` is only the optimizer weight for
    matching these binned means.
    """
    out = target.copy()
    support = out["weight_sum"].to_numpy(float)

    if MEAN_FIT_WEIGHT_MODE == "volume":
        fit_weight = support
    elif MEAN_FIT_WEIGHT_MODE == "volume_over_sigma2":
        sigma = mean_fit_sigma_from_target(out)
        fit_weight = support / sigma**2
    else:
        raise ValueError(f"Unknown MEAN_FIT_WEIGHT_MODE: {MEAN_FIT_WEIGHT_MODE}")

    out[mean_fit_weight_col] = fit_weight
    return out


def summarize_target_prediction_by_dh_category(target, model, fit_weight_column="weight_sum"):
    """Summarize target-cell prediction errors over low/mid/high |dh| bins."""
    pred_col = f"rho_pred_{model}"
    required = ["current_center_w", "rho_mean_w", pred_col, fit_weight_column, "support_ok"]
    missing = [c for c in required if c not in target.columns]
    if missing:
        raise KeyError(f"Missing columns for dh-category summary: {missing}")

    d = target.loc[target["support_ok"]].copy()
    d = d.loc[
        np.isfinite(d["current_center_w"])
        & np.isfinite(d["rho_mean_w"])
        & np.isfinite(d[pred_col])
        & np.isfinite(d[fit_weight_column])
        & (d[fit_weight_column] > 0)
    ].copy()
    d["dh_category"] = pd.cut(
        np.abs(d["current_center_w"].to_numpy(float)),
        bins=DH_CATEGORY_EDGES_M,
        labels=DH_CATEGORY_LABELS,
        include_lowest=True,
        right=False,
    )

    rows = []
    for category, group in d.dropna(subset=["dh_category"]).groupby("dh_category", observed=True, sort=False):
        w = group[fit_weight_column].to_numpy(float)
        residual = group[pred_col].to_numpy(float) - group["rho_mean_w"].to_numpy(float)
        rows.append({
            "model": model,
            "weight_column": fit_weight_column,
            "dh_category": str(category),
            "n_cells": len(group),
            "weight_sum": float(np.nansum(w)),
            "weighted_bias_kg_m3": weighted_mean(residual, w),
            "weighted_rmse_kg_m3": float(np.sqrt(np.nansum(w * residual**2) / np.nansum(w))),
        })
    return pd.DataFrame(rows)


def _dh_bin_label(left, right):
    """Return a compact text label for a dh interval."""
    def fmt(value):
        if np.isneginf(value):
            return "-inf"
        if np.isposinf(value):
            return "inf"
        return f"{float(value):g}"

    return f"{fmt(left)}:{fmt(right)}"


def summarize_target_prediction_by_period_dh_bin(
    target,
    model,
    fit_weight_column="weight_sum",
    signed=False,
):
    """
    Summarize target-cell prediction errors by period length and dh bin.

    :param target: Binned full model target table with attached predictions.
    :param model: Name of fitted model whose ``rho_pred_*`` column is used.
    :param fit_weight_column: Weight column used for the summary.
    :param signed: If True, bin signed current dh; otherwise bin |current dh|.
    """
    pred_col = f"rho_pred_{model}"
    required = ["period_years", "current_center_w", "memory_center_w", "rho_mean_w", pred_col, fit_weight_column, "support_ok"]
    missing = [c for c in required if c not in target.columns]
    if missing:
        raise KeyError(f"Missing columns for period-dh summary: {missing}")

    d = target.loc[target["support_ok"]].copy()
    d = d.loc[
        np.isfinite(d["period_years"])
        & np.isfinite(d["current_center_w"])
        & np.isfinite(d["memory_center_w"])
        & np.isfinite(d["rho_mean_w"])
        & np.isfinite(d[pred_col])
        & np.isfinite(d[fit_weight_column])
        & (d[fit_weight_column] > 0)
    ].copy()
    if d.empty:
        return pd.DataFrame()

    if signed:
        values = d["current_center_w"].to_numpy(float)
        edges = TARGET_SIGNED_DH_METRIC_EDGES_M
        bin_type = "signed_dh"
    else:
        values = np.abs(d["current_center_w"].to_numpy(float))
        edges = TARGET_ABS_DH_METRIC_EDGES_M
        bin_type = "abs_dh"

    d["_dh_bin"] = pd.cut(values, bins=edges, labels=False, include_lowest=True)
    d = d.dropna(subset=["_dh_bin"]).copy()
    d["_dh_bin"] = d["_dh_bin"].astype(int)
    total_weight = float(np.nansum(d[fit_weight_column].to_numpy(float)))

    rows = []
    for (period, ib), group in d.groupby(["period_years", "_dh_bin"], sort=True):
        ib = int(ib)
        left = float(edges[ib])
        right = float(edges[ib + 1])
        w = group[fit_weight_column].to_numpy(float)
        residual = group[pred_col].to_numpy(float) - group["rho_mean_w"].to_numpy(float)
        rows.append({
            "model": model,
            "weight_column": fit_weight_column,
            "period_years": float(period),
            "dh_bin_type": bin_type,
            "dh_bin": ib,
            "dh_bin_left_m": left,
            "dh_bin_right_m": right,
            "dh_bin_label": _dh_bin_label(left, right),
            "n_cells": len(group),
            "n_full_model_rows": int(np.nansum(group["n"].to_numpy(float))) if "n" in group else np.nan,
            "n_eff_sum": float(np.nansum(group["n_eff"].to_numpy(float))) if "n_eff" in group else np.nan,
            "weight_sum": float(np.nansum(w)),
            "weight_fraction": float(np.nansum(w) / total_weight) if total_weight > 0 else np.nan,
            "dh_center_w": weighted_mean(group["current_center_w"].to_numpy(float), w),
            "past_dh_center_w": weighted_mean(group["memory_center_w"].to_numpy(float), w),
            "rho_binned_w": weighted_mean(group["rho_mean_w"].to_numpy(float), w),
            "rho_pred_w": weighted_mean(group[pred_col].to_numpy(float), w),
            "weighted_bias_kg_m3": weighted_mean(residual, w),
            "weighted_rmse_kg_m3": float(np.sqrt(np.nansum(w * residual**2) / np.nansum(w))),
            "weighted_abs_bias_kg_m3": weighted_mean(np.abs(residual), w),
        })
    return pd.DataFrame(rows)


def summarize_target_prediction_period_dh_for_models(target, models, fit_weight_columns):
    """Build period-by-dh bias tables for several models and weight columns."""
    abs_tables = []
    signed_tables = []
    for model in models:
        for fit_weight_column in fit_weight_columns:
            if fit_weight_column not in target.columns:
                continue
            abs_tables.append(
                summarize_target_prediction_by_period_dh_bin(
                    target, model, fit_weight_column=fit_weight_column, signed=False
                )
            )
            signed_tables.append(
                summarize_target_prediction_by_period_dh_bin(
                    target, model, fit_weight_column=fit_weight_column, signed=True
                )
            )

    abs_out = pd.concat([t for t in abs_tables if len(t)], ignore_index=True) if abs_tables else pd.DataFrame()
    signed_out = pd.concat([t for t in signed_tables if len(t)], ignore_index=True) if signed_tables else pd.DataFrame()
    return abs_out, signed_out




def attach_predictions_to_target(target, fits, memory_col=None, period_col=None):
    """
    Attach fitted rho predictions to each target cell for every fitted model.

    Accept optional column arguments while using the target summary columns
    period_years, so those optional arguments are ignored unless alternate
    column names are explicitly supplied.
    """
    out = target.copy()
    x_col = "current_center_w"
    p_col = memory_col if memory_col is not None and memory_col in out.columns else "memory_center_w"
    t_col = period_col if period_col is not None and period_col in out.columns else "period_years"

    required = [x_col, p_col, t_col]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise KeyError(f"attach_predictions_to_target missing required columns: {missing}")

    x = out[x_col].to_numpy(float)
    p = out[p_col].to_numpy(float)
    T = out[t_col].to_numpy(float)

    for model, obj in fits.items():
        out[f"rho_pred_{model}"] = rho_model(
            obj["theta"],
            obj["spec"],
            obj["param_names"],
            x,
            p,
            T,
        )
    return out


def add_absdh_std_scale_to_target(target_fit):
    """
    Add a scale column for normalized model comparison.

    The scale is the weighted average raw rho STD in absolute-current-dh bins,
    not the individual 3D-cell STD. This targets the dominant heteroscedastic
    dependence on |dh| while avoiding overemphasis of tiny per-cell STD values.
    """
    tf = target_fit.copy()
    if "rho_std_absdh_w" in tf.columns:
        return tf

    if "rho_std_w" not in tf.columns:
        tf["rho_std_absdh_w"] = NORMALIZED_METRIC_SIGMA_FLOOR
        return tf

    abs_current = np.abs(tf["current_center_w"].to_numpy(float))
    edges = make_abs_dh_edges(abs_current)
    tf["_abs_current_bin_for_scale"] = pd.cut(abs_current, bins=edges, labels=False, include_lowest=True)

    scale = {}
    for ib, g in tf.dropna(subset=["_abs_current_bin_for_scale"]).groupby("_abs_current_bin_for_scale"):
        w = g["weight_sum"].to_numpy(float)
        svals = g["rho_std_w"].to_numpy(float)
        ok = np.isfinite(w) & (w > 0) & np.isfinite(svals) & (svals > 0)
        if np.any(ok):
            scale[int(ib)] = max(weighted_mean(svals[ok], w[ok]), NORMALIZED_METRIC_SIGMA_FLOOR)

    tf["rho_std_absdh_w"] = [
        scale.get(int(ib), NORMALIZED_METRIC_SIGMA_FLOOR) if np.isfinite(ib) else NORMALIZED_METRIC_SIGMA_FLOOR
        for ib in tf["_abs_current_bin_for_scale"].to_numpy(float)
    ]
    tf = tf.drop(columns=["_abs_current_bin_for_scale"])
    return tf



def fit_one_model(target_fit, spec):
    target_fit = add_absdh_std_scale_to_target(target_fit)
    param_names = params_for_forms(spec)
    x = target_fit["current_center_w"].to_numpy(dtype=float)
    p = target_fit["memory_center_w"].to_numpy(dtype=float)
    T = target_fit["period_years"].to_numpy(dtype=float)
    y = target_fit["rho_mean_w"].to_numpy(dtype=float)
    fit_weight_name = mean_fit_weight_col if mean_fit_weight_col in target_fit.columns else "weight_sum"
    w = target_fit[fit_weight_name].to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(p) & np.isfinite(T) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, p, T, y, w = x[ok], p[ok], T[ok], y[ok], w[ok]
    sqrtw = np.sqrt(w / np.nanmean(w))
    lo, hi = bounds_for_params(param_names)
    def pred(theta): return rho_model(theta, spec, param_names, x, p, T)
    def residual(theta):
        r = (pred(theta) - y) * sqrtw
        r[~np.isfinite(r)] = 1e12
        return r
    rng = np.random.default_rng(RANDOM_SEED)
    best, first_exception = None, None
    for theta0 in random_initials(rng, param_names, N_RANDOM_STARTS):
        try:
            if not np.all(np.isfinite(residual(theta0))): raise FloatingPointError("non-finite initial residual")
            res = least_squares(residual, theta0, bounds=(lo, hi), loss="soft_l1", f_scale=500.0, max_nfev=MAX_NFEV, x_scale="jac")
        except Exception as exc:
            if first_exception is None: first_exception = repr(exc)
            continue
        score = np.sum(res.fun**2)
        if best is None or score < best["score"]: best = {"theta": res.x, "score": score, "result": res}
    if best is None: raise RuntimeError(f"All starts failed for {spec['name']}. First exception: {first_exception}")
    yhat = pred(best["theta"])
    metrics = weighted_metrics(y, yhat, w, len(best["theta"]))

    if "rho_std_absdh_w" in target_fit.columns:
        sigma_scale = target_fit["rho_std_absdh_w"].to_numpy(float)[ok]
        sigma_scale = np.where(
            np.isfinite(sigma_scale) & (sigma_scale > NORMALIZED_METRIC_SIGMA_FLOOR),
            sigma_scale,
            NORMALIZED_METRIC_SIGMA_FLOOR,
        )
        nres = (yhat - y) / sigma_scale
        nwrmse = float(np.sqrt(np.sum(w * nres**2) / np.sum(w)))
        nwmae = float(np.sum(w * np.abs(nres)) / np.sum(w))
    else:
        nwrmse = np.nan
        nwmae = np.nan

    row = {
        "model": spec["name"],
        "n_params": len(best["theta"]),
        "mean_fit_weight_column": fit_weight_name,
        "score": float(best["score"]),
        "cost": float(best["result"].cost),
        "success": bool(best["result"].success),
        "nwrmse_absdh_std": nwrmse,
        "nwmae_absdh_std": nwmae,
        **metrics,
    }
    row.update({f"form_{k}": spec[k] for k in ["memory", "ratio", "damping", "period", "current"]})
    for name, val in zip(param_names, best["theta"]): row[f"param_{name}"] = float(val)
    return row, {"spec": spec, "theta": best["theta"], "param_names": param_names}


def fit_candidate_models(target_fit, candidate_models):
    rows, fits = [], {}
    for spec in candidate_models:
        log(f"Fitting model: {spec['name']}")
        row, obj = fit_one_model(target_fit, spec)
        rows.append(row); fits[spec["name"]] = obj
    return pd.DataFrame(rows).sort_values("wrmse").reset_index(drop=True), fits

# =============================================================================
# Diagnostics
# =============================================================================

def select_fit(summary, fits):
    model = str(summary.iloc[0]["model"]) if SELECTED_MODEL_FOR_DIAGNOSTICS is None else SELECTED_MODEL_FOR_DIAGNOSTICS
    if model not in fits: raise ValueError(f"Selected model {model!r} not in {list(fits)}")
    return model, fits[model]





def fit_or_find_paired_no_period_fit(selected_fit, summary, fit_objects, target_fit, t0=None):
    """
    Return a paired no-period fit for period-correction diagnostics.

    If an exact paired no-period model is already in the fitted candidates, use it.
    Otherwise, explicitly fit the same architecture with period='none' on the same
    target table. This is required when the main run skips the full model search
    and only fits the selected final period-dependent model.
    """
    no_period_model, no_period_fit = find_paired_no_period_fit(selected_fit, summary, fit_objects)
    if no_period_fit is not None:
        return no_period_model, no_period_fit

    spec = dict(selected_fit["spec"])
    spec["period"] = "none"
    spec["name"] = str(spec.get("name", "selected")) + "__paired_no_period"
    log(f"Fitting paired no-period model explicitly: {spec['name']}", t0)
    row, obj = fit_one_model(target_fit, spec)
    return spec["name"], obj

def find_paired_no_period_fit(selected_fit, summary, fit_objects):
    """Find a fitted no-period model matching the selected model's other components.

    If an exact paired model is not available, fall back to the best fitted
    period='none' model. Returns (model_name, fit_obj) or (None, None).
    """
    selected_spec = selected_fit["spec"]
    target = dict(selected_spec)
    target["period"] = "none"
    target.pop("name", None)
    for name, obj in fit_objects.items():
        sp = obj["spec"]
        if (
            sp.get("period") == "none"
            and sp.get("memory") == target.get("memory")
            and sp.get("ratio") == target.get("ratio")
            and sp.get("damping") == target.get("damping")
            and sp.get("current") == target.get("current")
        ):
            return name, obj
    no_period_models = [m for m, obj in fit_objects.items() if obj["spec"].get("period") == "none"]
    if not no_period_models:
        return None, None
    sub = summary.loc[summary["model"].isin(no_period_models)].sort_values("wrmse")
    if sub.empty:
        return None, None
    name = str(sub.iloc[0]["model"])
    return name, fit_objects[name]


def compute_row_predictions(df, fit_obj, memory_col):
    out = df.copy()
    x, p, T = out["signed_dh"].to_numpy(float), out[memory_col].to_numpy(float), out["period_years"].to_numpy(float)
    out["rho_pred"] = rho_model(fit_obj["theta"], fit_obj["spec"], fit_obj["param_names"], x, p, T)
    out["rho_raw"] = out[rho_col].astype(float)

    out["rho_resid"] = out["rho_raw"] - out["rho_pred"]
    out["rho_mean_removed"] = out["rho_resid"]
    comps = component_values(fit_obj["theta"], fit_obj["spec"], fit_obj["param_names"], x, p, T)
    for k, v in comps.items():
        out[k] = v
    out["abs_signed_dh"] = np.abs(out["signed_dh"])
    out["log10_area"] = np.log10(out[area_col].astype(float))
    out["period_plot_years"] = _map_periods_to_display_groups(out["period_years"].to_numpy(float))
    return out.loc[np.isfinite(out["rho_resid"])].copy()


def summarize_period_factor(df, factor_col, factor_name, edges=None, categorical=False):
    period_col = "period_plot_years" if "period_plot_years" in df.columns else "period_years"
    d = df.loc[np.isfinite(df["rho_resid"]) & np.isfinite(df[weight_col]) & (df[weight_col] > 0)].copy()
    d = d.loc[np.isfinite(d[period_col])].copy()

    if categorical:
        if factor_col == "period_years" and "period_plot_years" in d.columns:
            d["_factor_bin"] = d["period_plot_years"].astype(str)
            factor_source = "period_plot_years"
        else:
            d["_factor_bin"] = d[factor_col].astype(str)
            factor_source = factor_col
    else:
        d = d.loc[np.isfinite(d[factor_col])].copy()
        d["_factor_bin"] = pd.cut(d[factor_col], bins=edges, labels=False, include_lowest=True)
        d = d.dropna(subset=["_factor_bin"]).copy()
        d["_factor_bin"] = d["_factor_bin"].astype(int)
        factor_source = factor_col

    rows = []
    for (T, fb), g in d.groupby([period_col, "_factor_bin"], sort=True):
        w = g[weight_col].to_numpy(float)
        n_eff = weighted_effective_n(w)
        if len(g) < MIN_COUNT_DIAG_BIN or not np.isfinite(n_eff) or n_eff < MIN_NEFF_DIAG_BIN:
            continue
        center = float(T) if categorical else weighted_mean(g[factor_source].to_numpy(float), w)
        raw_vals = g["rho_raw"].to_numpy(float)
        resid_vals = g["rho_resid"].to_numpy(float)
        resid_std = weighted_std(resid_vals, w)
        resid_mean = weighted_mean(resid_vals, w)
        if "z_after" in g.columns:
            z_vals = g["z_after"].to_numpy(float)
            z_after_mean = weighted_mean(z_vals, w)
            z_after_std = weighted_std(z_vals, w)
        else:
            z_after_mean = resid_mean / resid_std if np.isfinite(resid_std) and resid_std > 0 else np.nan
            z_after_std = np.nan
        rows.append({
            "factor": factor_name,
            "factor_col": factor_col,
            "period_years": float(T),
            "factor_bin": str(fb),
            "factor_center_w": center,
            "n": len(g),
            "n_eff": n_eff,
            "weight_sum": np.nansum(w),
            "raw_mean_w": weighted_mean(raw_vals, w),
            "raw_std_w": weighted_std(raw_vals, w),
            "resid_mean_w": resid_mean,
            "resid_std_w": resid_std,
            "resid_abs_mean_w": weighted_mean(np.abs(resid_vals), w),
            "z_after_mean_w": z_after_mean,
            "z_after_std_w": z_after_std,
            "period_component_mean_w": weighted_mean(g["period_component"].to_numpy(float), w),
            "current_component_mean_w": weighted_mean(g["current_component"].to_numpy(float), w),
        })
    return pd.DataFrame(rows)

def summarize_all_factors(d_pred, memory_col):
    signed_edges = make_signed_log_dh_edges(d_pred["signed_dh"].to_numpy(float))
    abs_edges = make_abs_dh_edges(d_pred["abs_signed_dh"].to_numpy(float))
    mem_edges = make_quantile_edges(d_pred[memory_col].to_numpy(float), d_pred[weight_col].to_numpy(float), MEMORY_N_QUANTILE_BINS_FOR_FIT)
    area_edges = make_quantile_edges(d_pred["log10_area"].to_numpy(float), d_pred[weight_col].to_numpy(float), AREA_N_QUANTILE_BINS)
    parts = [summarize_period_factor(d_pred, "period_years", "Period length", categorical=True), summarize_period_factor(d_pred, "signed_dh", "Current signed dh", signed_edges), summarize_period_factor(d_pred, "abs_signed_dh", "Absolute current dh", abs_edges), summarize_period_factor(d_pred, memory_col, "Past elevation change", mem_edges), summarize_period_factor(d_pred, "log10_area", "Area", area_edges)]
    return pd.concat([p for p in parts if len(p)], ignore_index=True)


def summarize_residual_memory_scan(df_resid, all_rows):
    cumulative_rows, rate_rows = [], []
    for W in RESIDUAL_MEMORY_WINDOWS_TO_SCAN:
        dw = attach_exact_past_window(df_resid, all_rows, W, out_col=f"scan_past{W}_dh")
        mem_col = f"scan_past{W}_dh"; has_col = f"has_{mem_col}"
        dw = dw.loc[dw[has_col] & np.isfinite(dw[mem_col])].copy()
        if len(dw) == 0: continue
        dw[f"scan_past{W}_rate"] = dw[mem_col].astype(float) / float(W)
        for predictor_col, kind, rows in [(mem_col, "cumulative", cumulative_rows), (f"scan_past{W}_rate", "rate", rate_rows)]:
            edges = make_quantile_edges(dw[predictor_col].to_numpy(float), dw[weight_col].to_numpy(float), MEMORY_N_QUANTILE_BINS)
            if edges is None: continue
            tmp = dw.copy(); tmp["memory_bin"] = pd.cut(tmp[predictor_col], bins=edges, labels=False, include_lowest=True)
            tmp = tmp.dropna(subset=["memory_bin"]).copy(); tmp["memory_bin"] = tmp["memory_bin"].astype(int)
            for mb, g in tmp.groupby("memory_bin", sort=True):
                w = g[weight_col].to_numpy(float); n_eff = weighted_effective_n(w)
                support_ok = bool(len(g) >= MEMORY_MIN_COUNT_BIN and np.isfinite(n_eff) and n_eff >= MEMORY_MIN_NEFF_BIN)
                rows.append({"memory_window_years": int(W), "predictor_kind": kind, "memory_bin": int(mb), "memory_center_w": weighted_mean(g[predictor_col].to_numpy(float), w), "n": len(g), "n_eff": n_eff, "weight_sum": np.nansum(w), "resid_mean_w": weighted_mean(g["rho_resid"].to_numpy(float), w), "resid_std_w": weighted_std(g["rho_resid"].to_numpy(float), w), "resid_abs_mean_w": weighted_mean(np.abs(g["rho_resid"].to_numpy(float)), w), "support_ok": support_ok})
    cum, rate = pd.DataFrame(cumulative_rows), pd.DataFrame(rate_rows)
    return cum, _score_memory_scan(cum), rate, _score_memory_scan(rate)


def _score_memory_scan(summary):
    scores=[]
    if summary.empty: return pd.DataFrame()
    for W, g in summary.loc[summary["support_ok"]].groupby("memory_window_years", sort=True):
        w = g["weight_sum"].to_numpy(float); m = g["resid_mean_w"].to_numpy(float)
        ok = np.isfinite(w) & (w > 0) & np.isfinite(m)
        if np.any(ok): scores.append({"memory_window_years": int(W), "predictor_kind": str(g["predictor_kind"].iloc[0]), "wrms_binned_mean_residual": float(np.sqrt(np.sum(w[ok]*m[ok]**2)/np.sum(w[ok]))), "wmean_abs_binned_mean_residual": float(np.sum(w[ok]*np.abs(m[ok]))/np.sum(w[ok])), "range_binned_mean_residual": float(np.nanmax(m[ok])-np.nanmin(m[ok])), "n_bins": int(np.sum(ok)), "weight_sum": float(np.sum(w[ok]))})
    return pd.DataFrame(scores)

# =============================================================================
# Plotting
# =============================================================================


def _safe_name(name):
    """Return a filesystem-safe model name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))



def _aggregate_target_pair(target, model, x_col, x_bin_col, group_col, group_bin_col, fit_obj=None):
    """
    Collapse the 3D target table to a 2D plotting table.

    Important separation:
      - rho_obs_w is support-weighted because it represents the binned observed
        effective density.
      - rho_pred_w is retained for diagnostic compatibility: support-weighted
        average of target-cell predictions.
      - rho_pred_center is the model evaluated once at the aggregated predictor
        coordinates of the displayed bin. This is the fitted model value to use
        when the plotted quantity should not average fitted outputs.
    """
    pred_col = f"rho_pred_{model}"
    required = [x_col, x_bin_col, group_col, group_bin_col, "rho_mean_w", pred_col, "weight_sum", "support_ok"]
    tt = target.loc[target["support_ok"]].dropna(subset=[c for c in required if c in target.columns]).copy()
    if tt.empty:
        return pd.DataFrame()

    rows = []
    for (gb, xb), g in tt.groupby([group_bin_col, x_bin_col], sort=True):
        w = g["weight_sum"].to_numpy(float)
        if not np.isfinite(w).any() or np.nansum(w) <= 0:
            continue

        current_center = weighted_mean(g["current_center_w"].to_numpy(float), w) if "current_center_w" in g.columns else np.nan
        memory_center = weighted_mean(g["memory_center_w"].to_numpy(float), w) if "memory_center_w" in g.columns else np.nan
        period_center = weighted_mean(g["period_years"].to_numpy(float), w) if "period_years" in g.columns else np.nan

        row = {
            "group_bin": gb,
            "x_bin": xb,
            "x_center_w": weighted_mean(g[x_col].to_numpy(float), w),
            "group_center_w": weighted_mean(g[group_col].to_numpy(float), w),
            "current_center_w": current_center,
            "memory_center_w": memory_center,
            "period_center_w": period_center,
            "rho_obs_w": weighted_mean(g["rho_mean_w"].to_numpy(float), w),
            "rho_pred_w": weighted_mean(g[pred_col].to_numpy(float), w),
            "weight_sum": np.nansum(w),
            "n_cells": len(g),
        }

        if fit_obj is not None and np.isfinite(current_center) and np.isfinite(memory_center) and np.isfinite(period_center):
            row["rho_pred_center"] = float(rho_model(
                fit_obj["theta"],
                fit_obj["spec"],
                fit_obj["param_names"],
                np.array([current_center], dtype=float),
                np.array([memory_center], dtype=float),
                np.array([period_center], dtype=float),
            )[0])
        else:
            row["rho_pred_center"] = row["rho_pred_w"]

        rows.append(row)

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["group_bin", "x_bin"]).reset_index(drop=True)
    return out

def _plot_mean_fit_pair(plot_df, out_png, title, x_label, color_label, signed_x=True):
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 5.8), constrained_layout=True)

    cmap = plt.get_cmap(PLOT_CMAP)
    norm = mpl.colors.Normalize(
        vmin=np.nanmin(plot_df["group_center_w"]),
        vmax=np.nanmax(plot_df["group_center_w"]),
    )
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    colors = cmap(norm(plot_df["group_center_w"].to_numpy(float)))
    ax.scatter(
        plot_df["x_center_w"],
        plot_df["rho_obs_w"],
        s=28,
        c=colors,
        alpha=0.70,
        marker="o",
        linewidths=0,
        label="Observed bin mean",
    )

    for gb, sub in plot_df.groupby("group_bin", sort=True):
        sub = sub.sort_values("x_center_w")
        if len(sub) < 2:
            continue
        group_val = weighted_mean(sub["group_center_w"].to_numpy(float), sub["weight_sum"].to_numpy(float))
        col = cmap(norm(group_val))
        if signed_x:
            for mask in [sub["x_center_w"] < 0, sub["x_center_w"] > 0]:
                ss = sub.loc[mask].sort_values("x_center_w")
                if len(ss) < 2:
                    continue
                ax.plot(ss["x_center_w"], ss["rho_pred_w"], lw=2.0, color=col, alpha=0.95)
        else:
            ax.plot(sub["x_center_w"], sub["rho_pred_w"], lw=2.0, color=col, alpha=0.95)

    ax.axhline(RHO_ICE_FIXED, color="black", lw=1, ls="--")
    if signed_x:
        ax.axvline(0, color="black", lw=1)
        set_signed_log_xaxis(ax, plot_df["x_center_w"].to_numpy(float))
    ax.set_ylim(-2000, 3000)
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"Mean $\rho_{\Delta V}$")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.plot([], [], color="black", lw=2.0, label="Model fit")
    ax.legend(frameon=False, fontsize=8)

    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(color_label)
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_mean_fit_by_model(target, summary):
    """
    For each fitted candidate model, plot observed target-bin means and fitted values:
      1. current dh vs rho, colored by past elevation change rate;
      2. current dh vs rho, colored by period length.
    """
    out_mean_fit_dir.mkdir(parents=True, exist_ok=True)
    for model in summary["model"].tolist():
        safe = _safe_name(model)

        by_memory = _aggregate_target_pair(
            target=target,
            model=model,
            x_col="current_center_w",
            x_bin_col="current_bin",
            group_col="memory_center_w",
            group_bin_col="memory_bin",
        )
        _plot_mean_fit_pair(
            by_memory,
            out_mean_fit_dir / f"{safe}_mean_fit_currentdh_colored_by_memorydh.png",
            title=f"Mean fit: current dh colored by past elevation change rate ({model})",
            x_label="Current signed dh (m)",
            color_label="Past elevation change rate (m yr$^{-1}$)",
            signed_x=True,
        )

        by_period = _aggregate_target_pair(
            target=target,
            model=model,
            x_col="current_center_w",
            x_bin_col="current_bin",
            group_col="period_years",
            group_bin_col="period_years",
        )
        _plot_mean_fit_pair(
            by_period,
            out_mean_fit_dir / f"{safe}_mean_fit_currentdh_colored_by_period.png",
            title=f"Mean fit: current dh colored by period length ({model})",
            x_label="Current signed dh (m)",
            color_label="Period length (yr)",
            signed_x=True,
        )


def summarize_signed_absdh_residual_std(df, resid_col):
    """Summarize mean-removed residual STD separately on negative/positive dh branches."""
    d = df.loc[
        np.isfinite(df[resid_col])
        & np.isfinite(df["signed_dh"])
        & np.isfinite(df[weight_col])
        & (df[weight_col] > 0)
    ].copy()
    if d.empty:
        return pd.DataFrame()

    edges = make_signed_log_dh_edges(d["signed_dh"].to_numpy(float))
    d["current_bin"] = pd.cut(d["signed_dh"], bins=edges, labels=False, include_lowest=True)
    d = d.dropna(subset=["current_bin"]).copy()
    d["current_bin"] = d["current_bin"].astype(int)

    rows = []
    for ib, g in d.groupby("current_bin", sort=True):
        w = g[weight_col].to_numpy(float)
        n_eff = weighted_effective_n(w)
        if len(g) < MIN_COUNT_DIAG_BIN or not np.isfinite(n_eff) or n_eff < MIN_NEFF_DIAG_BIN:
            continue
        r = g[resid_col].to_numpy(float)
        rows.append({
            "current_bin": int(ib),
            "current_center_w": weighted_mean(g["signed_dh"].to_numpy(float), w),
            "abs_center_w": weighted_mean(np.abs(g["signed_dh"].to_numpy(float)), w),
            "resid_std_w": weighted_std(r, w),
            "n": len(g),
            "n_eff": n_eff,
            "weight_sum": np.nansum(w),
        })
    return pd.DataFrame(rows)


def plot_sigma_fit_signed_currentdh(df, theta_sig, out_png):
    """
    Plot sigma_rho(|dh|) fitted on mean-removed residuals, with binned residual
    STD shown separately for negative and positive current dh.
    """
    tab = summarize_signed_absdh_residual_std(df, "rho_mean_removed")
    if tab.empty:
        return

    vals = tab["current_center_w"].to_numpy(float)
    max_abs = np.nanmax(np.abs(vals[np.isfinite(vals)]))
    xmin = max(CURRENT_DH_MIN_ABS_FOR_FIT, 1e-4)
    xmax = max(max_abs, xmin * 10)
    xpos = np.geomspace(xmin, xmax, 400)
    xneg = -xpos[::-1]

    fig, ax = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)

    neg = tab.loc[tab["current_center_w"] < 0].sort_values("current_center_w")
    pos = tab.loc[tab["current_center_w"] > 0].sort_values("current_center_w")
    if len(neg):
        ax.scatter(neg["current_center_w"], neg["resid_std_w"], s=35, alpha=0.8, label="Binned STD, negative dh")
    if len(pos):
        ax.scatter(pos["current_center_w"], pos["resid_std_w"], s=35, alpha=0.8, label="Binned STD, positive dh")

    ax.plot(xneg, sigma_model(theta_sig, np.abs(xneg)), lw=2.0, label=r"$\sigma_\rho(|\Delta h|)$, negative branch")
    ax.plot(xpos, sigma_model(theta_sig, xpos), lw=2.0, label=r"$\sigma_\rho(|\Delta h|)$, positive branch")

    ax.axvline(0, color="black", lw=1)
    set_signed_log_xaxis(ax, vals)
    ax.set_xlabel("Current signed dh (m)")
    ax.set_ylabel(r"STD of mean-removed residual")
    ax.set_title(r"Uncertainty model $\sigma_\rho(|\Delta h|)$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)



def plot_model_metric_bar(summary):
    """
    Candidate model intercomparison.

    Primary panel uses bin-STD-normalized WRMSE so high-variance near-zero
    current-dh bins do not dominate the visual comparison. Secondary panel keeps
    raw WRMSE for reference.
    """
    if summary.empty:
        return
    sub = summary.sort_values("nwrmse_absdh_std" if "nwrmse_absdh_std" in summary.columns else "wrmse").copy()
    y = np.arange(len(sub))
    height = max(6.0, 0.42 * len(sub) + 2.0)

    fig, axes = plt.subplots(1, 2, figsize=(18, height), constrained_layout=True, sharey=True)

    if "nwrmse_absdh_std" in sub.columns:
        axes[0].barh(y, sub["nwrmse_absdh_std"])
        axes[0].set_xlabel("WRMSE / rho STD(|dh| bin)")
        axes[0].set_title("STD-scaled error")
    else:
        axes[0].barh(y, sub["wrmse"])
        axes[0].set_xlabel("WRMSE")
        axes[0].set_title("Raw error")

    axes[1].barh(y, sub["wrmse"])
    axes[1].set_xlabel("WRMSE (kg m$^{-3}$)")
    axes[1].set_title("Raw error")

    for ax in axes:
        ax.set_yticks(y)
        ax.set_yticklabels(sub["model"].tolist(), fontsize=8)
        ax.invert_yaxis()
        ax.grid(alpha=0.25, axis="x")

    fig.suptitle("Candidate model comparison", fontsize=13)
    fig.savefig(out_metric_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_period_lines(ax, suball, periods, x_col, y_col, cmap, norm, split_signed=False):
    """Plot lines by period. If split_signed=True, draw negative and positive branches separately."""
    for T in periods:
        sub = suball.loc[suball["period_years"] == T].copy()
        if sub.empty or sub["factor_bin"].nunique() < MIN_VALID_CELLS_PER_PERIOD_LINE:
            continue
        sub = sub.sort_values(x_col)
        if split_signed:
            for mask in [sub[x_col] < 0, sub[x_col] > 0]:
                ss = sub.loc[mask].sort_values(x_col)
                if len(ss) < 2:
                    continue
                ax.plot(ss[x_col], ss[y_col], marker="o", lw=1.3, ms=4, color=cmap(norm(T)))
        else:
            ax.plot(sub[x_col], sub[y_col], marker="o", lw=1.3, ms=4, color=cmap(norm(T)))


def plot_before_after_factor(diag, factor_name, out_png, x_label, signed_log_x=False, log_x=False):
    """
    Supplementary-friendly before/after mean plot.

    Both panels use the same y extent, derived from the upper/raw-rho panel.
    For signed current-dh plots, negative and positive branches are plotted
    separately to avoid artificial line segments across zero.
    """
    suball = diag.loc[diag["factor"] == factor_name].copy()
    if suball.empty:
        return
    periods = np.sort(suball["period_years"].unique())
    if PERIOD_LENGTHS_TO_PLOT is not None:
        periods = np.array([p for p in periods if p in PERIOD_LENGTHS_TO_PLOT], float)

    cmap = plt.get_cmap(PLOT_CMAP)
    norm = mpl.colors.Normalize(vmin=np.nanmin(periods), vmax=np.nanmax(periods))
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])

    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), constrained_layout=True, sharex=True)

    finite_raw = suball["raw_mean_w"].to_numpy(float)
    finite_raw = finite_raw[np.isfinite(finite_raw)]
    if finite_raw.size:
        lo, hi = np.nanpercentile(finite_raw, [0.5, 99.5])
        pad = 0.08 * max(hi - lo, 1.0)
        ylim_raw = (lo - pad, hi + pad)
        span = ylim_raw[1] - ylim_raw[0]
        ylim_resid = (-0.5 * span, 0.5 * span)
    else:
        ylim_raw = None
        ylim_resid = None

    split_signed = factor_name == "Current signed dh"

    for ax, col, ylabel, title, ylim in [
        (axes[0], "raw_mean_w", "Raw mean rho", f"Before: raw rho vs {factor_name}", ylim_raw),
        (axes[1], "resid_mean_w", "Mean residual", f"After: rho - model vs {factor_name}", ylim_resid),
    ]:
        _plot_period_lines(ax, suball, periods, "factor_center_w", col, cmap, norm, split_signed=split_signed)
        ax.axhline(0, color="black", lw=1)
        if signed_log_x:
            ax.axvline(0, color="black", lw=1)
            set_signed_log_xaxis(ax, suball["factor_center_w"].to_numpy(float))
        if log_x:
            ax.set_xscale("log")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel(x_label)
    cbar = fig.colorbar(sm, ax=axes, location="right", fraction=0.03, pad=0.02)
    cbar.set_label("Period length (yr)")
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_std_before_after_factor(diag, factor_name, out_png, x_label, signed_log_x=False, log_x=False):
    """
    Supplementary STD plot.

    Upper panel: mean-removed raw rho STD within each plotted bin.
    Lower panel: STD of standardized residuals after correction, using the
    fitted sigma_rho(|dh|) model applied at row level before binning.
    """
    suball = diag.loc[diag["factor"] == factor_name].copy()
    if suball.empty or "raw_std_w" not in suball.columns:
        return
    periods = np.sort(suball["period_years"].unique())
    if PERIOD_LENGTHS_TO_PLOT is not None:
        periods = np.array([p for p in periods if p in PERIOD_LENGTHS_TO_PLOT], float)

    cmap = plt.get_cmap(PLOT_CMAP)
    norm = mpl.colors.Normalize(vmin=np.nanmin(periods), vmax=np.nanmax(periods))
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])

    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), constrained_layout=True, sharex=True)

    before_vals = suball["raw_std_w"].to_numpy(float)
    before_vals = before_vals[np.isfinite(before_vals)]
    ylim_raw = None
    if before_vals.size:
        ymax = np.nanpercentile(before_vals, 99.5) * 1.10
        ymax = max(ymax, 1.0)
        ylim_raw = (0.0, ymax)

    split_signed = factor_name == "Current signed dh"

    _plot_period_lines(axes[0], suball, periods, "factor_center_w", "raw_std_w", cmap, norm, split_signed=split_signed)
    axes[0].set_ylabel("Mean-removed raw rho STD")
    axes[0].set_title(f"Before: mean-removed raw rho STD vs {factor_name}")
    if ylim_raw is not None:
        axes[0].set_ylim(*ylim_raw)

    if "z_after_std_w" in suball.columns and np.isfinite(suball["z_after_std_w"]).any():
        _plot_period_lines(axes[1], suball, periods, "factor_center_w", "z_after_std_w", cmap, norm, split_signed=split_signed)
        axes[1].axhline(1.0, color="black", lw=1)
        axes[1].set_ylim(0.0, 2.0)
        axes[1].set_ylabel("STD of standardized residual")
        axes[1].set_title(f"After: standardized mean-removed residual STD vs {factor_name}")
    else:
        _plot_period_lines(axes[1], suball, periods, "factor_center_w", "resid_std_w", cmap, norm, split_signed=split_signed)
        axes[1].set_ylabel("Residual STD")
        axes[1].set_title(f"After: mean-removed residual STD vs {factor_name}")
        if ylim_raw is not None:
            axes[1].set_ylim(*ylim_raw)

    for ax in axes:
        if signed_log_x:
            ax.axvline(0, color="black", lw=1)
            set_signed_log_xaxis(ax, suball["factor_center_w"].to_numpy(float))
        if log_x:
            ax.set_xscale("log")
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel(x_label)
    cbar = fig.colorbar(sm, ax=axes, location="right", fraction=0.03, pad=0.02)
    cbar.set_label("Period length (yr)")
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_standardized_after_factor(diag, factor_name, out_png, x_label, signed_log_x=False, log_x=False):
    """
    Plot standardized mean residual after correction using y extent [-2, 2].
    This uses z_after_mean_w = binned mean residual / binned residual STD.
    """
    suball = diag.loc[diag["factor"] == factor_name].copy()
    if suball.empty or "z_after_mean_w" not in suball.columns:
        return
    periods = np.sort(suball["period_years"].unique())
    if PERIOD_LENGTHS_TO_PLOT is not None:
        periods = np.array([p for p in periods if p in PERIOD_LENGTHS_TO_PLOT], float)

    cmap = plt.get_cmap(PLOT_CMAP)
    norm = mpl.colors.Normalize(vmin=np.nanmin(periods), vmax=np.nanmax(periods))
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])

    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    split_signed = factor_name == "Current signed dh"
    _plot_period_lines(ax, suball, periods, "factor_center_w", "z_after_mean_w", cmap, norm, split_signed=split_signed)
    ax.axhline(0, color="black", lw=1)
    ax.axhline(2, color="black", lw=0.8, ls="--")
    ax.axhline(-2, color="black", lw=0.8, ls="--")
    if signed_log_x:
        ax.axvline(0, color="black", lw=1)
        set_signed_log_xaxis(ax, suball["factor_center_w"].to_numpy(float))
    if log_x:
        ax.set_xscale("log")
    ax.set_ylim(-2, 2)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Binned mean residual / binned residual STD")
    ax.set_title(f"After: standardized mean residual vs {factor_name}")
    ax.grid(alpha=0.25)
    cbar = fig.colorbar(sm, ax=ax, location="right", fraction=0.03, pad=0.02)
    cbar.set_label("Period length (yr)")
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_component_by_period(fit_obj, out_png, component_name, title):
    vals=np.array([0.5,1,2,5,10,20,50,100,200,500],float); xgrid=np.concatenate([-vals[::-1],vals]); periods=np.array([1,2,5,10,20],float); pgrid=np.zeros_like(xgrid)
    fig,ax=plt.subplots(figsize=(12,5.5),constrained_layout=True); cmap=plt.get_cmap(PLOT_CMAP); norm=mpl.colors.Normalize(vmin=np.nanmin(periods),vmax=np.nanmax(periods)); sm=mpl.cm.ScalarMappable(cmap=cmap,norm=norm); sm.set_array([])
    for T in periods:
        Tgrid=np.full_like(xgrid,T,float); comp=component_values(fit_obj["theta"], fit_obj["spec"], fit_obj["param_names"], xgrid, pgrid, Tgrid); y=comp.get(component_name)
        ax.plot(xgrid,y,marker="o",lw=1.3,color=cmap(norm(T)))
    ax.axhline(0,color="black",lw=1); ax.axvline(0,color="black",lw=1); set_signed_log_xaxis(ax,xgrid); ax.set_ylabel(component_name); ax.set_xlabel("Current signed dh (m)"); ax.set_title(title); ax.grid(alpha=0.25); fig.colorbar(sm,ax=ax).set_label("Period length (yr)"); fig.savefig(out_png,dpi=DPI,bbox_inches="tight"); plt.close(fig)




def summarize_period_current_diagnostics(df, value_col, current_edges):
    """Summarize a row-level value by period length and signed current dh."""
    d = df.loc[
        np.isfinite(df[value_col])
        & np.isfinite(df["signed_dh"])
        & np.isfinite(df[weight_col])
        & (df[weight_col] > 0)
    ].copy()
    d["current_bin"] = pd.cut(d["signed_dh"], bins=current_edges, labels=False, include_lowest=True)
    d = d.dropna(subset=["current_bin"]).copy()
    d["current_bin"] = d["current_bin"].astype(int)
    rows = []
    for (T, ib), g in d.groupby(["period_years", "current_bin"], sort=True):
        w = g[weight_col].to_numpy(float)
        n_eff = weighted_effective_n(w)
        if len(g) < MIN_COUNT_DIAG_BIN or not np.isfinite(n_eff) or n_eff < MIN_NEFF_DIAG_BIN:
            continue
        rows.append({
            "period_years": float(T),
            "current_bin": int(ib),
            "current_center_w": weighted_mean(g["signed_dh"].to_numpy(float), w),
            "value_mean_w": weighted_mean(g[value_col].to_numpy(float), w),
            "value_std_w": weighted_std(g[value_col].to_numpy(float), w),
            "n": len(g),
            "n_eff": n_eff,
            "weight_sum": np.nansum(w),
        })
    return pd.DataFrame(rows)


def _plot_period_current_panel(ax, suball, periods, cmap, norm, ycol="value_mean_w"):
    """Plot period-current lines split into negative and positive branches."""
    for T in periods:
        sub = suball.loc[suball["period_years"] == T].sort_values("current_center_w")
        if sub.empty or sub["current_bin"].nunique() < MIN_VALID_CELLS_PER_PERIOD_LINE:
            continue
        for mask in [sub["current_center_w"] < 0, sub["current_center_w"] > 0]:
            ss = sub.loc[mask].sort_values("current_center_w")
            if len(ss) < 2:
                continue
            ax.plot(ss["current_center_w"], ss[ycol], marker="o", lw=1.3, ms=4, color=cmap(norm(T)))



def _build_period_correction_compare_rowwise(df_full, df_no_period):
    """
    Build row-wise compare table for period-correction diagnostics.

    The full and no-period predictions are computed from the same filtered input
    rows, so row-wise pairing is safer than merging on float/date keys.
    """
    if df_no_period is None or len(df_no_period) == 0:
        return pd.DataFrame(columns=[
            "signed_dh", "period_years", weight_col,
            "rho_pred_full", "rho_pred_no_period",
            "resid_full", "resid_no_period",
            "effective_period_correction", "resid_reconstructed",
        ])

    a = df_full.reset_index(drop=True).copy()
    b = df_no_period.reset_index(drop=True).copy()
    n = min(len(a), len(b))
    if n == 0:
        return pd.DataFrame(columns=[
            "signed_dh", "period_years", weight_col,
            "rho_pred_full", "rho_pred_no_period",
            "resid_full", "resid_no_period",
            "effective_period_correction", "resid_reconstructed",
        ])

    a = a.iloc[:n].copy()
    b = b.iloc[:n].copy()

    compare = pd.DataFrame({
        "signed_dh": a["signed_dh"].to_numpy(float),
        "period_years": a["period_years"].to_numpy(float),
        weight_col: a[weight_col].to_numpy(float),
        "rho_pred_full": a["rho_pred"].to_numpy(float),
        "rho_pred_no_period": b["rho_pred"].to_numpy(float),
        "resid_full": a["rho_resid"].to_numpy(float) if "rho_resid" in a.columns else np.full(n, np.nan),
        "resid_no_period": b["rho_resid"].to_numpy(float) if "rho_resid" in b.columns else np.full(n, np.nan),
    })

    # Keep display-period grouping available for downstream main-figure summaries
    if "period_plot_years" in a.columns:
        compare["period_plot_years"] = a["period_plot_years"].to_numpy(float)
    else:
        compare["period_plot_years"] = _map_periods_to_display_groups(compare["period_years"].to_numpy(float))

    compare["effective_period_correction"] = compare["rho_pred_full"] - compare["rho_pred_no_period"]
    compare["resid_reconstructed"] = compare["resid_no_period"] - compare["effective_period_correction"]
    return compare


def plot_period_correction_performance(df_full, df_no_period, selected_name, no_period_name, out_png):
    """
    Plot no-period residuals, effective fitted correction, and full residuals.

    The effective fitted correction is defined as:
        correction = prediction_full - prediction_no_period
    so:
        full residual = no-period residual - correction

    This avoids sign ambiguity and includes any parameter adjustment between
    paired fits, not only the explicit period component from the full model.
    """
    if df_no_period is None or len(df_no_period) == 0:
        return

    current_edges = make_signed_log_dh_edges(df_full["signed_dh"].to_numpy(float))
    compare = _build_period_correction_compare_rowwise(df_full, df_no_period)

    nores = summarize_period_current_diagnostics(compare, "resid_no_period", current_edges)
    effcorr = summarize_period_current_diagnostics(compare, "effective_period_correction", current_edges)
    fullres = summarize_period_current_diagnostics(compare, "resid_full", current_edges)
    recon = summarize_period_current_diagnostics(compare, "resid_reconstructed", current_edges)

    nores["panel"] = "No-period residual: rho - model(no period)"
    effcorr["panel"] = "Effective fitted period correction: model(full) - model(no period)"
    fullres["panel"] = "Full residual: rho - model(full)"
    recon["panel"] = "Reconstructed full residual: no-period residual - correction"

    diag = pd.concat([nores, effcorr, fullres, recon], ignore_index=True)
    diag.to_csv(out_no_period_diag_csv, index=False)
    if diag.empty:
        return

    periods = np.sort(diag["period_years"].unique())
    if PERIOD_LENGTHS_TO_PLOT is not None:
        periods = np.array([p for p in periods if p in PERIOD_LENGTHS_TO_PLOT], dtype=float)
    cmap = plt.get_cmap(PLOT_CMAP)
    norm = mpl.colors.Normalize(vmin=np.nanmin(periods), vmax=np.nanmax(periods))
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])

    panels = [
        "No-period residual: rho - model(no period)",
        "Effective fitted period correction: model(full) - model(no period)",
        "Full residual: rho - model(full)",
    ]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10.5), constrained_layout=True, sharex=True)

    # Use same y extent across panels, derived from the no-period residual
    base_vals = nores["value_mean_w"].to_numpy(float)
    common_ylim = robust_symmetric_ylim(base_vals, min_halfspan=20) if np.isfinite(base_vals).any() else None

    for ax, panel in zip(axes, panels):
        suball = diag.loc[diag["panel"] == panel]
        _plot_period_current_panel(ax, suball, periods, cmap, norm)
        ax.axhline(0, color="black", lw=1)
        ax.axvline(0, color="black", lw=1)
        set_signed_log_xaxis(ax, diag["current_center_w"].to_numpy(float))
        if common_ylim is not None:
            ax.set_ylim(*common_ylim)
        ax.set_ylabel("Weighted mean")
        ax.set_title(panel)
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Current signed dh (m)")
    fig.suptitle(f"Period correction performance: full={selected_name}; no-period={no_period_name}", fontsize=12)
    cbar = fig.colorbar(sm, ax=axes, location="right", fraction=0.03, pad=0.02)
    cbar.set_label("Period length (yr)")
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_memory_scan(summary, score, out_png, out_score_png, predictor_kind):
    if summary.empty: return
    windows=np.sort(summary["memory_window_years"].unique()); cmap=plt.get_cmap(PLOT_CMAP); norm=mpl.colors.Normalize(vmin=np.nanmin(windows),vmax=np.nanmax(windows)); sm=mpl.cm.ScalarMappable(cmap=cmap,norm=norm); sm.set_array([])
    fig,ax=plt.subplots(figsize=(11.5,6.0),constrained_layout=True); vals=[]
    for W in windows:
        sub=summary.loc[(summary["memory_window_years"]==W)&(summary["support_ok"])].sort_values("memory_bin")
        if len(sub)<3: continue
        x=sub["memory_center_w"].to_numpy(float); y=sub["resid_mean_w"].to_numpy(float); vals.append(y[np.isfinite(y)]); ax.plot(x,y,marker="o",lw=1.2,ms=3.5,color=cmap(norm(W)))
    ax.axhline(0,color="black",lw=1); ax.axvline(0,color="black",lw=1); ax.set_xlabel("Cumulative elevation change over earlier window (m)" if predictor_kind=="cumulative" else "Past elevation change rate over window (m yr$^{-1}$)"); ax.set_ylabel("Weighted mean residual"); ax.set_title(f"Residual dependency on {predictor_kind} memory windows"); ax.grid(alpha=0.25)
    if vals: ax.set_ylim(*robust_symmetric_ylim(np.concatenate(vals),min_halfspan=20))
    fig.colorbar(sm,ax=ax).set_label("Memory window length (yr)"); fig.savefig(out_png,dpi=DPI,bbox_inches="tight"); plt.close(fig)
    if not score.empty:
        fig,ax=plt.subplots(figsize=(8,4.8),constrained_layout=True); score=score.sort_values("memory_window_years"); ax.plot(score["memory_window_years"],score["wrms_binned_mean_residual"],marker="o",lw=1.5); ax.set_xlabel("Memory window length (yr)"); ax.set_ylabel("WRMS of binned mean residual"); ax.set_title(f"Residual memory score ({predictor_kind})"); ax.grid(alpha=0.25); fig.savefig(out_score_png,dpi=DPI,bbox_inches="tight"); plt.close(fig)


def plot_memory_profile(profile):
    if profile.empty: return
    fig,ax=plt.subplots(figsize=(9.5,5),constrained_layout=True)
    for mode,sub in profile.groupby("memory_mode"):
        x=sub["memory_window_years"] if mode=="fixed_window" else sub["memory_tau_years"]
        ax.plot(x,sub["wrmse"],marker="o",lw=1.5,label=mode)
    ax.set_xlabel("Memory parameter"); ax.set_ylabel("Weighted RMSE"); ax.set_title("Memory-profile model fit score"); ax.grid(alpha=0.25); ax.legend(frameon=False); fig.savefig(out_memory_profile_png,dpi=DPI,bbox_inches="tight"); plt.close(fig)


# =============================================================================
# Main and supplementary paper figures
# =============================================================================






def _target_period_array():
    return np.asarray(PERIOD_LENGTHS_TO_PLOT if PERIOD_LENGTHS_TO_PLOT is not None else [], dtype=float)

def _map_periods_to_display_groups(values):
    values = np.asarray(values, dtype=float)
    targets = _target_period_array()
    out = np.full(values.shape, np.nan, dtype=float)
    if values.size == 0 or targets.size == 0:
        return out
    ok = np.isfinite(values)
    if np.any(ok):
        vv = values[ok][:, None]
        tt = targets[None, :]
        out[ok] = targets[np.argmin(np.abs(vv - tt), axis=1)]
    return out

def _filter_display_periods(periods):
    targets = _target_period_array()
    if targets.size:
        return targets
    periods = np.asarray(periods, dtype=float)
    periods = periods[np.isfinite(periods)]
    return np.sort(np.unique(periods))

def _pick_representative_memory_groups(values, targets=None):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values = np.sort(np.unique(values))
    if values.size == 0:
        return np.array([]), np.array([])
    if targets is None:
        targets = MAIN_MEMORY_TARGET_LABELS
    targets = np.asarray(targets, dtype=float)

    selected_vals = []
    selected_labels = []
    for t in targets:
        idx = int(np.argmin(np.abs(values - t)))
        v = float(values[idx])
        if v not in selected_vals:
            selected_vals.append(v)
            selected_labels.append(float(t))
    return np.asarray(selected_vals, dtype=float), np.asarray(selected_labels, dtype=float)

def _format_memory_label(v):
    if abs(v) >= 1:
        return f"{int(np.round(v))} m yr-1"
    return f"{v:g} m yr-1"

def _format_period_label(v):
    return f"{int(round(v))} yr" if np.isfinite(v) else "NA"

def _drop_small_abs_current_display_bin(df, xcol):
    """
    Remove the noisy smallest current-dh display bin (0.1 m, and anything close to it).
    """
    if df is None or len(df) == 0 or xcol not in df.columns:
        return df
    x = np.abs(df[xcol].to_numpy(float))
    return df.loc[~(x < 0.2)].copy()

def _dense_signed_curve_from_points(x, y, n=240, pad_frac=0.08):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & (x != 0)
    x = x[ok]
    y = y[ok]
    if len(x) < 2:
        return []
    curves = []
    for sign in (-1.0, 1.0):
        mask = np.sign(x) == sign
        if np.count_nonzero(mask) < 2:
            continue
        xb = np.abs(x[mask])
        yb = y[mask]
        ord_ = np.argsort(xb)
        xb = xb[ord_]
        yb = yb[ord_]
        t = np.log10(np.clip(xb, 1e-12, None))
        tu, idx = np.unique(t, return_index=True)
        yu = yb[idx]
        if len(tu) < 2:
            continue
        span = float(tu[-1] - tu[0])
        pad = pad_frac * span if span > 0 else 0.0
        tmin = max(np.log10(max(CURRENT_DH_MIN_ABS_FOR_FIT, 1e-6)), tu[0] - pad)
        tmax = tu[-1] + pad
        tg = np.linspace(tmin, tmax, n)
        yg = np.interp(tg, tu, yu)
        xg = (10.0 ** tg) * sign
        ord2 = np.argsort(xg)
        curves.append((xg[ord2], yg[ord2]))
    return curves



def _plot_group_scatter_and_dense_fit_from_support(ax, df, group_key_col, x_col, y_obs_col, y_fit_col,
                                                   cmap, norm, split_signed=True, scatter_size=26):
    """
    Plot observed points and interpolate fitted values on a dense support, separately
    for negative and positive current dh.
    """
    if df is None or df.empty:
        return
    for gval, sub in df.groupby(group_key_col, sort=True):
        try:
            gv = float(gval)
        except Exception:
            continue
        if not np.isfinite(gv):
            continue
        col = cmap(norm(gv))
        sub = sub.sort_values(x_col)
        ax.scatter(sub[x_col], sub[y_obs_col], s=scatter_size, color=col, alpha=0.75, edgecolors="none")
        x = sub[x_col].to_numpy(float)
        y = sub[y_fit_col].to_numpy(float)
        if split_signed:
            for sign in (-1.0, 1.0):
                mask = np.isfinite(x) & np.isfinite(y) & (np.sign(x) == sign)
                if np.count_nonzero(mask) < 2:
                    continue
                xb = np.abs(x[mask])
                yb = y[mask]
                ord_ = np.argsort(xb)
                xb = xb[ord_]
                yb = yb[ord_]
                xb_u, idx = np.unique(xb, return_index=True)
                yb_u = yb[idx]
                if len(xb_u) < 2:
                    continue
                xg = np.geomspace(np.nanmin(xb_u), np.nanmax(xb_u), 300)
                yg = np.interp(np.log10(xg), np.log10(xb_u), yb_u)
                ax.plot(sign * xg, yg, lw=2.0, color=col)
        else:
            ok = np.isfinite(x) & np.isfinite(y)
            if np.count_nonzero(ok) >= 2:
                xok = x[ok]
                yok = y[ok]
                ord_ = np.argsort(xok)
                xok = xok[ord_]
                yok = yok[ord_]
                xg = np.geomspace(np.nanmin(xok), np.nanmax(xok), 300)
                yg = np.interp(np.log10(xg), np.log10(xok), yok)
                ax.plot(xg, yg, lw=2.0, color=col)

def _plot_group_scatter_and_dense_fit(ax, df, group_col, x_col, y_obs_col, y_fit_col, cmap, norm, split_signed=True, scatter_size=26):
    if df is None or df.empty:
        return
    for gval, sub in df.groupby(group_col, sort=True):
        try:
            gv = float(gval)
        except Exception:
            continue
        if not np.isfinite(gv):
            continue
        col = cmap(norm(gv))
        sub = sub.sort_values(x_col)
        ax.scatter(sub[x_col], sub[y_obs_col], s=scatter_size, color=col, alpha=0.75, edgecolors="none")
        x = sub[x_col].to_numpy(float)
        y = sub[y_fit_col].to_numpy(float)
        if split_signed:
            for xg, yg in _dense_signed_curve_from_points(x, y):
                ax.plot(xg, yg, lw=2.0, color=col)
        else:
            ok = np.isfinite(x) & np.isfinite(y)
            if np.count_nonzero(ok) >= 2:
                ord_ = np.argsort(x[ok])
                ax.plot(x[ok][ord_], y[ok][ord_], lw=2.0, color=col)







def _main_signed_dense_xgrid():
    """Dense signed current-dh support for main fitted curves."""
    xpos = np.geomspace(MAIN_DENSE_X_MIN, MAIN_DENSE_X_MAX, int(MAIN_DENSE_X_N))
    return np.concatenate([-xpos[::-1], xpos])


def _weighted_period_distribution_for_memory_bin(target, memory_bin_value):
    """
    Return period values and normalized weights conditional on a given memory bin.
    This matches the left main-panel curves more closely to the corresponding
    memory-colored binned summaries than a global period distribution does.
    """
    if target is None or target.empty:
        return np.array([1.0]), np.array([1.0])
    tt = target.loc[target["support_ok"]].copy() if "support_ok" in target.columns else target.copy()
    if "memory_bin" not in tt.columns:
        return _weighted_period_distribution_from_target(target)
    tt = tt.loc[tt["memory_bin"] == memory_bin_value].copy()
    ok = np.isfinite(tt["period_years"]) & np.isfinite(tt["weight_sum"]) & (tt["weight_sum"] > 0)
    tt = tt.loc[ok]
    if tt.empty:
        return _weighted_period_distribution_from_target(target)
    g = tt.groupby("period_years", as_index=False)["weight_sum"].sum()
    T = g["period_years"].to_numpy(float)
    w = g["weight_sum"].to_numpy(float)
    w = w / np.sum(w)
    return T, w

def _plot_dense_line_through_zero(ax, xgrid, ygrid, color, lw=2.0):
    """
    Plot a continuous dense line across the full signed current-dh axis, including x=0.
    """
    xgrid = np.asarray(xgrid, dtype=float)
    ygrid = np.asarray(ygrid, dtype=float)
    ok = np.isfinite(xgrid) & np.isfinite(ygrid)
    if np.count_nonzero(ok) < 2:
        return
    xx = np.concatenate([xgrid[ok], np.array([0.0])])
    # Bridge zero by nearest finite mean when y varies
    if np.allclose(ygrid[ok], ygrid[ok][0], equal_nan=False):
        yy0 = ygrid[ok][0]
    else:
        # Take the mean of the closest negative/positive finite values
        neg = ygrid[(xgrid < 0) & np.isfinite(ygrid)]
        pos = ygrid[(xgrid > 0) & np.isfinite(ygrid)]
        if len(neg) and len(pos):
            yy0 = 0.5 * (neg[-1] + pos[0])
        elif len(neg):
            yy0 = neg[-1]
        elif len(pos):
            yy0 = pos[0]
        else:
            yy0 = np.nanmean(ygrid[ok])
    yy = np.concatenate([ygrid[ok], np.array([yy0])])
    order = np.argsort(xx)
    ax.plot(xx[order], yy[order], lw=lw, color=color)

def _weighted_period_distribution_from_target(target):
    """Return period values and normalized weights from the fitted target table."""
    if target is None or target.empty:
        return np.array([1.0]), np.array([1.0])
    tt = target.loc[target["support_ok"]].copy() if "support_ok" in target.columns else target.copy()
    ok = np.isfinite(tt["period_years"]) & np.isfinite(tt["weight_sum"]) & (tt["weight_sum"] > 0)
    tt = tt.loc[ok]
    if tt.empty:
        return np.array([1.0]), np.array([1.0])
    g = tt.groupby("period_years", as_index=False)["weight_sum"].sum()
    T = g["period_years"].to_numpy(float)
    w = g["weight_sum"].to_numpy(float)
    w = w / np.sum(w)
    return T, w

def _dense_mean_curve_memory(fit_obj, xgrid, memory_value, period_values, period_weights):
    """
    Dense mean model curve for a fixed past elevation change rate, averaged over the empirical
    period distribution.
    """
    y = np.zeros_like(xgrid, dtype=float)
    for T, wt in zip(period_values, period_weights):
        y += wt * rho_model(
            fit_obj["theta"],
            fit_obj["spec"],
            fit_obj["param_names"],
            xgrid,
            np.full_like(xgrid, float(memory_value), dtype=float),
            np.full_like(xgrid, float(T), dtype=float),
        )
    return y

def _dense_period_component_curve(fit_obj, xgrid, period_value):
    """Dense fitted additional period component for the right mean panel."""
    comps = component_values(
        fit_obj["theta"],
        fit_obj["spec"],
        fit_obj["param_names"],
        xgrid,
        np.zeros_like(xgrid, dtype=float),
        np.full_like(xgrid, float(period_value), dtype=float),
    )
    return comps["period_component"]

def _plot_signed_dense_line(ax, xgrid, ygrid, color, lw=2.0, alpha=1.0):
    xgrid = np.asarray(xgrid, dtype=float)
    ygrid = np.asarray(ygrid, dtype=float)
    for mask in [(xgrid < 0), (xgrid > 0)]:
        ok = mask & np.isfinite(ygrid)
        if np.count_nonzero(ok) >= 2:
            ax.plot(xgrid[ok], ygrid[ok], lw=lw, color=color, alpha=alpha)



def _main_style_legend_handles():
    from matplotlib.lines import Line2D
    return [
        Line2D([], [], linestyle="-", linewidth=2.4, color="black", label="Model fit"),
        Line2D([], [], linestyle="None", marker="o", markersize=6, color="black", label="Binned estimate"),
    ]


def _set_compact_linear_y_limits(ax, arrays, include_ice=True, min_pad=None):
    """Set compact linear y-limits from plotted finite values."""
    vals = []
    for arr in arrays:
        if arr is None:
            continue
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        if len(a):
            vals.append(a)
    if include_ice:
        vals.append(np.array([RHO_ICE_FIXED], dtype=float))
    if not vals:
        return
    yy = np.concatenate(vals)
    yy = yy[np.isfinite(yy)]
    if len(yy) == 0:
        return
    lo = float(np.nanpercentile(yy, 1.0))
    hi = float(np.nanpercentile(yy, 99.0))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return
    if hi <= lo:
        lo = float(np.nanmin(yy))
        hi = float(np.nanmax(yy))
    span = max(hi - lo, 1.0)
    pad = max(float(MAIN_MEAN_DYNAMIC_Y_MIN_PAD if min_pad is None else min_pad), 0.07 * span)
    ax.set_ylim(lo - pad, hi + pad)



def _effective_period_compare_table(d_pred_full, d_pred_no_period):
    """
    Build the same comparison quantity used in the dedicated period-correction diagnostic.

    Important: both prediction tables are computed from the same input row table.
    We therefore pair them row-wise after a light identity sanity check, rather
    than merging on float/date keys.
    """
    if d_pred_no_period is None or len(d_pred_no_period) == 0:
        return pd.DataFrame()

    a = d_pred_full.reset_index(drop=True).copy()
    b = d_pred_no_period.reset_index(drop=True).copy()
    n = min(len(a), len(b))
    a = a.iloc[:n].copy()
    b = b.iloc[:n].copy()

    # Light sanity check on identity columns where present
    id_cols = [c for c in ["rgiid", variant_col, "start_date", "end_date"] if c in a.columns and c in b.columns]
    if id_cols:
        mismatch = np.zeros(n, dtype=bool)
        for c in id_cols:
            av = a[c].to_numpy()
            bv = b[c].to_numpy()
            if np.issubdtype(np.asarray(av).dtype, np.number):
                mismatch |= ~(np.isclose(av.astype(float), bv.astype(float), equal_nan=True))
            else:
                mismatch |= (av != bv)
        keep = ~mismatch
        a = a.loc[keep].reset_index(drop=True)
        b = b.loc[keep].reset_index(drop=True)

    if len(a) == 0:
        return pd.DataFrame()

    compare = pd.DataFrame({
        "rgiid": a["rgiid"].to_numpy() if "rgiid" in a.columns else np.arange(len(a)),
        variant_col: a[variant_col].to_numpy() if variant_col in a.columns else np.repeat("", len(a)),
        "start_date": a["start_date"].to_numpy() if "start_date" in a.columns else np.arange(len(a), dtype=float),
        "end_date": a["end_date"].to_numpy() if "end_date" in a.columns else np.arange(len(a), dtype=float),
        "signed_dh": a["signed_dh"].to_numpy(float),
        "period_years": a["period_years"].to_numpy(float),
        "period_plot_years": a["period_plot_years"].to_numpy(float) if "period_plot_years" in a.columns else _map_periods_to_display_groups(a["period_years"].to_numpy(float)),
        weight_col: a[weight_col].to_numpy(float),
        "rho_pred_full": a["rho_pred"].to_numpy(float),
        "rho_pred_no_period": b["rho_pred"].to_numpy(float),
    })
    if "rho_resid" in a.columns:
        compare["resid_full"] = a["rho_resid"].to_numpy(float)
    if "rho_resid" in b.columns:
        compare["resid_no_period"] = b["rho_resid"].to_numpy(float)
    compare["effective_period_correction"] = compare["rho_pred_full"] - compare["rho_pred_no_period"]
    return compare

def _summarize_effective_period_correction(compare):
    """
    Summarize the effective fitted period correction for the main right panel.

    This uses a permissive row-level aggregation directly on the display periods
    so the panel does not collapse to empty when exact-period bins are sparse.
    """
    if compare is None or len(compare) == 0:
        return pd.DataFrame()

    d = compare.loc[
        np.isfinite(compare["effective_period_correction"])
        & np.isfinite(compare["signed_dh"])
        & np.isfinite(compare["period_years"])
        & np.isfinite(compare[weight_col])
        & (compare[weight_col] > 0)
    ].copy()
    if d.empty:
        return pd.DataFrame()

    d["group_bin"] = _map_periods_to_display_groups(d["period_years"].to_numpy(float))
    d = d.loc[np.isfinite(d["group_bin"])].copy()
    if d.empty:
        return pd.DataFrame()

    current_edges = make_signed_log_dh_edges(d["signed_dh"].to_numpy(float))
    d["current_bin"] = pd.cut(d["signed_dh"], bins=current_edges, labels=False, include_lowest=True)
    d = d.dropna(subset=["current_bin"]).copy()
    d["current_bin"] = d["current_bin"].astype(int)

    rows = []
    for (Tdisp, ib), g in d.groupby(["group_bin", "current_bin"], sort=True):
        w = g[weight_col].to_numpy(float)
        n_eff = weighted_effective_n(w)
        if len(g) < MAIN_RIGHTPANEL_MIN_COUNT or not np.isfinite(n_eff) or n_eff < MAIN_RIGHTPANEL_MIN_NEFF:
            continue
        xw = weighted_mean(g["signed_dh"].to_numpy(float), w)
        xw = float(_recenter_small_display_bin(np.array([xw], dtype=float))[0])
        rows.append({
            "group_bin": float(Tdisp),
            "x_bin": int(ib),
            "x_center_w": xw,
            "rho_obs_w": weighted_mean(g["effective_period_correction"].to_numpy(float), w),
            "weight_sum": np.nansum(w),
            "n_cells": len(g),
            "n_eff_sum": float(n_eff),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["group_bin", "x_bin"]).reset_index(drop=True)
    return out


def _memory_distribution_for_period(d_pred, memory_col, period_value):
    """
    Weighted empirical memory distribution for a displayed period group.

    Returns a small set of weighted-memory representative values and equal
    quadrature weights for dense period-correction curve evaluation.
    """
    if d_pred is None or len(d_pred) == 0 or memory_col not in d_pred.columns:
        return np.array([0.0]), np.array([1.0])

    if "period_plot_years" in d_pred.columns:
        per = d_pred["period_plot_years"].to_numpy(float)
    else:
        per = _map_periods_to_display_groups(d_pred["period_years"].to_numpy(float))

    sub = d_pred.loc[np.isfinite(per) & np.isclose(per, float(period_value))].copy()
    sub = sub.loc[
        np.isfinite(sub[memory_col])
        & np.isfinite(sub[weight_col])
        & (sub[weight_col] > 0)
    ].copy()
    if sub.empty:
        return np.array([0.0]), np.array([1.0])

    q = np.array([0.05, 0.20, 0.40, 0.60, 0.80, 0.95], dtype=float)
    vals = weighted_quantile(
        sub[memory_col].to_numpy(float),
        q,
        sample_weight=sub[weight_col].to_numpy(float),
    )
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.array([0.0]), np.array([1.0])

    # Deduplicate quantiles, preserving order
    vals = np.unique(np.round(vals, decimals=8))
    w = np.ones(len(vals), dtype=float) / float(len(vals))
    return vals, w

def _dense_effective_period_correction_curve(full_fit_obj, no_period_fit_obj, xgrid, period_value, memory_values, memory_weights):
    if no_period_fit_obj is None:
        return np.full_like(xgrid, np.nan, dtype=float)
    y = np.zeros_like(xgrid, dtype=float)
    for p, wt in zip(memory_values, memory_weights):
        yf = rho_model(full_fit_obj["theta"], full_fit_obj["spec"], full_fit_obj["param_names"],
                       xgrid, np.full_like(xgrid, float(p), dtype=float), np.full_like(xgrid, float(period_value), dtype=float))
        yn = rho_model(no_period_fit_obj["theta"], no_period_fit_obj["spec"], no_period_fit_obj["param_names"],
                       xgrid, np.full_like(xgrid, float(p), dtype=float), np.full_like(xgrid, float(period_value), dtype=float))
        y += wt * (yf - yn)
    return y

def _aggregate_period_adjusted_panel(d_pred):
    """
    Build the right-panel main-mean table showing the additional period dependency
    after removing the current+memory contribution.

    y_obs = rho_raw - (rho_pred - period_component)
    y_fit = period_component
    """
    period_col = "period_plot_years" if "period_plot_years" in d_pred.columns else "period_years"
    d = d_pred.loc[
        np.isfinite(d_pred["signed_dh"])
        & np.isfinite(d_pred[period_col])
        & np.isfinite(d_pred["rho_raw"])
        & np.isfinite(d_pred["rho_pred"])
        & np.isfinite(d_pred["period_component"])
        & np.isfinite(d_pred[weight_col])
        & (d_pred[weight_col] > 0)
    ].copy()
    if d.empty:
        return pd.DataFrame()

    d["y_obs_period_only"] = d["rho_raw"] - (d["rho_pred"] - d["period_component"])
    d["y_fit_period_only"] = d["period_component"]
    signed_edges = make_signed_log_dh_edges(d["signed_dh"].to_numpy(float))
    d["current_bin"] = pd.cut(d["signed_dh"], bins=signed_edges, labels=False, include_lowest=True)
    d = d.dropna(subset=["current_bin"]).copy()
    d["current_bin"] = d["current_bin"].astype(int)

    rows = []
    for (Tdisp, ib), g in d.groupby([period_col, "current_bin"], sort=True):
        w = g[weight_col].to_numpy(float)
        if not np.isfinite(w).any() or np.nansum(w) <= 0:
            continue
        xw = weighted_mean(g["signed_dh"].to_numpy(float), w)
        if np.abs(xw) < 0.2:
            continue
        rows.append({
            "group_bin": float(Tdisp),
            "group_center_w": float(Tdisp),
            "x_bin": int(ib),
            "x_center_w": xw,
            "rho_obs_w": weighted_mean(g["y_obs_period_only"].to_numpy(float), w),
            "rho_pred_w": weighted_mean(g["y_fit_period_only"].to_numpy(float), w),
            "weight_sum": np.nansum(w),
            "n_cells": len(g),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["group_bin", "x_bin"]).reset_index(drop=True)
    return out






def _recenter_small_display_bin(x):
    """
    For display only, force the [0.05, 0.2] and [-0.2, -0.05] bins to be shown
    at +/-0.1 m so that their weighted-center bias near zero does not distort
    the main figure.
    """
    x = np.asarray(x, dtype=float).copy()
    okp = (x >= MAIN_SMALLBIN_EDGE_LOW) & (x < MAIN_SMALLBIN_EDGE_HIGH)
    okn = (x <= -MAIN_SMALLBIN_EDGE_LOW) & (x > -MAIN_SMALLBIN_EDGE_HIGH)
    x[okp] = MAIN_SMALLBIN_DISPLAY_CENTER
    x[okn] = -MAIN_SMALLBIN_DISPLAY_CENTER
    return x



def _add_large_negative_inset(ax, scatter_triplets, line_dict):
    """
    Add a true zoomed inset for the negative large-dh branch using the exact same
    scatter and line arrays as the main panel. The inset shows only [-50, -10] m
    and is placed in the upper-right corner.
    """
    try:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    except Exception:
        return None

    axins = inset_axes(ax, width="38%", height="40%", loc="upper right", borderpad=1.0)
    yvals = []

    for xs, ys, col in scatter_triplets:
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        ok = np.isfinite(xs) & np.isfinite(ys) & (xs <= -10.0) & (xs >= -50.0)
        if np.any(ok):
            xabs = -xs[ok]
            axins.scatter(xabs, ys[ok], s=16, color=col, alpha=0.75, edgecolors="none")
            yvals.append(ys[ok])

    for _lab, (xl, yl, col) in line_dict.items():
        xl = np.asarray(xl, dtype=float)
        yl = np.asarray(yl, dtype=float)
        ok = np.isfinite(xl) & np.isfinite(yl) & (xl <= -10.0) & (xl >= -50.0)
        if np.any(ok):
            xabs = -xl[ok]
            order = np.argsort(xabs)
            axins.plot(xabs[order], yl[ok][order], lw=1.4, color=col)
            yvals.append(yl[ok][order])

    axins.set_xscale("log")
    axins.set_xlim(10.0, 50.0)
    axins.set_xticks([10, 20, 50])
    axins.set_xticklabels(["-10", "-20", "-50"])
    axins.grid(alpha=0.2)
    axins.axhline(RHO_ICE_FIXED, color="black", lw=0.8, ls="--")
    axins.tick_params(labelsize=7)
    axins.set_title("[-50, -10] m", fontsize=7, pad=2)

    if yvals:
        yy = np.concatenate([np.asarray(v, dtype=float) for v in yvals])
        yy = yy[np.isfinite(yy)]
        if len(yy):
            pad = 0.03 * max(20.0, float(np.nanmax(yy) - np.nanmin(yy)))
            axins.set_ylim(float(np.nanmin(yy) - pad), float(np.nanmax(yy) + pad))
    return axins



def _collapse_exact_period_diag_to_display(diag):
    """
    Collapse exact-period current-dh diagnostic summaries to the displayed period
    groups (e.g. 1, 2, 4, 7, 10, 15 yr), preserving the same current-dh binning.
    """
    if diag is None or len(diag) == 0:
        return pd.DataFrame()

    out = diag.copy()
    out["period_plot_years"] = _map_periods_to_display_groups(out["period_years"].to_numpy(float))
    out = out.loc[np.isfinite(out["period_plot_years"])].copy()
    if out.empty:
        return pd.DataFrame()

    rows = []
    for (Tdisp, ib), g in out.groupby(["period_plot_years", "current_bin"], sort=True):
        w = g["weight_sum"].to_numpy(float)
        if not np.isfinite(w).any() or np.nansum(w) <= 0:
            continue
        xw = weighted_mean(g["current_center_w"].to_numpy(float), w)
        xw = float(_recenter_small_display_bin(np.array([xw], dtype=float))[0])
        rows.append({
            "period_years": float(Tdisp),
            "current_bin": int(ib),
            "current_center_w": xw,
            "value_mean_w": weighted_mean(g["value_mean_w"].to_numpy(float), w),
            "value_std_w": weighted_mean(g["value_std_w"].to_numpy(float), w),
            "n": int(np.nansum(g["n"].to_numpy(float))) if "n" in g else len(g),
            "n_eff": float(np.nansum(g["n_eff"].to_numpy(float))) if "n_eff" in g else np.nan,
            "weight_sum": np.nansum(w),
        })
    return pd.DataFrame(rows)




def _plot_binned_points_and_interpolated_line(ax, x, y_points, y_line, color, marker_size=24, lw=2.0):
    """
    Plot binned points and a visually denser line interpolated from the same
    binned model values. This preserves the exact data used by the diagnostic
    plots, while avoiding a jagged low-resolution visual line in the main figure.
    """
    x = np.asarray(x, dtype=float)
    y_points = np.asarray(y_points, dtype=float)
    y_line = np.asarray(y_line, dtype=float)

    okp = np.isfinite(x) & np.isfinite(y_points)
    if np.any(okp):
        ax.scatter(x[okp], y_points[okp], s=marker_size, color=color, alpha=0.75, edgecolors="none")

    for sign in (-1.0, 1.0):
        ok = np.isfinite(x) & np.isfinite(y_line) & (np.sign(x) == sign)
        if np.count_nonzero(ok) < 2:
            continue
        xa = np.abs(x[ok])
        yl = y_line[ok]
        order = np.argsort(xa)
        xa = xa[order]
        yl = yl[order]
        xa_u, idx = np.unique(xa, return_index=True)
        yl_u = yl[idx]
        if len(xa_u) < 2:
            continue
        xg_abs = np.geomspace(np.nanmin(xa_u), np.nanmax(xa_u), 240)
        yg = np.interp(np.log10(xg_abs), np.log10(xa_u), yl_u)
        xg = sign * xg_abs
        order2 = np.argsort(xg)
        ax.plot(xg[order2], yg[order2], lw=lw, color=color, alpha=0.95)



def _plot_signed_dense_line_through_zero(ax, xgrid, ygrid, color, lw=1.8, alpha=0.95):
    """
    Plot a dense signed-x line and bridge across x=0 using the nearest finite
    negative/positive values so the line is visually continuous.
    """
    xgrid = np.asarray(xgrid, dtype=float)
    ygrid = np.asarray(ygrid, dtype=float)
    ok = np.isfinite(xgrid) & np.isfinite(ygrid)
    if np.count_nonzero(ok) < 2:
        return

    xn = xgrid[(xgrid < 0) & ok]
    yn = ygrid[(xgrid < 0) & ok]
    xp = xgrid[(xgrid > 0) & ok]
    yp = ygrid[(xgrid > 0) & ok]

    xx_parts = []
    yy_parts = []
    if len(xn):
        ordn = np.argsort(xn)
        xx_parts.append(xn[ordn]); yy_parts.append(yn[ordn])
    if len(xn) and len(xp):
        y0 = 0.5 * (yn[np.argmax(xn)] + yp[np.argmin(xp)])
        xx_parts.append(np.array([0.0])); yy_parts.append(np.array([y0]))
    elif len(xn):
        xx_parts.append(np.array([0.0])); yy_parts.append(np.array([yn[np.argmax(xn)]]))
    elif len(xp):
        xx_parts.append(np.array([0.0])); yy_parts.append(np.array([yp[np.argmin(xp)]]))
    if len(xp):
        ordp = np.argsort(xp)
        xx_parts.append(xp[ordp]); yy_parts.append(yp[ordp])

    xx = np.concatenate(xx_parts)
    yy = np.concatenate(yy_parts)
    order = np.argsort(xx)
    ax.plot(xx[order], yy[order], color=color, lw=lw, alpha=alpha)




def _main_dense_signed_support():
    xpos = np.geomspace(MAIN_DENSE_X_MIN, MAIN_DENSE_X_MAX, int(MAIN_DENSE_X_N))
    return np.concatenate([-xpos[::-1], xpos])

def _target_period_distribution_for_memory_group(target, memory_group):
    """
    Period distribution conditional on the same memory-bin group used for the
    binned diagnostic points.
    """
    tt = target.loc[target["support_ok"]].copy() if "support_ok" in target.columns else target.copy()
    if "memory_bin" in tt.columns:
        tt = tt.loc[tt["memory_bin"] == memory_group].copy()
    ok = np.isfinite(tt["period_years"]) & np.isfinite(tt["weight_sum"]) & (tt["weight_sum"] > 0)
    tt = tt.loc[ok]
    if tt.empty:
        return np.array([1.0]), np.array([1.0])
    g = tt.groupby("period_years", as_index=False)["weight_sum"].sum()
    T = g["period_years"].to_numpy(float)
    w = g["weight_sum"].to_numpy(float)
    sw = np.nansum(w)
    if not np.isfinite(sw) or sw <= 0:
        return np.array([1.0]), np.array([1.0])
    return T, w / sw


def _dense_mean_fit_for_group(fit_obj, plot_sub, xgrid):
    """
    Analytical surrogate curve for one displayed memory group.

    Uses the actual fitted predictor values represented by the plotted group:
      - current dh varies over dense xgrid;
      - past elevation change rate is fixed to the support-weighted group center;
      - period is fixed to the support-weighted group-period center.

    This is not a support-weighted average of fitted output values and is not an
    interpolation of bin predictions.
    """
    if plot_sub is None or len(plot_sub) == 0:
        return np.full_like(xgrid, np.nan, dtype=float)

    w = plot_sub["weight_sum"].to_numpy(float)
    if not np.isfinite(w).any() or np.nansum(w) <= 0:
        w = np.ones(len(plot_sub), dtype=float)

    if "memory_center_w" in plot_sub.columns:
        memory_value = weighted_mean(plot_sub["memory_center_w"].to_numpy(float), w)
    else:
        memory_value = weighted_mean(plot_sub["group_center_w"].to_numpy(float), w)

    if "period_center_w" in plot_sub.columns:
        period_value = weighted_mean(plot_sub["period_center_w"].to_numpy(float), w)
    elif "period_years" in plot_sub.columns:
        period_value = weighted_mean(plot_sub["period_years"].to_numpy(float), w)
    else:
        period_value = 1.0

    return rho_model(
        fit_obj["theta"],
        fit_obj["spec"],
        fit_obj["param_names"],
        xgrid,
        np.full_like(xgrid, float(memory_value), dtype=float),
        np.full_like(xgrid, float(period_value), dtype=float),
    )


def _memory_distribution_for_period_group(d_pred, memory_col, period_value):
    """
    Representative memory values for an analytical period-correction curve.

    This is only for displaying the analytical correction line. It avoids
    volume change weighting of fitted output values; the curve is an unweighted
    average over representative memory states within the displayed period group.
    """
    if d_pred is None or len(d_pred) == 0 or memory_col not in d_pred.columns:
        return np.array([0.0]), np.array([1.0])

    if "period_plot_years" in d_pred.columns:
        per = d_pred["period_plot_years"].to_numpy(float)
    else:
        per = _map_periods_to_display_groups(d_pred["period_years"].to_numpy(float))

    sub = d_pred.loc[np.isfinite(per) & np.isclose(per, float(period_value))].copy()
    sub = sub.loc[np.isfinite(sub[memory_col])].copy()
    if sub.empty:
        return np.array([0.0]), np.array([1.0])

    vals = np.nanquantile(sub[memory_col].to_numpy(float), [0.10, 0.30, 0.50, 0.70, 0.90])
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.array([0.0]), np.array([1.0])

    vals = np.unique(np.round(vals, decimals=8))
    w = np.ones(len(vals), dtype=float) / float(len(vals))
    return vals, w


def _dense_period_correction_for_group(full_fit_obj, no_period_fit_obj, d_pred, memory_col, period_value, xgrid):
    """
    Analytical effective period correction model(full)-model(no period), evaluated
    on dense current-dh support for the displayed period group.

    The line is analytical, not an interpolation of binned summaries. Memory is
    represented by an unweighted set of quantile states to avoid support-weighted
    averaging of fitted output values.
    """
    if no_period_fit_obj is None:
        return np.full_like(xgrid, np.nan, dtype=float)

    mem_vals, mem_w = _memory_distribution_for_period_group(d_pred, memory_col, period_value)
    y = np.zeros_like(xgrid, dtype=float)
    for p, wt in zip(mem_vals, mem_w):
        yf = rho_model(
            full_fit_obj["theta"], full_fit_obj["spec"], full_fit_obj["param_names"],
            xgrid,
            np.full_like(xgrid, float(p), dtype=float),
            np.full_like(xgrid, float(period_value), dtype=float),
        )
        yn = rho_model(
            no_period_fit_obj["theta"], no_period_fit_obj["spec"], no_period_fit_obj["param_names"],
            xgrid,
            np.full_like(xgrid, float(p), dtype=float),
            np.full_like(xgrid, float(period_value), dtype=float),
        )
        y += wt * (yf - yn)
    return y

def _plot_dense_signed_curve(ax, xgrid, ygrid, color, lw=2.0, alpha=0.95, connect_zero=False):
    xgrid = np.asarray(xgrid, dtype=float)
    ygrid = np.asarray(ygrid, dtype=float)
    if connect_zero:
        ok = np.isfinite(xgrid) & np.isfinite(ygrid)
        if np.count_nonzero(ok) < 2:
            return
        xx = [xgrid[ok]]
        yy = [ygrid[ok]]
        neg = ygrid[(xgrid < 0) & np.isfinite(ygrid)]
        pos = ygrid[(xgrid > 0) & np.isfinite(ygrid)]
        if len(neg) and len(pos):
            y0 = 0.5 * (neg[-1] + pos[0])
            xx.append(np.array([0.0]))
            yy.append(np.array([y0]))
        elif len(neg):
            xx.append(np.array([0.0])); yy.append(np.array([neg[-1]]))
        elif len(pos):
            xx.append(np.array([0.0])); yy.append(np.array([pos[0]]))
        xx = np.concatenate(xx)
        yy = np.concatenate(yy)
        order = np.argsort(xx)
        ax.plot(xx[order], yy[order], lw=lw, color=color, alpha=alpha)
    else:
        for mask in [xgrid < 0, xgrid > 0]:
            ok = mask & np.isfinite(ygrid)
            if np.count_nonzero(ok) >= 2:
                ax.plot(xgrid[ok], ygrid[ok], lw=lw, color=color, alpha=alpha)



def _signed_log_coordinate(x):
    """
    Monotonic plotting coordinate close to matplotlib symlog, used only for
    interpolation of fitted anchor values on dense display support.
    """
    x = np.asarray(x, dtype=float)
    lin = float(SIGNED_DH_SYMLOG_LINTHRESH)
    return np.sign(x) * np.log10(1.0 + np.abs(x) / lin)

def _dense_interpolate_signed_line(x_anchor, y_anchor, xgrid):
    """
    Smoothly interpolate fitted anchor values in signed-log coordinate,
    separately on each side of zero.

    The anchors remain the same support-weighted fitted quantities used by the
    diagnostic mean-fit plot; this function only changes the visual connection
    between anchors. A shape-preserving PCHIP spline is used when possible, with
    linear interpolation for sparse branches.
    """
    x_anchor = np.asarray(x_anchor, dtype=float)
    y_anchor = np.asarray(y_anchor, dtype=float)
    xgrid = np.asarray(xgrid, dtype=float)
    ygrid = np.full_like(xgrid, np.nan, dtype=float)

    try:
        from scipy.interpolate import PchipInterpolator
    except Exception:  # Pragma: no cover - Use interpolation if SciPy interpolation is unavailable
        PchipInterpolator = None

    for positive in [False, True]:
        if positive:
            amask = x_anchor > 0
            gmask = xgrid > 0
        else:
            amask = x_anchor < 0
            gmask = xgrid < 0

        ok = amask & np.isfinite(x_anchor) & np.isfinite(y_anchor)
        if np.count_nonzero(ok) < 2:
            continue

        xa = _signed_log_coordinate(x_anchor[ok])
        ya = y_anchor[ok]
        order = np.argsort(xa)
        xa = xa[order]
        ya = ya[order]

        # Collapse duplicate display coordinates by averaging their fitted values
        xu = np.unique(xa)
        if len(xu) < 2:
            continue
        yu = np.array([np.nanmean(ya[xa == x]) for x in xu], dtype=float)
        ok2 = np.isfinite(xu) & np.isfinite(yu)
        xu = xu[ok2]
        yu = yu[ok2]
        if len(xu) < 2:
            continue

        xg = _signed_log_coordinate(xgrid[gmask])
        inside = (xg >= np.nanmin(xu)) & (xg <= np.nanmax(xu))
        vals = np.full_like(xg, np.nan, dtype=float)
        if np.any(inside):
            if PchipInterpolator is not None and len(xu) >= 3:
                interp = PchipInterpolator(xu, yu, extrapolate=False)
                vals[inside] = interp(xg[inside])
            else:
                vals[inside] = np.interp(xg[inside], xu, yu)
        ygrid[gmask] = vals

    return ygrid



def _dense_grid_within_anchor_support(x_anchor, xgrid, min_abs=None):
    """
    Return a dense signed-dh grid restricted to the support covered by the
    displayed binned estimates, separately on each side of zero.

    This lets the main figure show a smooth analytical model curve without
    extrapolating the singular interval model into unsupported near-zero space.
    """
    x_anchor = np.asarray(x_anchor, dtype=float)
    xgrid = np.asarray(xgrid, dtype=float)
    out_parts = []
    min_abs_val = 0.0 if min_abs is None else float(min_abs)

    for positive in [False, True]:
        if positive:
            xa = x_anchor[np.isfinite(x_anchor) & (x_anchor > 0)]
            if len(xa) < 2:
                continue
            lo = max(float(np.nanmin(xa)), min_abs_val)
            hi = float(np.nanmax(xa))
            mask = np.isfinite(xgrid) & (xgrid > 0) & (xgrid >= lo) & (xgrid <= hi)
        else:
            xa = x_anchor[np.isfinite(x_anchor) & (x_anchor < 0)]
            if len(xa) < 2:
                continue
            lo = float(np.nanmin(xa))
            hi = min(float(np.nanmax(xa)), -min_abs_val)
            mask = np.isfinite(xgrid) & (xgrid < 0) & (xgrid >= lo) & (xgrid <= hi)

        if np.count_nonzero(mask) >= 2:
            out_parts.append(xgrid[mask])

    if not out_parts:
        return np.array([], dtype=float)
    return np.concatenate(out_parts)




def _plot_signed_anchor_line(ax, x_anchor, y_anchor, color, lw=2.0, alpha=0.95):
    """Plot model anchors separately on negative and positive signed-dh branches."""
    x_anchor = np.asarray(x_anchor, dtype=float)
    y_anchor = np.asarray(y_anchor, dtype=float)
    for mask in [x_anchor < 0, x_anchor > 0]:
        ok = mask & np.isfinite(x_anchor) & np.isfinite(y_anchor)
        if np.count_nonzero(ok) < 2:
            continue
        order = np.argsort(x_anchor[ok])
        ax.plot(x_anchor[ok][order], y_anchor[ok][order], lw=lw, color=color, alpha=alpha)


def _plot_mean_model_from_anchors(ax, x_anchor, y_anchor, color, xgrid, min_abs=None,
                                  mode="smooth_binned", lw=2.0, alpha=0.95):
    """
    Plot the main-paper mean model line using the same fitted quantities as the
    diagnostic mean-fit plot.

    mode='binned' draws the fitted model at displayed bin centers.
    mode='smooth_binned' draws a smooth shape-preserving interpolation through those same
    fitted anchors in signed-log x coordinates. This avoids evaluating the
    singular interval model at unsupported pointwise predictor combinations,
    which was the source of the distorted smooth curves.
    mode='analytical' is intentionally not implemented here; use the dedicated
    diagnostic functions for pointwise analytical checks.
    """
    x_anchor = np.asarray(x_anchor, dtype=float)
    y_anchor = np.asarray(y_anchor, dtype=float)
    ok = np.isfinite(x_anchor) & np.isfinite(y_anchor)
    if np.count_nonzero(ok) < 2:
        return np.array([], dtype=float), np.array([], dtype=float)

    if mode == "binned":
        _plot_signed_anchor_line(ax, x_anchor[ok], y_anchor[ok], color, lw=lw, alpha=alpha)
        return x_anchor[ok], y_anchor[ok]

    if mode != "smooth_binned":
        raise ValueError(f"Unknown main mean model-line mode: {mode}")

    xline = _dense_grid_within_anchor_support(x_anchor[ok], xgrid, min_abs=min_abs)
    if len(xline) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    yline = _dense_interpolate_signed_line(x_anchor[ok], y_anchor[ok], xline)
    _plot_dense_signed_curve(ax, xline, yline, color, lw=lw, alpha=alpha, connect_zero=False)
    return xline, yline


def _with_suffix_before_ext(path, suffix):
    path = Path(path)
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")

def _plot_mean_fit_pair_on_axis(ax, plot_df, cmap, norm, signed_x=True, y_limit=True,
                                fit_obj=None, target=None, dense_model=False):
    """
    Axis-level mean-fit plot.

    Dots are support-weighted observed bin means. Fitted anchor values remain
    model-at-bin-center values, but dense_model=True draws the analytical
    surrogate on dense signed-dh support using the actual group predictor values.
    """
    if plot_df is None or plot_df.empty:
        return

    color_col = "color_center_w" if "color_center_w" in plot_df.columns else "group_center_w"
    colors = cmap(norm(plot_df[color_col].to_numpy(float)))
    xdisp = _recenter_small_display_bin(plot_df["x_center_w"].to_numpy(float))
    ax.scatter(
        xdisp,
        plot_df["rho_obs_w"],
        s=28,
        c=colors,
        alpha=0.70,
        marker="o",
        linewidths=0,
    )

    xgrid = _main_dense_signed_support()
    pred_line_col = "rho_pred_center" if "rho_pred_center" in plot_df.columns else "rho_pred_w"

    for gb, sub in plot_df.groupby("group_bin", sort=True):
        sub = sub.sort_values("x_center_w").copy()
        if len(sub) < 2:
            continue
        color_val = weighted_mean(sub[color_col].to_numpy(float), sub["weight_sum"].to_numpy(float))
        col = cmap(norm(color_val))

        if dense_model and fit_obj is not None:
            ygrid = _dense_mean_fit_for_group(fit_obj, sub, xgrid)
            _plot_dense_signed_curve(ax, xgrid, ygrid, col, lw=2.0, alpha=0.95, connect_zero=False)
            continue

        x = _recenter_small_display_bin(sub["x_center_w"].to_numpy(float))
        y = sub[pred_line_col].to_numpy(float)
        if signed_x:
            for mask in [x < 0, x > 0]:
                xx = x[mask]
                yy = y[mask]
                ok = np.isfinite(xx) & np.isfinite(yy)
                if np.count_nonzero(ok) < 2:
                    continue
                order = np.argsort(xx[ok])
                ax.plot(xx[ok][order], yy[ok][order], lw=2.0, color=col, alpha=0.95)
        else:
            ok = np.isfinite(x) & np.isfinite(y)
            if np.count_nonzero(ok) >= 2:
                order = np.argsort(x[ok])
                ax.plot(x[ok][order], y[ok][order], lw=2.0, color=col, alpha=0.95)

    ax.axhline(RHO_ICE_FIXED, color="black", lw=1, ls="--")
    if signed_x:
        ax.axvline(0, color="black", lw=1)
        set_signed_log_xaxis(ax, np.concatenate([xdisp, np.array([-50.0, 50.0])]))
    if y_limit:
        ax.set_ylim(-2000, 3000)
    ax.grid(alpha=0.25)



def _undamped_current_period_component(fit_obj, x, T):
    """
    Undamped bracket-scale current + period correction:
        C_h(dh) + P_t(dt)
    not multiplied by R_h(dh).
    """
    pars = unpack(fit_obj["theta"], fit_obj["param_names"])
    x = np.asarray(x, dtype=float)
    T = np.asarray(T, dtype=float)
    return (
        current_component(pars, fit_obj["spec"]["current"], x)
        + period_component_form(pars, fit_obj["spec"]["period"], T)
    )

def _main_current_period_component_table(d_pred, fit_obj):
    """
    Build binned observations for the right MAIN mean panel on the same scale as
    C_h(dh) + P_t(dt).

    Since the full model is
        rho = rho_ice + R_h(dh) [M_mem + Q_mem + P_t + C_h],
    the corresponding observed bracket-scale estimate after removing the memory
    terms is
        [(rho - rho_ice) - (R_h M_mem + R_h Q_mem)] / R_h.
    """
    required = [
        "signed_dh", "period_plot_years", "rho_raw", "damping",
        "memory_component", "ratio_component", weight_col,
    ]
    missing = [c for c in required if c not in d_pred.columns]
    if missing:
        raise KeyError(f"_main_current_period_component_table missing columns: {missing}")

    d = d_pred.loc[
        np.isfinite(d_pred["signed_dh"])
        & np.isfinite(d_pred["period_plot_years"])
        & np.isfinite(d_pred["rho_raw"])
        & np.isfinite(d_pred["damping"])
        & (d_pred["damping"] > 0)
        & np.isfinite(d_pred["memory_component"])
        & np.isfinite(d_pred["ratio_component"])
        & np.isfinite(d_pred[weight_col])
        & (d_pred[weight_col] > 0)
    ].copy()
    if d.empty:
        return pd.DataFrame()

    d["current_period_obs"] = (
        d["rho_raw"].to_numpy(float)
        - RHO_ICE_FIXED
        - d["memory_component"].to_numpy(float)
        - d["ratio_component"].to_numpy(float)
    ) / d["damping"].to_numpy(float)

    d["current_period_model"] = _undamped_current_period_component(
        fit_obj,
        d["signed_dh"].to_numpy(float),
        d["period_years"].to_numpy(float),
    )

    current_edges = make_signed_log_dh_edges(d["signed_dh"].to_numpy(float))
    d["current_bin"] = pd.cut(d["signed_dh"], bins=current_edges, labels=False, include_lowest=True)
    d = d.dropna(subset=["current_bin"]).copy()
    d["current_bin"] = d["current_bin"].astype(int)

    rows = []
    for (Tdisp, ib), g in d.groupby(["period_plot_years", "current_bin"], sort=True):
        w = g[weight_col].to_numpy(float)
        n_eff = weighted_effective_n(w)
        if len(g) < MIN_COUNT_DIAG_BIN or not np.isfinite(n_eff) or n_eff < MIN_NEFF_DIAG_BIN:
            continue
        if not np.isfinite(w).any() or np.nansum(w) <= 0:
            continue
        rows.append({
            "period_years": float(Tdisp),
            "current_bin": int(ib),
            "current_center_w": weighted_mean(g["signed_dh"].to_numpy(float), w),
            "value_mean_w": weighted_mean(g["current_period_obs"].to_numpy(float), w),
            "model_center_w": weighted_mean(g["current_period_model"].to_numpy(float), w),
            "weight_sum": np.nansum(w),
            "n": len(g),
            "n_eff": n_eff,
        })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["period_years", "current_bin"]).reset_index(drop=True)
    return out

def _dense_current_period_component_for_period(fit_obj, xgrid, period_value):
    """
    Dense analytical C_h(dh)+P_t(dt) curve for one displayed period.
    """
    return _undamped_current_period_component(
        fit_obj,
        xgrid,
        np.full_like(xgrid, float(period_value), dtype=float),
    )


def _plot_period_current_panel_on_axis(ax, suball, periods, cmap, norm, ycol="value_mean_w",
                                       dense_model=False, full_fit_obj=None,
                                       no_period_fit_obj=None, d_pred=None,
                                       memory_col=None):
    """
    Right MAIN mean panel: undamped current + period correction C_h(dh)+P_t(dt).

    Dots are binned observed bracket-scale estimates after removing the memory
    terms and dividing by R_h(dh). Lines are the analytical undamped
    C_h(dh)+P_t(dt) curves.
    """
    xgrid = _main_dense_signed_support()
    for T in periods:
        sub = suball.loc[suball["period_years"] == T].sort_values("current_center_w").copy()
        if sub.empty or sub["current_bin"].nunique() < MIN_VALID_CELLS_PER_PERIOD_LINE:
            continue
        col = cmap(norm(T))
        ax.scatter(
            _recenter_small_display_bin(sub["current_center_w"].to_numpy(float)),
            sub[ycol].to_numpy(float),
            s=24,
            color=col,
            alpha=0.75,
            edgecolors="none",
        )

        if dense_model and full_fit_obj is not None:
            ygrid = _dense_current_period_component_for_period(full_fit_obj, xgrid, float(T))
            _plot_dense_signed_curve(ax, xgrid, ygrid, col, lw=1.8, alpha=0.95, connect_zero=True)
            continue

        x = _recenter_small_display_bin(sub["current_center_w"].to_numpy(float))
        y = sub["model_center_w"].to_numpy(float) if "model_center_w" in sub.columns else sub[ycol].to_numpy(float)
        for mask in [x < 0, x > 0]:
            xx = x[mask]
            yy = y[mask]
            ok = np.isfinite(xx) & np.isfinite(yy)
            if np.count_nonzero(ok) < 2:
                continue
            order = np.argsort(xx[ok])
            ax.plot(xx[ok][order], yy[ok][order], lw=1.8, color=col, alpha=0.95)



def _dense_mean_neutral_memory_for_period(fit_obj, xgrid, period_value):
    """
    Simplified mean model for neutral antecedent state:
        mu_rho(dh, dh_mem=0, dt)

    period_value may be either a scalar or an array with the same shape as xgrid.
    """
    xgrid = np.asarray(xgrid, dtype=float)
    period_arr = np.asarray(period_value, dtype=float)
    if period_arr.ndim == 0:
        period_arr = np.full_like(xgrid, float(period_arr), dtype=float)
    else:
        period_arr = np.broadcast_to(period_arr, xgrid.shape).astype(float)

    return rho_model(
        fit_obj["theta"],
        fit_obj["spec"],
        fit_obj["param_names"],
        xgrid,
        np.zeros_like(xgrid, dtype=float),
        period_arr,
    )

def _neutral_memory_mean_panel_table(d_pred, fit_obj, selected_model=None):
    """
    Build binned anchors for the neutral-memory mean panel using all row-level
    data transformed to the equivalent dh_mem = 0 state.

    The full mean model can be written as:
        rho = rho_ice + R(dh) [M_mem(dh, dh_mem) + P_t(dt) + C_h(dh)].

    After subtracting the fitted damped memory contribution,
        R(dh) M_mem(dh, dh_mem),
    each row is on the equivalent neutral-memory scale:
        rho_mem0 = rho - memory_component - ratio_component
                 = rho_ice + R(dh) [P_t(dt) + C_h(dh)] + residual.

    This avoids requiring observations with past elevation change rate close to zero.
    """
    required = [
        "signed_dh", "period_plot_years", "period_years", "rho_raw",
        "memory_component", "ratio_component", weight_col
    ]
    missing = [c for c in required if c not in d_pred.columns]
    if missing:
        raise KeyError(f"_neutral_memory_mean_panel_table missing required columns: {missing}")

    d = d_pred.loc[
        np.isfinite(d_pred["signed_dh"])
        & np.isfinite(d_pred["period_plot_years"])
        & np.isfinite(d_pred["period_years"])
        & np.isfinite(d_pred["rho_raw"])
        & np.isfinite(d_pred["memory_component"])
        & np.isfinite(d_pred["ratio_component"])
        & np.isfinite(d_pred[weight_col])
        & (d_pred[weight_col] > 0)
    ].copy()
    if d.empty:
        return pd.DataFrame()

    d["rho_mem0"] = (
        d["rho_raw"].to_numpy(float)
        - d["memory_component"].to_numpy(float)
        - d["ratio_component"].to_numpy(float)
    )
    d["rho_mem0_model"] = _dense_mean_neutral_memory_for_period(
        fit_obj,
        d["signed_dh"].to_numpy(float),
        d["period_years"].to_numpy(float),
    )

    current_edges = make_signed_log_dh_edges(d["signed_dh"].to_numpy(float))
    d["current_bin"] = pd.cut(d["signed_dh"], bins=current_edges, labels=False, include_lowest=True)
    d = d.dropna(subset=["current_bin"]).copy()
    d["current_bin"] = d["current_bin"].astype(int)

    rows = []
    for (Tdisp, xb), g in d.groupby(["period_plot_years", "current_bin"], sort=True):
        w = g[weight_col].to_numpy(float)
        if not np.isfinite(w).any() or np.nansum(w) <= 0:
            continue
        n_eff = weighted_effective_n(w)
        if len(g) < MIN_COUNT_DIAG_BIN or not np.isfinite(n_eff) or n_eff < MIN_NEFF_DIAG_BIN:
            continue

        dhc = weighted_mean(g["signed_dh"].to_numpy(float), w)
        Tc = weighted_mean(g["period_years"].to_numpy(float), w)
        obs = weighted_mean(g["rho_mem0"].to_numpy(float), w)
        mod = float(_dense_mean_neutral_memory_for_period(fit_obj, np.array([dhc], dtype=float), Tc)[0])

        rows.append({
            "period_years": float(Tdisp),
            "period_center_w": Tc,
            "current_bin": int(xb),
            "current_center_w": dhc,
            "rho_obs_w": obs,
            "rho_pred_center": mod,
            "weight_sum": np.nansum(w),
            "n": len(g),
            "n_eff": n_eff,
        })

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["period_years", "current_bin"]).reset_index(drop=True)
    return out


def _filter_main_mean_right_panel_target_to_period(target, target_year=5.0):
    """Return a target table restricted to the period slice used in main mean panel b.
    Prefer an exact period_years match; otherwise use the closest available period_center_w/value.
    """
    if target is None or len(target) == 0:
        return target
    df = target.copy()
    # Exact period-year slice when available
    for col in ["period_years", "period_center_w"]:
        if col in df.columns:
            vals = df[col].to_numpy(float)
            finite = vals[np.isfinite(vals)]
            if finite.size == 0:
                continue
            if np.any(np.isclose(finite, float(target_year), atol=1e-9, rtol=0.0)):
                return df.loc[np.isclose(df[col].to_numpy(float), float(target_year), atol=1e-9, rtol=0.0)].copy()
    # Default: closest available period slice
    for col in ["period_center_w", "period_years"]:
        if col in df.columns:
            vals = df[col].to_numpy(float)
            finite = vals[np.isfinite(vals)]
            if finite.size == 0:
                continue
            u = np.unique(finite)
            chosen = float(u[np.argmin(np.abs(u - float(target_year)))])
            return df.loc[np.isclose(df[col].to_numpy(float), chosen, atol=1e-9, rtol=0.0)].copy()
    return df

def _normality_stage_arrays(df, value_col):
    vals = df[value_col].to_numpy(float)
    w = df[weight_col].to_numpy(float)
    vals, w = _valid_xy(vals, w)
    return vals, w


def _robust_normality_sample(values, weights, qlo=0.005, qhi=0.995):
    """
    Return the central weighted distribution used for robust normality metrics.

    The robust metrics are computed on the same central distribution displayed
    in the histogram, so extreme weighted outliers do not dominate the moments.
    """
    values, weights = _valid_xy(values, weights)
    if values.size == 0:
        return values, weights, np.nan, np.nan
    lo, hi = weighted_quantile(values, [qlo, qhi], weights)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return values, weights, np.nanmin(values), np.nanmax(values)
    keep = (values >= lo) & (values <= hi)
    return values[keep], weights[keep], float(lo), float(hi)


def _normality_metrics(values, weights):
    # Mean, standard deviation and excess kurtosis use the full
    # Volume change weighted distribution. Skewness is reported as Bowley
    # Quantile skewness to avoid domination by a few extreme outliers
    mu = weighted_mean(values, weights)
    sd = weighted_std(values, weights)
    sk = weighted_quantile_skewness(values, weights)
    ek = weighted_excess_kurtosis(values, weights)
    return mu, sd, sk, ek


def _plot_weighted_hist_with_normal(ax, values, weights, x_label, row_label, show_legend=False):
    v_rob, w_rob, q_lo, q_hi = _robust_normality_sample(values, weights)
    mu, sd, sk, ek = _normality_metrics(values, weights)
    if len(v_rob) == 0:
        return
    bins = np.linspace(q_lo, q_hi, 45)
    ax.hist(v_rob, bins=bins, weights=w_rob, density=True, alpha=0.55, edgecolor='none', label='Volume change weighted\nhistogram')
    if np.isfinite(mu) and np.isfinite(sd) and sd > 0:
        xx = np.linspace(q_lo, q_hi, 500)
        ax.plot(xx, stats.norm.pdf(xx, loc=mu, scale=sd), lw=2.0, label='Normal fit')
    txt = (f"Mean = {mu:.2f}\n"
           f"STD = {sd:.2f}\n"
           f"Robust skewness = {sk:.3f}\n"
           f"Excess kurtosis = {ek:.3f}")
    ax.text(0.02, 0.50, txt, transform=ax.transAxes, ha='left', va='center', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.8', alpha=0.95))
    if show_legend:
        ax.legend(frameon=False, loc='upper right', fontsize=8)
    ax.set_ylabel(row_label)
    ax.set_xlabel(x_label)
    ax.grid(alpha=0.2)


def _short_normality_label(x_label):
    if 'z_' in x_label or 'z_{' in x_label:
        return r'of standardized residual $z_{\rho}$'
    if 'residual' in x_label or 'Residual' in x_label:
        return r'after subtracting $\mu_{\rho}$'
    return r'of raw $\rho_{\Delta V}$'


def _plot_weighted_qq(ax, values, weights, x_label):
    # Use the full weighted distribution for quantiles; only the displayed axes
    # Are clipped to +/-4 to make panels directly comparable
    mu, sd, _, _ = _normality_metrics(values, weights)
    if len(values) == 0 or not np.isfinite(sd) or sd <= 0:
        return
    probs = np.linspace(0.001, 0.999, 399)
    obs = weighted_quantile(values, probs, weights)
    theo = stats.norm.ppf(probs)
    obs_std = (obs - mu) / sd
    ok = np.isfinite(obs_std) & np.isfinite(theo)
    obs_std, theo = obs_std[ok], theo[ok]
    if len(obs_std) == 0:
        return
    ax.scatter(theo, obs_std, s=12, alpha=0.8, linewidths=0)
    ax.plot([-3, 3], [-3, 3], color='black', lw=1)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel('Theoretical normal quantiles')
    ax.set_ylabel('Observed quantiles\n' + _short_normality_label(x_label))
    ax.grid(alpha=0.2)


def plot_supp_normality_figure(d_pred, out_png):
    stages = [
        ('rho_raw', r'Effective density $\rho_{\Delta V}$ (kg m$^{-3}$)', r'Raw $\rho_{\Delta V}$'),
        ('rho_mean_removed', r'Mean-removed residual $\rho_{\Delta V} - \mu_{\rho}$ (kg m$^{-3}$)', r'After subtracting $\mu_{\rho}$'),
        ('z_after', r'Standardized residual $z_{\rho}$', r'After standardizing by $\mu_{\rho}$ and $\sigma_{\rho}$'),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(10.0, 10.4), constrained_layout=True)
    axes[0, 0].set_title('Normal fit to volume change weighted distribution')
    axes[0, 1].set_title('Q-Q plot')
    letters = iter(list('abcdef'))
    for i, (col, xlab, rowlab) in enumerate(stages):
        values, weights = _normality_stage_arrays(d_pred, col)
        _plot_weighted_hist_with_normal(axes[i, 0], values, weights, xlab, rowlab, show_legend=(i == 0))
        _plot_weighted_qq(axes[i, 1], values, weights, xlab)
        _add_panel_letter(axes[i, 0], next(letters))
        _add_panel_letter(axes[i, 1], next(letters))
    fig.savefig(out_png, dpi=DPI, bbox_inches='tight')
    plt.close(fig)

def plot_main_rho_mean_figure(target, selected_model, d_pred, fit_obj, d_pred_no_period, no_period_fit, memory_col, out_png, model_line_mode="smooth_binned", x_axis_mode="log"):
    """
    Main rho-mean figure.

    Panel a: simplified neutral-memory mean mu(dh, dh_mem=0, dt), colored by
    period length. This is the user-facing baseline dependence.

    Panel b: full mean model at a fixed 5-year period, colored by past elevation change rate.
    This shows the additional antecedent-state dependence.

    model_line_mode='analytical' draws the continuous fitted function along a
    representative predictor path, so curves can extend beyond the displayed
    bins. model_line_mode='smooth_binned' draws a smooth shape-preserving
    interpolation through diagnostic fitted anchors. model_line_mode='binned'
    draws the fitted anchors directly.
    """
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.5), constrained_layout=True)
    xgrid = _main_dense_signed_support()

    # ------------------------------------------------------------------
    # Panel a: simplified mean for neutral past elevation change rate = 0, colored by period
    # Use all data transformed to equivalent past elevation change rate = 0
    # ------------------------------------------------------------------
    neutral_tab = _neutral_memory_mean_panel_table(d_pred, fit_obj, selected_model)
    neutral_tab.to_csv(out_plot_dir / f"{run_label}_{main_suffix}_main_neutral_memory_mean_panel_summary.csv", index=False)

    if neutral_tab.empty:
        axes[0].text(0.5, 0.5, "No neutral-memory transformed data", ha="center", va="center", transform=axes[0].transAxes)
    else:
        neutral_plot = neutral_tab.loc[
            np.isfinite(neutral_tab["current_center_w"])
            & (np.abs(neutral_tab["current_center_w"].to_numpy(float)) >= MAIN_MEAN_LEFT_MIN_ABS_DH_DISPLAY)
        ].copy()
        periods = np.sort(neutral_plot["period_years"].unique()) if len(neutral_plot) else np.array([], dtype=float)
        if PERIOD_LENGTHS_TO_PLOT is not None:
            periods = np.array([p for p in periods if p in PERIOD_LENGTHS_TO_PLOT], dtype=float)

        if neutral_plot.empty or len(periods) == 0:
            axes[0].text(0.5, 0.5, "No neutral-memory bins with |dh| >= 1 m", ha="center", va="center", transform=axes[0].transAxes)
        else:
            cmap_per = plt.get_cmap(PLOT_CMAP_PERIOD)
            norm_per = mpl.colors.Normalize(vmin=np.nanmin(periods), vmax=np.nanmax(periods))
            y_for_limits = []

            # Plot the fitted curve at the same bin centers as the displayed
            # Neutral-memory binned estimates. This matches the dedicated
            # Diagnostic figures and avoids drawing the singular interval model
            # Into the intentionally omitted |dh| < 1 m region
            for T in periods:
                sub = neutral_plot.loc[neutral_plot["period_years"] == T].sort_values("current_center_w")
                if sub.empty:
                    continue
                col = cmap_per(norm_per(float(T)))
                xobs = sub["current_center_w"].to_numpy(float)
                yobs = sub["rho_obs_w"].to_numpy(float)
                yfit_anchor = sub["rho_pred_center"].to_numpy(float)

                axes[0].scatter(xobs, yobs, s=24, color=col, alpha=0.75, edgecolors="none")

                if model_line_mode == "analytical":
                    # Continuous fitted function for the panel with zero past elevation change rate
                    # Here dh_p = 0 exactly, so the quotient-like past elevation change rate term is zero
                    # And the function is finite as dh approaches zero
                    xfit = xgrid.copy()
                    yfit_plot = rho_model(
                        fit_obj["theta"], fit_obj["spec"], fit_obj["param_names"],
                        xfit,
                        np.zeros_like(xfit, dtype=float),
                        np.full_like(xfit, float(T), dtype=float),
                    )
                    _plot_dense_signed_curve(axes[0], xfit, yfit_plot, col, lw=1.8, alpha=0.95, connect_zero=False)
                else:
                    # Diagnostic-style model line: fitted values at the same binned
                    # Support as the binned estimates, optionally smoothed visually
                    xfit, yfit_plot = _plot_mean_model_from_anchors(
                        axes[0], xobs, yfit_anchor, col, xgrid,
                        min_abs=MAIN_MEAN_LEFT_MIN_ABS_DH_DISPLAY,
                        mode=model_line_mode, lw=1.8, alpha=0.95
                    )

                y_for_limits.append(yobs)
                y_for_limits.append(yfit_anchor)
                # For analytical curves, avoid letting the intentional near-zero
                # Behaviour set an unreadably large y extent. The curve is still
                # Drawn and clipped by the compact axis limits
                if model_line_mode == "analytical":
                    y_for_limits.append(yfit_plot[np.abs(xfit) >= MAIN_MEAN_LEFT_MIN_ABS_DH_DISPLAY])
                else:
                    y_for_limits.append(yfit_plot)

            axes[0].axhline(RHO_ICE_FIXED, color="black", lw=1, ls="--")
            axes[0].axvline(0, color="black", lw=1)
            set_signed_log_xaxis(axes[0], np.concatenate([neutral_plot["current_center_w"].to_numpy(float), np.array([-50.0, 50.0])]))
            axes[0].set_xlim(-50.0, 50.0)
            _set_compact_linear_y_limits(axes[0], y_for_limits, include_ice=True)
            axes[0].set_xlabel("Current dh (m)")
            axes[0].set_ylabel(r"Mean $\rho_{\Delta V}$ for $\Delta h^{\rm p}=0$ (kg m$^{-3}$)")
            axes[0].set_title(r"Current and period dependence")
            axes[0].grid(alpha=0.25)

            color_handles = [
                Line2D([], [], marker="o", ms=5, lw=1.8, color=cmap_per(norm_per(float(v))),
                       label=_format_period_label(float(v)))
                for v in periods
            ]
            leg1 = axes[0].legend(handles=color_handles, title="Period length", frameon=False, fontsize=8,
                                  title_fontsize=9, loc="upper left", ncol=1)
            axes[0].add_artist(leg1)
            axes[0].legend(handles=_main_style_legend_handles(), frameon=False, fontsize=10, loc="lower right")

    # ------------------------------------------------------------------
    # Panel b: full mean at dt = 5 yr, colored by past elevation change rate
    # IMPORTANT: this panel is intentionally restricted to the 5-year
    # Period slice; mixing periods here makes the displayed binned estimates
    # Inconsistent with any single analytical model curve
    # ------------------------------------------------------------------
    target_right = _filter_main_mean_right_panel_target_to_period(target, target_year=5.0)
    by_memory = _aggregate_target_pair(
        target=target_right,
        model=selected_model,
        x_col="current_center_w",
        x_bin_col="current_bin",
        group_col="memory_center_w",
        group_bin_col="memory_bin",
        fit_obj=fit_obj,
    )

    if by_memory.empty:
        axes[1].text(0.5, 0.5, "No target data", ha="center", va="center", transform=axes[1].transAxes)
    else:
        mem_lookup = (
            by_memory.groupby("group_bin", as_index=False)
            .agg(group_center_ref=("group_center_w", "mean"))
            .sort_values("group_center_ref")
        )
        actual_vals, actual_labels = _pick_representative_memory_groups(mem_lookup["group_center_ref"].to_numpy(float))
        sel_bins = []
        sel_map = {}
        sel_actual = {}
        for v, lab in zip(actual_vals, actual_labels):
            idx = int(np.argmin(np.abs(mem_lookup["group_center_ref"].to_numpy(float) - v)))
            gb = mem_lookup["group_bin"].iloc[idx]
            if gb not in sel_bins:
                sel_bins.append(gb)
                sel_map[gb] = lab
                sel_actual[gb] = mem_lookup["group_center_ref"].iloc[idx]

        plot_df = by_memory.loc[by_memory["group_bin"].isin(sel_bins)].copy()

        cmap_mem = plt.get_cmap(PLOT_CMAP_MEMORY)
        label_vals = np.asarray(MAIN_MEMORY_TARGET_LABELS, dtype=float)
        vmax = np.nanmax(np.abs(label_vals)) if len(label_vals) else 1.0
        vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
        norm_mem = mpl.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

        plot_df["color_center_w"] = plot_df["group_bin"].map(sel_map).astype(float)
        colors = cmap_mem(norm_mem(plot_df["color_center_w"].to_numpy(float)))
        y_for_limits = [plot_df["rho_obs_w"].to_numpy(float)]
        axes[1].scatter(
            _recenter_small_display_bin(plot_df["x_center_w"].to_numpy(float)),
            plot_df["rho_obs_w"].to_numpy(float),
            s=28,
            c=colors,
            alpha=0.70,
            marker="o",
            linewidths=0,
        )

        for gb, sub in plot_df.groupby("group_bin", sort=True):
            mem_val = float(sel_actual.get(gb, weighted_mean(sub["group_center_w"].to_numpy(float), sub["weight_sum"].to_numpy(float))))
            lab_val = float(sel_map.get(gb, mem_val))
            col = cmap_mem(norm_mem(lab_val))

            xfit_anchor = sub["x_center_w"].to_numpy(float)
            yfit_anchor = sub["rho_pred_w"].to_numpy(float)

            if model_line_mode == "analytical":
                # Continuous fitted function for the displayed past elevation change rate group
                # At the fixed 5-year slice used by panel b
                period_val = 5.0
                xfit = xgrid.copy()
                yfit_plot = rho_model(
                    fit_obj["theta"], fit_obj["spec"], fit_obj["param_names"],
                    xfit,
                    np.full_like(xfit, float(mem_val), dtype=float),
                    np.full_like(xfit, float(period_val), dtype=float),
                )
                _plot_dense_signed_curve(axes[1], xfit, yfit_plot, col, lw=2.0, alpha=0.95, connect_zero=False)
            else:
                # Diagnostic-style model line: fitted values at the same binned
                # Support as the binned estimates, optionally smoothed visually
                xfit, yfit_plot = _plot_mean_model_from_anchors(
                    axes[1], xfit_anchor, yfit_anchor, col, xgrid,
                    min_abs=None, mode=model_line_mode, lw=2.0, alpha=0.95
                )

            y_for_limits.append(yfit_anchor)
            if model_line_mode == "analytical":
                y_for_limits.append(yfit_plot[np.abs(xfit) >= MAIN_MEAN_LEFT_MIN_ABS_DH_DISPLAY])
            else:
                y_for_limits.append(yfit_plot)

        axes[1].axhline(RHO_ICE_FIXED, color="black", lw=1, ls="--")
        axes[1].axvline(0, color="black", lw=1)
        set_signed_log_xaxis(axes[1], np.concatenate([plot_df["x_center_w"].to_numpy(float), np.array([-50.0, 50.0])]))
        axes[1].set_xlim(-50.0, 50.0)
        _set_compact_linear_y_limits(axes[1], y_for_limits, include_ice=True)
        axes[1].set_xlabel("Current dh (m)")
        axes[1].set_ylabel(r"Mean $\rho_{\Delta V}$ (kg m$^{-3}$)")
        axes[1].set_title(r"Additional past elevation change rate dependence (5-year period)")
        axes[1].grid(alpha=0.25)

        color_handles = [
            Line2D([], [], marker="o", ms=5, lw=1.8, color=cmap_mem(norm_mem(float(lbl))),
                   label=_format_memory_label(float(lbl)))
            for lbl in label_vals
        ]
        leg1 = axes[1].legend(handles=color_handles, title="Past elevation change rate", frameon=False, fontsize=8,
                              title_fontsize=9, loc="upper left", ncol=1)
        axes[1].add_artist(leg1)
        axes[1].legend(handles=_main_style_legend_handles(), frameon=False, fontsize=10, loc="lower right")

    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

def plot_main_rho_std_figure(d_pred, theta_sig, out_png, x_axis_mode="log"):
    """
    Main STD figure: selected residual-spread model with period dependence.
    """
    from matplotlib.lines import Line2D

    tab = summarize_absdh_residual_std_for_sigma(d_pred, "rho_mean_removed")
    if tab.empty:
        return
    tab = tab.loc[np.isfinite(tab["abs_center_w"]) & (tab["abs_center_w"] <= SIGMA_MAX_ABS_DH_FOR_FIT)].copy()
    if tab.empty:
        return

    periods = np.sort(tab["period_years"].unique()) if "period_years" in tab.columns else np.array([np.nan])
    periods = _filter_display_periods(periods)

    cmap = plt.get_cmap(PLOT_CMAP_PERIOD)
    norm = mpl.colors.Normalize(vmin=np.nanmin(periods), vmax=np.nanmax(periods)) if len(periods) and np.all(np.isfinite(periods)) else mpl.colors.Normalize(vmin=0, vmax=1)

    if x_axis_mode not in ("log", "linear"):
        raise ValueError(f"Unknown x_axis_mode: {x_axis_mode}")
    xmax = min(max(np.nanmax(tab["abs_center_w"].to_numpy(float)), CURRENT_DH_MIN_ABS_FOR_FIT * 10), SIGMA_MAX_ABS_DH_FOR_FIT)
    xmin = max(CURRENT_DH_MIN_ABS_FOR_FIT, 1e-4)
    if x_axis_mode == "linear":
        xline = np.linspace(xmin, xmax * 1.02, 600)
    else:
        xline = np.geomspace(xmin, xmax * 1.02, 600)

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.35), constrained_layout=True)
    sigma_form_for_plot, sigma_params_for_plot = _resolve_sigma_form_and_params(theta_sig)
    y_all = []

    if "period_years" in tab.columns:
        avail = np.sort(tab["period_years"].dropna().unique().astype(float))
        for T in periods:
            Tuse = avail[int(np.argmin(np.abs(avail - float(T))))] if len(avail) else T
            sub = tab.loc[np.isclose(tab["period_years"], Tuse)].sort_values("abs_center_w")
            if sub.empty:
                continue
            col = cmap(norm(T)) if np.isfinite(T) else "black"
            ax.scatter(sub["abs_center_w"], sub["resid_std_w"], s=24, color=col, alpha=0.8, edgecolors="none")
            yline = np.maximum(
                sigma_model_form(
                    theta_sig,
                    xline,
                    period_years=np.full_like(xline, float(Tuse), dtype=float),
                    form=sigma_form_for_plot,
                    param_names=sigma_params_for_plot,
                ),
                SIGMA_NUMERIC_FLOOR,
            )
            ok = np.isfinite(yline) & (yline > 0)
            if np.any(ok):
                ax.plot(xline[ok], yline[ok], lw=1.8, color=col)
                y_all.append(yline[ok])
            vals = sub["resid_std_w"].to_numpy(float)
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if len(vals):
                y_all.append(vals)

    if x_axis_mode == "linear":
        ax.set_xscale("linear")
    else:
        ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$|\Delta h|$ (m)")
    ax.set_ylabel(r"STD of mean-removed residual (kg m$^{-3}$)")
    ax.set_title(r"Residual-spread model")
    ax.grid(alpha=0.25)

    if y_all:
        yy = np.concatenate([np.asarray(v, dtype=float) for v in y_all])
        yy = yy[np.isfinite(yy) & (yy > 0)]
        if len(yy):
            ax.set_ylim(max(np.nanmin(yy) * 0.85, 1e-3), np.nanmax(yy) * 1.15)

    if len(periods) > 1:
        handles = [
            Line2D([], [], marker="o", ms=5, lw=1.8, color=cmap(norm(float(v))), label=_format_period_label(float(v)))
            for v in periods
        ]
        leg1 = ax.legend(handles=handles, title="Period length", frameon=False, fontsize=8,
                         title_fontsize=9, loc="upper right", ncol=1)
        ax.add_artist(leg1)
    ax.legend(handles=_main_style_legend_handles(), frameon=False, fontsize=10, loc="lower left")

    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

def _plot_supp_panel(ax, suball, ycol, periods, x_label, y_label, title,
                     signed_log_x=False, log_x=False, hline=None, ylim=None):
    cmap = plt.get_cmap(PLOT_CMAP_PERIOD)
    norm = mpl.colors.Normalize(vmin=np.nanmin(periods), vmax=np.nanmax(periods))
    for T in periods:
        sub = suball.loc[suball["period_years"] == T].sort_values("factor_center_w")
        if sub.empty:
            continue
        col = cmap(norm(T))
        split_signed = signed_log_x
        if split_signed:
            for mask in [sub["factor_center_w"] < 0, sub["factor_center_w"] > 0]:
                ss = sub.loc[mask].sort_values("factor_center_w")
                if len(ss) >= 2:
                    ax.plot(ss["factor_center_w"], ss[ycol], marker="o", ms=3.0, lw=1.0, color=col)
        elif len(sub) >= 2:
            ax.plot(sub["factor_center_w"], sub[ycol], marker="o", ms=3.0, lw=1.0, color=col)
    if hline is not None:
        ax.axhline(hline, color="black", lw=0.9, ls="--")
    if signed_log_x:
        ax.axvline(0, color="black", lw=0.8)
        set_signed_log_xaxis(ax, suball["factor_center_w"].to_numpy(float))
    if log_x:
        ax.set_xscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    return sm




def plot_supp_rho_mean_figure(diag, out_png):
    """Supplementary mean diagnostics for current change, past elevation change rate, and area."""
    diag = diag.copy()
    diag["factor"] = diag["factor"].replace({"Model memory dh": "Past elevation change"})
    specs = [
        ("Current signed dh", "Current dh", "Current dh (m)", True, False),
        ("Past elevation change", "Past elevation change rate", "Past elevation change rate (m yr$^{-1}$)", True, False),
        ("Area", "Area", "Area", False, True),
    ]
    d = diag.loc[diag["factor"].isin([sp[0] for sp in specs])].copy()
    if d.empty:
        return

    mcur = d["factor"].eq("Current signed dh")
    d = d.loc[~(mcur & (np.abs(d["factor_center_w"].to_numpy(float)) < 0.2))].copy()

    periods = _filter_display_periods(np.sort(d["period_years"].unique()))
    if len(periods) == 0:
        return

    raw = d["raw_mean_w"].to_numpy(float)
    resid = d["resid_mean_w"].to_numpy(float)
    raw = raw[np.isfinite(raw)]
    resid = resid[np.isfinite(resid)]
    halfspan = 100.0
    if len(raw):
        halfspan = max(halfspan, float(np.nanmax(np.abs(raw - RHO_ICE_FIXED))))
    if len(resid):
        halfspan = max(halfspan, float(np.nanmax(np.abs(resid))))
    halfspan *= 1.05
    ylim_top = (RHO_ICE_FIXED - halfspan, RHO_ICE_FIXED + halfspan)
    ylim_bottom = (-halfspan, halfspan)

    fig, axes = plt.subplots(2, 3, figsize=(12.6, 6.6), constrained_layout=True, sharey="row")
    sm = None
    for j, (factor, title, xlabel, signed_x, log_x) in enumerate(specs):
        sub = d.loc[d["factor"] == factor]
        sm = _plot_supp_panel(
            axes[0, j], sub, "raw_mean_w", periods, "", r"Mean $\rho_{\Delta V}$" if j == 0 else "",
            title, signed_x, log_x, hline=RHO_ICE_FIXED, ylim=ylim_top
        )
        _plot_supp_panel(
            axes[1, j], sub, "resid_mean_w", periods, xlabel, r"Mean residual" if j == 0 else "",
            "", signed_x, log_x, hline=0, ylim=ylim_bottom
        )
    if sm is not None:
        fig.colorbar(sm, ax=axes, pad=0.015, shrink=0.95).set_label("Period length (yr)")
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_supp_rho_std_figure(diag, out_png):
    """Supplementary STD diagnostics: raw STD and standardized residual STD."""
    diag = diag.copy()
    diag["factor"] = diag["factor"].replace({"Model memory dh": "Past elevation change"})
    specs = [
        ("Current signed dh", "Current dh", "Current dh (m)", True, False),
        ("Past elevation change", "Past elevation change rate", "Past elevation change rate (m yr$^{-1}$)", True, False),
        ("Area", "Area", "Area", False, True),
    ]
    d = diag.loc[diag["factor"].isin([sp[0] for sp in specs])].copy()
    if d.empty:
        return

    mcur = d["factor"].eq("Current signed dh")
    d = d.loc[~(mcur & (np.abs(d["factor_center_w"].to_numpy(float)) < 0.2))].copy()

    periods = _filter_display_periods(np.sort(d["period_years"].unique()))
    if len(periods) == 0:
        return

    # Auto upper-row range from raw STD; fixed lower-row range for standardized residual STD
    rawv = d["raw_std_w"].to_numpy(float)
    rawv = rawv[np.isfinite(rawv)]
    if len(rawv):
        top_ylim = (0.0, float(np.nanmax(rawv) * 1.05))
    else:
        top_ylim = None

    fig, axes = plt.subplots(2, 3, figsize=(12.6, 6.6), constrained_layout=True, sharey="row")
    sm = None
    for j, (factor, title, xlabel, signed_x, log_x) in enumerate(specs):
        sub = d.loc[d["factor"] == factor]
        sm = _plot_supp_panel(
            axes[0, j], sub, "raw_std_w", periods, "", r"Raw STD" if j == 0 else "",
            title, signed_x, log_x, ylim=top_ylim
        )
        ycol = "z_after_std_w" if "z_after_std_w" in sub.columns else "resid_std_w"
        ylab = "STD of standardized residual" if ycol == "z_after_std_w" else "Residual STD"
        _plot_supp_panel(
            axes[1, j], sub, ycol, periods, xlabel, ylab if j == 0 else "",
            "", signed_x, log_x, hline=1 if ycol == "z_after_std_w" else None, ylim=(0, 5)
        )
    if sm is not None:
        fig.colorbar(sm, ax=axes, pad=0.015, shrink=0.95).set_label("Period length (yr)")
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Pipeline
# =============================================================================


def _write_memory_period_availability_diagnostics(df_mem, d_used, target, memory_col):
    """
    Write period-level availability diagnostics so period losses can be traced.
    """
    try:
        rows = []
        all_periods = np.sort(df_mem["period_years"].dropna().unique().astype(float))
        has_col = f"has_{memory_col}"
        for T in all_periods:
            g0 = df_mem.loc[np.isclose(df_mem["period_years"], T)]
            g1 = g0.loc[g0[has_col]] if has_col in g0.columns else g0.iloc[0:0]
            g2 = d_used.loc[np.isclose(d_used["period_years"], T)] if len(d_used) else d_used
            gt = target.loc[np.isclose(target["period_years"], T)] if len(target) else target
            rows.append({
                "period_years": float(T),
                "n_input": int(len(g0)),
                "n_has_memory": int(len(g1)),
                "n_used_after_absdh": int(len(g2)),
                "weighted_input": float(np.nansum(g0[weight_col].to_numpy(float))) if len(g0) else 0.0,
                "weighted_has_memory": float(np.nansum(g1[weight_col].to_numpy(float))) if len(g1) else 0.0,
                "weighted_used_after_absdh": float(np.nansum(g2[weight_col].to_numpy(float))) if len(g2) else 0.0,
                "n_target_cells": int(len(gt)),
                "n_supported_target_cells": int(np.nansum(gt["support_ok"].to_numpy(bool))) if len(gt) and "support_ok" in gt else 0,
            })
        pd.DataFrame(rows).to_csv(
            out_plot_dir / f"{run_label}_{main_suffix}_memory_period_availability.csv",
            index=False,
        )
    except Exception as exc:
        log(f"Could not write memory-period availability diagnostics: {exc!r}")


def build_fit_target_for_memory(all_rows, mode, window=None, tau=None):
    df_mem, memory_col = attach_model_memory(all_rows, all_rows, mode, window=window, tau=tau)
    has_col = f"has_{memory_col}"

    share = np.average(df_mem[has_col].astype(float), weights=df_mem[weight_col])
    log(f"Memory availability {memory_col}: weighted share = {share:.3f}")

    try:
        rows = []
        for T, g in df_mem.groupby("period_years", sort=True):
            if PERIOD_LENGTHS_TO_PLOT is not None and T not in PERIOD_LENGTHS_TO_PLOT:
                continue
            rows.append(f"{float(T):g}yr:{np.average(g[has_col].astype(float), weights=g[weight_col]):.2f}")
        if rows:
            log("Memory availability by requested period: " + ", ".join(rows))
    except Exception:
        pass

    d = df_mem.loc[df_mem[has_col] & np.isfinite(df_mem[memory_col])].copy()
    d = d.loc[np.abs(d["signed_dh"].to_numpy(float)) >= CURRENT_DH_MIN_ABS_FOR_FIT].copy()
    if len(d) == 0:
        raise ValueError(f"No rows for memory {memory_col}")

    current_edges = make_signed_log_dh_edges(d["signed_dh"].to_numpy(float))
    mem_edges = make_quantile_edges(
        d[memory_col].to_numpy(float),
        d[weight_col].to_numpy(float),
        MEMORY_N_QUANTILE_BINS_FOR_FIT,
    )
    if mem_edges is None:
        raise ValueError(f"Could not build memory bins for {memory_col}")

    target = summarize_target_3d(prepare_binned_rows(d, current_edges, mem_edges, memory_col), memory_col)
    target = attach_mean_fit_weights(target)
    target_fit = target.loc[target["support_ok"]].copy()

    _write_memory_period_availability_diagnostics(df_mem, d, target, memory_col)
    return d, memory_col, target, target_fit

def run_one_memory_fit(all_rows, mode, window=None, tau=None, candidate_models=None):
    if candidate_models is None: candidate_models=CANDIDATE_MODELS
    d,memory_col,target,target_fit=build_fit_target_for_memory(all_rows,mode,window=window,tau=tau)
    if len(target_fit)<8: raise ValueError(f"Too few supported cells for {memory_col}: {len(target_fit)}")
    candidates=[]
    for spec in candidate_models:
        sp=dict(spec); sp["memory_mode"]=mode; sp["memory_window_years"]=window if mode=="fixed_window" else np.nan; sp["memory_tau_years"]=tau if mode=="exp_cumulative" else np.nan; candidates.append(sp)
    summary,fits=fit_candidate_models(target_fit,candidates); target=attach_predictions_to_target(target,fits)
    return d,memory_col,target,target_fit,summary,fits




def write_final_joint_rho_model_files(
    selected_model,
    fit_obj,
    theta_sig,
    memory_mode,
    memory_window,
    memory_tau,
    memory_col,
    sigma_resid_col="rho_mean_removed",
):
    """
    Write a single final joint rho model parameter table and a structured JSON
    sidecar containing both the rho mean and rho STD models.

    The CSV is long-format and deliberately separates numeric values from text
    values so it can be read robustly by downstream scripts.
    """
    sigma_form = globals().get("SIGMA_SELECTED_FORM", SIGMA_MODEL_FOR_DIAGNOSTICS)
    if sigma_form == "best":
        sigma_form = SIGMA_MODEL_FOR_DIAGNOSTICS
    sigma_param_names = list(globals().get("SIGMA_SELECTED_PARAMS", SIGMA_PARAMS_BY_FORM.get(sigma_form, SIGMA_PARAMS)))

    mean_params = unpack(fit_obj["theta"], fit_obj["param_names"])
    sigma_params = {
        name: float(val)
        for name, val in zip(sigma_param_names, np.asarray(theta_sig, dtype=float))
    }

    mean_spec = dict(fit_obj.get("spec", {}))
    memory_meta = {
        "memory_mode": memory_mode,
        "memory_window_years": memory_window,
        "memory_tau_years": memory_tau,
        "memory_col": memory_col,
    }

    fixed_meta = {
        "rho_ice_fixed": float(RHO_ICE_FIXED),
        "sigma_numeric_floor": float(SIGMA_NUMERIC_FLOOR),
        "sigma_residual_column": sigma_resid_col,
        "mean_target_support_weight_column": weight_col,
        "mean_target_fit_weight_column": mean_fit_weight_col,
        "mean_target_fit_weight_mode": MEAN_FIT_WEIGHT_MODE,
        "rho_column": rho_col,
    }

    records = []

    def add_record(component, model, parameter, value, role="", units="", value_type=None):
        if isinstance(value, (np.integer, int, np.floating, float)) and np.isfinite(float(value)):
            value_numeric = float(value)
            value_text = ""
            vt = value_type or "numeric"
        elif value is None:
            value_numeric = np.nan
            value_text = ""
            vt = value_type or "none"
        else:
            value_numeric = np.nan
            value_text = str(value)
            vt = value_type or "text"
        records.append({
            "component": component,
            "model": model,
            "parameter": parameter,
            "value_numeric": value_numeric,
            "value_text": value_text,
            "value_type": vt,
            "role": role,
            "units": units,
        })

    add_record("rho_mean", selected_model, "selected_model", selected_model, role="model identifier", value_type="text")
    for k, v in mean_spec.items():
        add_record("rho_mean", selected_model, f"spec.{k}", v, role="mean model form/specification")
    for k, v in mean_params.items():
        add_record("rho_mean", selected_model, k, v, role="fitted mean parameter")

    add_record("rho_std", sigma_form, "selected_sigma_form", sigma_form, role="sigma model identifier", value_type="text")
    for k, v in sigma_params.items():
        add_record("rho_std", sigma_form, k, v, role="fitted sigma parameter")

    for k, v in memory_meta.items():
        add_record("memory", "memory_definition", k, v, role="memory predictor metadata")

    for k, v in fixed_meta.items():
        add_record("fixed", "global_constants", k, v, role="fixed constant or column metadata")

    pd.DataFrame(records).to_csv(out_final_joint_model_csv, index=False)

    def _json_safe(obj):
        if isinstance(obj, dict):
            return {str(k): _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_json_safe(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return [_json_safe(v) for v in obj.tolist()]
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            val = float(obj)
            return val if np.isfinite(val) else None
        if obj is None:
            return None
        return str(obj) if not isinstance(obj, (str, bool)) else obj

    payload = {
        "rho_mean": {
            "selected_model": selected_model,
            "spec": mean_spec,
            "param_names": list(fit_obj["param_names"]),
            "parameters": mean_params,
        },
        "rho_std": {
            "selected_sigma_form": sigma_form,
            "param_names": sigma_param_names,
            "parameters": sigma_params,
            "residual_column": sigma_resid_col,
        },
        "memory": memory_meta,
        "fixed": fixed_meta,
        "files": {
            "csv": str(out_final_joint_model_csv),
            "json": str(out_final_joint_model_json),
        },
    }
    with open(out_final_joint_model_json, "w") as f:
        json.dump(_json_safe(payload), f, indent=2, sort_keys=True)

    return pd.DataFrame(records)

def run_main_analysis(all_rows, memory_mode=None, memory_window=None, memory_tau=None):
    t0 = perf_counter()
    if memory_mode is None:
        memory_mode = MODEL_MEMORY_MODE
    if memory_window is None:
        memory_window = MODEL_MEMORY_WINDOW_YEARS
    if memory_tau is None:
        memory_tau = MODEL_MEMORY_TAU_YEARS

    log(
        f"Building main fit target and fitting candidate models "
        f"(memory_mode={memory_mode}, window={memory_window}, tau={memory_tau})",
        t0
    )

    d, memory_col, target, target_fit, summary, fits = run_one_memory_fit(
        all_rows, memory_mode, window=memory_window, tau=memory_tau
    )
    summary.to_csv(out_fit_summary_csv, index=False)
    target.to_csv(out_target_csv, index=False)
    if WRITE_FIT_DIAGNOSTIC_PLOTS:
        plot_mean_fit_by_model(target, summary)
        log(f"Saved rho-mean fit plots in: {out_mean_fit_dir}", t0)

    params = []
    for model, obj in fits.items():
        params.append({"model": model, **unpack(obj["theta"], obj["param_names"])})
    pd.DataFrame(params).to_csv(out_params_csv, index=False)

    selected_model, fit_obj = select_fit(summary, fits)
    log(f"Selected model for diagnostics: {selected_model}", t0)

    dh_category_metrics = pd.concat(
        [
            summarize_target_prediction_by_dh_category(target, selected_model, fit_weight_column="weight_sum"),
            summarize_target_prediction_by_dh_category(target, selected_model, fit_weight_column=mean_fit_weight_col),
        ],
        ignore_index=True,
    )
    dh_category_metrics.to_csv(out_target_dh_category_csv, index=False)
    log(f"Saved target low/mid/high |dh| metrics: {out_target_dh_category_csv}", t0)

    period_absdh_metrics, period_signeddh_metrics = summarize_target_prediction_period_dh_for_models(
        target,
        [selected_model],
        ["weight_sum", mean_fit_weight_col],
    )
    period_absdh_metrics.to_csv(out_target_period_absdh_csv, index=False)
    period_signeddh_metrics.to_csv(out_target_period_signeddh_csv, index=False)
    log(f"Saved target period x |dh| metrics: {out_target_period_absdh_csv}", t0)
    log(f"Saved target period x signed-dh metrics: {out_target_period_signeddh_csv}", t0)

    if RUN_NO_PERIOD_DIAGNOSTIC:
        no_period_model, no_period_fit = fit_or_find_paired_no_period_fit(fit_obj, summary, fits, target_fit, t0=t0)
        if no_period_fit is not None:
            log(f"Paired no-period model for diagnostics: {no_period_model}", t0)
        else:
            log("No no-period model available for period-correction diagnostics", t0)
    else:
        no_period_model, no_period_fit = None, None
        log("Skipping paired no-period diagnostics to reduce memory", t0)

    d_pred = compute_row_predictions(d, fit_obj, memory_col)
    if not RUN_NO_PERIOD_DIAGNOSTIC:
        del d
        gc.collect()

    theta_sig_after, sigma_abs_table_after = fit_sigma_model_from_rows(d_pred, resid_col="rho_mean_removed")
    sigma_abs_table_after.to_csv(out_sigma_abs_table_csv, index=False)
    pd.DataFrame([{
        **{"sigma_form": globals().get("SIGMA_SELECTED_FORM", SIGMA_MODEL_FOR_DIAGNOSTICS)},
        **{name: val for name, val in zip(globals().get("SIGMA_SELECTED_PARAMS", SIGMA_PARAMS), theta_sig_after)}
    }]).to_csv(out_sigma_params_csv, index=False)
    globals().get("SIGMA_MODEL_SELECTION_TABLE", pd.DataFrame()).to_csv(out_sigma_model_selection_csv, index=False)

    write_final_joint_rho_model_files(
        selected_model=selected_model,
        fit_obj=fit_obj,
        theta_sig=theta_sig_after,
        memory_mode=memory_mode,
        memory_window=memory_window,
        memory_tau=memory_tau,
        memory_col=memory_col,
        sigma_resid_col="rho_mean_removed",
    )
    try:
        from glacier_density_surrogate import write_packaged_params

        package_json = write_packaged_params(parameter_path=out_final_joint_model_csv)
        log(f"Updated packaged surrogate parameters: {package_json}", t0)
    except Exception as exc:
        log(f"Could not update packaged surrogate parameters: {exc!r}", t0)

    log(f"Saved final joint rho model parameters: {out_final_joint_model_csv}", t0)
    log(f"Saved final joint rho model JSON: {out_final_joint_model_json}", t0)

    d_pred["sigma_after_model"] = sigma_model_safe(
        theta_sig_after,
        d_pred["abs_signed_dh"].to_numpy(float),
        d_pred["period_years"].to_numpy(float),
    )
    d_pred["z_after"] = d_pred["rho_mean_removed"].to_numpy(float) / d_pred["sigma_after_model"].to_numpy(float)

    path_sigma_tab, path_sigma_selection = run_pathlength_sigma_diagnostic(
        d_pred,
        all_rows,
        resid_col="rho_mean_removed",
    )

    if WRITE_FIT_DIAGNOSTIC_PLOTS:
        plot_sigma_fit_signed_currentdh(d_pred, theta_sig_after, out_sigma_fit_signed_png)
        plot_sigma_fit_by_period(sigma_abs_table_after, theta_sig_after, out_sigma_fit_by_period_png)
        plot_sigma_fit_all_models(
            sigma_abs_table_after,
            globals().get("SIGMA_MODEL_SELECTION_TABLE", pd.DataFrame()),
            out_sigma_fit_by_model_dir,
        )

    d_pred_no_period = compute_row_predictions(d, no_period_fit, memory_col) if no_period_fit is not None else None
    if d_pred_no_period is not None:
        d_pred_no_period["rho_mean_removed"] = d_pred_no_period["rho_resid"]
        theta_sig_no_period, _ = fit_sigma_model_from_rows(d_pred_no_period, resid_col="rho_mean_removed")
        d_pred_no_period["sigma_after_model"] = sigma_model_safe(
            theta_sig_no_period,
            d_pred_no_period["abs_signed_dh"].to_numpy(float),
            d_pred_no_period["period_years"].to_numpy(float),
        )
        d_pred_no_period["z_after"] = (
            d_pred_no_period["rho_mean_removed"].to_numpy(float)
            / d_pred_no_period["sigma_after_model"].to_numpy(float)
        )

    sample_cols = [
        "rgiid", variant_col, "start_date", "end_date", "period_years", rho_col,
        "rho_pred", "rho_resid", "rho_mean_removed", "signed_dh", memory_col, weight_col,
        "memory_component", "ratio_component", "period_component", "current_component",
        "damping", "sigma_after_model", "z_after"
    ]
    d_pred[[c for c in sample_cols if c in d_pred.columns]].head(MAX_ROW_SAMPLE_OUTPUT).to_csv(
        out_row_sample_csv, index=False
    )

    # Full row-level output for spatial and temporal correlation diagnostics
    # This file is intended as the standard input for downstream residual-
    # Correlation scripts
    standardized_cols = [
        "rgiid", variant_col, "start_date", "end_date", "period_years",
        rho_col, b_col, area_col, "signed_dh", "abs_signed_dh", memory_col,
        weight_col, "rho_pred", "rho_mean_removed", "sigma_after_model", "z_after",
    ]
    standardized_cols += [c for c in OPTIONAL_OUTPUT_METADATA_COLS if c in d_pred.columns]
    standardized_cols = [c for c in standardized_cols if c in d_pred.columns]
    d_pred.loc[:, standardized_cols].rename(columns={"z_after": "z_rho"}).to_csv(
        out_standardized_residuals_csv, index=False
    )
    if LOW_MEMORY_MODE:
        drop_cols = [c for c in OPTIONAL_OUTPUT_METADATA_COLS if c in d_pred.columns]
        if drop_cols:
            d_pred.drop(columns=drop_cols, inplace=True)
            gc.collect()

    diag = summarize_all_factors(d_pred, memory_col)
    diag.to_csv(out_diag_csv, index=False)

    if RUN_RESIDUAL_MEMORY_SCAN_DIAGNOSTICS:
        mem_cum, mem_cum_score, mem_rate, mem_rate_score = summarize_residual_memory_scan(d_pred, all_rows)
    else:
        mem_cum, mem_cum_score, mem_rate, mem_rate_score = (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    mem_cum.to_csv(out_memory_scan_csv, index=False)
    mem_rate.to_csv(out_memory_rate_scan_csv, index=False)
    mem_cum_score.to_csv(out_memory_score_csv, index=False)
    mem_rate_score.to_csv(out_memory_rate_score_csv, index=False)

    if WRITE_FIT_DIAGNOSTIC_PLOTS:
        log("Writing diagnostic plots", t0)
        plot_model_metric_bar(summary)

        plot_before_after_factor(diag, "Current signed dh", out_current_png, "Current signed dh (m)", signed_log_x=True)
        plot_before_after_factor(diag, "Absolute current dh", out_abs_current_png, "|Current dh| (m)", log_x=True)
        plot_before_after_factor(diag, "Past elevation change", out_memory_png, "Past elevation change rate (m yr$^{-1}$)", signed_log_x=True)
        plot_before_after_factor(diag, "Area", out_area_png, "Area", log_x=True)
        plot_before_after_factor(diag, "Period length", out_period_png, "Period length (yr)")

        plot_std_before_after_factor(diag, "Current signed dh", out_std_current_png, "Current signed dh (m)", signed_log_x=True)
        plot_std_before_after_factor(diag, "Absolute current dh", out_std_abs_current_png, "|Current dh| (m)", log_x=True)
        plot_std_before_after_factor(diag, "Past elevation change", out_std_memory_png, "Past elevation change rate (m yr$^{-1}$)", signed_log_x=True)
        plot_std_before_after_factor(diag, "Area", out_std_area_png, "Area", log_x=True)
        plot_std_before_after_factor(diag, "Period length", out_std_period_png, "Period length (yr)")

        plot_standardized_after_factor(diag, "Current signed dh", out_z_current_png, "Current signed dh (m)", signed_log_x=True)
        plot_standardized_after_factor(diag, "Absolute current dh", out_z_abs_current_png, "|Current dh| (m)", log_x=True)
        plot_standardized_after_factor(diag, "Past elevation change", out_z_memory_png, "Past elevation change rate (m yr$^{-1}$)", signed_log_x=True)
        plot_standardized_after_factor(diag, "Area", out_z_area_png, "Area", log_x=True)
        plot_standardized_after_factor(diag, "Period length", out_z_period_png, "Period length (yr)")

        plot_component_by_period(fit_obj, out_period_component_png, "period_component", "Fitted period component")
        plot_component_by_period(fit_obj, out_current_component_png, "current_component", "Fitted even current-dh component")
        plot_period_correction_performance(
            d_pred, d_pred_no_period, selected_model, no_period_model, out_period_correction_performance_png
        )
        plot_memory_scan(mem_cum, mem_cum_score, out_memory_scan_png, out_memory_score_png, "cumulative")
        plot_memory_scan(mem_rate, mem_rate_score, out_memory_rate_scan_png, out_memory_rate_score_png, "rate")

    if RUN_PAPER_FIGURES:
        # Main paper figures are written in two x-axis variants:
        #   - Default/log: signed-log x-axis for mean, log x-axis for uncertainty;
        #   - Linear: identical content with linear x-axis for visual comparison
        plot_main_rho_mean_figure(
            target, selected_model, d_pred, fit_obj, d_pred_no_period, no_period_fit,
            memory_col, out_main_mean_figure_png, model_line_mode="analytical", x_axis_mode="log"
        )
        plot_main_rho_mean_figure(
            target, selected_model, d_pred, fit_obj, d_pred_no_period, no_period_fit,
            memory_col, out_main_mean_figure_linear_png, model_line_mode="analytical", x_axis_mode="linear"
        )
        plot_main_rho_std_figure(d_pred, theta_sig_after, out_main_std_figure_png, x_axis_mode="log")
        plot_main_rho_std_figure(d_pred, theta_sig_after, out_main_std_figure_linear_png, x_axis_mode="linear")
        plot_supp_normality_figure(d_pred, out_supp_normality_figure_png)
        plot_supp_rho_mean_figure(diag, out_supp_mean_figure_png)
        plot_supp_rho_std_figure(diag, out_supp_std_figure_png)

    print("\nFit summary:", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("\nSelected parameters:", flush=True)
    print(pd.DataFrame([{
        **{"model": selected_model, "memory_mode": memory_mode, "memory_window_years": memory_window, "memory_tau_years": memory_tau},
        **unpack(fit_obj["theta"], fit_obj["param_names"])
    }]).to_string(index=False), flush=True)

    return selected_model, dict(fit_obj["spec"]), {
        "memory_mode": memory_mode,
        "memory_window_years": memory_window,
        "memory_tau_years": memory_tau,
        "memory_col": memory_col,
    }

def run_memory_profile(all_rows):
    """
    Refit the full candidate set for multiple memory definitions, then score:
      - fit error;
      - residual structure against all fixed diagnostic memory windows;
      - residual structure against past elevation change rates over fixed windows;
      - residual structure against current dh, period length, and area.

    This is the decision layer for choosing the memory-dh definition.
    """
    if not RUN_MEMORY_PROFILE:
        return pd.DataFrame()

    t0 = perf_counter()
    profile_rows = []
    long_rows = []
    scan_cum_by_label = {}
    scan_rate_by_label = {}

    for W in MEMORY_PROFILE_FIXED_WINDOWS:
        log(f"Memory profile window W={W}", t0)
        try:
            row, long, mem_cum, mem_rate = _fit_and_score_one_memory(
                all_rows,
                "fixed_window",
                window=W,
                tau=None,
                candidate_models=CANDIDATE_MODELS,
                t0=t0,
            )
            profile_rows.append(row)
            if len(long):
                long_rows.append(long)
            scan_cum_by_label[row["memory_label"]] = mem_cum
            scan_rate_by_label[row["memory_label"]] = mem_rate
        except Exception as exc:
            profile_rows.append({
                "memory_label": f"window_{W}yr",
                "memory_mode": "fixed_window",
                "memory_parameter": float(W),
                "memory_window_years": W,
                "memory_tau_years": np.nan,
                "best_model": "FAILED",
                "error": repr(exc),
            })

    if RUN_EXP_MEMORY_PROFILE:
        for tau in MEMORY_PROFILE_EXP_TAUS:
            log(f"Memory profile exponential tau={tau:g}", t0)
            try:
                row, long, mem_cum, mem_rate = _fit_and_score_one_memory(
                    all_rows,
                    "exp_cumulative",
                    window=None,
                    tau=tau,
                    candidate_models=CANDIDATE_MODELS,
                    t0=t0,
                )
                profile_rows.append(row)
                if len(long):
                    long_rows.append(long)
                scan_cum_by_label[row["memory_label"]] = mem_cum
                scan_rate_by_label[row["memory_label"]] = mem_rate
            except Exception as exc:
                profile_rows.append({
                    "memory_label": f"exp_tau{tau:g}yr",
                    "memory_mode": "exp_cumulative",
                    "memory_parameter": float(tau),
                    "memory_window_years": np.nan,
                    "memory_tau_years": tau,
                    "best_model": "FAILED",
                    "error": repr(exc),
                })

    profile_df = pd.DataFrame(profile_rows)
    profile_df.to_csv(out_memory_profile_summary_csv, index=False)
    profile_df.to_csv(out_memory_profile_csv, index=False)  # Standard filename

    if long_rows:
        long_df = pd.concat(long_rows, ignore_index=True)
    else:
        long_df = pd.DataFrame()
    long_df.to_csv(out_memory_profile_diagnostic_long_csv, index=False)

    plot_memory_profile_scores(profile_df)
    plot_memory_score_heatmap(long_df, "cumulative", out_memory_heatmap_cumulative_png)
    plot_memory_score_heatmap(long_df, "rate", out_memory_heatmap_rate_png)
    plot_best_memory_residual_scans(profile_df, scan_cum_by_label, scan_rate_by_label)

    log(f"Saved memory profile summary: {out_memory_profile_summary_csv}", t0)
    log(f"Saved memory profile diagnostics: {out_memory_profile_diagnostic_long_csv}", t0)
    return profile_df



def validate_memory_profile_outputs():
    """Fail early if memory-profile output variables are missing."""
    required = [
        "out_memory_profile_nwrmse_png",
        "out_memory_profile_resid_score_png",
        "out_memory_heatmap_cumulative_png",
        "out_memory_heatmap_rate_png",
        "out_memory_profile_diagnostic_long_csv",
        "out_memory_profile_summary_csv",
        "out_memory_profile_best_cumulative_png",
        "out_memory_profile_best_rate_png",
    ]
    missing = [name for name in required if name not in globals()]
    if missing:
        raise RuntimeError(f"Missing memory-profile output variables: {missing}")



def validate_output_paths_for_plots():
    """
    Fail early if plot output variables referenced by run_main_analysis are not defined.
    """
    required = [
        "out_metric_png",
        "out_current_png",
        "out_abs_current_png",
        "out_memory_png",
        "out_area_png",
        "out_period_png",
        "out_std_current_png",
        "out_std_abs_current_png",
        "out_std_memory_png",
        "out_std_area_png",
        "out_std_period_png",
        "out_z_current_png",
        "out_z_abs_current_png",
        "out_z_memory_png",
        "out_z_area_png",
        "out_z_period_png",
        "out_period_component_png",
        "out_current_component_png",
        "out_period_correction_performance_png",
        "out_memory_scan_png",
        "out_memory_score_png",
        "out_memory_rate_scan_png",
        "out_memory_rate_score_png",
        "out_mean_fit_dir",
        "out_sigma_fit_signed_png",
        "out_sigma_fit_by_model_dir",
        "out_main_mean_figure_png",
        "out_main_std_figure_png",
        "out_supp_mean_figure_png",
        "out_supp_std_figure_png",
        "out_supp_normality_figure_png",
        "out_standardized_residuals_csv",
    ]
    missing = [name for name in required if name not in globals()]
    if missing:
        raise RuntimeError(f"Missing output-path variables: {missing}")



def validate_attach_predictions_signature():
    """Check target prediction helper call signatures."""
    dummy_target = pd.DataFrame({
        "current_center_w": np.array([-2.0, 2.0]),
        "memory_center_w": np.array([-1.0, 1.0]),
        "period_years": np.array([1.0, 5.0]),
        "rho_mean_w": np.array([800.0, 850.0]),
        "support_ok": np.array([True, True]),
        "weight_sum": np.array([1.0, 1.0]),
    })
    spec = dict(CANDIDATE_MODELS[0])
    names = params_for_forms(spec)
    theta = initial_for_params(names)
    fits = {"dummy": {"spec": spec, "theta": theta, "param_names": names}}
    out = attach_predictions_to_target(dummy_target, fits)
    out_optional = attach_predictions_to_target(dummy_target, fits, None, None)
    for obj in [out, out_optional]:
        if "rho_pred_dummy" not in obj.columns:
            raise RuntimeError("attach_predictions_to_target did not create prediction column.")
        if not np.all(np.isfinite(obj["rho_pred_dummy"].to_numpy(float))):
            raise RuntimeError("attach_predictions_to_target produced non-finite predictions.")



def validate_required_globals():
    """Fail early if critical global constants used by diagnostics are missing."""
    required = [
        "MEMORY_N_QUANTILE_BINS",
        "MEMORY_MIN_COUNT_BIN",
        "MEMORY_MIN_NEFF_BIN",
        "AREA_N_QUANTILE_BINS",
        "CURRENT_DH_MIN_ABS_FOR_FIT",
        "NORMALIZED_METRIC_SIGMA_FLOOR",
        "SIGMA_NUMERIC_FLOOR",
        "SIGMA_CANDIDATE_FORMS",
        "SIGMA_MODEL_FOR_DIAGNOSTICS",
        "SIGMA_LOGLINEAR_X0",
        "SIGMA_TWOSLOPE_ETA_FIXED",
        "SKIP_FULL_MEAN_MODEL_SEARCH",
        "FINAL_MEAN_MODEL_SPEC",
        "RUN_PAPER_FIGURES",
    ]
    missing = [name for name in required if name not in globals()]
    if missing:
        raise RuntimeError(f"Missing required global constants: {missing}")




def plot_sigma_fit_by_period(tab, theta_sig, out_png):
    """Plot candidate-selected sigma fit against binned residual STD by period."""
    if tab is None or tab.empty:
        return
    if "period_years" not in tab.columns:
        return
    periods = np.sort(tab["period_years"].unique())
    if len(periods) == 0:
        return
    cmap = plt.get_cmap(PLOT_CMAP)
    norm = mpl.colors.Normalize(vmin=np.nanmin(periods), vmax=np.nanmax(periods))
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    xmin = max(CURRENT_DH_MIN_ABS_FOR_FIT, 1e-4)
    xmax = max(np.nanmax(tab["abs_center_w"].to_numpy(float)), xmin * 10)
    xline = np.geomspace(xmin, xmax, 500)

    fig, ax = plt.subplots(figsize=(8.0, 5.5), constrained_layout=True)
    for T in periods:
        sub = tab.loc[tab["period_years"] == T].sort_values("abs_center_w")
        if sub.empty:
            continue
        col = cmap(norm(T))
        ax.scatter(sub["abs_center_w"], sub["resid_std_w"], s=22, color=col, alpha=0.8, edgecolors="none")
        yline = sigma_model(theta_sig, xline, np.full_like(xline, T, dtype=float))
        ax.plot(xline, yline, lw=1.8, color=col)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$|\Delta h|$ (m)")
    ax.set_ylabel(r"STD of mean-removed residual (kg m$^{-3}$)")
    ax.set_title(f"Selected sigma model: {globals().get('SIGMA_SELECTED_FORM', SIGMA_MODEL_FOR_DIAGNOSTICS)}")
    ax.grid(alpha=0.25)
    fig.colorbar(sm, ax=ax, pad=0.02).set_label("Period length (yr)")
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Optional joint exponential-vs-Gaussian-vs-spherical memory fit
# =============================================================================
def attach_annual_lag_matrix(current_rows, all_rows, kmax):
    """
    Attach annual lag dh columns for joint memory optimization.

    Lag k contains dh(t-k, t-k+1) for each current-period row.
    """
    out = current_rows.copy()
    base = out[["rgiid", variant_col, "start_date"]].copy()
    base["_row_id"] = np.arange(len(out))

    annual = all_rows.loc[np.isclose(all_rows["period_years"], 1.0), [
        "rgiid", variant_col, "start_date", "end_date", "signed_dh"
    ]].copy()
    annual = annual.drop_duplicates(subset=["rgiid", variant_col, "start_date", "end_date"])

    for k in range(1, int(kmax) + 1):
        lookup = annual.rename(columns={
            "start_date": "_annual_start",
            "end_date": "_annual_end",
            "signed_dh": f"lagdh_{k}",
        })
        need = base.copy()
        need["_annual_start"] = need["start_date"] - float(k)
        need["_annual_end"] = need["start_date"] - float(k - 1)
        m = need.merge(
            lookup[["rgiid", variant_col, "_annual_start", "_annual_end", f"lagdh_{k}"]],
            on=["rgiid", variant_col, "_annual_start", "_annual_end"],
            how="left",
        ).sort_values("_row_id")
        out[f"lagdh_{k}"] = m[f"lagdh_{k}"].to_numpy(float)

    lag_cols = [f"lagdh_{k}" for k in range(1, int(kmax) + 1)]
    out["has_joint_memory_lags"] = np.all(np.isfinite(out[lag_cols].to_numpy(float)), axis=1)
    return out, lag_cols


def joint_kernel_memory_weights(kmax, tau, kernel, normalize=None):
    """
    Fixed-shape memory weights.

    exponential:
        w_k = exp(-k/tau)

    spherical:
        w_k = 1 - 1.5 h + 0.5 h^3 for h=k/tau < 1, else 0
    """
    if normalize is None:
        normalize = JOINT_KERNEL_MEMORY_NORMALIZE_WEIGHTS

    k = np.arange(1, int(kmax) + 1, dtype=float)
    tau = float(tau)
    if tau <= 0 or not np.isfinite(tau):
        return np.full_like(k, np.nan, dtype=float)

    if kernel == "exponential":
        w = np.exp(-k / tau)
    elif kernel == "gaussian":
        w = np.exp(-((k / tau) ** 2))
    elif kernel == "spherical":
        h = k / tau
        w = np.where(h < 1.0, 1.0 - 1.5 * h + 0.5 * h**3, 0.0)
    else:
        raise ValueError(f"Unknown joint memory kernel: {kernel}")

    w = np.asarray(w, dtype=float)
    w[~np.isfinite(w)] = 0.0
    w[w < 0] = 0.0

    if normalize:
        sw = np.sum(w)
        if not np.isfinite(sw) or sw <= 0:
            return np.full_like(k, np.nan, dtype=float)
        w = w / sw

    return w


def joint_kernel_memory_from_lags(lag_mat, tau, kernel, normalize=None):
    weights = joint_kernel_memory_weights(
        lag_mat.shape[1], tau=tau, kernel=kernel, normalize=normalize
    )
    if not np.all(np.isfinite(weights)):
        return np.full(lag_mat.shape[0], np.nan, dtype=float)
    return lag_mat @ weights


def _build_joint_kernel_arrays(all_rows, t0=None):
    d = all_rows.loc[np.abs(all_rows["signed_dh"].to_numpy(float)) >= CURRENT_DH_MIN_ABS_FOR_FIT].copy()
    d = d.loc[
        np.isfinite(d["signed_dh"])
        & np.isfinite(d["period_years"])
        & np.isfinite(d[rho_col])
        & np.isfinite(d[weight_col])
        & (d[weight_col] > 0)
    ].copy()

    d, lag_cols = attach_annual_lag_matrix(d, all_rows, JOINT_KERNEL_MEMORY_KMAX_YEARS)
    d = d.loc[d["has_joint_memory_lags"]].copy()
    if d.empty:
        raise ValueError("No rows have complete annual lag history for joint kernel memory fit.")

    if JOINT_KERNEL_MEMORY_MAX_ROWS is not None and len(d) > int(JOINT_KERNEL_MEMORY_MAX_ROWS):
        d = d.sample(n=int(JOINT_KERNEL_MEMORY_MAX_ROWS), random_state=JOINT_KERNEL_MEMORY_RANDOM_SEED).copy()

    log(f"Joint kernel memory fit rows: {len(d):,}", t0)
    lag_mat = d[lag_cols].to_numpy(float)
    x = d["signed_dh"].to_numpy(float)
    T = d["period_years"].to_numpy(float)
    y = d[rho_col].to_numpy(float)
    w = d[weight_col].to_numpy(float)
    sqrtw = np.sqrt(w / np.nanmean(w))
    return d, lag_mat, x, T, y, w, sqrtw


def _unpack_joint_kernel_theta(theta_joint, base_param_names):
    theta_joint = np.asarray(theta_joint, dtype=float)
    theta_base = theta_joint[:len(base_param_names)]
    tau = float(theta_joint[len(base_param_names)])
    return theta_base, tau


def fit_one_joint_kernel_memory_model(shared, base_spec, kernel, t0=None):
    d, lag_mat, x, T, y, w, sqrtw = shared

    spec = dict(base_spec)
    spec["name"] = f"joint_{kernel}_memory"
    base_param_names = params_for_forms(spec)

    theta0_base = initial_for_params(base_param_names)
    lo_base, hi_base = bounds_for_params(base_param_names)

    tau0 = JOINT_KERNEL_MEMORY_TAU_INITIALS.get(kernel, 5.0)
    tau0 = float(np.clip(tau0, JOINT_KERNEL_MEMORY_TAU_BOUNDS[0], JOINT_KERNEL_MEMORY_TAU_BOUNDS[1]))

    lo = np.concatenate([lo_base, [JOINT_KERNEL_MEMORY_TAU_BOUNDS[0]]])
    hi = np.concatenate([hi_base, [JOINT_KERNEL_MEMORY_TAU_BOUNDS[1]]])

    def pred(theta_joint):
        theta_base, tau = _unpack_joint_kernel_theta(theta_joint, base_param_names)
        p = joint_kernel_memory_from_lags(lag_mat, tau=tau, kernel=kernel)
        return rho_model(theta_base, spec, base_param_names, x, p, T)

    eval_counter = {"n": 0}

    def residual(theta_joint):
        eval_counter["n"] += 1
        if JOINT_KERNEL_MEMORY_VERBOSE_EVERY_N_EVAL and eval_counter["n"] % int(JOINT_KERNEL_MEMORY_VERBOSE_EVERY_N_EVAL) == 0:
            log(f"Joint {kernel} eval {eval_counter['n']}: tau={theta_joint[-1]:.3g}", t0)
        r = (pred(theta_joint) - y) * sqrtw
        r[~np.isfinite(r)] = 1e12
        return r

    rng = np.random.default_rng(
        JOINT_KERNEL_MEMORY_RANDOM_SEED + {"exponential": 101, "gaussian": 151, "spherical": 202}.get(kernel, 303)
    )
    n_starts = max(1, int(JOINT_KERNEL_MEMORY_MAX_STARTS))
    base_starts = random_initials(rng, base_param_names, n_starts - 1)
    tau_starts = [tau0]
    for _ in range(n_starts - 1):
        tau_starts.append(float(np.exp(rng.uniform(
            np.log(JOINT_KERNEL_MEMORY_TAU_BOUNDS[0]),
            np.log(JOINT_KERNEL_MEMORY_TAU_BOUNDS[1]),
        ))))
    starts = [np.concatenate([bs, [ts]]) for bs, ts in zip(base_starts, tau_starts)]

    best = None
    first_exception = None
    for istart, theta0 in enumerate(starts, start=1):
        eval_counter["n"] = 0
        log(f"Joint {kernel} optimizer start {istart}/{len(starts)}: tau0={theta0[-1]:.3g}", t0)
        try:
            if not np.all(np.isfinite(residual(theta0))):
                raise FloatingPointError("non-finite initial residual")
            res = least_squares(
                residual,
                theta0,
                bounds=(lo, hi),
                loss="soft_l1",
                f_scale=500.0,
                max_nfev=JOINT_KERNEL_MEMORY_MAX_NFEV,
                ftol=JOINT_KERNEL_MEMORY_FTOL,
                xtol=JOINT_KERNEL_MEMORY_XTOL,
                gtol=JOINT_KERNEL_MEMORY_GTOL,
                x_scale="jac",
            )
        except Exception as exc:
            log(f"Joint {kernel} optimizer start {istart} failed after {eval_counter['n']} evals: {exc!r}", t0)
            if first_exception is None:
                first_exception = repr(exc)
            continue

        score = float(np.sum(res.fun ** 2))
        log(
            f"Joint {kernel} optimizer start {istart} done: evals={eval_counter['n']}, "
            f"cost={res.cost:.6g}, score={score:.6g}, tau={res.x[-1]:.3g}, "
            f"status={res.status}, optimality={res.optimality:.3g}",
            t0,
        )
        if best is None or score < best["score"]:
            best = {"theta": res.x, "score": score, "result": res}

    if best is None:
        raise RuntimeError(f"All joint {kernel} memory starts failed. First exception: {first_exception}")

    theta_base, tau_fit = _unpack_joint_kernel_theta(best["theta"], base_param_names)
    p_fit = joint_kernel_memory_from_lags(lag_mat, tau=tau_fit, kernel=kernel)
    yhat = rho_model(theta_base, spec, base_param_names, x, p_fit, T)
    metrics = weighted_metrics(y, yhat, w, len(best["theta"]))

    out = d.copy()
    out["joint_kernel"] = kernel
    out["joint_memory_dh"] = p_fit
    out["rho_pred_joint"] = yhat
    out["rho_resid_joint"] = y - yhat
    out["abs_signed_dh"] = np.abs(out["signed_dh"].to_numpy(float))
    out["log10_area"] = np.log10(out[area_col].astype(float))

    param_rows = []
    for name, val in zip(base_param_names, theta_base):
        param_rows.append({"kernel": kernel, "parameter": name, "value": float(val)})
    param_rows.append({"kernel": kernel, "parameter": "tau_mem", "value": float(tau_fit)})

    weights = joint_kernel_memory_weights(
        JOINT_KERNEL_MEMORY_KMAX_YEARS, tau=tau_fit, kernel=kernel
    )
    weight_rows = pd.DataFrame({
        "kernel": kernel,
        "tau_mem": float(tau_fit),
        "lag_years": np.arange(1, JOINT_KERNEL_MEMORY_KMAX_YEARS + 1, dtype=int),
        "weight": weights,
    })

    summary = {
        "model": f"joint_{kernel}_memory",
        "kernel": kernel,
        "n_rows": int(len(out)),
        "n_params": int(len(best["theta"])),
        "tau_mem": float(tau_fit),
        "memory_weights_normalized": bool(JOINT_KERNEL_MEMORY_NORMALIZE_WEIGHTS),
        "score": float(best["score"]),
        "base_memory_form": spec.get("memory"),
        "base_ratio_form": spec.get("ratio"),
        "base_damping_form": spec.get("damping"),
        "base_period_form": spec.get("period"),
        "base_current_form": spec.get("current"),
        **metrics,
    }

    log(f"Joint {kernel} memory fit complete: tau={tau_fit:.3g}", t0)
    return {"kernel": kernel, "summary": summary, "params": param_rows, "weights": weight_rows, "rows": out}


def summarize_joint_memory_diagnostics(df, value_col="rho_resid_joint"):
    d = df.loc[
        np.isfinite(df[value_col])
        & np.isfinite(df["joint_memory_dh"])
        & np.isfinite(df[weight_col])
        & (df[weight_col] > 0)
    ].copy()
    if d.empty:
        return pd.DataFrame()
    edges = make_signed_log_dh_edges(d["joint_memory_dh"].to_numpy(float))
    d["memory_bin_diag"] = pd.cut(d["joint_memory_dh"], bins=edges, labels=False, include_lowest=True)
    d = d.dropna(subset=["memory_bin_diag"]).copy()
    d["memory_bin_diag"] = d["memory_bin_diag"].astype(int)
    rows = []
    for (T, ib), g in d.groupby(["period_years", "memory_bin_diag"], sort=True):
        w = g[weight_col].to_numpy(float)
        n_eff = weighted_effective_n(w)
        if len(g) < MIN_COUNT_DIAG_BIN or not np.isfinite(n_eff) or n_eff < MIN_NEFF_DIAG_BIN:
            continue
        rows.append({
            "period_years": float(T),
            "memory_bin": int(ib),
            "memory_center_w": weighted_mean(g["joint_memory_dh"].to_numpy(float), w),
            "value_mean_w": weighted_mean(g[value_col].to_numpy(float), w),
            "value_std_w": weighted_std(g[value_col].to_numpy(float), w),
            "n": len(g),
            "n_eff": n_eff,
            "weight_sum": np.nansum(w),
        })
    return pd.DataFrame(rows)


def _plot_joint_memory_period_lines(ax, tab, ycol, title):
    if tab.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return None
    periods = np.sort(tab["period_years"].unique())
    cmap = plt.get_cmap(PLOT_CMAP)
    norm = mpl.colors.Normalize(vmin=np.nanmin(periods), vmax=np.nanmax(periods))
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    for T in periods:
        sub = tab.loc[tab["period_years"] == T].sort_values("memory_center_w")
        col = cmap(norm(T))
        for mask in [sub["memory_center_w"] < 0, sub["memory_center_w"] > 0]:
            ss = sub.loc[mask]
            if len(ss) >= 2:
                ax.plot(ss["memory_center_w"], ss[ycol], marker="o", lw=1.2, ms=3.5, color=col)
    ax.axhline(0, color="black", lw=1)
    ax.axvline(0, color="black", lw=1)
    set_signed_log_xaxis(ax, tab["memory_center_w"].to_numpy(float))
    ax.set_xlabel("Joint fitted past elevation change rate")
    ax.set_ylabel("Weighted mean residual")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    return sm


def plot_joint_kernel_memory_weights(weight_df, out_png):
    if weight_df is None or weight_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    for kernel, sub in weight_df.groupby("kernel", sort=False):
        sub = sub.sort_values("lag_years")
        tau = float(sub["tau_mem"].iloc[0])
        ax.plot(sub["lag_years"], sub["weight"], marker="o", lw=1.8, label=f"{kernel} (tau={tau:.3g})")
    ax.set_xlabel("Lag (yr)")
    ax.set_ylabel("Memory weight")
    ax.set_title("Joint-fitted memory-kernel weights")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_joint_kernel_memory_residuals(results, out_png):
    if not results:
        return
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(6.2 * n, 8.5), constrained_layout=True, squeeze=False)
    for j, res in enumerate(results):
        kernel = res["kernel"]
        out = res["rows"]

        current_edges = make_signed_log_dh_edges(out["signed_dh"].to_numpy(float))
        cur = summarize_period_current_diagnostics(out, "rho_resid_joint", current_edges)
        ax = axes[0, j]
        if not cur.empty:
            periods = np.sort(cur["period_years"].unique())
            cmap = plt.get_cmap(PLOT_CMAP)
            norm = mpl.colors.Normalize(vmin=np.nanmin(periods), vmax=np.nanmax(periods))
            _plot_period_current_panel(ax, cur, periods, cmap, norm, ycol="value_mean_w")
            sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
            fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02).set_label("Period length (yr)")
        ax.axhline(0, color="black", lw=1)
        ax.axvline(0, color="black", lw=1)
        set_signed_log_xaxis(ax, out["signed_dh"].to_numpy(float))
        ax.set_xlabel("Current dh (m)")
        ax.set_ylabel("Weighted mean residual")
        ax.set_title(f"{kernel}: residual vs current dh")
        ax.grid(alpha=0.25)

        mem_diag = summarize_joint_memory_diagnostics(out)
        _plot_joint_memory_period_lines(
            axes[1, j], mem_diag, "value_mean_w", f"{kernel}: residual vs fitted past elevation change rate"
        )
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_joint_kernel_observed_predicted(results, out_png):
    if not results:
        return
    fig, axes = plt.subplots(1, len(results), figsize=(5.6 * len(results), 5.2), constrained_layout=True, squeeze=False)
    axes = axes[0]
    for ax, res in zip(axes, results):
        out = res["rows"]
        sample = out
        if len(sample) > 500000:
            sample = sample.sample(n=500000, random_state=JOINT_KERNEL_MEMORY_RANDOM_SEED)
        ax.scatter(sample[rho_col], sample["rho_pred_joint"], s=2, alpha=0.08, linewidths=0)
        vals = np.concatenate([
            sample[rho_col].to_numpy(float),
            sample["rho_pred_joint"].to_numpy(float),
        ])
        vals = vals[np.isfinite(vals)]
        if len(vals):
            lo, hi = np.nanpercentile(vals, [0.5, 99.5])
            ax.plot([lo, hi], [lo, hi], color="black", lw=1)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
        ax.set_xlabel("Observed rho")
        ax.set_ylabel("Predicted rho")
        ax.set_title(res["kernel"])
        ax.grid(alpha=0.25)
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fit_joint_kernel_memory_models(all_rows, base_spec, t0=None):
    if not RUN_JOINT_KERNEL_MEMORY_FIT:
        return None

    log(f"Starting joint kernel memory comparison: {JOINT_KERNEL_MEMORY_FORMS}", t0)
    shared = _build_joint_kernel_arrays(all_rows, t0=t0)

    results = []
    errors = []
    for kernel in JOINT_KERNEL_MEMORY_FORMS:
        try:
            results.append(fit_one_joint_kernel_memory_model(shared, base_spec, kernel, t0=t0))
        except Exception as exc:
            log(f"Joint {kernel} memory fit failed: {exc!r}", t0)
            errors.append({"kernel": kernel, "error": repr(exc)})

    if results:
        summary = pd.DataFrame([r["summary"] for r in results]).sort_values("wrmse")
        params = pd.DataFrame([row for r in results for row in r["params"]])
        weights = pd.concat([r["weights"] for r in results], ignore_index=True)

        summary.to_csv(out_joint_kernel_memory_summary_csv, index=False)
        params.to_csv(out_joint_kernel_memory_params_csv, index=False)
        weights.to_csv(out_joint_kernel_memory_weights_csv, index=False)

        plot_joint_kernel_memory_weights(weights, out_joint_kernel_memory_weights_png)
        plot_joint_kernel_memory_residuals(results, out_joint_kernel_memory_residual_png)
        plot_joint_kernel_observed_predicted(results, out_joint_kernel_memory_scatter_png)
        log(f"Saved joint kernel memory comparison: {out_joint_kernel_memory_summary_csv}", t0)
    else:
        pd.DataFrame(errors).to_csv(out_joint_kernel_memory_summary_csv, index=False)

    if errors:
        pd.DataFrame(errors).to_csv(
            out_plot_dir / f"{run_label}_{main_suffix}_joint_kernel_memory_errors.csv",
            index=False,
        )

    return results




def choose_diagnostic_memory_from_joint_results(results, preferred_kernel="exponential", t0=None):
    """
    Pick the final memory configuration to use for all diagnostics / paper figures.

    Preference:
      1) the requested preferred kernel (default: exponential), if available;
      2) otherwise the best-performing joint kernel by WRMSE.

    Returns a dict with keys:
      mode, window, tau, kernel, wrmse
    or None if no valid joint fit is available.
    """
    if results is None or len(results) == 0:
        return None

    valid = []
    for r in results:
        try:
            kernel = str(r["kernel"])
            tau = float(r["summary"]["tau_mem"])
            wrmse = float(r["summary"]["wrmse"])
        except Exception:
            continue
        if np.isfinite(tau) and np.isfinite(wrmse):
            valid.append({"kernel": kernel, "tau": tau, "wrmse": wrmse})

    if len(valid) == 0:
        return None

    chosen = None
    for row in valid:
        if row["kernel"] == preferred_kernel:
            chosen = row
            break
    if chosen is None:
        chosen = sorted(valid, key=lambda d: d["wrmse"])[0]

    log(
        f"Using joint-memory kernel for diagnostics: {chosen['kernel']} "
        f"(tau={chosen['tau']:.3g}, wrmse={chosen['wrmse']:.6g})",
        t0
    )
    return {
        "mode": "exp_cumulative",
        "window": None,
        "tau": float(chosen["tau"]),
        "kernel": chosen["kernel"],
        "wrmse": float(chosen["wrmse"]),
    }

def maybe_run_joint_kernel_memory_fit(all_rows, selected_spec, t0=None):
    if not RUN_JOINT_KERNEL_MEMORY_FIT:
        return None
    try:
        return fit_joint_kernel_memory_models(all_rows, selected_spec, t0=t0)
    except Exception as exc:
        log(f"Joint kernel memory comparison failed: {exc!r}", t0)
        pd.DataFrame([{"error": repr(exc)}]).to_csv(out_joint_kernel_memory_summary_csv, index=False)
        return None


# =============================================================================
# Validation and main
# =============================================================================

def validate_model_forms():
    x=np.array([-10,-2,2,10],float); p=np.array([-5,1,-1,5],float); T=np.array([1,2,5,10],float)
    for spec in CANDIDATE_MODELS:
        names=params_for_forms(spec); theta=initial_for_params(names); y=rho_model(theta,spec,names,x,p,T)
        if not np.all(np.isfinite(y)): raise RuntimeError(f"Initial {spec['name']} returns non-finite values")
        xp,xn=np.abs(x),-np.abs(x); comp_p=component_values(theta,spec,names,xp,p,T); comp_n=component_values(theta,spec,names,xn,p,T)
        if not np.allclose(comp_p["period_component"],comp_n["period_component"],equal_nan=True): raise RuntimeError(f"Period component not even for {spec['name']}")
        if not np.allclose(comp_p["current_component"],comp_n["current_component"],equal_nan=True): raise RuntimeError(f"Current component not even for {spec['name']}")






# -----------------------------------------------------------------------------
# Plot refinements overrides (paper polish)
# -----------------------------------------------------------------------------
PLOT_CMAP_MEMORY = "coolwarm_r"
MAIN_SMALLBIN_DISPLAY_CENTER = 0.5
MAIN_SMALLBIN_EDGE_LOW = 0.05
MAIN_SMALLBIN_EDGE_HIGH = 1.0
MAIN_MEAN_LEFT_MIN_ABS_DH_DISPLAY = 0.05
SUPP_MIN_COUNT_MULTIPLIER = 2


def _add_panel_letter(ax, letter):
    ax.text(0.02, 0.98, letter, transform=ax.transAxes, ha="left", va="top", fontsize=14, fontweight="bold")


def _collapse_nearzero_current_display(df, x_col, weight_col="weight_sum",
                                       y_cols=("rho_obs_w", "rho_pred_center", "rho_pred_w"),
                                       group_cols=("period_years",), threshold=1.0, display_center=0.5):
    """Collapse bins with 0<|current dh|<threshold into one display bin per sign and group."""
    if df is None or len(df) == 0:
        return df
    d = df.copy()
    use_group_cols = [c for c in group_cols if c in d.columns]
    out = []
    groups = d.groupby(use_group_cols, dropna=False, sort=False) if use_group_cols else [((), d)]
    for _, sub in groups:
        x = sub[x_col].to_numpy(float)
        keep = ~np.isfinite(x) | (np.abs(x) >= threshold) | (x == 0)
        if np.any(keep):
            out.append(sub.loc[keep].copy())
        for sgn in (-1.0, 1.0):
            mask = np.isfinite(x) & (x * sgn > 0) & (np.abs(x) < threshold)
            if not np.any(mask):
                continue
            g = sub.loc[mask].copy()
            row = g.iloc[[0]].copy()
            w = g[weight_col].to_numpy(float) if weight_col in g.columns else np.ones(len(g), dtype=float)
            if (not np.isfinite(w).any()) or np.nansum(w) <= 0:
                w = np.ones(len(g), dtype=float)
            row.loc[:, x_col] = sgn * float(display_center)
            if "x_center_w" in row.columns:
                row.loc[:, "x_center_w"] = sgn * float(display_center)
            if "current_center_w" in row.columns:
                row.loc[:, "current_center_w"] = sgn * float(display_center)
            if weight_col in row.columns:
                row.loc[:, weight_col] = float(np.nansum(w))
            if "n" in row.columns:
                row.loc[:, "n"] = int(np.nansum(g["n"].to_numpy(float))) if "n" in g.columns else len(g)
            if "n_eff" in row.columns:
                row.loc[:, "n_eff"] = float(np.nansum(g["n_eff"].to_numpy(float)))
            for yc in y_cols:
                if yc in g.columns:
                    row.loc[:, yc] = weighted_mean(g[yc].to_numpy(float), w)
            out.append(row)
    if not out:
        return d.iloc[0:0].copy()
    out = pd.concat(out, ignore_index=True)
    sort_cols = [c for c in [*use_group_cols, x_col] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def _filter_supp_support(df):
    if df is None or len(df) == 0:
        return df
    d = df.copy()
    min_n = int(MIN_COUNT_DIAG_BIN * SUPP_MIN_COUNT_MULTIPLIER)
    min_neff = float(MIN_NEFF_DIAG_BIN * SUPP_MIN_COUNT_MULTIPLIER)
    if "n" in d.columns:
        d = d.loc[d["n"].to_numpy(float) >= min_n].copy()
    if "n_eff" in d.columns:
        d = d.loc[d["n_eff"].to_numpy(float) >= min_neff].copy()
    return d




def _set_sparse_signed_ticks(ax):
    ticks = np.array([-50.0, -10.0, -2.0, -0.5, 0.0, 0.5, 2.0, 10.0, 50.0], dtype=float)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])


def _set_main_mean_signed_ticks(ax):
    ticks = np.array([-50.0, -20.0, -5.0, -2.0, -0.5, 0.0, 0.5, 2, 5, 20.0, 50.0], dtype=float)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])


def _set_main_mean_linear_ticks(ax):
    ticks = np.array([-50.0, -25.0, 0.0, 25.0, 50.0], dtype=float)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])

def plot_main_rho_mean_figure(target, selected_model, d_pred, fit_obj, d_pred_no_period, no_period_fit, memory_col, out_png, model_line_mode="smooth_binned", x_axis_mode="log"):
    """Main rho-mean figure polished for the paper."""
    from matplotlib.lines import Line2D

    if x_axis_mode not in ("log", "linear"):
        raise ValueError(f"Unknown x_axis_mode: {x_axis_mode}")
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.55), constrained_layout=True, sharey=True)
    ax_mem, ax_per = axes
    if x_axis_mode == "linear":
        xgrid = np.concatenate([
            np.linspace(-MAIN_DENSE_X_MAX, -MAIN_DENSE_X_MIN, int(MAIN_DENSE_X_N)),
            np.linspace(MAIN_DENSE_X_MIN, MAIN_DENSE_X_MAX, int(MAIN_DENSE_X_N)),
        ])
    else:
        xgrid = _main_dense_signed_support()

    # Panel a: past elevation change rate dependence at fixed 5-year period
    target_right = _filter_main_mean_right_panel_target_to_period(target, target_year=5.0)
    by_memory = _aggregate_target_pair(
        target=target_right,
        model=selected_model,
        x_col="current_center_w",
        x_bin_col="current_bin",
        group_col="memory_center_w",
        group_bin_col="memory_bin",
        fit_obj=fit_obj,
    )
    if by_memory.empty:
        ax_mem.text(0.5, 0.5, "No target data", ha="center", va="center", transform=ax_mem.transAxes)
    else:
        mem_lookup = (
            by_memory.groupby("group_bin", as_index=False)
            .agg(group_center_ref=("group_center_w", "mean"))
            .sort_values("group_center_ref")
        )
        actual_vals, actual_labels = _pick_representative_memory_groups(mem_lookup["group_center_ref"].to_numpy(float))
        sel_bins = []
        sel_map = {}
        sel_actual = {}
        for v, lab in zip(actual_vals, actual_labels):
            idx = int(np.argmin(np.abs(mem_lookup["group_center_ref"].to_numpy(float) - v)))
            gb = mem_lookup["group_bin"].iloc[idx]
            if gb not in sel_bins:
                sel_bins.append(gb)
                sel_map[gb] = lab
                sel_actual[gb] = mem_lookup["group_center_ref"].iloc[idx]
        plot_df = by_memory.loc[by_memory["group_bin"].isin(sel_bins)].copy()
        label_vals = np.asarray(MAIN_MEMORY_TARGET_LABELS, dtype=float)
        vmax = np.nanmax(np.abs(label_vals)) if len(label_vals) else 1.0
        vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
        cmap_mem = plt.get_cmap(PLOT_CMAP_MEMORY)
        norm_mem = mpl.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        plot_df["color_center_w"] = plot_df["group_bin"].map(sel_map).astype(float)
        colors = cmap_mem(norm_mem(plot_df["color_center_w"].to_numpy(float)))
        ax_mem.scatter(
            plot_df["x_center_w"].to_numpy(float),
            plot_df["rho_obs_w"].to_numpy(float),
            s=28, c=colors, alpha=0.70, marker="o", linewidths=0,
        )
        for gb, sub in plot_df.groupby("group_bin", sort=True):
            mem_val = float(sel_actual.get(gb, weighted_mean(sub["group_center_w"].to_numpy(float), sub["weight_sum"].to_numpy(float))))
            lab_val = float(sel_map.get(gb, mem_val))
            col = cmap_mem(norm_mem(lab_val))
            xfit_anchor = sub["x_center_w"].to_numpy(float)
            yfit_anchor = sub["rho_pred_w"].to_numpy(float)
            if model_line_mode == "analytical":
                period_val = 5.0
                yfit_plot = rho_model(
                    fit_obj["theta"], fit_obj["spec"], fit_obj["param_names"],
                    xgrid,
                    np.full_like(xgrid, float(mem_val), dtype=float),
                    np.full_like(xgrid, float(period_val), dtype=float),
                )
                _plot_dense_signed_curve(ax_mem, xgrid, yfit_plot, col, lw=2.0, alpha=0.95, connect_zero=False)
            else:
                _plot_mean_model_from_anchors(ax_mem, xfit_anchor, yfit_anchor, col, xgrid, min_abs=None, mode=model_line_mode, lw=2.0, alpha=0.95)
        ax_mem.axhline(RHO_ICE_FIXED, color="black", lw=1, ls="--")
        ax_mem.axvline(0, color="black", lw=1)
        if x_axis_mode == "linear":
            ax_mem.set_xscale("linear")
            _set_main_mean_linear_ticks(ax_mem)
        else:
            set_signed_log_xaxis(ax_mem, np.concatenate([plot_df["x_center_w"].to_numpy(float), np.array([-50.0, 50.0])]))
            _set_main_mean_signed_ticks(ax_mem)
        ax_mem.set_xlim(-50.0, 50.0)
        ax_mem.grid(alpha=0.25)
        color_handles = [
            Line2D([], [], marker="o", ms=5, lw=1.8, color=cmap_mem(norm_mem(float(lbl))), label=_format_memory_label(float(lbl)))
            for lbl in label_vals
        ]
        leg1 = ax_mem.legend(handles=color_handles, title="Past elevation\nchange rate " + r"$\dot{h}^{\rm p}$", frameon=False, fontsize=9, title_fontsize=10, loc="lower left")
        ax_mem.add_artist(leg1)
        ax_mem.legend(handles=_main_style_legend_handles(), frameon=False, fontsize=10, loc="lower right")

    # Panel b: neutral past elevation change rate dependence colored by period
    neutral_tab = _neutral_memory_mean_panel_table(d_pred, fit_obj, selected_model)
    neutral_tab.to_csv(out_plot_dir / f"{run_label}_{main_suffix}_main_neutral_memory_mean_panel_summary.csv", index=False)
    if neutral_tab.empty:
        ax_per.text(0.5, 0.5, "No neutral-memory transformed data", ha="center", va="center", transform=ax_per.transAxes)
    else:
        neutral_plot = neutral_tab.loc[np.isfinite(neutral_tab["current_center_w"])].copy()
        if PERIOD_LENGTHS_TO_PLOT is not None:
            neutral_plot = neutral_plot.loc[neutral_plot["period_years"].isin(PERIOD_LENGTHS_TO_PLOT)].copy()
        # Keep the native current-dh bins on this panel, removing only the noisy
        # Smallest near-zero bin (centered around 0.1 m)
        neutral_plot = neutral_plot.loc[np.abs(neutral_plot["current_center_w"].to_numpy(float)) >= 0.2].copy()
        periods = np.sort(neutral_plot["period_years"].unique()) if len(neutral_plot) else np.array([], dtype=float)
        if neutral_plot.empty or len(periods) == 0:
            ax_per.text(0.5, 0.5, "No transformed data", ha="center", va="center", transform=ax_per.transAxes)
        else:
            cmap_per = plt.get_cmap(PLOT_CMAP_PERIOD)
            norm_per = mpl.colors.Normalize(vmin=np.nanmin(periods), vmax=np.nanmax(periods))
            for T in periods:
                sub = neutral_plot.loc[neutral_plot["period_years"] == T].sort_values("current_center_w")
                if sub.empty:
                    continue
                col = cmap_per(norm_per(float(T)))
                xobs = sub["current_center_w"].to_numpy(float)
                yobs = sub["rho_obs_w"].to_numpy(float)
                yfit_anchor = sub["rho_pred_center"].to_numpy(float)
                ax_per.scatter(xobs, yobs, s=24, color=col, alpha=0.75, edgecolors="none")
                if model_line_mode == "analytical":
                    yfit_plot = rho_model(
                        fit_obj["theta"], fit_obj["spec"], fit_obj["param_names"],
                        xgrid,
                        np.zeros_like(xgrid, dtype=float),
                        np.full_like(xgrid, float(T), dtype=float),
                    )
                    _plot_dense_signed_curve(ax_per, xgrid, yfit_plot, col, lw=1.8, alpha=0.95, connect_zero=False)
                else:
                    _plot_mean_model_from_anchors(ax_per, xobs, yfit_anchor, col, xgrid, min_abs=None, mode=model_line_mode, lw=1.8, alpha=0.95)
            ax_per.axhline(RHO_ICE_FIXED, color="black", lw=1, ls="--")
            ax_per.axvline(0, color="black", lw=1)
            if x_axis_mode == "linear":
                ax_per.set_xscale("linear")
                _set_main_mean_linear_ticks(ax_per)
            else:
                set_signed_log_xaxis(ax_per, np.concatenate([neutral_plot["current_center_w"].to_numpy(float), np.array([-50.0, 50.0])]))
                _set_main_mean_signed_ticks(ax_per)
            ax_per.set_xlim(-50.0, 50.0)
            ax_per.grid(alpha=0.25)
            color_handles = [
                Line2D([], [], marker="o", ms=5, lw=1.8, color=cmap_per(norm_per(float(v))), label=_format_period_label(float(v)))
                for v in periods
            ]
            leg1 = ax_per.legend(handles=color_handles, title=r"Period length $\Delta t$", frameon=False, fontsize=9, title_fontsize=10, loc="lower left")
            ax_per.add_artist(leg1)
            ax_per.legend(handles=_main_style_legend_handles(), frameon=False, fontsize=10, loc="lower right")

    ax_mem.set_ylim(-100.0, 1700.0)
    ax_per.set_ylim(-100.0, 1700.0)
    ax_mem.text(0.98, 0.98, "Period length = 5 yr", transform=ax_mem.transAxes, ha="right", va="top", fontsize=10)
    ax_per.text(0.98, 0.98, "Past elevation change rate = 0 m yr$^{-1}$", transform=ax_per.transAxes, ha="right", va="top", fontsize=10)
    ax_mem.set_xlabel(r"Elevation change $\Delta h$ (m)")
    ax_per.set_xlabel(r"Elevation change $\Delta h$ (m)")
    ax_mem.set_ylabel(r"Mean of effective density $\mu_{\rho}$ (kg m$^{-3}$)")
    _add_panel_letter(ax_mem, "a")
    _add_panel_letter(ax_per, "b")
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _apply_decimal_log_ticks(ax, axis="both", ticks=None):
    if ticks is None:
        ticks = [0.1, 1, 10, 100, 1000, 10000]
    ticks = [float(t) for t in ticks]
    fmt = mpl.ticker.FuncFormatter(lambda x, pos: (f"{x:g}" if np.isfinite(x) and x > 0 else ""))
    if axis in ("x", "both"):
        xmin, xmax = ax.get_xlim()
        xt = [t for t in ticks if xmin <= t <= xmax]
        if xt:
            ax.set_xticks(xt)
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    if axis in ("y", "both"):
        ymin, ymax = ax.get_ylim()
        yt = [t for t in ticks if ymin <= t <= ymax]
        if yt:
            ax.set_yticks(yt)
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())


def _apply_area_log_ticks(ax, values=None):
    if values is not None:
        vv = np.asarray(values, float)
        vv = vv[np.isfinite(vv) & (vv > 0)]
    else:
        vv = np.array([], dtype=float)
    if len(vv):
        lo = 10 ** np.floor(np.log10(np.nanmin(vv)))
        hi = 10 ** np.ceil(np.log10(np.nanmax(vv)))
    else:
        lo, hi = ax.get_xlim()
    ticks = []
    p0 = int(np.floor(np.log10(lo))) if lo > 0 else 0
    p1 = int(np.ceil(np.log10(hi))) if hi > 0 else 4
    for p in range(p0, p1 + 1):
        for m in (1, 2, 5):
            t = m * (10 ** p)
            if lo * 0.999 <= t <= hi * 1.001:
                ticks.append(float(t))
    _apply_decimal_log_ticks(ax, axis="x", ticks=sorted(set(ticks)))


def plot_main_rho_std_figure(d_pred, theta_sig, out_png, x_axis_mode="log"):
    """Main STD figure: single-panel polished version."""
    from matplotlib.lines import Line2D
    tab = summarize_absdh_residual_std_for_sigma(d_pred, "rho_mean_removed")
    if tab.empty:
        return
    tab = tab.loc[np.isfinite(tab["abs_center_w"]) & (tab["abs_center_w"] <= SIGMA_MAX_ABS_DH_FOR_FIT)].copy()
    if tab.empty:
        return
    periods = np.sort(tab["period_years"].unique()) if "period_years" in tab.columns else np.array([np.nan])
    periods = _filter_display_periods(periods)
    cmap = plt.get_cmap(PLOT_CMAP_PERIOD)
    norm = mpl.colors.Normalize(vmin=np.nanmin(periods), vmax=np.nanmax(periods)) if len(periods) and np.all(np.isfinite(periods)) else mpl.colors.Normalize(vmin=0, vmax=1)
    if x_axis_mode not in ("log", "linear"):
        raise ValueError(f"Unknown x_axis_mode: {x_axis_mode}")
    xmax = min(max(np.nanmax(tab["abs_center_w"].to_numpy(float)), CURRENT_DH_MIN_ABS_FOR_FIT * 10), SIGMA_MAX_ABS_DH_FOR_FIT)
    xmin = max(CURRENT_DH_MIN_ABS_FOR_FIT, 1e-4)
    if x_axis_mode == "linear":
        xline = np.linspace(xmin, xmax * 1.02, 600)
    else:
        xline = np.geomspace(xmin, xmax * 1.02, 600)
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 5.0), constrained_layout=True)
    sigma_form_for_plot, sigma_params_for_plot = _resolve_sigma_form_and_params(theta_sig)
    y_all = []
    avail = np.sort(tab["period_years"].dropna().unique().astype(float)) if "period_years" in tab.columns else np.array([])
    for T in periods:
        Tuse = avail[int(np.argmin(np.abs(avail - float(T))))] if len(avail) else T
        sub = tab.loc[np.isclose(tab["period_years"], Tuse)].sort_values("abs_center_w") if "period_years" in tab.columns else tab.sort_values("abs_center_w")
        if sub.empty:
            continue
        col = cmap(norm(T)) if np.isfinite(T) else "black"
        ax.scatter(sub["abs_center_w"], sub["resid_std_w"], s=24, color=col, alpha=0.8, edgecolors="none")
        yline = np.maximum(sigma_model_form(theta_sig, xline, period_years=np.full_like(xline, float(Tuse), dtype=float), form=sigma_form_for_plot, param_names=sigma_params_for_plot), SIGMA_NUMERIC_FLOOR)
        ok = np.isfinite(yline) & (yline > 0)
        if np.any(ok):
            ax.plot(xline[ok], yline[ok], lw=1.8, color=col)
            y_all.append(yline[ok])
        vals = sub["resid_std_w"].to_numpy(float)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if len(vals):
            y_all.append(vals)
    if x_axis_mode == "linear":
        ax.set_xscale("linear")
    else:
        ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Absolute elevation change $|\Delta h|$ (m)")
    ax.set_ylabel(r"Uncertainty in effective density $\sigma_{\rho}$ (kg m$^{-3}$)")
    ax.grid(alpha=0.25)
    if y_all:
        yy = np.concatenate([np.asarray(v, dtype=float) for v in y_all])
        yy = yy[np.isfinite(yy) & (yy > 0)]
        if len(yy):
            ax.set_ylim(max(np.nanmin(yy) * 0.85, 1e-3), np.nanmax(yy) * 1.15)
    if x_axis_mode == "linear":
        ax.set_xticks([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
        ax.set_xticklabels(["0", "10", "20", "30", "40", "50"])
        _apply_decimal_log_ticks(ax, axis="y")
    else:
        _apply_decimal_log_ticks(ax, axis="both")
    if len(periods) > 1:
        handles = [Line2D([], [], marker="o", ms=5, lw=1.8, color=cmap(norm(float(v))), label=_format_period_label(float(v))) for v in periods]
        leg1 = ax.legend(handles=handles, title=r"Period length $\Delta t$", frameon=False, fontsize=9, title_fontsize=10, loc="upper right")
        ax.add_artist(leg1)
    ax.legend(handles=_main_style_legend_handles(), frameon=False, fontsize=10, loc="lower left")
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_supp_panel(ax, suball, ycol, periods, x_label, y_label,
                     signed_log_x=False, log_x=False, hline=None, ylim=None):
    cmap = plt.get_cmap(PLOT_CMAP_PERIOD)
    norm = mpl.colors.Normalize(vmin=np.nanmin(periods), vmax=np.nanmax(periods))
    for T in periods:
        sub = suball.loc[suball["period_years"] == T].sort_values("factor_center_w")
        if sub.empty:
            continue
        col = cmap(norm(T))
        if signed_log_x:
            for mask in [sub["factor_center_w"] < 0, sub["factor_center_w"] > 0]:
                ss = sub.loc[mask].sort_values("factor_center_w")
                if len(ss) >= 2:
                    ax.plot(ss["factor_center_w"], ss[ycol], marker="o", ms=3.0, lw=1.0, color=col)
        elif len(sub) >= 2:
            ax.plot(sub["factor_center_w"], sub[ycol], marker="o", ms=3.0, lw=1.0, color=col)
    if hline is not None:
        ax.axhline(hline, color="black", lw=0.9, ls="--")
    if signed_log_x:
        ax.axvline(0, color="black", lw=0.8)
        set_signed_log_xaxis(ax, suball["factor_center_w"].to_numpy(float))
    if log_x:
        ax.set_xscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    return sm


def plot_supp_rho_mean_figure(diag, out_png):
    specs = [
        ("Current signed dh", r"Elevation change $\Delta h$ (m)", True, False),
        ("Past elevation change", r"Past elevation change rate $\dot{h}^{\rm p}$ (m yr$^{-1}$)", False, False),
        ("Area", r"Glacier area $A$", False, True),
    ]
    d = diag.loc[diag["factor"].isin([sp[0] for sp in specs])].copy()
    d = _filter_supp_support(d)
    if d.empty:
        return
    mcur = d["factor"].eq("Current signed dh")
    d = d.loc[~(mcur & (np.abs(d["factor_center_w"].to_numpy(float)) < 0.2))].copy()
    periods = _filter_display_periods(np.sort(d["period_years"].unique()))
    if len(periods) == 0:
        return
    raw = d["raw_mean_w"].to_numpy(float); raw = raw[np.isfinite(raw)]
    resid = d["resid_mean_w"].to_numpy(float); resid = resid[np.isfinite(resid)]
    halfspan = 100.0
    if len(raw): halfspan = max(halfspan, float(np.nanmax(np.abs(raw - RHO_ICE_FIXED))))
    if len(resid): halfspan = max(halfspan, float(np.nanmax(np.abs(resid))))
    halfspan *= 1.05
    ylim_top = (RHO_ICE_FIXED - halfspan, RHO_ICE_FIXED + halfspan)
    ylim_bottom = (-halfspan, halfspan)
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 6.6), constrained_layout=True, sharey="row")
    sm = None
    labels = list("abcdef")
    k = 0
    for j, (factor, xlabel, signed_x, log_x) in enumerate(specs):
        sub = d.loc[d["factor"] == factor]
        ylab_top = "Binned mean of\neffective density " + r"$\mu_{\rho}$" + "\n" + r"(kg m$^{-3}$)" if j == 0 else ""
        sm = _plot_supp_panel(axes[0, j], sub, "raw_mean_w", periods, "", ylab_top, signed_log_x=signed_x, log_x=log_x, hline=RHO_ICE_FIXED, ylim=ylim_top)
        ylab_bot = "Residual mean after\nfitted model " + r"(kg m$^{-3}$)" if j == 0 else ""
        _plot_supp_panel(axes[1, j], sub, "resid_mean_w", periods, xlabel, ylab_bot, signed_log_x=signed_x, log_x=log_x, hline=0, ylim=ylim_bottom)
        if factor == "Current signed dh":
            _set_sparse_signed_ticks(axes[0, j])
            _set_sparse_signed_ticks(axes[1, j])
        if factor == "Area":
            _apply_area_log_ticks(axes[0, j], sub["factor_center_w"].to_numpy(float))
            _apply_area_log_ticks(axes[1, j], sub["factor_center_w"].to_numpy(float))
        _add_panel_letter(axes[0, j], labels[k]); k += 1
        _add_panel_letter(axes[1, j], labels[k]); k += 1
    if sm is not None:
        fig.colorbar(sm, ax=axes, pad=0.015, shrink=0.95).set_label(r"Period length $\Delta t$ (yr)")
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_supp_rho_std_figure(diag, out_png):
    specs = [
        ("Current signed dh", r"Elevation change $\Delta h$ (m)", True, False),
        ("Past elevation change", r"Past elevation change rate $\dot{h}^{\rm p}$ (m yr$^{-1}$)", False, False),
        ("Area", r"Glacier area $A$", False, True),
    ]
    d = diag.loc[diag["factor"].isin([sp[0] for sp in specs])].copy()
    d = _filter_supp_support(d)
    if d.empty:
        return
    mcur = d["factor"].eq("Current signed dh")
    d = d.loc[~(mcur & (np.abs(d["factor_center_w"].to_numpy(float)) < 0.2))].copy()
    periods = _filter_display_periods(np.sort(d["period_years"].unique()))
    if len(periods) == 0:
        return
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 6.6), constrained_layout=True, sharey="row")
    sm = None
    labels = list("abcdef")
    k = 0
    for j, (factor, xlabel, signed_x, log_x) in enumerate(specs):
        sub = d.loc[d["factor"] == factor]
        ylab_top = "Binned uncertainty in\neffective density " + r"$\sigma_{\rho}$" + "\n" + r"(kg m$^{-3}$)" if j == 0 else ""
        sm = _plot_supp_panel(axes[0, j], sub, "raw_std_w", periods, "", ylab_top, signed_log_x=signed_x, log_x=log_x, ylim=None)
        ycol = "z_after_std_w" if "z_after_std_w" in sub.columns else "resid_std_w"
        ylab = "STD after standardization\nby fitted model " + r"$z_{\rho}$" if ycol == "z_after_std_w" else "Residual STD"
        _plot_supp_panel(axes[1, j], sub, ycol, periods, xlabel, ylab if j == 0 else "", signed_log_x=signed_x, log_x=log_x, hline=1 if ycol == "z_after_std_w" else None, ylim=(0, 5))
        if factor == "Current signed dh":
            _set_sparse_signed_ticks(axes[0, j])
            _set_sparse_signed_ticks(axes[1, j])
        if factor == "Area":
            _apply_area_log_ticks(axes[0, j], sub["factor_center_w"].to_numpy(float))
            _apply_area_log_ticks(axes[1, j], sub["factor_center_w"].to_numpy(float))
        _add_panel_letter(axes[0, j], labels[k]); k += 1
        _add_panel_letter(axes[1, j], labels[k]); k += 1
    if sm is not None:
        fig.colorbar(sm, ax=axes, pad=0.015, shrink=0.95).set_label(r"Period length $\Delta t$ (yr)")
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Joint-memory final-run overrides and memory-lean implementation
# -----------------------------------------------------------------------------
# The final model run must perform the joint kernel-memory fit. Keep row-heavy
# Joint diagnostic plots disabled by default; the fitted tau/parameters are still
# Saved, and the selected tau is used for all final diagnostics and figures
RUN_JOINT_KERNEL_MEMORY_FIT = True
USE_JOINT_EXPONENTIAL_MEMORY_FOR_DIAGNOSTICS = True
JOINT_KERNEL_MEMORY_MAX_ROWS = None
RUN_JOINT_KERNEL_DIAGNOSTIC_PLOTS = False


def _build_joint_kernel_arrays(all_rows, t0=None):
    """
    Memory-lean construction of arrays for the joint memory-kernel fit.

    The original implementation appended all lag columns to a full row-level
    DataFrame, which is expensive for tens of millions of rows. This version
    keeps only the columns needed by the optimizer and stores the annual-lag
    matrix as a single float32 NumPy array.
    """
    needed_cols = [
        "rgiid", variant_col, "start_date", "end_date", "period_years",
        "signed_dh", rho_col, area_col, weight_col,
    ]
    needed_cols = [c for c in needed_cols if c in all_rows.columns]

    mask = (
        (np.abs(all_rows["signed_dh"].to_numpy(float)) >= CURRENT_DH_MIN_ABS_FOR_FIT)
        & np.isfinite(all_rows["signed_dh"].to_numpy(float))
        & np.isfinite(all_rows["period_years"].to_numpy(float))
        & np.isfinite(all_rows[rho_col].to_numpy(float))
        & np.isfinite(all_rows[weight_col].to_numpy(float))
        & (all_rows[weight_col].to_numpy(float) > 0)
    )
    d = all_rows.loc[mask, needed_cols].copy()
    del mask
    gc.collect()

    if d.empty:
        raise ValueError("No eligible rows for joint kernel memory fit.")

    d["_row_id"] = np.arange(len(d), dtype=np.int64)
    n = len(d)
    kmax = int(JOINT_KERNEL_MEMORY_KMAX_YEARS)
    lag_mat = np.empty((n, kmax), dtype=np.float32)
    valid = np.ones(n, dtype=bool)

    base = d[["rgiid", variant_col, "start_date", "_row_id"]].copy()
    annual = all_rows.loc[
        np.isclose(all_rows["period_years"].to_numpy(float), 1.0),
        ["rgiid", variant_col, "start_date", "end_date", "signed_dh"],
    ].copy()
    annual = annual.drop_duplicates(subset=["rgiid", variant_col, "start_date", "end_date"])

    for k in range(1, kmax + 1):
        log(f"Building joint-memory annual lag {k}/{kmax}", t0)
        lookup = annual.rename(columns={
            "start_date": "_annual_start",
            "end_date": "_annual_end",
            "signed_dh": "_lagdh",
        })
        need = base.copy()
        need["_annual_start"] = need["start_date"] - float(k)
        need["_annual_end"] = need["start_date"] - float(k - 1)
        m = need.merge(
            lookup[["rgiid", variant_col, "_annual_start", "_annual_end", "_lagdh"]],
            on=["rgiid", variant_col, "_annual_start", "_annual_end"],
            how="left",
        ).sort_values("_row_id")
        vals = m["_lagdh"].to_numpy(dtype=np.float32, copy=True)
        lag_mat[:, k - 1] = vals
        valid &= np.isfinite(vals)
        del lookup, need, m, vals
        gc.collect()

    del annual, base
    gc.collect()

    if not np.any(valid):
        raise ValueError("No rows have complete annual lag history for joint kernel memory fit.")

    if np.count_nonzero(valid) < len(valid):
        d = d.loc[valid].copy()
        lag_mat = lag_mat[valid, :].copy()
        valid = None
        gc.collect()

    if JOINT_KERNEL_MEMORY_MAX_ROWS is not None and len(d) > int(JOINT_KERNEL_MEMORY_MAX_ROWS):
        rng = np.random.default_rng(JOINT_KERNEL_MEMORY_RANDOM_SEED)
        take = np.sort(rng.choice(len(d), size=int(JOINT_KERNEL_MEMORY_MAX_ROWS), replace=False))
        d = d.iloc[take].copy()
        lag_mat = lag_mat[take, :].copy()
        gc.collect()

    log(f"Joint kernel memory fit rows: {len(d):,}", t0)
    x = d["signed_dh"].to_numpy(dtype=np.float64, copy=True)
    T = d["period_years"].to_numpy(dtype=np.float64, copy=True)
    y = d[rho_col].to_numpy(dtype=np.float64, copy=True)
    w = d[weight_col].to_numpy(dtype=np.float64, copy=True)
    sqrtw = np.sqrt(w / np.nanmean(w))

    # Keep only lightweight identity columns for optional post-fit diagnostics
    keep = [c for c in ["rgiid", variant_col, "start_date", "end_date", "period_years", "signed_dh", rho_col, area_col, weight_col] if c in d.columns]
    d = d[keep].copy()
    gc.collect()
    return d, lag_mat, x, T, y, w, sqrtw


def joint_kernel_memory_from_lags(lag_mat, tau, kernel, normalize=None):
    weights = joint_kernel_memory_weights(
        lag_mat.shape[1], tau=tau, kernel=kernel, normalize=normalize
    )
    if not np.all(np.isfinite(weights)):
        return np.full(lag_mat.shape[0], np.nan, dtype=float)
    return lag_mat @ weights.astype(lag_mat.dtype, copy=False)


class _JointKernelEarlyStop(RuntimeError):
    """Controlled early stop for flat joint-memory least-squares fits."""


def fit_one_joint_kernel_memory_model(shared, base_spec, kernel, t0=None):
    d, lag_mat, x, T, y, w, sqrtw = shared

    spec = dict(base_spec)
    spec["name"] = f"joint_{kernel}_memory"
    base_param_names = params_for_forms(spec)

    theta0_base = initial_for_params(base_param_names)
    lo_base, hi_base = bounds_for_params(base_param_names)

    tau0 = JOINT_KERNEL_MEMORY_TAU_INITIALS.get(kernel, 5.0)
    tau0 = float(np.clip(tau0, JOINT_KERNEL_MEMORY_TAU_BOUNDS[0], JOINT_KERNEL_MEMORY_TAU_BOUNDS[1]))

    lo = np.concatenate([lo_base, [JOINT_KERNEL_MEMORY_TAU_BOUNDS[0]]])
    hi = np.concatenate([hi_base, [JOINT_KERNEL_MEMORY_TAU_BOUNDS[1]]])

    def pred(theta_joint):
        theta_base, tau = _unpack_joint_kernel_theta(theta_joint, base_param_names)
        p = joint_kernel_memory_from_lags(lag_mat, tau=tau, kernel=kernel)
        return rho_model(theta_base, spec, base_param_names, x, p, T)

    eval_counter = {"n": 0}
    tracker = {
        "best_score": np.inf,
        "best_theta": None,
        "best_fun": None,
        "progress_score": np.inf,
        "last_progress_eval": 0,
    }

    def residual(theta_joint):
        eval_counter["n"] += 1
        r = (pred(theta_joint) - y) * sqrtw
        r[~np.isfinite(r)] = 1e12
        score = float(np.sum(r ** 2))

        if score < tracker["best_score"]:
            tracker["best_score"] = score
            tracker["best_theta"] = np.asarray(theta_joint, dtype=float).copy()
            tracker["best_fun"] = np.asarray(r, dtype=float).copy()

        rel_improve = (
            np.inf if not np.isfinite(tracker["progress_score"])
            else (tracker["progress_score"] - score) / max(abs(tracker["progress_score"]), 1.0)
        )
        if rel_improve > JOINT_KERNEL_MEMORY_EARLY_STOP_REL_IMPROVEMENT:
            tracker["progress_score"] = score
            tracker["last_progress_eval"] = eval_counter["n"]

        if JOINT_KERNEL_MEMORY_VERBOSE_EVERY_N_EVAL and eval_counter["n"] % int(JOINT_KERNEL_MEMORY_VERBOSE_EVERY_N_EVAL) == 0:
            log(
                f"Joint {kernel} eval {eval_counter['n']}: tau={theta_joint[-1]:.3g}, "
                f"score={score:.6g}, best={tracker['best_score']:.6g}",
                t0,
            )

        max_raw = globals().get("JOINT_KERNEL_MEMORY_MAX_RESIDUAL_EVALS", None)
        if max_raw is not None and eval_counter["n"] >= int(max_raw):
            raise _JointKernelEarlyStop(
                f"reached {eval_counter['n']} raw residual calls; best_score={tracker['best_score']:.6g}"
            )

        patience = globals().get("JOINT_KERNEL_MEMORY_EARLY_STOP_PATIENCE_EVALS", None)
        if patience is not None and eval_counter["n"] - tracker["last_progress_eval"] >= int(patience):
            raise _JointKernelEarlyStop(
                f"no relative score improvement > {JOINT_KERNEL_MEMORY_EARLY_STOP_REL_IMPROVEMENT:g} "
                f"for {patience} raw residual calls; best_score={tracker['best_score']:.6g}"
            )

        return r

    rng = np.random.default_rng(
        JOINT_KERNEL_MEMORY_RANDOM_SEED + {"exponential": 101, "gaussian": 151, "spherical": 202}.get(kernel, 303)
    )
    n_starts = max(1, int(JOINT_KERNEL_MEMORY_MAX_STARTS))
    base_starts = random_initials(rng, base_param_names, n_starts - 1)
    tau_starts = [tau0]
    for _ in range(n_starts - 1):
        tau_starts.append(float(np.exp(rng.uniform(
            np.log(JOINT_KERNEL_MEMORY_TAU_BOUNDS[0]),
            np.log(JOINT_KERNEL_MEMORY_TAU_BOUNDS[1]),
        ))))
    starts = [np.concatenate([bs, [ts]]) for bs, ts in zip(base_starts, tau_starts)]

    best = None
    first_exception = None
    for istart, theta0 in enumerate(starts, start=1):
        eval_counter["n"] = 0
        tracker.update({
            "best_score": np.inf,
            "best_theta": None,
            "best_fun": None,
            "progress_score": np.inf,
            "last_progress_eval": 0,
        })
        log(f"Joint {kernel} optimizer start {istart}/{len(starts)}: tau0={theta0[-1]:.3g}", t0)
        try:
            if not np.all(np.isfinite(residual(theta0))):
                raise FloatingPointError("non-finite initial residual")
            res = least_squares(
                residual,
                theta0,
                bounds=(lo, hi),
                loss="soft_l1",
                f_scale=500.0,
                max_nfev=JOINT_KERNEL_MEMORY_MAX_NFEV,
                ftol=JOINT_KERNEL_MEMORY_FTOL,
                xtol=JOINT_KERNEL_MEMORY_XTOL,
                gtol=JOINT_KERNEL_MEMORY_GTOL,
                x_scale="jac",
            )
        except _JointKernelEarlyStop as exc:
            log(f"Joint {kernel} optimizer start {istart} early-stopped after {eval_counter['n']} evals: {exc}", t0)
            if tracker["best_theta"] is None or tracker["best_fun"] is None:
                if first_exception is None:
                    first_exception = repr(exc)
                continue
            res = SimpleNamespace(
                x=tracker["best_theta"],
                fun=tracker["best_fun"],
                cost=0.5 * tracker["best_score"],
                status=99,
                optimality=np.nan,
                success=True,
                message=str(exc),
            )
        except Exception as exc:
            log(f"Joint {kernel} optimizer start {istart} failed after {eval_counter['n']} evals: {exc!r}", t0)
            if first_exception is None:
                first_exception = repr(exc)
            continue

        score = float(np.sum(res.fun ** 2))
        log(
            f"Joint {kernel} optimizer start {istart} done: evals={eval_counter['n']}, "
            f"cost={res.cost:.6g}, score={score:.6g}, tau={res.x[-1]:.3g}, "
            f"status={res.status}, optimality={getattr(res, 'optimality', np.nan):.3g}",
            t0,
        )
        if best is None or score < best["score"]:
            best = {"theta": res.x, "score": score, "result": res}

    if best is None:
        raise RuntimeError(f"All joint {kernel} memory starts failed. First exception: {first_exception}")

    theta_base, tau_fit = _unpack_joint_kernel_theta(best["theta"], base_param_names)
    p_fit = joint_kernel_memory_from_lags(lag_mat, tau=tau_fit, kernel=kernel)
    yhat = rho_model(theta_base, spec, base_param_names, x, p_fit, T)
    metrics = weighted_metrics(y, yhat, w, len(best["theta"]))

    out = None
    if RUN_JOINT_KERNEL_DIAGNOSTIC_PLOTS:
        out = d.copy()
        out["joint_kernel"] = kernel
        out["joint_memory_dh"] = p_fit
        out["rho_pred_joint"] = yhat
        out["rho_resid_joint"] = y - yhat
        out["abs_signed_dh"] = np.abs(out["signed_dh"].to_numpy(float))
        out["log10_area"] = np.log10(out[area_col].astype(float))

    param_rows = []
    for name, val in zip(base_param_names, theta_base):
        param_rows.append({"kernel": kernel, "parameter": name, "value": float(val)})
    param_rows.append({"kernel": kernel, "parameter": "tau_mem", "value": float(tau_fit)})

    weights = joint_kernel_memory_weights(
        JOINT_KERNEL_MEMORY_KMAX_YEARS, tau=tau_fit, kernel=kernel
    )
    weight_rows = pd.DataFrame({
        "kernel": kernel,
        "tau_mem": float(tau_fit),
        "lag_years": np.arange(1, JOINT_KERNEL_MEMORY_KMAX_YEARS + 1, dtype=int),
        "weight": weights,
    })

    summary = {
        "model": f"joint_{kernel}_memory",
        "kernel": kernel,
        "n_rows": int(len(d)),
        "n_params": int(len(best["theta"])),
        "tau_mem": float(tau_fit),
        "memory_weights_normalized": bool(JOINT_KERNEL_MEMORY_NORMALIZE_WEIGHTS),
        "score": float(best["score"]),
        "base_memory_form": spec.get("memory"),
        "base_ratio_form": spec.get("ratio"),
        "base_damping_form": spec.get("damping"),
        "base_period_form": spec.get("period"),
        "base_current_form": spec.get("current"),
        **metrics,
    }

    del p_fit, yhat
    gc.collect()
    log(f"Joint {kernel} memory fit complete: tau={tau_fit:.3g}", t0)
    return {"kernel": kernel, "summary": summary, "params": param_rows, "weights": weight_rows, "rows": out}


def fit_joint_kernel_memory_models(all_rows, base_spec, t0=None):
    if not RUN_JOINT_KERNEL_MEMORY_FIT:
        return None

    log(f"Starting joint kernel memory comparison: {JOINT_KERNEL_MEMORY_FORMS}", t0)
    shared = _build_joint_kernel_arrays(all_rows, t0=t0)

    results = []
    errors = []
    for kernel in JOINT_KERNEL_MEMORY_FORMS:
        try:
            results.append(fit_one_joint_kernel_memory_model(shared, base_spec, kernel, t0=t0))
        except Exception as exc:
            log(f"Joint {kernel} memory fit failed: {exc!r}", t0)
            errors.append({"kernel": kernel, "error": repr(exc)})

    # Drop the shared lag matrix immediately after fitting
    del shared
    gc.collect()

    if results:
        summary = pd.DataFrame([r["summary"] for r in results]).sort_values("wrmse")
        params = pd.DataFrame([row for r in results for row in r["params"]])
        weights = pd.concat([r["weights"] for r in results], ignore_index=True)

        summary.to_csv(out_joint_kernel_memory_summary_csv, index=False)
        params.to_csv(out_joint_kernel_memory_params_csv, index=False)
        weights.to_csv(out_joint_kernel_memory_weights_csv, index=False)

        plot_joint_kernel_memory_weights(weights, out_joint_kernel_memory_weights_png)
        if RUN_JOINT_KERNEL_DIAGNOSTIC_PLOTS and any(r.get("rows") is not None for r in results):
            plot_joint_kernel_memory_residuals(results, out_joint_kernel_memory_residual_png)
            plot_joint_kernel_observed_predicted(results, out_joint_kernel_memory_scatter_png)
        log(f"Saved joint kernel memory comparison: {out_joint_kernel_memory_summary_csv}", t0)
    else:
        pd.DataFrame(errors).to_csv(out_joint_kernel_memory_summary_csv, index=False)

    if errors:
        pd.DataFrame(errors).to_csv(
            out_plot_dir / f"{run_label}_{main_suffix}_joint_kernel_memory_errors.csv",
            index=False,
        )

    return results

def main():
    validate_required_globals()
    validate_model_forms()
    validate_output_paths_for_plots()
    validate_attach_predictions_signature()
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = perf_counter()

    print("\nConfiguration:", flush=True)
    for name, val in [
        ("input_csv", input_csv),
        ("out_dir", str(out_dir)),
        ("MODEL_MEMORY_MODE", MODEL_MEMORY_MODE),
        ("MODEL_MEMORY_WINDOW_YEARS", MODEL_MEMORY_WINDOW_YEARS),
        ("MODEL_MEMORY_TAU_YEARS", MODEL_MEMORY_TAU_YEARS),
        ("USE_JOINT_EXPONENTIAL_MEMORY_FOR_DIAGNOSTICS", USE_JOINT_EXPONENTIAL_MEMORY_FOR_DIAGNOSTICS),
        ("RUN_MEMORY_PROFILE", RUN_MEMORY_PROFILE),
        ("SKIP_FULL_MEAN_MODEL_SEARCH", SKIP_FULL_MEAN_MODEL_SEARCH),
        ("FINAL_MEAN_MODEL_SPEC", FINAL_MEAN_MODEL_SPEC["name"]),
        ("RUN_JOINT_KERNEL_MEMORY_FIT", RUN_JOINT_KERNEL_MEMORY_FIT),
        ("JOINT_KERNEL_MEMORY_FORMS", JOINT_KERNEL_MEMORY_FORMS),
        ("CANDIDATE_MODELS", [m["name"] for m in CANDIDATE_MODELS]),
    ]:
        print(f"  {name} = {val}", flush=True)

    log("Reading and preparing input rows", t0)
    all_rows = prepare_dataframe()
    log(f"Prepared {len(all_rows):,} rows", t0)

    # Use the final selected mean-form directly for the joint memory-kernel comparison,
    # Then generate diagnostics only once with the chosen memory representation
    base_spec = dict(FINAL_MEAN_MODEL_SPEC if SKIP_FULL_MEAN_MODEL_SEARCH else CANDIDATE_MODELS[0])
    base_spec["name"] = str(base_spec.get("name", "base_spec"))

    chosen = None
    if USE_JOINT_EXPONENTIAL_MEMORY_FOR_DIAGNOSTICS and RUN_JOINT_KERNEL_MEMORY_FIT:
        joint_results = maybe_run_joint_kernel_memory_fit(all_rows, base_spec, t0=t0)
        chosen = choose_diagnostic_memory_from_joint_results(
            joint_results, preferred_kernel="exponential", t0=t0
        )
        del joint_results
        gc.collect()

    if chosen is not None:
        log(
            f"Running diagnostics and paper figures with final memory model: "
            f"{chosen['kernel']} kernel, tau={chosen['tau']:.3g}",
            t0,
        )
        selected_model, selected_spec, used_memory = run_main_analysis(
            all_rows,
            memory_mode=chosen["mode"],
            memory_window=chosen["window"],
            memory_tau=chosen["tau"],
        )
    else:
        log("Falling back to configured memory for diagnostics.", t0)
        selected_model, selected_spec, used_memory = run_main_analysis(all_rows)

    run_memory_profile(all_rows)
    log("Done", t0)

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        main()
