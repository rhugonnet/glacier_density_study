
#!/usr/bin/env python3
"""
Spatial and temporal correlation of standardized effective-density residuals.

This version fits one duration-invariant spatial correlation model to direct
pair-product estimates of the standardized effective-density residuals.

Key assumptions and outputs
---------------------------
- Residual variants (iteration9, sensmin, sensmax) are split before pair
  products are computed.
- Spatial correlations are estimated within one variant and one exact
  observation interval from products z_rho(g1) * z_rho(g2).
- Exact-interval estimates are aggregated by target period length for
  diagnostics, but one common spatial model is fitted across all selected
  period lengths.
- The fitted spatial model contains a nugget, a 200-km exponential component,
  and a 1500-km exponential component with non-negative fractions summing to 1.
- Temporal correlations are estimated from annual residuals with a simple
  nugget plus exponential correlation model for diagnostic purposes. The
  operational covariance model assumes zero temporal covariance at non-zero lag.
- The fitted spatial function is the elementary standardized-density-residual
  correlation. Exact propagation across arbitrary temporal partitions must be
  performed in mass-error covariance space by summing same-elementary-period
  covariance contributions.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

STUDY_DIR = Path(__file__).resolve().parents[1]
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

from study_paths import (
    STANDARDIZED_RESIDUALS_PATH,
    CORRELATION_DIAGNOSTICS_DIR,
    CORRELATION_MAIN_PATH,
)

# =============================================================================
# Paths
# =============================================================================

RUN_LABEL = "rho_error_correlation_standardized_residuals_directcorr_constantspace_temporaldiagnostic"

IN_CSV = STANDARDIZED_RESIDUALS_PATH

OUT_DIR = CORRELATION_DIAGNOSTICS_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_MAIN_CSV = CORRELATION_MAIN_PATH

OUT_SPATIAL_EMPIRICAL_CSV = OUT_DIR / f"{RUN_LABEL}_spatial_empirical_groups.csv"
OUT_SPATIAL_AGG_CSV = OUT_DIR / f"{RUN_LABEL}_spatial_empirical_aggregated.csv"
OUT_SPATIAL_USED_CSV = OUT_DIR / f"{RUN_LABEL}_spatial_empirical_aggregated_used_for_fit.csv"
OUT_SPATIAL_PERIOD_DIAGNOSTICS_CSV = OUT_DIR / f"{RUN_LABEL}_spatial_period_diagnostics.csv"
OUT_SPATIAL_CANDIDATES_CSV = OUT_DIR / f"{RUN_LABEL}_spatial_candidate_fits.csv"
OUT_SPATIAL_PARAMS_CSV = OUT_DIR / f"{RUN_LABEL}_spatial_fit_parameters.csv"
OUT_SPATIAL_PARAMS_JSON = OUT_DIR / f"{RUN_LABEL}_spatial_fit_parameters.json"

OUT_TEMPORAL_EMPIRICAL_CSV = OUT_DIR / f"{RUN_LABEL}_temporal_empirical_glacier_runs.csv"
OUT_TEMPORAL_AGG_CSV = OUT_DIR / f"{RUN_LABEL}_temporal_empirical_aggregated.csv"
OUT_TEMPORAL_PARAMS_CSV = OUT_DIR / f"{RUN_LABEL}_temporal_fit_parameters.csv"
OUT_TEMPORAL_PARAMS_JSON = OUT_DIR / f"{RUN_LABEL}_temporal_fit_parameters.json"

OUT_MAIN_FIG = OUT_DIR / f"{RUN_LABEL}_FIG_main_spatial_temporal_correlation.png"
OUT_SPATIAL_RAW_FIG = OUT_DIR / f"{RUN_LABEL}_FIG_spatial_common_fit_residuals.png"

WRITE_FIT_DIAGNOSTIC_PLOTS = True
UPDATE_PACKAGED_PARAMS = True
REUSE_EMPIRICAL_ESTIMATES = False
PACKAGED_PARAM_JSON = STUDY_DIR.parent / "glacier_density_surrogate" / "parameters" / "final_parameters.json"

# =============================================================================
# Input columns and controls
# =============================================================================

Z_COL_CANDIDATES = ["z_rho", "z_after", "drho_std"]
WEIGHT_COL_CANDIDATES = ["abs_dV_weight", "weight", "w"]
LAT_COL_CANDIDATES = ["lat", "cenlat", "center_lat"]
LON_COL_CANDIDATES = ["lon", "cenlon", "center_lon"]
REGION_COL_CANDIDATES = ["region", "rgi_region", "O1Region"]

RGI_COL = "rgiid"
START_COL = "start_date"
END_COL = "end_date"
PERIOD_COL = "period_years"
RUN_SOURCE_COL = "rho_variant"
RUN_COL = "_run_label"
RUN_LABELS_TO_USE = ["iteration9", "sensmin", "sensmax"]

PERIOD_LENGTHS_TO_USE = [1, 2, 4, 7, 10, 15]
SPATIAL_PERIOD_LENGTHS_TO_PLOT = [1, 2, 4, 7, 10, 15]
PERIOD_ATOL = 1e-6
USE_SPATIAL_PERIOD_WINDOWS = True
SPATIAL_PERIOD_WINDOWS = {
    1.0: (1.0, 1.0),
    2.0: (2.0, 2.0),
    4.0: (4.0, 4.0),
    7.0: (7.0, 7.0),
    10.0: (8.0, 12.0),
    15.0: (13.0, 17.0),
}

# Spatial grouping and pair sampling
MIN_GROUP_SIZE_SPATIAL = 8
RANDOM_SEED = 42
USE_REGION_GROUPS_FOR_SPATIAL = False
CHECK_UNIQUE_RGIID_PER_SPATIAL_GROUP = True
DUPLICATE_RGIID_SPATIAL_ACTION = "raise"

SPATIAL_PAIR_SELECTION_MODE = "bin_stratified"
EXACT_ALL_PAIRS_MAX_N = 2500
MAX_RANDOM_PAIRS_PER_GROUP = 500_000
MAX_PAIRS_PER_SPATIAL_BIN_PER_GROUP = 20_000
NEAR_PAIR_MAX_DISTANCE_KM = 300.0
RANDOM_FAR_PAIR_BATCH_SIZE = 300_000
MAX_RANDOM_FAR_PAIR_CANDIDATES = 10_000_000

SPATIAL_LAG_EDGES_KM = np.unique(
    np.concatenate(
        (
            [0.0],
            np.geomspace(5.0, 200.0, 16),
            np.array(
                [320.0, 520.0, 850.0, 1400.0, 2300.0, 3800.0, 6200.0, 9000.0, 13_000.0, 20_000.0],
                dtype=float,
            ),
        )
    ).astype(float)
)
MIN_RAW_PAIRS_PER_SPATIAL_BIN = 20

# Direct spatial-correlation estimator
CENTER_Z_WITHIN_EXACT_INTERVAL = False
NORMALIZE_Z_RMS_WITHIN_EXACT_INTERVAL = True
FILTER_STANDARDIZED_RESIDUAL_OUTLIERS = False
STANDARDIZED_OUTLIER_NMAD_THRESHOLD = 3.0
STANDARDIZED_OUTLIER_CENTER = "zero"
STANDARDIZED_OUTLIER_RECOMPUTE_NMAD_AFTER_SCALING = False
MIN_ROWS_AFTER_OUTLIER_FILTER = 8

# Aggregation of exact-interval estimates within period/lag bins:
# "equal_interval", "sqrt_n_pairs", or "n_pairs"
SPATIAL_INTERVAL_AGGREGATION = "equal_interval"

# Duration-invariant spatial fit
SPATIAL_CANDIDATE_FORMS = ["exponential", "gaussian", "spherical"]
SPATIAL_CANDIDATE_COMPONENTS = [1, 2, 3]
SPATIAL_RANGE_BOUNDS_KM = (20.0, 20_000.0)
SPATIAL_EXPONENTIAL_TIE_TOLERANCE = 0.001
SPATIAL_FIT_PERIODS = None  # None fits all period lengths
GLOBAL_MAX_SPATIAL_LAG_FOR_CORR_KM = 10_500.0
MIN_EXACT_INTERVALS_PER_SPATIAL_BIN_FOR_FIT = 1
SPATIAL_DIRECT_CORR_SHORT_WEIGHT_SCALE_KM = 80.0
SPATIAL_DIRECT_CORR_SHORT_WEIGHT_FACTOR = 1.5
SPATIAL_DIRECT_CORR_F_SCALE = 0.03
SPATIAL_FIT_LOSS = "soft_l1"
EQUALIZE_PERIOD_WEIGHTS_IN_SPATIAL_FIT = True

# Kept for helper compatibility; pair weights are not used for direct
# standardized-residual correlations.
USE_PAIR_WEIGHTS = False
PAIR_WEIGHT_MODE = "none"

REGION_GROUP_MAP = {
    1: 20, 2: 20, 3: 23, 4: 23, 5: 23,
    13: 21, 14: 21, 15: 21, 16: 22, 17: 22,
}

# =============================================================================
# Temporal diagnostic controls
# =============================================================================

SEMIVARIOGRAM_ESTIMATOR = "cressie"
TEMPORAL_PERIOD_LENGTH_YEARS = 1.0
TEMPORAL_MAX_LAG_YEARS = 14
TEMPORAL_LAGS_YEARS = np.arange(1, TEMPORAL_MAX_LAG_YEARS + 1, dtype=int)
MIN_GLACIER_OBS_TEMPORAL = 4
MAX_TEMPORAL_GLACIERS_PER_RUN = 10_000
TEMPORAL_RANDOM_SEED = 913
MIN_RAW_PAIRS_PER_TEMPORAL_LAG = 1
TEMPORAL_HIGH_LAG_COUNT_FOR_SILL = 5

TEMPORAL_FIT_WEIGHT_DECAY_YEARS = 4.0
TEMPORAL_FIT_WEIGHT_DECAY_POWER = 1.5
TEMPORAL_FORCE_FINAL_SILL = True
TEMPORAL_FINAL_SILL_LAG_YEARS = float(TEMPORAL_MAX_LAG_YEARS)
TEMPORAL_FINAL_SILL_ANCHOR_WEIGHT_FACTOR = 4.0
TEMPORAL_FIT_LOSS = "soft_l1"
TEMPORAL_FIT_F_SCALE = 0.05

# =============================================================================
# Plot styling
# =============================================================================

PLOT_SCATTER_SIZE = 24
PLOT_MODEL_LINEWIDTH = 2.2
PLOT_LEGEND_MARKERSIZE = 5.0
PLOT_LEGEND_LINEWIDTH = 2.0
PLOT_LEGEND_FONTSIZE = 9
PLOT_LEGEND_TITLE_FONTSIZE = 9
SPATIAL_PLOT_MAX_LAG_KM = 10_000.0

# =============================================================================
# Small helpers
# =============================================================================


def log(msg: str, t0: float | None = None) -> None:
    if t0 is None:
        print(f"[progress] {msg}", flush=True)
    else:
        print(f"[progress] {msg} | elapsed {perf_counter() - t0:.1f} s", flush=True)


def first_existing(columns: pd.Index, candidates: list[str], required: bool = True) -> str | None:
    for col in candidates:
        if col in columns:
            return col
    if required:
        raise KeyError(f"None of these columns exist: {candidates}")
    return None


def robust_nmad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.nanmedian(x)
    return float(1.4826 * np.nanmedian(np.abs(x - med)))


def add_scaled_residual_and_filter(
    df: pd.DataFrame,
    input_col: str,
    output_col: str,
) -> tuple[pd.DataFrame, dict]:
    """Apply NMAD scaling and optional robust outlier filtering.

    The input residuals are already standardized by the surrogate uncertainty.
    This second scaling normalizes the
    residual distribution within each period length before estimating empirical
    variograms.  The optional outlier filter is then applied in this scaled
    residual space, before any spatial or temporal pair differences are formed.
    """
    out = df.copy()
    z = out[input_col].to_numpy(float)
    sig = robust_nmad(z)
    if np.isfinite(sig) and sig > 0:
        z_scaled = z / sig
    else:
        z_scaled = z.copy()
    out[output_col] = z_scaled

    stats = {
        "n_before": int(len(out)),
        "n_after": int(len(out)),
        "n_removed": 0,
        "scale_nmad": float(sig) if np.isfinite(sig) else np.nan,
        "filter_enabled": bool(FILTER_STANDARDIZED_RESIDUAL_OUTLIERS),
    }

    if not FILTER_STANDARDIZED_RESIDUAL_OUTLIERS:
        return out, stats

    if STANDARDIZED_OUTLIER_CENTER == "zero":
        center = 0.0
    elif STANDARDIZED_OUTLIER_CENTER == "median":
        center = float(np.nanmedian(z_scaled))
    else:
        raise ValueError(STANDARDIZED_OUTLIER_CENTER)

    if STANDARDIZED_OUTLIER_RECOMPUTE_NMAD_AFTER_SCALING:
        scale = robust_nmad(z_scaled)
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
    else:
        scale = 1.0

    threshold = float(STANDARDIZED_OUTLIER_NMAD_THRESHOLD) * scale
    keep = np.isfinite(z_scaled) & (np.abs(z_scaled - center) <= threshold)
    # Do not accidentally delete an entire period/run subset if the scale is
    # Pathological; keep the unfiltered scaled residuals in that rare case
    if np.count_nonzero(keep) < MIN_ROWS_AFTER_OUTLIER_FILTER:
        stats.update({
            "center": center,
            "filter_scale": scale,
            "threshold": threshold,
            "n_after": int(len(out)),
            "n_removed": 0,
            "filter_skipped_reason": "too_few_rows_after_filter",
        })
        return out, stats

    filtered = out.loc[keep].copy()
    stats.update({
        "center": center,
        "filter_scale": scale,
        "threshold": threshold,
        "n_after": int(len(filtered)),
        "n_removed": int(len(out) - len(filtered)),
        "filter_skipped_reason": "",
    })
    return filtered, stats


def coerce_year_like(s: pd.Series) -> pd.Series:
    """Coerce numeric years or date-like strings to decimal-ish year labels.

    The input tables use numeric year columns. This helper keeps numeric years as
    numeric and falls back to datetime year for date strings.
    """
    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().mean() > 0.95:
        return numeric.astype(float)
    dt = pd.to_datetime(s, errors="coerce")
    out = dt.dt.year.astype(float)
    return out


def derive_period_years(df: pd.DataFrame) -> pd.Series:
    if START_COL in df.columns and END_COL in df.columns:
        start = coerce_year_like(df[START_COL])
        end = coerce_year_like(df[END_COL])
        period = end - start
        rounded = np.round(period)
        period = np.where(np.isfinite(period) & (np.abs(period - rounded) < 1e-6), rounded, period)
        return pd.Series(period, index=df.index, dtype=float)
    if PERIOD_COL in df.columns:
        return pd.to_numeric(df[PERIOD_COL], errors="coerce").astype(float)
    raise KeyError("Input must contain start_date/end_date or period_years.")


def spatial_period_window(target_period: float) -> tuple[float, float]:
    """Return true-period bounds pooled into a target spatial period label."""
    if USE_SPATIAL_PERIOD_WINDOWS:
        lo, hi = SPATIAL_PERIOD_WINDOWS.get(float(target_period), (float(target_period), float(target_period)))
        return float(lo), float(hi)
    p = float(target_period)
    return p, p


def select_spatial_period_window(df: pd.DataFrame, target_period: float) -> pd.DataFrame:
    """Select rows for one target period, optionally pooling nearby durations."""
    p = df[PERIOD_COL].to_numpy(float)
    lo, hi = spatial_period_window(float(target_period))
    if np.isclose(lo, hi, atol=PERIOD_ATOL):
        return df.loc[np.isclose(p, float(target_period), atol=PERIOD_ATOL)].copy()
    return df.loc[(p >= lo - PERIOD_ATOL) & (p <= hi + PERIOD_ATOL)].copy()


def haversine_km(lat1, lon1, lat2, lon2):
    r_earth_km = 6371.0088
    lat1 = np.deg2rad(np.asarray(lat1, dtype=float))
    lon1 = np.deg2rad(np.asarray(lon1, dtype=float))
    lat2 = np.deg2rad(np.asarray(lat2, dtype=float))
    lon2 = np.deg2rad(np.asarray(lon2, dtype=float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * r_earth_km * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def semivariance_from_diffs(
    diff: np.ndarray,
    weights: np.ndarray | None = None,
    estimator: str = SEMIVARIOGRAM_ESTIMATOR,
) -> float:
    """Estimate semivariance from pair differences.

    The estimators are implemented directly so that the result is independent
    of SciKit-GStat conventions. For independent unit-normal residuals, all
    three estimators should be close to a unit sill in expectation.

    - ``matheron``: 0.5 * mean(diff**2), optionally weighted.
    - ``dowd``: 0.5 * (median(abs(diff)) / 0.67448975)**2. No extra /2
      correction is applied.
    - ``cressie``: Cressie-Hawkins robust semivariance,
      mean(abs(diff)**0.5)**4 / [2 * (0.457 + 0.494/n + 0.045/n**2)].
      With weights, the mean is weighted and n is replaced by Kish effective n.
    """
    estimator = estimator.lower()
    diff = np.asarray(diff, dtype=float)
    if weights is None:
        weights = np.ones_like(diff, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    ok = np.isfinite(diff) & np.isfinite(weights) & (weights > 0)
    diff = diff[ok]
    weights = weights[ok]
    if diff.size == 0:
        return np.nan

    if estimator == "matheron":
        return float(0.5 * np.average(diff**2, weights=weights))

    if estimator == "dowd":
        # The Dowd median estimator is kept unweighted by design; weighted
        # Medians would create another estimator choice. This is the closest
        # Custom robust variogram estimator
        sigma_pair = np.nanmedian(np.abs(diff)) / 0.6744897501960817
        return float(0.5 * sigma_pair**2)

    if estimator == "cressie":
        mean_sqrt_abs = np.average(np.abs(diff) ** 0.5, weights=weights)
        # Kish effective sample size for the small-sample correction when
        # Weights are used; equal to n for uniform weights
        wsum = float(np.sum(weights))
        neff = wsum**2 / float(np.sum(weights**2)) if wsum > 0 else float(diff.size)
        correction = 0.457 + 0.494 / neff + 0.045 / (neff**2)
        return float((mean_sqrt_abs**4) / (2.0 * correction))

    raise ValueError(f"Unknown SEMIVARIOGRAM_ESTIMATOR: {estimator!r}. Use 'dowd', 'matheron', or 'cressie'.")


def sample_pairs(n: int, max_pairs: int | None, rng: np.random.Generator):
    total = n * (n - 1) // 2
    if max_pairs is None or total <= max_pairs:
        return np.triu_indices(n, k=1)
    # Draw random unique unordered pairs. Oversample to compensate for self-pairs
    # And duplicates, but never materialize the full pair matrix
    need = max_pairs
    pairs_out = []
    seen = set()
    while len(pairs_out) < need:
        batch = max(need * 3, 10_000)
        i = rng.integers(0, n, size=batch)
        j = rng.integers(0, n, size=batch)
        ok = i != j
        lo = np.minimum(i[ok], j[ok])
        hi = np.maximum(i[ok], j[ok])
        for a, b in zip(lo, hi):
            key = (int(a), int(b))
            if key not in seen:
                seen.add(key)
                pairs_out.append(key)
                if len(pairs_out) >= need:
                    break
    arr = np.asarray(pairs_out, dtype=int)
    return arr[:, 0], arr[:, 1]


def latlon_to_unit_xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Convert lat/lon in degrees to 3D unit-sphere coordinates."""
    lat_r = np.deg2rad(np.asarray(lat, dtype=float))
    lon_r = np.deg2rad(np.asarray(lon, dtype=float))
    coslat = np.cos(lat_r)
    return np.column_stack((coslat * np.cos(lon_r), coslat * np.sin(lon_r), np.sin(lat_r)))


def chord_radius_for_distance_km(distance_km: float) -> float:
    """Chord distance on the unit sphere corresponding to a great-circle distance."""
    r_earth_km = 6371.0088
    theta = float(distance_km) / r_earth_km
    return float(2.0 * np.sin(theta / 2.0))


def select_spatial_pairs(lat: np.ndarray, lon: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, str]:
    """Select pairs for one spatial variogram group.

    Region-group mode uses uniform random-pair sampling. Direct-global
    mode can use a bin-stratified sampler so each spatial lag bin has support.
    This keeps short-lag bins from being under-sampled when uniform random sampling
    represented mostly far pairs and left short-distance bins noisy or empty.

    For small groups, all pairs are used exactly.  For larger groups, pairs up
    to ``NEAR_PAIR_MAX_DISTANCE_KM`` are queried from a KD-tree on the unit
    sphere and capped by bin, while farther bins are filled by random batches.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    n = len(lat)
    if n < 2:
        return np.array([], dtype=int), np.array([], dtype=int), "none"

    if USE_REGION_GROUPS_FOR_SPATIAL or SPATIAL_PAIR_SELECTION_MODE == "random":
        i, j = sample_pairs(n, MAX_RANDOM_PAIRS_PER_GROUP, rng)
        return i.astype(int), j.astype(int), "random"

    if SPATIAL_PAIR_SELECTION_MODE != "bin_stratified":
        raise ValueError(f"Unknown SPATIAL_PAIR_SELECTION_MODE: {SPATIAL_PAIR_SELECTION_MODE!r}")

    if n <= EXACT_ALL_PAIRS_MAX_N:
        i, j = np.triu_indices(n, k=1)
        return i.astype(int), j.astype(int), "all_pairs_exact"

    nbins = len(SPATIAL_LAG_EDGES_KM) - 1
    bins_i: list[list[np.ndarray]] = [[] for _ in range(nbins)]
    bins_j: list[list[np.ndarray]] = [[] for _ in range(nbins)]
    bin_counts = np.zeros(nbins, dtype=int)
    target = int(MAX_PAIRS_PER_SPATIAL_BIN_PER_GROUP)

    def bins_need_more() -> np.ndarray:
        return bin_counts < target

    def add_candidate_pairs(ii: np.ndarray, jj: np.ndarray, dist: np.ndarray) -> None:
        nonlocal bin_counts
        if ii.size == 0:
            return
        b = np.digitize(dist, SPATIAL_LAG_EDGES_KM) - 1
        ok = (b >= 0) & (b < nbins) & np.isfinite(dist)
        if not np.any(ok):
            return
        ii = ii[ok]
        jj = jj[ok]
        b = b[ok]
        for bb in np.unique(b):
            need = target - int(bin_counts[bb])
            if need <= 0:
                continue
            idx = np.flatnonzero(b == bb)
            if idx.size > need:
                idx = rng.choice(idx, size=need, replace=False)
            bins_i[bb].append(ii[idx].astype(int, copy=False))
            bins_j[bb].append(jj[idx].astype(int, copy=False))
            bin_counts[bb] += int(idx.size)

    # Fill close-distance bins using an actual spatial-neighbour query.  This is
    # The critical difference from uniform global random sampling: short lags are
    # Not left to chance
    near_max = min(float(NEAR_PAIR_MAX_DISTANCE_KM), float(SPATIAL_LAG_EDGES_KM[-1]))
    near_bin_mask = SPATIAL_LAG_EDGES_KM[1:] <= near_max + 1e-12
    if np.any(near_bin_mask):
        xyz = latlon_to_unit_xyz(lat, lon)
        tree = cKDTree(xyz)
        r_chord = chord_radius_for_distance_km(near_max)
        order = rng.permutation(n)
        for a in order:
            if np.all(bin_counts[near_bin_mask] >= target):
                break
            neigh = tree.query_ball_point(xyz[a], r_chord)
            if len(neigh) <= 1:
                continue
            jj = np.asarray(neigh, dtype=int)
            jj = jj[jj > a]
            if jj.size == 0:
                continue
            ii = np.full(jj.size, int(a), dtype=int)
            dist = haversine_km(lat[ii], lon[ii], lat[jj], lon[jj])
            add_candidate_pairs(ii, jj, dist)

    # Fill remaining bins with random global pairs.  This still samples direct
    # Global pairs, but caps each distance bin separately so no lag dominates the
    # Estimator solely because it is common in the global distance distribution
    n_candidates = 0
    while np.any(bins_need_more()) and n_candidates < int(MAX_RANDOM_FAR_PAIR_CANDIDATES):
        batch = int(min(RANDOM_FAR_PAIR_BATCH_SIZE, int(MAX_RANDOM_FAR_PAIR_CANDIDATES) - n_candidates))
        if batch <= 0:
            break
        ii = rng.integers(0, n, size=batch)
        jj = rng.integers(0, n, size=batch)
        ok = ii != jj
        ii = ii[ok]
        jj = jj[ok]
        lo = np.minimum(ii, jj)
        hi = np.maximum(ii, jj)
        dist = haversine_km(lat[lo], lon[lo], lat[hi], lon[hi])
        add_candidate_pairs(lo, hi, dist)
        n_candidates += batch

    out_i = [np.concatenate(x) for x in bins_i if len(x)]
    out_j = [np.concatenate(x) for x in bins_j if len(x)]
    if not out_i:
        return np.array([], dtype=int), np.array([], dtype=int), "bin_stratified_empty"
    return np.concatenate(out_i).astype(int), np.concatenate(out_j).astype(int), "bin_stratified_global"


def pair_weights(wi: np.ndarray, wj: np.ndarray) -> np.ndarray:
    if not USE_PAIR_WEIGHTS:
        return np.ones_like(wi, dtype=float)
    if PAIR_WEIGHT_MODE == "product":
        return wi * wj
    if PAIR_WEIGHT_MODE == "sqrt_product":
        return np.sqrt(wi * wj)
    if PAIR_WEIGHT_MODE == "none":
        return np.ones_like(wi, dtype=float)
    raise ValueError(PAIR_WEIGHT_MODE)


def assign_region_group(df: pd.DataFrame, region_col: str | None) -> pd.Series:
    if not USE_REGION_GROUPS_FOR_SPATIAL or region_col is None or region_col not in df.columns:
        return pd.Series(0, index=df.index, dtype="int16")
    reg = pd.to_numeric(df[region_col], errors="coerce")
    out = reg.copy()
    for old, new in REGION_GROUP_MAP.items():
        out.loc[reg == old] = new
    return out.fillna(-1).astype("int16")


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(ok):
        return np.nan
    return float(np.average(values[ok], weights=weights[ok]))


def weighted_quantile(values: np.ndarray, q: float, weights: np.ndarray | None = None) -> float:
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(values)
    if weights is None:
        vals = values[ok]
        return float(np.nanquantile(vals, q)) if vals.size else np.nan
    weights = np.asarray(weights, dtype=float)
    ok &= np.isfinite(weights) & (weights > 0)
    vals = values[ok]
    ww = weights[ok]
    if vals.size == 0:
        return np.nan
    order = np.argsort(vals)
    vals = vals[order]
    ww = ww[order]
    cdf = np.cumsum(ww) / np.sum(ww)
    return float(np.interp(q, cdf, vals))


# =============================================================================
# Data preparation
# =============================================================================


def read_standardized_residuals() -> tuple[pd.DataFrame, str | None]:
    header = pd.read_csv(IN_CSV, nrows=0)
    columns = header.columns

    z_col = first_existing(columns, Z_COL_CANDIDATES)
    lat_col = first_existing(columns, LAT_COL_CANDIDATES)
    lon_col = first_existing(columns, LON_COL_CANDIDATES)
    region_col = first_existing(columns, REGION_COL_CANDIDATES, required=False)
    weight_col = first_existing(columns, WEIGHT_COL_CANDIDATES, required=False)

    required = [z_col, lat_col, lon_col, RGI_COL, START_COL, END_COL, RUN_SOURCE_COL]
    optional = [region_col, weight_col, PERIOD_COL]
    usecols = []
    for col in required + optional:
        if col is not None and col in columns and col not in usecols:
            usecols.append(col)

    df = pd.read_csv(IN_CSV, usecols=usecols)
    if weight_col is None:
        df["_unit_weight"] = 1.0
        weight_col = "_unit_weight"

    df[START_COL] = coerce_year_like(df[START_COL])
    df[END_COL] = coerce_year_like(df[END_COL])
    df[PERIOD_COL] = derive_period_years(df)

    df[RUN_COL] = df[RUN_SOURCE_COL].astype(str).str.lower()
    if RUN_LABELS_TO_USE is not None:
        keep = df[RUN_COL].isin([r.lower() for r in RUN_LABELS_TO_USE])
        df = df.loc[keep].copy()
        if df.empty:
            raise RuntimeError(f"No rows remain after filtering {RUN_SOURCE_COL} to {RUN_LABELS_TO_USE}.")

    rename = {z_col: "z", lat_col: "lat", lon_col: "lon", weight_col: "w"}
    if region_col is not None:
        rename[region_col] = "region_raw"
        region_col = "region_raw"
    df = df.rename(columns=rename)

    for col in ["z", "lat", "lon", "w", START_COL, END_COL, PERIOD_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    need = ["z", "lat", "lon", "w", RGI_COL, START_COL, END_COL, PERIOD_COL, RUN_COL]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=need).copy()
    df = df.loc[df["w"] > 0].copy()

    if region_col is not None:
        df[region_col] = pd.to_numeric(df[region_col], errors="coerce")
    df["region_group"] = assign_region_group(df, region_col)

    df[RGI_COL] = df[RGI_COL].astype(str)
    df[RUN_COL] = df[RUN_COL].astype("category")
    return df, region_col



# =============================================================================
# Direct spatial correlation
# =============================================================================


def check_unique_rgiid(g: pd.DataFrame, meta: dict) -> pd.DataFrame:
    if not CHECK_UNIQUE_RGIID_PER_SPATIAL_GROUP:
        return g
    dup = g[RGI_COL].duplicated(keep=False)
    if not dup.any():
        return g
    examples = sorted(g.loc[dup, RGI_COL].astype(str).unique().tolist())[:15]
    msg = (
        "Duplicate RGIIDs found within one spatial correlation group "
        f"(run_label={meta.get('run_label')}, period_years={meta.get('period_years')}, "
        f"start_date={meta.get('start_date')}, end_date={meta.get('end_date')}). "
        f"Examples: {examples}"
    )
    if DUPLICATE_RGIID_SPATIAL_ACTION == "raise":
        raise ValueError(msg)
    if DUPLICATE_RGIID_SPATIAL_ACTION == "drop_duplicates":
        print("[warning] " + msg + " Dropping duplicate rows, keeping first.", flush=True)
        return g.drop_duplicates(subset=[RGI_COL], keep="first").copy()
    raise ValueError(DUPLICATE_RGIID_SPATIAL_ACTION)


def prepare_exact_interval_z(g: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Prepare standardized residuals for one exact interval.

    The input z values are already standardized by the surrogate uncertainty.
    Optional RMS normalization removes small realization-specific departures
    from unit variance without introducing a period-specific variogram sill.
    """
    out = g.copy()
    z = out["z"].to_numpy(float)
    keep = np.isfinite(z)

    if FILTER_STANDARDIZED_RESIDUAL_OUTLIERS:
        if STANDARDIZED_OUTLIER_CENTER == "zero":
            center_filter = 0.0
        elif STANDARDIZED_OUTLIER_CENTER == "median":
            center_filter = float(np.nanmedian(z[keep]))
        else:
            raise ValueError(STANDARDIZED_OUTLIER_CENTER)
        scale_filter = robust_nmad(z[keep])
        if not np.isfinite(scale_filter) or scale_filter <= 0:
            scale_filter = 1.0
        keep &= np.abs(z - center_filter) <= float(STANDARDIZED_OUTLIER_NMAD_THRESHOLD) * scale_filter

    if np.count_nonzero(keep) < MIN_ROWS_AFTER_OUTLIER_FILTER:
        keep = np.isfinite(z)

    out = out.loc[keep].copy()
    z = out["z"].to_numpy(float)

    raw_mean = float(np.nanmean(z))
    raw_rms = float(np.sqrt(np.nanmean(z**2)))
    raw_std = float(np.nanstd(z))

    if CENTER_Z_WITHIN_EXACT_INTERVAL:
        z = z - float(np.nanmean(z))

    normalization = 1.0
    if NORMALIZE_Z_RMS_WITHIN_EXACT_INTERVAL:
        normalization = float(np.sqrt(np.nanmean(z**2)))
        if np.isfinite(normalization) and normalization > 0:
            z = z / normalization
        else:
            normalization = 1.0

    out["z_spatial"] = z
    stats = {
        "n_rows": int(len(out)),
        "raw_mean": raw_mean,
        "raw_rms": raw_rms,
        "raw_std": raw_std,
        "normalization": normalization,
        "prepared_mean": float(np.nanmean(z)),
        "prepared_rms": float(np.sqrt(np.nanmean(z**2))),
    }
    return out, stats


def binned_direct_spatial_correlation(
    dist_km: np.ndarray,
    product: np.ndarray,
    edges_km: np.ndarray,
) -> list[dict]:
    ok = np.isfinite(dist_km) & np.isfinite(product)
    dist_km = np.asarray(dist_km, dtype=float)[ok]
    product = np.asarray(product, dtype=float)[ok]
    if dist_km.size == 0:
        return []

    bin_id = np.digitize(dist_km, edges_km) - 1
    ok = (bin_id >= 0) & (bin_id < len(edges_km) - 1)
    dist_km = dist_km[ok]
    product = product[ok]
    bin_id = bin_id[ok]

    rows: list[dict] = []
    for bb in np.unique(bin_id):
        mask = bin_id == bb
        n_pairs = int(np.count_nonzero(mask))
        if n_pairs < MIN_RAW_PAIRS_PER_SPATIAL_BIN:
            continue
        vals = product[mask]
        rows.append(
            {
                "lag_bin_left_km": float(edges_km[bb]),
                "lag_bin_right_km": float(edges_km[bb + 1]),
                "lag_center_km": float(np.nanmedian(dist_km[mask])),
                "corr": float(np.nanmean(vals)),
                "corr_product_std": float(np.nanstd(vals)),
                "corr_product_se_naive": float(np.nanstd(vals) / np.sqrt(max(n_pairs, 1))),
                "n_pairs": n_pairs,
                "median_distance_km": float(np.nanmedian(dist_km[mask])),
                "estimator": "mean_pair_product",
            }
        )
    return rows


def empirical_spatial_correlation(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    rng = np.random.default_rng(RANDOM_SEED)

    for period in PERIOD_LENGTHS_TO_USE:
        pmin, pmax = spatial_period_window(float(period))
        dper_all = select_spatial_period_window(df, float(period))
        if dper_all.empty:
            log(f"Spatial target period={period:g} yr: no rows")
            continue

        source_periods = sorted(
            np.unique(dper_all[PERIOD_COL].dropna().astype(float).round(6)).tolist()
        )
        for run_label, dper in dper_all.groupby(RUN_COL, sort=True, observed=True):
            log(
                f"Spatial target period={period:g} yr, run={run_label}, "
                f"true periods={source_periods}: {len(dper):,} rows"
            )
            group_cols = [START_COL, END_COL]
            if USE_REGION_GROUPS_FOR_SPATIAL:
                group_cols.append("region_group")

            n_groups = 0
            for key, g in dper.groupby(group_cols, sort=True):
                if len(g) < MIN_GROUP_SIZE_SPATIAL:
                    continue

                if USE_REGION_GROUPS_FOR_SPATIAL:
                    start, end, region_group = key
                    region_group_meta = int(region_group)
                else:
                    start, end = key
                    region_group_meta = 0

                source_period = float(np.nanmedian(g[PERIOD_COL].to_numpy(float)))
                meta = {
                    "run_label": str(run_label),
                    "period_years": float(period),
                    "source_period_years": source_period,
                    "period_window_min_years": float(pmin),
                    "period_window_max_years": float(pmax),
                    "start_date": float(start),
                    "end_date": float(end),
                    "region_group": region_group_meta,
                    "spatial_grouping": (
                        "region_group" if USE_REGION_GROUPS_FOR_SPATIAL else "global"
                    ),
                }
                g = check_unique_rgiid(g, meta)
                g, z_stats = prepare_exact_interval_z(g)
                if len(g) < MIN_GROUP_SIZE_SPATIAL:
                    continue

                lat = g["lat"].to_numpy(float)
                lon = g["lon"].to_numpy(float)
                z = g["z_spatial"].to_numpy(float)

                ii, jj, pair_selection = select_spatial_pairs(lat, lon, rng)
                if len(ii) == 0:
                    continue
                dist = haversine_km(lat[ii], lon[ii], lat[jj], lon[jj])
                product = z[ii] * z[jj]

                for row in binned_direct_spatial_correlation(
                    dist, product, SPATIAL_LAG_EDGES_KM
                ):
                    row.update(meta)
                    row["pair_selection"] = pair_selection
                    row.update({f"z_{k}": v for k, v in z_stats.items()})
                    rows.append(row)
                n_groups += 1

            log(
                f"Spatial target period={period:g} yr, run={run_label}: "
                f"processed {n_groups} exact intervals"
            )

    return pd.DataFrame(rows)


def _aggregation_weights(g: pd.DataFrame) -> np.ndarray:
    if SPATIAL_INTERVAL_AGGREGATION == "equal_interval":
        return np.ones(len(g), dtype=float)
    if SPATIAL_INTERVAL_AGGREGATION == "sqrt_n_pairs":
        return np.sqrt(g["n_pairs"].to_numpy(float))
    if SPATIAL_INTERVAL_AGGREGATION == "n_pairs":
        return g["n_pairs"].to_numpy(float)
    raise ValueError(SPATIAL_INTERVAL_AGGREGATION)


def aggregate_spatial_empirical(emp: pd.DataFrame) -> pd.DataFrame:
    if emp.empty:
        return emp

    rows: list[dict] = []
    for (period, lag_right), g in emp.groupby(
        ["period_years", "lag_bin_right_km"], sort=True
    ):
        g = g.loc[np.isfinite(g["corr"]) & (g["n_pairs"] > 0)].copy()
        if g.empty:
            continue
        weights = _aggregation_weights(g)
        corr_values = g["corr"].to_numpy(float)
        corr_mean = weighted_mean(corr_values, weights)
        weight_sum = float(np.sum(weights))
        if weight_sum > 0:
            corr_var = float(
                np.average((corr_values - corr_mean) ** 2, weights=weights)
            )
        else:
            corr_var = np.nan

        pair_weights_arr = g["n_pairs"].to_numpy(float)
        rows.append(
            {
                "period_years": float(period),
                "lag_bin_left_km": float(g["lag_bin_left_km"].iloc[0]),
                "lag_bin_right_km": float(lag_right),
                "lag_center_km": weighted_mean(
                    g["median_distance_km"].to_numpy(float), pair_weights_arr
                ),
                "corr": corr_mean,
                "corr_equal_interval": float(np.nanmean(corr_values)),
                "corr_pair_weighted": weighted_mean(corr_values, pair_weights_arr),
                "corr_between_interval_std": float(np.sqrt(corr_var))
                if np.isfinite(corr_var)
                else np.nan,
                "n_pairs": int(np.sum(g["n_pairs"].to_numpy(int))),
                "n_exact_intervals": int(len(g)),
                "run_labels": ",".join(
                    sorted(g["run_label"].astype(str).unique().tolist())
                ),
                "source_period_min_years": float(
                    np.nanmin(g["source_period_years"].to_numpy(float))
                ),
                "source_period_max_years": float(
                    np.nanmax(g["source_period_years"].to_numpy(float))
                ),
                "source_period_values": ",".join(
                    f"{v:g}"
                    for v in sorted(
                        np.unique(
                            g["source_period_years"]
                            .dropna()
                            .astype(float)
                            .round(6)
                        ).tolist()
                    )
                ),
                "aggregation_mode": SPATIAL_INTERVAL_AGGREGATION,
                "used_for_corr": False,
            }
        )

    out = (
        pd.DataFrame(rows)
        .sort_values(["period_years", "lag_center_km"])
        .reset_index(drop=True)
    )
    return mark_spatial_bins_used_for_fit(out)


def mark_spatial_bins_used_for_fit(agg: pd.DataFrame) -> pd.DataFrame:
    out = agg.copy()
    used = np.isfinite(out["corr"].to_numpy(float))
    used &= out["n_pairs"].to_numpy(float) > 0
    used &= (
        out["n_exact_intervals"].to_numpy(float)
        >= float(MIN_EXACT_INTERVALS_PER_SPATIAL_BIN_FOR_FIT)
    )
    used &= out["lag_center_km"].to_numpy(float) > 0
    used &= (
        out["lag_center_km"].to_numpy(float)
        <= float(GLOBAL_MAX_SPATIAL_LAG_FOR_CORR_KM)
    )
    if SPATIAL_FIT_PERIODS is not None:
        used &= out["period_years"].isin(
            [float(v) for v in SPATIAL_FIT_PERIODS]
        ).to_numpy(bool)
    out["used_for_corr"] = used
    return out


def _softmax(theta: np.ndarray, n: int) -> np.ndarray:
    logits = np.zeros(n, dtype=float)
    logits[1:] = np.asarray(theta, dtype=float)
    logits -= np.max(logits)
    ex = np.exp(logits)
    return ex / np.sum(ex)


def spatial_component(d_km: np.ndarray, range_km: float, form: str) -> np.ndarray:
    d = np.asarray(d_km, dtype=float)
    h = np.maximum(d, 0.0) / max(float(range_km), 1.0e-12)
    if form == "exponential":
        return np.exp(-3.0 * h)
    if form == "gaussian":
        return np.exp(-3.0 * h**2)
    if form == "spherical":
        return np.where(h < 1.0, 1.0 - 1.5 * h + 0.5 * h**3, 0.0)
    raise ValueError(f"Unsupported spatial correlation form: {form}")


def spatial_corr_model(d_km: np.ndarray, params: dict) -> np.ndarray:
    d = np.asarray(d_km, dtype=float)
    form = str(params.get("spatial_corr_form", "exponential"))
    n_components = int(float(params.get("n_spatial_components", 2)))
    out = np.zeros_like(d, dtype=float)
    for i in range(1, n_components + 1):
        q = float(params.get(f"q{i}_range_fraction", params.get("q1_short_range_fraction" if i == 1 else "q2_long_range_fraction", 0.0)))
        r = float(params.get(f"r{i}_km", params.get("r1_km" if i == 1 else "r2_km", np.nan)))
        if np.isfinite(r) and q > 0:
            out = out + q * spatial_component(d, r, form)
    return np.clip(out, 0.0, 1.0)


def _spatial_fit_weights(d: pd.DataFrame) -> np.ndarray:
    distance = d["lag_center_km"].to_numpy(float)
    support = np.sqrt(np.maximum(d["n_exact_intervals"].to_numpy(float), 1.0))

    if EQUALIZE_PERIOD_WEIGHTS_IN_SPATIAL_FIT:
        period_norm = (
            d.assign(_support2=support**2)
            .groupby("period_years")["_support2"]
            .transform("sum")
            .to_numpy(float)
        )
        support = support / np.sqrt(np.maximum(period_norm, 1e-12))

    support *= 1.0 + SPATIAL_DIRECT_CORR_SHORT_WEIGHT_FACTOR * np.exp(
        -distance / SPATIAL_DIRECT_CORR_SHORT_WEIGHT_SCALE_KM
    )
    positive = support[np.isfinite(support) & (support > 0)]
    if positive.size:
        support = support / np.nanmedian(positive)
    return support


def _fit_spatial_candidate(
    distance: np.ndarray,
    y: np.ndarray,
    support: np.ndarray,
    *,
    form: str,
    n_components: int,
) -> tuple[dict, np.ndarray]:
    lo, hi = np.log(SPATIAL_RANGE_BOUNDS_KM[0]), np.log(SPATIAL_RANGE_BOUNDS_KM[1])

    def unpack(theta: np.ndarray) -> dict:
        q = _softmax(theta[:n_components], n_components + 1)
        ranges = np.sort(np.exp(theta[n_components:]))
        return {
            "spatial_corr_form": form,
            "n_spatial_components": int(n_components),
            "q0_nugget_fraction": float(q[0]),
            "q1_short_range_fraction": float(q[1]),
            "q2_long_range_fraction": float(q[2]) if n_components >= 2 else 0.0,
            "r1_km": float(ranges[0]),
            "r2_km": float(ranges[1]) if n_components >= 2 else np.nan,
            "q1_range_fraction": float(q[1]),
            "q2_range_fraction": float(q[2]) if n_components >= 2 else 0.0,
            "q3_range_fraction": float(q[3]) if n_components >= 3 else 0.0,
            "r3_km": float(ranges[2]) if n_components >= 3 else np.nan,
        }

    def residual(theta: np.ndarray) -> np.ndarray:
        return support * (spatial_corr_model(distance, unpack(theta)) - y)

    if n_components == 1:
        start_ranges = [(100.0,), (300.0,), (800.0,), (2000.0,)]
    elif n_components == 2:
        start_ranges = [(80.0, 800.0), (150.0, 1500.0), (200.0, 3000.0), (500.0, 5000.0)]
    else:
        start_ranges = [(100.0, 800.0, 5000.0), (150.0, 1000.0, 10_000.0), (250.0, 1500.0, 12_000.0), (100.0, 1500.0, 15_000.0)]
    best: tuple[float, dict, np.ndarray] | None = None
    for ranges in start_ranges:
        if n_components == 1:
            theta0 = np.array([1.0, np.log(ranges[0])], dtype=float)
            lower = np.array([-20.0, lo], dtype=float)
            upper = np.array([20.0, hi], dtype=float)
        elif n_components == 2:
            theta0 = np.array([0.0, -1.0, np.log(ranges[0]), np.log(ranges[1])], dtype=float)
            lower = np.array([-20.0, -20.0, lo, lo], dtype=float)
            upper = np.array([20.0, 20.0, hi, hi], dtype=float)
        else:
            theta0 = np.array([0.0, -1.0, -2.0, np.log(ranges[0]), np.log(ranges[1]), np.log(ranges[2])], dtype=float)
            lower = np.array([-20.0, -20.0, -20.0, lo, lo, lo], dtype=float)
            upper = np.array([20.0, 20.0, 20.0, hi, hi, hi], dtype=float)
        result = least_squares(
            residual,
            x0=theta0,
            bounds=(lower, upper),
            loss=SPATIAL_FIT_LOSS,
            f_scale=SPATIAL_DIRECT_CORR_F_SCALE,
            max_nfev=20_000,
        )
        params = unpack(result.x)
        pred = spatial_corr_model(distance, params)
        weighted_rmse = float(
            np.sqrt(np.average((pred - y) ** 2, weights=np.maximum(support, 1e-12)))
        )
        if best is None or weighted_rmse < best[0]:
            best = (weighted_rmse, params, pred)
    if best is None:
        raise RuntimeError("No spatial candidate fit succeeded.")
    return best[1], best[2]


def fit_constant_spatial_model(
    spatial_agg: pd.DataFrame,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = spatial_agg.loc[spatial_agg["used_for_corr"].astype(bool)].copy()
    d = d.dropna(
        subset=["lag_center_km", "period_years", "corr", "n_exact_intervals"]
    )
    if len(d) < 5:
        raise RuntimeError("Too few spatial correlation bins for fitting.")

    distance = d["lag_center_km"].to_numpy(float)
    y = d["corr"].to_numpy(float)
    support = _spatial_fit_weights(d)

    candidate_rows: list[dict] = []
    fitted: list[tuple[dict, np.ndarray]] = []
    for form in SPATIAL_CANDIDATE_FORMS:
        for n_components in SPATIAL_CANDIDATE_COMPONENTS:
            params, pred_candidate = _fit_spatial_candidate(
                distance,
                y,
                support,
                form=form,
                n_components=n_components,
            )
            err = pred_candidate - y
            row = dict(params)
            row.update(
                {
                    "n_fit_parameters": int(2 * n_components),
                    "rmse": float(np.sqrt(np.nanmean(err**2))),
                    "weighted_rmse": float(
                        np.sqrt(
                            np.average(
                                err**2,
                                weights=np.maximum(support, 1e-12),
                            )
                        )
                    ),
                    "mae": float(np.nanmean(np.abs(err))),
                    "bias_model_minus_empirical": float(np.nanmean(err)),
                    "bias_model_minus_empirical_200_5000km": float(
                        np.nanmean(err[distance >= 200.0])
                    ),
                }
            )
            candidate_rows.append(row)
            fitted.append((params, pred_candidate))

    candidate_diag = pd.DataFrame(candidate_rows).sort_values(
        ["weighted_rmse", "n_fit_parameters"],
        kind="mergesort",
    ).reset_index(drop=True)
    best_rmse = float(candidate_diag["weighted_rmse"].iloc[0])
    exponential_candidates = candidate_diag.loc[
        (candidate_diag["spatial_corr_form"] == "exponential")
        & (
            candidate_diag["weighted_rmse"]
            <= best_rmse + float(SPATIAL_EXPONENTIAL_TIE_TOLERANCE)
        )
    ]
    best_idx = (
        int(exponential_candidates.index[0])
        if not exponential_candidates.empty
        else int(candidate_diag.index[0])
    )
    best_signature = candidate_diag.loc[best_idx, ["spatial_corr_form", "n_spatial_components"]].to_dict()
    best_pos = next(
        i for i, (params, _) in enumerate(fitted)
        if params["spatial_corr_form"] == best_signature["spatial_corr_form"]
        and params["n_spatial_components"] == best_signature["n_spatial_components"]
    )
    diagnostics, pred = fitted[best_pos]
    fit_err = pred - y

    period_rows: list[dict] = []
    for period, gp in d.assign(pred=pred).groupby("period_years", sort=True):
        period_err = gp["pred"].to_numpy(float) - gp["corr"].to_numpy(float)
        period_rows.append(
            {
                "period_years": float(period),
                "n_fit_bins": int(len(gp)),
                "rmse": float(np.sqrt(np.nanmean(period_err**2))),
                "mae": float(np.nanmean(np.abs(period_err))),
                "bias_model_minus_empirical": float(np.nanmean(period_err)),
                "max_abs_difference": float(np.nanmax(np.abs(period_err))),
            }
        )
    period_diag = pd.DataFrame(period_rows)

    diagnostics = {
        **diagnostics,
        "n_fit_bins": int(len(d)),
        "weighted_rmse": float(
            np.sqrt(np.average(fit_err**2, weights=np.maximum(support, 1e-12)))
        ),
        "rmse": float(np.sqrt(np.nanmean(fit_err**2))),
        "mae": float(np.nanmean(np.abs(fit_err))),
        "bias_model_minus_empirical": float(np.nanmean(fit_err)),
        "bias_model_minus_empirical_200_5000km": float(
            np.nanmean(fit_err[distance >= 200.0])
        ),
        "fit_periods": (
            "all"
            if SPATIAL_FIT_PERIODS is None
            else ",".join(str(v) for v in SPATIAL_FIT_PERIODS)
        ),
        "estimator": "direct_mean_pair_product",
        "interval_aggregation": SPATIAL_INTERVAL_AGGREGATION,
        "center_z_within_exact_interval": bool(
            CENTER_Z_WITHIN_EXACT_INTERVAL
        ),
        "normalize_z_rms_within_exact_interval": bool(
            NORMALIZE_Z_RMS_WITHIN_EXACT_INTERVAL
        ),
        "temporal_covariance_assumption_for_application": (
            "zero_for_nonzero_elementary_period_lags"
        ),
    }
    return diagnostics, period_diag, candidate_diag, d.assign(pred=pred)

# =============================================================================
# Temporal empirical variogram
# =============================================================================


def empirical_temporal_variogram(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    d_all = df.loc[np.isclose(df[PERIOD_COL].to_numpy(float), TEMPORAL_PERIOD_LENGTH_YEARS, atol=PERIOD_ATOL)].copy()
    if d_all.empty:
        return pd.DataFrame()

    for run_label, d in d_all.groupby(RUN_COL, sort=True, observed=True):
        d, filter_stats = add_scaled_residual_and_filter(d, input_col="z", output_col="z_temporal")
        log(
            f"Temporal run={run_label}: "
            f"{filter_stats['n_after']:,}/{filter_stats['n_before']:,} annual rows after residual filter "
            f"(removed {filter_stats['n_removed']:,})"
        )
        if d.empty:
            continue

        rgiids = pd.unique(d[RGI_COL])
        if MAX_TEMPORAL_GLACIERS_PER_RUN is not None and len(rgiids) > MAX_TEMPORAL_GLACIERS_PER_RUN:
            run_hash = int.from_bytes(hashlib.blake2b(str(run_label).encode("utf-8"), digest_size=4).digest(), "little")
            rng = np.random.default_rng(int(np.random.SeedSequence([TEMPORAL_RANDOM_SEED, run_hash]).generate_state(1)[0]))
            rgiids = rng.choice(rgiids, size=MAX_TEMPORAL_GLACIERS_PER_RUN, replace=False)

        log(f"Temporal run={run_label}: {len(rgiids):,} glaciers sampled")

        for rgiid in rgiids:
            g = d.loc[d[RGI_COL] == rgiid, [START_COL, END_COL, "z_temporal"]].copy()
            # Annual series should have one residual per glacier/run/start/end. If
            # Duplicates exist within the same run, fail because temporal pairs
            # Would otherwise include duplicate self-periods
            dup = g.duplicated(subset=[START_COL, END_COL], keep=False)
            if dup.any():
                ex = g.loc[dup, [START_COL, END_COL]].head(5).to_dict("records")
                raise ValueError(
                    f"Duplicate annual temporal rows for rgiid={rgiid}, run_label={run_label}. "
                    f"Examples start/end: {ex}"
                )
            g = g.dropna().sort_values(START_COL)
            if len(g) < MIN_GLACIER_OBS_TEMPORAL:
                continue

            start = g[START_COL].to_numpy(float)
            vals = g["z_temporal"].to_numpy(float)
            ii, jj = np.triu_indices(len(g), k=1)
            tau = np.abs(start[jj] - start[ii])
            tau_i = np.rint(tau).astype(int)
            exact = np.isclose(tau, tau_i, atol=PERIOD_ATOL)

            gamma_by_lag = []
            count_by_lag = []
            for lag in TEMPORAL_LAGS_YEARS:
                m = exact & (tau_i == lag)
                count = int(np.count_nonzero(m))
                if count < MIN_RAW_PAIRS_PER_TEMPORAL_LAG:
                    gamma_by_lag.append(np.nan)
                    count_by_lag.append(0)
                    continue
                diff = vals[jj[m]] - vals[ii[m]]
                gamma_by_lag.append(semivariance_from_diffs(diff))
                count_by_lag.append(count)

            gamma_by_lag = np.asarray(gamma_by_lag, dtype=float)
            count_by_lag = np.asarray(count_by_lag, dtype=int)
            finite = np.isfinite(gamma_by_lag) & (count_by_lag > 0)
            if not np.any(finite):
                continue

            # Reference workflow scaled by the last five lags. If those are unavailable
            # Use the highest available finite lags rather than
            # Dropping the entire glacier-run variogram
            finite_idx = np.where(finite)[0]
            high_idx = finite_idx[-min(TEMPORAL_HIGH_LAG_COUNT_FOR_SILL, len(finite_idx)) :]
            sill = float(np.nanmean(gamma_by_lag[high_idx]))
            if not np.isfinite(sill) or sill <= 0:
                continue
            gamma_by_lag = gamma_by_lag / sill

            for lag, gamma, count in zip(TEMPORAL_LAGS_YEARS, gamma_by_lag, count_by_lag):
                if not np.isfinite(gamma) or count <= 0:
                    continue
                rows.append(
                    {
                        "run_label": str(run_label),
                        RGI_COL: str(rgiid),
                        "lag_center_yr": float(lag),
                        "gamma": float(gamma),
                        "corr": float(1.0 - gamma),
                        "n_pairs": int(count),
                        "semivariogram_estimator": SEMIVARIOGRAM_ESTIMATOR,
                    }
                )

    return pd.DataFrame(rows)


def aggregate_temporal_empirical(emp: pd.DataFrame) -> pd.DataFrame:
    if emp.empty:
        return emp
    rows = []
    for lag, g in emp.groupby("lag_center_yr", sort=True):
        gg = g.loc[np.isfinite(g["gamma"])]
        if gg.empty:
            continue
        gamma = float(np.nanmean(gg["gamma"].to_numpy(float)))
        rows.append(
            {
                "lag_center_yr": float(lag),
                "gamma": gamma,
                "corr": float(1.0 - gamma),
                "err_gamma": float(np.nanstd(gg["gamma"].to_numpy(float))),
                "n_pairs": int(np.nansum(gg["n_pairs"].to_numpy(float))),
                "n_glacier_runs": int(len(gg)),
                "n_glaciers": int(gg[RGI_COL].nunique()),
                "run_labels": ",".join(sorted(gg["run_label"].astype(str).unique().tolist())),
            }
        )
    return pd.DataFrame(rows)


def temporal_variogram_model(t_yr: np.ndarray, nugget: float, range_yr: float) -> np.ndarray:
    """Temporal variogram with a nugget plus one exponential component.

    For positive lags, gamma(t) = n + (1 - n) [1 - exp(-3t / r)], where n is
    the nugget and r is the correlation range. The empirical temporal points
    start at one year, so the fitted nugget represents unresolved decorrelation
    between exact zero lag and the first annual lag.
    """
    t_yr = np.maximum(np.asarray(t_yr, dtype=float), 0.0)
    nugget = float(np.clip(nugget, 0.0, 1.0))
    range_yr = max(float(range_yr), 1.0e-12)
    g = 1.0 - np.exp(-3.0 * t_yr / range_yr)
    return np.clip(nugget + (1.0 - nugget) * g, 0.0, 1.0)


def temporal_corr_model(t_yr: np.ndarray, nugget: float, range_yr: float) -> np.ndarray:
    """Temporal correlation corresponding to ``temporal_variogram_model``.

    This returns the fitted positive-lag correlation, including the nugget.
    Therefore the curve extrapolated to zero lag starts at 1 - nugget, not at
    the exact self-correlation r(0)=1. This is the quantity shown on the plot.
    """
    t_yr = np.asarray(t_yr, dtype=float)
    corr = 1.0 - temporal_variogram_model(t_yr, nugget, range_yr)
    return np.clip(corr, 0.0, 1.0)

def temporal_fit_weights(lag_yr: np.ndarray, n_glacier_runs: np.ndarray) -> np.ndarray:
    """Fit weights that decrease with lag after accounting for sample support."""
    lag_yr = np.asarray(lag_yr, dtype=float)
    n_glacier_runs = np.asarray(n_glacier_runs, dtype=float)
    support_w = np.sqrt(np.maximum(n_glacier_runs, 1.0))
    if support_w.size:
        support_w = np.minimum(support_w, np.nanpercentile(support_w, 90))
    lag_w = 1.0 / (1.0 + lag_yr / float(TEMPORAL_FIT_WEIGHT_DECAY_YEARS)) ** float(TEMPORAL_FIT_WEIGHT_DECAY_POWER)
    w = support_w * lag_w
    # Keep weights on a stable numerical scale for least_squares/f_scale
    med = np.nanmedian(w[np.isfinite(w) & (w > 0)]) if np.any(np.isfinite(w) & (w > 0)) else 1.0
    if np.isfinite(med) and med > 0:
        w = w / med
    return w


def fit_temporal_model(agg: pd.DataFrame) -> np.ndarray:
    d = agg.dropna(subset=["lag_center_yr", "gamma"]).copy()
    d = d.loc[d["lag_center_yr"] > 0]
    if d.empty:
        raise RuntimeError("No temporal empirical bins available for fitting.")

    x = d["lag_center_yr"].to_numpy(float)
    # Clip only for fitting: the empirical CSV still stores the raw aggregate
    y = np.clip(d["gamma"].to_numpy(float), 0.0, 1.0)
    w = temporal_fit_weights(x, d["n_glacier_runs"].to_numpy(float))

    def resid(theta):
        return w * (temporal_variogram_model(x, *theta) - y)

    res = least_squares(
        resid,
        # Nugget and exponential correlation range.
        x0=np.array([0.70, 4.0], dtype=float),
        bounds=([0.0, 0.05], [0.95, 50.0]),
        loss=TEMPORAL_FIT_LOSS,
        f_scale=TEMPORAL_FIT_F_SCALE,
        max_nfev=10000,
    )
    return res.x.astype(float)


def update_packaged_correlation_params(spatial_out: dict, temporal_out: dict) -> None:
    """Update only correlation parameters in the packaged surrogate JSON."""
    if not PACKAGED_PARAM_JSON.exists():
        raise FileNotFoundError(PACKAGED_PARAM_JSON)
    with open(PACKAGED_PARAM_JSON) as f:
        params = json.load(f)

    spatial_keys = [
        "spatial_corr_form",
        "n_spatial_components",
        "q0_nugget_fraction",
        "q1_range_fraction",
        "q2_range_fraction",
        "q3_range_fraction",
        "q1_short_range_fraction",
        "q2_long_range_fraction",
        "r1_km",
        "r2_km",
        "r3_km",
    ]
    for key in spatial_keys:
        if key in spatial_out:
            params[key] = spatial_out[key]

    params.update(
        {
            "temporal_model_form": temporal_out["temporal_model_form"],
            "temporal_nugget": temporal_out["empirical_nugget"],
            "amplitude_after_nugget": temporal_out["empirical_sill"],
            "temporal_timescale_yr": temporal_out["empirical_range_yr"],
            "temporal_exponent": temporal_out["empirical_exponent"],
            "applied_temporal_correlation_at_positive_lag": temporal_out[
                "applied_temporal_correlation_at_positive_lag"
            ],
            "applied_temporal_covariance_assumption": temporal_out[
                "applied_temporal_covariance_assumption"
            ],
            "semivariogramestimator": temporal_out["semivariogram_estimator"],
        }
    )

    with open(PACKAGED_PARAM_JSON, "w") as f:
        json.dump(params, f, indent=2, sort_keys=True)
        f.write("\n")



# =============================================================================
# Plotting
# =============================================================================


def add_panel_letter(ax, letter: str) -> None:
    ax.text(
        0.02,
        0.98,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
    )


def plot_spatial_fit_residuals(
    spatial_agg: pd.DataFrame,
    spatial_params: dict,
) -> None:
    periods = SPATIAL_PERIOD_LENGTHS_TO_PLOT
    cmap = plt.get_cmap("viridis")
    norm = mpl.colors.Normalize(vmin=min(periods), vmax=max(periods))

    fig, ax = plt.subplots(figsize=(6.4, 4.7), constrained_layout=True)
    for period in periods:
        sub = spatial_agg.loc[
            np.isclose(
                spatial_agg["period_years"].to_numpy(float),
                float(period),
                atol=PERIOD_ATOL,
            )
            & spatial_agg["used_for_corr"].astype(bool)
        ].sort_values("lag_center_km")
        if sub.empty:
            continue
        pred = spatial_corr_model(sub["lag_center_km"].to_numpy(float), spatial_params)
        ax.plot(
            sub["lag_center_km"],
            sub["corr"].to_numpy(float) - pred,
            marker="o",
            ms=3,
            lw=1.1,
            color=cmap(norm(period)),
            label=f"{period:g} yr",
        )
    ax.axhline(0.0, color="black", lw=1.0, ls=":")
    ax.set_xscale("log")
    ax.set_xlim(max(1.0, SPATIAL_LAG_EDGES_KM[1]), SPATIAL_PLOT_MAX_LAG_KM)
    ax.set_xlabel(r"Spatial lag $d$ (km)")
    ax.set_ylabel(r"Empirical correlation minus common fit")
    ax.grid(alpha=0.25)
    ax.legend(
        frameon=False,
        title=r"Period length $\Delta t$",
        fontsize=PLOT_LEGEND_FONTSIZE,
        title_fontsize=PLOT_LEGEND_TITLE_FONTSIZE,
    )
    fig.savefig(OUT_SPATIAL_RAW_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_joint_correlation_figure(
    spatial_agg: pd.DataFrame,
    spatial_params: dict,
    temporal_agg: pd.DataFrame,
    temporal_params: np.ndarray,
) -> None:
    periods = SPATIAL_PERIOD_LENGTHS_TO_PLOT
    cmap = plt.get_cmap("viridis")
    norm = mpl.colors.Normalize(vmin=min(periods), vmax=max(periods))

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0), constrained_layout=True)
    ax_s, ax_t = axes

    for period in periods:
        sub = spatial_agg.loc[
            np.isclose(
                spatial_agg["period_years"].to_numpy(float),
                float(period),
                atol=PERIOD_ATOL,
            )
            & spatial_agg["used_for_corr"].astype(bool)
        ].sort_values("lag_center_km")
        if sub.empty:
            continue
        color = cmap(norm(period))
        ax_s.scatter(
            sub["lag_center_km"],
            sub["corr"],
            s=PLOT_SCATTER_SIZE,
            color=color,
            alpha=0.75,
            edgecolors="none",
        )
        ax_s.plot(
            sub["lag_center_km"],
            sub["corr"],
            color=color,
            lw=1.0,
            alpha=0.65,
        )

    x_space = np.geomspace(
        max(1.0, SPATIAL_LAG_EDGES_KM[1]),
        SPATIAL_PLOT_MAX_LAG_KM,
        800,
    )
    ax_s.plot(
        x_space,
        spatial_corr_model(x_space, spatial_params),
        color="black",
        lw=PLOT_MODEL_LINEWIDTH,
        label="Common spatial fit",
    )
    ax_s.axhline(0, color="black", lw=0.9, ls=":")
    ax_s.set_xscale("log")
    ax_s.set_xlim(
        max(1.0, SPATIAL_LAG_EDGES_KM[1]),
        SPATIAL_PLOT_MAX_LAG_KM,
    )
    ax_s.set_ylim(-0.08, 1.02)
    ax_s.set_xlabel(r"Spatial lag $d$ (km)")
    ax_s.set_ylabel(r"Correlation of standardized residuals $r_{\rho}^{\rm s}$")
    ax_s.grid(alpha=0.25)

    period_handles = [
        mpl.lines.Line2D(
            [],
            [],
            marker="o",
            markersize=PLOT_LEGEND_MARKERSIZE,
            lw=1.3,
            color=cmap(norm(period)),
            label=f"{period:g} yr",
        )
        for period in periods
    ]
    legend_period = ax_s.legend(
        handles=period_handles,
        title=r"Period length $\Delta t$",
        frameon=False,
        fontsize=PLOT_LEGEND_FONTSIZE,
        title_fontsize=PLOT_LEGEND_TITLE_FONTSIZE,
        loc="upper right",
    )
    ax_s.add_artist(legend_period)
    ax_s.legend(
        handles=[
            mpl.lines.Line2D(
                [],
                [],
                lw=PLOT_MODEL_LINEWIDTH,
                color="black",
                label="Common spatial fit",
            )
        ],
        frameon=False,
        fontsize=PLOT_LEGEND_FONTSIZE,
        loc="upper center",
    )

    if not temporal_agg.empty:
        sub = temporal_agg.sort_values("lag_center_yr")
        ax_t.scatter(
            sub["lag_center_yr"],
            sub["corr"],
            s=32,
            color="black",
            alpha=0.75,
            edgecolors="none",
            label="Binned estimate",
        )

    x_time = np.linspace(0.0, TEMPORAL_MAX_LAG_YEARS, 500)
    y_empirical = temporal_corr_model(x_time, *temporal_params)
    ax_t.plot(
        x_time,
        y_empirical,
        color="0.35",
        lw=2.0,
        ls="--",
        label="Empirical diagnostic fit",
    )
    ax_t.axhline(
        0.0,
        color="black",
        lw=2.0,
        label=r"Applied model: $r_{\rho}^{\rm t}(\tau>0)=0$",
    )
    ax_t.scatter([0.0], [1.0], color="black", s=24, zorder=4)
    ax_t.set_xlim(0, TEMPORAL_MAX_LAG_YEARS)
    ax_t.set_ylim(-0.08, 1.02)
    ax_t.set_xlabel(r"Temporal lag $\tau$ (yr)")
    ax_t.set_ylabel(r"Correlation of standardized residuals $r_{\rho}^{\rm t}$")
    ax_t.grid(alpha=0.25)
    ax_t.legend(
        frameon=False,
        fontsize=PLOT_LEGEND_FONTSIZE,
        loc="upper right",
    )

    add_panel_letter(ax_s, "a")
    add_panel_letter(ax_t, "b")
    fig.savefig(OUT_MAIN_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    t0 = perf_counter()
    df = None
    if REUSE_EMPIRICAL_ESTIMATES and OUT_SPATIAL_AGG_CSV.exists():
        log(f"Reusing spatial empirical bins from {OUT_SPATIAL_AGG_CSV}", t0)
        spatial_agg = pd.read_csv(OUT_SPATIAL_AGG_CSV)
        spatial_agg = mark_spatial_bins_used_for_fit(spatial_agg)
    else:
        log(f"Reading {IN_CSV}", t0)
        df, _ = read_standardized_residuals()
        log(
            f"Prepared {len(df):,} rows across run labels: "
            f"{sorted(df[RUN_COL].astype(str).unique().tolist())}",
            t0,
        )

        log("Estimating direct empirical spatial correlations", t0)
        spatial_emp = empirical_spatial_correlation(df)
        spatial_emp.to_csv(OUT_SPATIAL_EMPIRICAL_CSV, index=False)
        if spatial_emp.empty:
            raise RuntimeError("No spatial empirical correlation rows were produced.")

        spatial_agg = aggregate_spatial_empirical(spatial_emp)
    if spatial_agg.empty:
        raise RuntimeError("No spatial empirical bins survived aggregation.")

    log("Fitting one duration-invariant spatial correlation model", t0)
    spatial_out, spatial_period_diag, spatial_candidate_diag, _ = fit_constant_spatial_model(spatial_agg)
    spatial_agg.to_csv(OUT_SPATIAL_AGG_CSV, index=False)
    spatial_agg.loc[spatial_agg["used_for_corr"].astype(bool)].to_csv(
        OUT_SPATIAL_USED_CSV, index=False
    )
    spatial_period_diag.to_csv(OUT_SPATIAL_PERIOD_DIAGNOSTICS_CSV, index=False)
    spatial_candidate_diag.to_csv(OUT_SPATIAL_CANDIDATES_CSV, index=False)
    pd.DataFrame([spatial_out]).to_csv(OUT_SPATIAL_PARAMS_CSV, index=False)
    with open(OUT_SPATIAL_PARAMS_JSON, "w") as f:
        json.dump(spatial_out, f, indent=2, sort_keys=True)
    log(f"Spatial parameters: {spatial_out}", t0)

    if REUSE_EMPIRICAL_ESTIMATES and OUT_TEMPORAL_AGG_CSV.exists():
        log(f"Reusing temporal empirical bins from {OUT_TEMPORAL_AGG_CSV}", t0)
        temporal_agg = pd.read_csv(OUT_TEMPORAL_AGG_CSV)
    else:
        if df is None:
            log(f"Reading {IN_CSV}", t0)
            df, _ = read_standardized_residuals()
            log(
                f"Prepared {len(df):,} rows across run labels: "
                f"{sorted(df[RUN_COL].astype(str).unique().tolist())}",
                t0,
            )
        log("Estimating empirical temporal correlations for diagnostics", t0)
        temporal_emp = empirical_temporal_variogram(df)
        temporal_emp.to_csv(OUT_TEMPORAL_EMPIRICAL_CSV, index=False)
        if temporal_emp.empty:
            annual = df.loc[
                np.isclose(
                    df[PERIOD_COL].to_numpy(float),
                    TEMPORAL_PERIOD_LENGTH_YEARS,
                    atol=PERIOD_ATOL,
                )
            ]
            raise RuntimeError(
                "No temporal empirical glacier-run correlations were produced. "
                f"Annual rows available: {len(annual):,}."
            )

        temporal_agg = aggregate_temporal_empirical(temporal_emp)
        temporal_agg.to_csv(OUT_TEMPORAL_AGG_CSV, index=False)
    if temporal_agg.empty:
        raise RuntimeError("No temporal empirical bins survived aggregation.")

    temporal_params = fit_temporal_model(temporal_agg)
    temporal_out = {
        "temporal_model_form": "nugget_exponential",
        "empirical_nugget": float(temporal_params[0]),
        "empirical_sill": float(1.0 - temporal_params[0]),
        "empirical_range_yr": float(temporal_params[1]),
        "empirical_timescale_yr": float(temporal_params[1]),
        "empirical_exponent": 1.0,
        "empirical_amplitude_after_nugget": float(1.0 - temporal_params[0]),
        "n_fit_bins": int(len(temporal_agg)),
        "applied_temporal_correlation_at_positive_lag": 0.0,
        "applied_temporal_covariance_assumption": "zero_for_nonzero_lags",
        "semivariogram_estimator": SEMIVARIOGRAM_ESTIMATOR,
    }
    pd.DataFrame([temporal_out]).to_csv(OUT_TEMPORAL_PARAMS_CSV, index=False)
    with open(OUT_TEMPORAL_PARAMS_JSON, "w") as f:
        json.dump(temporal_out, f, indent=2, sort_keys=True)
    log(f"Temporal diagnostic parameters: {temporal_out}", t0)

    combined_out = {
        **{f"spatial_{key}": value for key, value in spatial_out.items()},
        **{f"temporal_{key}": value for key, value in temporal_out.items()},
    }
    pd.DataFrame([combined_out]).to_csv(OUT_MAIN_CSV, index=False)
    log(f"Saved primary correlation output: {OUT_MAIN_CSV}", t0)

    if UPDATE_PACKAGED_PARAMS:
        update_packaged_correlation_params(spatial_out, temporal_out)
        log(f"Updated packaged correlation parameters: {PACKAGED_PARAM_JSON}", t0)

    if WRITE_FIT_DIAGNOSTIC_PLOTS:
        plot_joint_correlation_figure(
            spatial_agg,
            spatial_out,
            temporal_agg,
            temporal_params,
        )
        plot_spatial_fit_residuals(spatial_agg, spatial_out)
        log(f"Saved main correlation figure: {OUT_MAIN_FIG}", t0)
        log(f"Saved spatial fit residual figure: {OUT_SPATIAL_RAW_FIG}", t0)


if __name__ == "__main__":
    main()
