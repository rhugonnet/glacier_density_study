#!/usr/bin/env python3
"""Supplementary figure S3: normality check."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

STUDY_DIR = Path(__file__).resolve().parents[1]
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

from study_paths import FIGURES_DIR, STANDARDIZED_RESIDUALS_PATH

# Define project paths

# Define plotting constants
DPI = 300

# Keep typography consistent with fit-script figures
mpl.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def add_panel_letter(ax, letter: str) -> None:
    """Add a panel label

    :param ax: Matplotlib axis to label
    :param letter: Panel letter
    """
    ax.text(0.02, 0.98, letter, transform=ax.transAxes, ha="left", va="top", fontsize=14, fontweight="bold")


def valid_xy(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep finite values with positive weights

    :param values: Sample values
    :param weights: Sample weights
    """
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return values[ok], weights[ok]


def weighted_quantile(values: np.ndarray, quantiles: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute weighted quantiles

    :param values: Sample values
    :param quantiles: Quantiles between zero and one
    :param weights: Non-negative sample weights
    """
    values, weights = valid_xy(values, weights)
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cdf = np.cumsum(w) - 0.5 * w
    cdf = cdf / np.sum(w)
    return np.interp(quantiles, cdf, v)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute a weighted mean

    :param values: Sample values
    :param weights: Sample weights
    """
    values, weights = valid_xy(values, weights)
    return float(np.average(values, weights=weights))


def weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute a weighted standard deviation

    :param values: Sample values
    :param weights: Sample weights
    """
    values, weights = valid_xy(values, weights)
    mu = weighted_mean(values, weights)
    return float(np.sqrt(np.average((values - mu) ** 2, weights=weights)))


def weighted_excess_kurtosis(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute weighted excess kurtosis

    :param values: Sample values
    :param weights: Sample weights
    """
    values, weights = valid_xy(values, weights)
    mu = weighted_mean(values, weights)
    sd = weighted_std(values, weights)
    if sd <= 0 or not np.isfinite(sd):
        return np.nan
    return float(np.average(((values - mu) / sd) ** 4, weights=weights) - 3.0)


def weighted_quantile_skewness(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute Bowley quantile skewness

    :param values: Sample values
    :param weights: Sample weights
    """
    q25, q50, q75 = weighted_quantile(values, np.array([0.25, 0.50, 0.75]), weights)
    denom = q75 - q25
    if denom <= 0 or not np.isfinite(denom):
        return np.nan
    return float((q75 + q25 - 2.0 * q50) / denom)


def robust_normality_sample(values: np.ndarray, weights: np.ndarray, qlo: float = 0.005, qhi: float = 0.995) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return the central weighted distribution shown in histograms

    :param values: Sample values
    :param weights: Sample weights
    :param qlo: Lower clipping quantile
    :param qhi: Upper clipping quantile
    """
    values, weights = valid_xy(values, weights)
    lo, hi = weighted_quantile(values, np.array([qlo, qhi]), weights)
    keep = (values >= lo) & (values <= hi)
    return values[keep], weights[keep], float(lo), float(hi)


def normality_metrics(values: np.ndarray, weights: np.ndarray) -> tuple[float, float, float, float]:
    """Compute displayed normality metrics

    :param values: Sample values
    :param weights: Sample weights
    """
    return (
        weighted_mean(values, weights),
        weighted_std(values, weights),
        weighted_quantile_skewness(values, weights),
        weighted_excess_kurtosis(values, weights),
    )


def plot_weighted_hist_with_normal(ax, values: np.ndarray, weights: np.ndarray, x_label: str, row_label: str, show_legend: bool = False) -> None:
    """Plot weighted histogram, normal fit and metrics

    :param ax: Matplotlib axis to draw into
    :param values: Sample values
    :param weights: Sample weights
    :param x_label: X-axis label
    :param row_label: Row label for the y axis
    """
    v_rob, w_rob, q_lo, q_hi = robust_normality_sample(values, weights)
    mu, sd, sk, ek = normality_metrics(values, weights)
    if len(v_rob) == 0:
        return
    bins = np.linspace(q_lo, q_hi, 45)
    ax.hist(v_rob, bins=bins, weights=w_rob, density=True, alpha=0.55, edgecolor="none", label="Volume change weighted\nhistogram")
    if np.isfinite(mu) and np.isfinite(sd) and sd > 0:
        xx = np.linspace(q_lo, q_hi, 500)
        ax.plot(xx, stats.norm.pdf(xx, loc=mu, scale=sd), lw=2.0, label="Normal fit")
    txt = f"Mean = {mu:.2f}\nSTD = {sd:.2f}\nRobust skewness = {sk:.3f}\nExcess kurtosis = {ek:.3f}"
    ax.text(0.02, 0.50, txt, transform=ax.transAxes, ha="left", va="center", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.95))
    if show_legend:
        ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.set_ylabel(row_label)
    ax.set_xlabel(x_label)
    ax.grid(alpha=0.2)


def short_normality_label(x_label: str) -> str:
    """Return a compact Q-Q y-label suffix

    :param x_label: Full histogram x-label
    """
    if "z_" in x_label or "z_{" in x_label:
        return r"of standardized residual $z_{\rho}$"
    if "residual" in x_label or "Residual" in x_label:
        return r"after subtracting $\mu_{\rho}$"
    return r"of raw $\rho_{\Delta V}$"


def plot_weighted_qq(ax, values: np.ndarray, weights: np.ndarray, x_label: str) -> None:
    """Plot standardized weighted normal Q-Q points

    :param ax: Matplotlib axis to draw into
    :param values: Sample values
    :param weights: Sample weights
    :param x_label: Full histogram x-label
    """
    values, weights = valid_xy(values, weights)
    mu, sd, _, _ = normality_metrics(values, weights)
    if len(values) == 0 or not np.isfinite(sd) or sd <= 0:
        return
    probs = np.linspace(0.001, 0.999, 399)
    obs = weighted_quantile(values, probs, weights)
    theo = stats.norm.ppf(probs)
    obs_std = (obs - mu) / sd
    ok = np.isfinite(obs_std) & np.isfinite(theo)
    obs_std = obs_std[ok]
    theo = theo[ok]
    ax.scatter(theo, obs_std, s=12, alpha=0.8, linewidths=0)
    ax.plot([-3, 3], [-3, 3], color="black", lw=1)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel("Theoretical normal quantiles")
    ax.set_ylabel("Observed quantiles\n" + short_normality_label(x_label))
    ax.grid(alpha=0.2)


# Read standardized residual output
cols = ["rho", "rho_mean_removed", "z_rho", "abs_dV_weight"]
df = pd.read_csv(STANDARDIZED_RESIDUALS_PATH, usecols=cols)
weights = df["abs_dV_weight"].to_numpy(float)
stages = [
    ("rho", r"Effective density $\rho_{\Delta V}$ (kg m$^{-3}$)", r"Raw $\rho_{\Delta V}$"),
    ("rho_mean_removed", r"Mean-removed residual $\rho_{\Delta V} - \mu_{\rho}$ (kg m$^{-3}$)", r"After subtracting $\mu_{\rho}$"),
    ("z_rho", r"Standardized residual $z_{\rho}$", r"After standardizing by $\mu_{\rho}$ and $\sigma_{\rho}$"),
]

# Plot distribution and Q-Q rows
fig, axes = plt.subplots(3, 2, figsize=(10.0, 10.4), constrained_layout=True)
axes[0, 0].set_title("Normal fit to volume change weighted distribution")
axes[0, 1].set_title("Q-Q plot")
letters = iter(list("abcdef"))
for i, (col, xlabel, row_label) in enumerate(stages):
    values = df[col].to_numpy(float)
    plot_weighted_hist_with_normal(axes[i, 0], values, weights, xlabel, row_label, show_legend=(i == 0))
    plot_weighted_qq(axes[i, 1], values, weights, xlabel)
    add_panel_letter(axes[i, 0], next(letters))
    add_panel_letter(axes[i, 1], next(letters))

# Save figure
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
out = FIGURES_DIR / "FIG_supp_03_normality_check.pdf"
fig.savefig(out, dpi=DPI, bbox_inches="tight")
plt.close(fig)
print(f"[done] Wrote figure: {out}")
