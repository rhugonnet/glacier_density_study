#!/usr/bin/env python3
"""
Run the final 2026 analysis and figure pipeline.

The objective of this script is to keep the final full study run reproducible from Python only.
All choices/order of scripts is encoded below.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


STUDY_DIR = Path(__file__).resolve().parent
REPO_DIR = STUDY_DIR.parent

ANALYSIS_SCRIPTS = [
    "analysis/fit_mean_uncertainty_rho.py",
    "analysis/fit_spatial_temporal_correlation.py",
    "analysis/surrogate_full_model_agreement.py",
    "analysis/apply_surrogate_hugonnet2021.py",
]

MANUSCRIPT_VALUE_SCRIPT = "figures/manuscript_values.py"
TABLE_SCRIPTS = [
    "figures/table_s1_agreement_categories.py",
]

FIGURE_SCRIPTS = [
    "figures/fig_0_surrogate_flowchart.py",
    "figures/fig_1_mean_function.py",
    "figures/fig_2_uncertainty_function.py",
    "figures/fig_3_correlation_functions.py",
    "figures/fig_4_uncertain_dh.py",
    "figures/fig_5_temporal_closure.py",
    "figures/fig_6_tile_area.py",
    "figures/fig_7_regional_full_period_boxplot.py",
    "figures/fig_s1_mean_residuals.py",
    "figures/fig_s2_uncertainty_residuals.py",
    "figures/fig_s3_normality_check.py",
    "figures/fig_s4_decadal_change_maps.py",
    "figures/fig_s5_standardized_residual_map.py",
]


def run_script(relative_path: str) -> None:
    """Run one pipeline script with the current Python interpreter."""
    script = STUDY_DIR / relative_path
    print(f"\n[run] {script}", flush=True)
    subprocess.run([sys.executable, str(script)], cwd=REPO_DIR, check=True)


def main() -> None:
    """Run analyses, manuscript values, and all figures in dependency order."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    for script in ANALYSIS_SCRIPTS:
        run_script(script)

    run_script(MANUSCRIPT_VALUE_SCRIPT)

    for script in TABLE_SCRIPTS:
        run_script(script)

    for script in FIGURE_SCRIPTS:
        run_script(script)

    print("\n[done] Final pipeline completed.", flush=True)


if __name__ == "__main__":
    main()
