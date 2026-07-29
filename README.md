# glacier_density_study

Code for **Huss, Hugonnet, et al., _Converting glacier volume change to mass change: a global assessment_**.

This repository contains the study scripts and the reusable Python package `glacier_density_surrogate`. The package predicts the effective density of glacier volume change and its uncertainty from elevation-change observations, without running the full mass-balance and firn-densification model used for calibration.

Below a short guide to: use the surrogate on your own elevation-change data, reproduce the study processing steps, and regenerate the figures, tables and manuscript values.

## Use the surrogate

### Setup environment

Install the package from the repository root:

```sh
python -m pip install -e .
```

For the study scripts and tests, install the optional dependencies:

```sh
python -m pip install -e ".[study,test]"
```

The package uses the finalized surrogate parameters bundled in `glacier_density_surrogate/parameters/final_parameters.json`. Users normally do not need to load parameter files manually.

### Input format

For a glacier time series, use one row per annual or multi-annual elevation-change period:

```csv
start_year,end_year,dh_m,sigma_dh_m,area_m2
2000,2001,-0.55,0.18,25000000
2001,2002,-0.35,0.18,25000000
2002,2003,0.10,0.18,25000000
2003,2004,-0.85,0.20,25000000
2004,2005,-0.65,0.20,25000000
```

The example file is available in `examples/synthetic_elevation_timeseries.csv`.

Required columns:
* `start_year`: start of the observation period.
* `end_year`: end of the observation period.
* `dh_m`: glacier-wide elevation change over the period, in metres.
* `sigma_dh_m`: one-sigma uncertainty of `dh_m`, in metres.

Optional columns:
* `area_m2`: glacier area in square metres. If omitted, the model returns effective density but not mass change.
* `past_dh_m`: past elevation-change predictor, in metres. For multi-year observations, this should normally be an annual-equivalent recent trend.
* `sigma_past_dh_m`: one-sigma uncertainty of `past_dh_m`, in metres.

If past elevation change is not provided, the default assumption is persistence: the first period uses the current-period trend, `dh_m / period_years`, and later periods use an exponentially weighted mean of preceding rows. Use `--past-missing zero` to set missing past change to zero instead. This assumption should be reported, as past conditions affect the predicted mean density.

### Command-line interface

Apply the surrogate to a CSV:

```sh
glacier-density-surrogate \
  examples/synthetic_elevation_timeseries.csv \
  examples/synthetic_elevation_timeseries_predictions.csv
```

The output CSV contains independent and temporally reconciled estimates, including:
* `mu_rho_independent_kg_m3`, `sigma_rho_independent_kg_m3`
* `mu_rho_closed_kg_m3`, `sigma_rho_closed_kg_m3`
* `dV_m3`, `dM_closed_kg`

For a time series, temporal reconciliation is applied automatically when the input rows are consecutive non-overlapping periods. Use `--no-expand-periods` to evaluate only the input rows.

Useful options:

```sh
glacier-density-surrogate input.csv output.csv \
  --past-missing current \
  --dh-error-corr-form exponential \
  --dh-error-corr-range 5
```

A single period can also be evaluated directly:

```sh
glacier-density-surrogate --dh -1.0 --sigma-dh 0.2 --dt 5 --area-m2 1000000
```

### Python interface

For one glacier and one observation period:

```python
from glacier_density_surrogate import RhoSurrogate

model = RhoSurrogate()

out = model.predict(
    dh=-1.0,
    sigma_dh=0.2,
    dt=5,
    area_m2=1_000_000,
)

print(out["mu_rho_kg_m3"], out["sigma_rho_kg_m3"])
```

For a time series:

```python
import pandas as pd

from glacier_density_surrogate import RhoSurrogate, make_dh_error_correlation

observations = pd.read_csv("examples/synthetic_elevation_timeseries.csv")
model = RhoSurrogate()

predictions = model.predict_timeseries(
    observations,
    dh_error_corr=make_dh_error_correlation("none"),
)

print(predictions[[
    "start",
    "end",
    "dh_m",
    "mu_rho_closed_kg_m3",
    "sigma_rho_closed_kg_m3",
]])
```

Advanced users can call lower-level helpers directly:
* `model.mu_rho(...)`
* `model.sigma_rho(...)`
* `model.integrated_mu(...)`
* `model.integrated_sigma(...)`
* `model.spatial_corr(...)`
* `model.temporal_corr(...)`

### Synthetic example

The synthetic example plot is generated from the example CSV with:

```sh
python examples/plot_synthetic_example.py
```

<img src="examples/synthetic_elevation_timeseries_surrogate.png" width="780">

### What the surrogate does

For each observation period, the surrogate:
1. predicts mean effective density from current elevation change, past elevation change and period length;
2. predicts effective-density uncertainty from elevation change and period length;
3. integrates both quantities over uncertainty in observed elevation change;
4. reconciles nested or consecutive time periods when a time series is provided;
5. provides spatial and temporal error-correlation functions for uncertainty propagation over multiple glaciers and periods.

The model is intended as a practical replacement for fixed density-conversion factors such as `850 +/- 60 kg m-3`. It remains a statistical surrogate of the full model, so input assumptions such as missing past elevation change and elevation-change error correlation should be documented in applications.

## Reproduce the study

### Data and paths

The study scripts use the final manuscript workspace defined in `scripts_study_2026/study_paths.py`:

```text
/home/atom/ongoing/own/glacier_density_study/final
```

Main inputs are read from:

```text
/home/atom/ongoing/own/glacier_density_study/final/data/
```

Main results are written to:

```text
/home/atom/ongoing/own/glacier_density_study/final/results/
```

Figures and manuscript-ready table files are written to:

```text
/home/atom/ongoing/own/glacier_density_study/final/figures/
```

The public data archive and DOI will be added here once the dataset is finalized.

### Setup study environment

A Conda environment file is provided in `scripts_study_2026/environment.yml`:

```sh
conda env create -f scripts_study_2026/environment.yml
conda activate glacier-density
python -m pip install --no-build-isolation -e ".[study,test]"
```

The last command installs this repository in editable mode, so changes to `glacier_density_surrogate/` are used immediately by the scripts.

### Processing scripts

Study processing scripts are in `scripts_study_2026/analysis/`:
* `convert_fullmodel_outputs_to_dataframe.py`: converts full-model outputs to the analysis dataframe.
* `fit_mean_uncertainty_rho.py`: fits the surrogate mean and uncertainty functions and writes standardized residuals.
* `fit_spatial_temporal_correlation.py`: fits spatial and temporal error-correlation functions.
* `surrogate_full_model_agreement.py`: compares surrogate predictions to full-model estimates after glacier and regional aggregation.
* `apply_surrogate_hugonnet2021.py`: applies the final surrogate to Hugonnet-style global volume-change estimates.
* `firn_parametrization_variance_contribution.py`: diagnostic script for firn-parametrization uncertainty contribution.

Final fit outputs used by later scripts are:
* `results/mu_sigma_model_parameters.csv`
* `results/diagnostics/correlations/rho_error_correlation_standardized_residuals_directcorr_constantspace_temporaldiagnostic_spatial_fit_parameters.csv`
* `results/diagnostics/correlations/rho_error_correlation_standardized_residuals_directcorr_constantspace_temporaldiagnostic_temporal_fit_parameters.csv`

After updating the fit outputs, update the bundled JSON parameters with:

```python
from pathlib import Path

from glacier_density_surrogate import write_packaged_params

results_dir = Path("/home/atom/ongoing/own/glacier_density_study/final/results")
correlation_dir = results_dir / "diagnostics" / "correlations"

write_packaged_params(
    results_dir / "mu_sigma_model_parameters.csv",
    correlation_dir / "rho_error_correlation_standardized_residuals_directcorr_constantspace_temporaldiagnostic_spatial_fit_parameters.csv",
    correlation_dir / "rho_error_correlation_standardized_residuals_directcorr_constantspace_temporaldiagnostic_temporal_fit_parameters.csv",
)
```

### Figures, tables and manuscript values

Figure, table and manuscript-value scripts are in `scripts_study_2026/figures/`:
* `fig_0_surrogate_flowchart.py`
* `fig_1_mean_function.py`
* `fig_2_uncertainty_function.py`
* `fig_3_correlation_functions.py`
* `fig_4_uncertain_dh.py`
* `fig_5_temporal_closure.py`
* `fig_6_tile_area.py`
* `fig_7_regional_full_period_boxplot.py`
* `fig_s1_mean_residuals.py`
* `fig_s2_uncertainty_residuals.py`
* `fig_s3_normality_check.py`
* `fig_s4_decadal_change_maps.py`
* `fig_s5_standardized_residual_map.py`
* `table_s1_agreement_categories.py`
* `manuscript_values.py`

Run any script directly, for example:

```sh
python scripts_study_2026/figures/fig_1_mean_function.py
python scripts_study_2026/figures/table_s1_agreement_categories.py
python scripts_study_2026/figures/manuscript_values.py
```

Main manuscript values are written to:

```text
/home/atom/ongoing/own/glacier_density_study/final/results/manuscript_values.csv
```

Supplementary Table 1 is written to:

```text
/home/atom/ongoing/own/glacier_density_study/final/figures/TABLE_supp_01_agreement_categories.tex
/home/atom/ongoing/own/glacier_density_study/final/figures/TABLE_supp_01_agreement_categories.csv
```

### Run the full pipeline

The full final pipeline is:

```sh
python scripts_study_2026/run_final_pipeline.py
```

The pipeline runs the final analysis scripts, manuscript-value script, Supplementary Table 1 script and all final figure scripts in dependency order.

## Repository layout

```text
glacier_density_surrogate/
  surrogate.py       # scalar model, integration and prediction API
  timeseries.py      # vectorized prediction and temporal reconciliation
  aggregation.py     # spatial and regional/global propagation helpers
  cli.py             # command-line interface
  parameters/
    final_parameters.json

examples/
  synthetic_elevation_timeseries.csv
  plot_synthetic_example.py
  synthetic_elevation_timeseries_surrogate.png

scripts_study_2026/
  environment.yml
  study_paths.py
  run_final_pipeline.py
  analysis/
  figures/

tests/
  test_surrogate.py
```
