"""Run the README examples and save glacier and regional predictions."""

import json
from pathlib import Path

import pandas as pd

from glacier_density_surrogate import RhoSurrogate

inputs = Path(__file__).resolve().parent / "inputs"
outputs = Path(__file__).resolve().parent / "outputs"
outputs.mkdir(exist_ok=True)
model = RhoSurrogate()

# One glacier and observation period
result = model.predict(dh=-1.0, sigma_dh=0.2, dt=5, area_m2=1_000_000)
(outputs / "single_period_prediction.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

# A single glacier elevation time series
observations = pd.read_csv(inputs / "dh_timeseries.csv")
predictions = model.predict_timeseries(observations)
predictions.to_csv(outputs / "dh_predictions.csv", index=False)

# Multiple glacier time series, then regional aggregation
observations = pd.read_csv(inputs / "regional_dh_timeseries.csv")
predictions = model.predict_timeseries(observations)
regional = model.aggregate_regions(predictions)
predictions.to_csv(outputs / "glacier_predictions.csv", index=False)
regional.to_csv(outputs / "regional_predictions.csv", index=False)
