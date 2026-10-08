"""Tests for CLI interface with JSON and CSV files."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from glacier_density_surrogate import RhoSurrogate
from glacier_density_surrogate.cli import main


def test_main__single_period_json(capsys) -> None:
    """Checks that direct observations produce JSON with consistent volume change and mass change."""

    # Convert a five-year elevation loss over a known glacier area
    status = main(["--dh", "-1", "--sigma-dh", "0.2", "--dt", "5", "--area-m2", "1000000"])
    result = json.loads(capsys.readouterr().out)

    # Check the assumed past elevation change rate and derived mass change
    assert status == 0
    assert result["past_dh_m"] == -0.2
    assert result["dV_m3"] == -1_000_000
    assert np.isfinite(result["mu_rho_kg_m3"])
    assert result["dM_kg"] == result["mu_rho_kg_m3"] * result["dV_m3"]
    assert result["sigma_dM_rho_kg"] > 0
    assert result["sigma_dM_dh_kg"] > 0
    assert result["sigma_dM_total_kg"] > result["sigma_dM_rho_kg"]
    assert result["sigma_dV_m3"] == 200_000


def test_module__single_period_json() -> None:
    """Checks that python -m glacier_density_surrogate prints the same prediction as the API."""

    # Use the documented example to check the module entry point with the active interpreter
    arguments = [
        sys.executable, "-m", "glacier_density_surrogate",
        "--dh", "-1.0", "--sigma-dh", "0.2", "--dt", "5", "--area-m2", "1000000",
    ]
    expected = RhoSurrogate().predict(dh=-1.0, sigma_dh=0.2, dt=5, area_m2=1_000_000)

    # A successful process must emit a JSON prediction without extra output
    process = subprocess.run(arguments, capture_output=True, text=True, check=True)
    result = json.loads(process.stdout)
    assert result == expected
    assert process.stderr == ""


@pytest.mark.parametrize("sigma_dh", ["0", "0.2"])
def test_main__zero_change_valid_json(capsys, sigma_dh) -> None:
    """Checks that zero change emits JSON null for undefined density and finite mass change."""

    # Zero net volume change makes effective density undefined for exact or uncertain change
    status = main(["--dh", "0", "--sigma-dh", sigma_dh, "--dt", "1", "--area-m2", "1000000"])
    output = capsys.readouterr().out

    # JSON must represent undefined density as null, without NaN or +/-inf
    def reject_constant(value):
        raise ValueError(f"Invalid JSON constant: {value}")

    result = json.loads(output, parse_constant=reject_constant)
    assert status == 0
    assert result["mu_rho_kg_m3"] is None
    assert result["sigma_rho_kg_m3"] is None
    assert result["dV_m3"] == 0
    assert np.isfinite(result["dM_kg"])
    assert np.isfinite(result["sigma_dM_rho_kg"])
    assert result["sigma_dM_rho_kg"] > 0
    assert np.isfinite(result["sigma_dM_dh_kg"])
    assert np.isfinite(result["sigma_dM_total_kg"])


def test_main__csv_reconciles_contiguous_periods(tmp_path, capsys) -> None:
    """Checks that a CSV produces annual and combined periods whose mass changes add."""

    # Write two consecutive observations using the documented CSV column names
    input_path = tmp_path / "observations.csv"
    output_path = tmp_path / "predictions" / "density.csv"
    observations = pd.DataFrame({
        "start_year": [2000, 2001],
        "end_year": [2001, 2002],
        "dh_m": [-0.5, -0.4],
        "sigma_dh_m": [0.1, 0.1],
        "area_m2": [1_000_000, 1_000_000],
    })
    observations.to_csv(input_path, index=False)

    # Run the CSV interface and read its saved predictions
    status = main([str(input_path), str(output_path)])
    predictions = pd.read_csv(output_path)

    # The mass change over both years must equal the sum of the annual mass changes
    annual_mass = predictions.loc[predictions["period_years"] == 1, "dM_kg"].sum()
    combined_mass = predictions.loc[predictions["period_years"] == 2, "dM_kg"].iloc[0]
    assert status == 0
    assert len(predictions) == 3
    np.testing.assert_allclose(combined_mass, annual_mass, rtol=1e-12)
    assert str(output_path) in capsys.readouterr().out


@pytest.mark.parametrize("change", ["-1", "0"])
def test_main__error_missing_dt(capsys, change):
    """Checks an error is raised for direct observations without an explicit period duration."""

    # Nonzero and zero changes both require a measured observation period
    with pytest.raises(SystemExit) as error:
        main(["--dh", change])

    # Report the missing argument before a prediction can be printed
    output = capsys.readouterr()
    assert error.value.code == 2
    assert "--dt is required with --dh" in output.err
    assert output.out == ""


class TestMultipleGlacierCSV:
    """Test module for shared glacier observations and separate regional CSV output."""

    @pytest.mark.parametrize("id_col", ["glacier_id", "name"])
    def test_main__glacier_and_regional_outputs(self, tmp_path, capsys, id_col):
        """Checks that one CSV produces independent glacier predictions and additive regional estimates."""

        # Different histories, areas and coordinates describe two glaciers in one region
        first = pd.DataFrame({
            id_col: ["g1", "g1"], "start_year": [2000, 2001], "end_year": [2001, 2002],
            "dh_m": [-1.0, -0.4], "sigma_dh_m": [0.1, 0.2], "area_m2": [1e6, 1e6],
            "lat": [28.0, 28.0], "lon": [87.0, 87.0], "region_group": ["A", "A"],
        })
        second = first.assign(**{id_col: "g2"}, dh_m=[-0.2, 0.6], area_m2=2e6, lat=29.0)
        observations = pd.concat([first, second], ignore_index=True)
        input_path = tmp_path / "observations.csv"
        glacier_path = tmp_path / "glaciers.csv"
        regional_path = tmp_path / "region" / "predictions.csv"
        observations.to_csv(input_path, index=False)

        # Request regional propagation alongside the normal output from the shared input
        arguments = [str(input_path), str(glacier_path), "--regional-output", str(regional_path)]
        if id_col == "name":
            arguments.extend(["--id-col", id_col])
        assert main(arguments) == 0
        glaciers = pd.read_csv(glacier_path)
        regional = pd.read_csv(regional_path)

        # Shared predictions agree with the API, and regional changes count each glacier once
        expected = RhoSurrogate().predict_timeseries(observations, id_col=id_col)
        pd.testing.assert_frame_equal(glaciers, expected, check_dtype=False)
        assert len(glaciers) == 6
        assert len(regional) == 3
        annual = regional.loc[regional["period_years"] == 1]
        full = regional.loc[regional["period_years"] == 2].iloc[0]
        np.testing.assert_allclose(full["dM_kg"], annual["dM_kg"].sum())
        np.testing.assert_allclose(full["sigma_dM_rho_kg"] ** 2, np.sum(annual["sigma_dM_rho_kg"] ** 2))
        glacier_full = glaciers.loc[glaciers["period_years"] == 2]
        expected_input = np.linalg.norm(glacier_full["sigma_dM_dh_kg"])
        np.testing.assert_allclose(full["sigma_dM_dh_kg"], expected_input)
        np.testing.assert_allclose(full["sigma_dM_total_kg"], np.hypot(full["sigma_dM_rho_kg"], expected_input))
        assert str(regional_path) in capsys.readouterr().out

    def test_main__error_regional_metadata_before_writing(self, tmp_path):
        """Checks an error is raised for missing coordinates before either output CSV is written."""

        # Glacier predictions work without coordinates, but spatial propagation requires them
        observations = pd.DataFrame({
            "glacier_id": ["g1"], "start_year": [2000], "end_year": [2001],
            "dh_m": [-1.0], "area_m2": [1e6],
        })
        input_path = tmp_path / "input.csv"
        output_path = tmp_path / "glaciers.csv"
        regional_path = tmp_path / "regional.csv"
        observations.to_csv(input_path, index=False)

        with pytest.raises(ValueError, match="lat, lon"):
            main([str(input_path), str(output_path), "--regional-output", str(regional_path)])
        assert not output_path.exists()
        assert not regional_path.exists()

    def test_main__error_identical_output_paths(self, tmp_path):
        """Checks an error is raised before regional results can overwrite the glacier output."""

        output_path = tmp_path / "predictions.csv"
        with pytest.raises(SystemExit):
            main(["input.csv", str(output_path), "--regional-output", str(output_path)])
