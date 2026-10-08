#!/usr/bin/env python3
"""Command-line tool to apply the effective density surrogate to direct observations or CSV files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from .surrogate import RhoSurrogate, make_dh_error_correlation


def build_parser() -> argparse.ArgumentParser:
    """
    Define the CSV and single-period command-line options.

    :returns: Configured argument parser.
    """

    parser = argparse.ArgumentParser(
        prog="glacier-density-surrogate",
        description="Estimate effective density from glacier elevation change observations.",
    )
    parser.add_argument("input", nargs="?", help="Input CSV with period rows")
    parser.add_argument("output", nargs="?", help="Output CSV for surrogate predictions")
    parser.add_argument("--id-col", default=None, help="Glacier identifier column; detects glacier_id or rgiid")
    parser.add_argument("--regional-output", default=None, help="Optional CSV for regional mass change and uncertainty")
    parser.add_argument("--dh", type=float, help="Elevation change over the observation period in metres")
    parser.add_argument("--sigma-dh", type=float, default=0.0, help="Elevation change uncertainty in metres")
    parser.add_argument("--dt", type=float, help="Observation period duration in years; required with --dh")
    parser.add_argument("--past-dhdt", type=float, default=None, help="Past elevation change rate in m yr-1")
    parser.add_argument("--sigma-past-dhdt", type=float, default=None, help="Past elevation change rate uncertainty in m yr-1")
    parser.add_argument("--area-m2", type=float, default=None, help="Constant glacier area in square metres")
    parser.add_argument("--start-col", default=None, help="Input column for the observation start year")
    parser.add_argument("--end-col", default=None, help="Input column for the observation end year")
    parser.add_argument("--dt-col", default=None, help="Input period duration column")
    parser.add_argument("--dh-col", default=None, help="Input elevation change column")
    parser.add_argument("--sigma-dh-col", default=None, help="Input elevation change uncertainty column")
    parser.add_argument("--past-dhdt-col", default=None, help="Input past elevation change rate column")
    parser.add_argument("--sigma-past-dhdt-col", default=None, help="Input past elevation change rate uncertainty column")
    parser.add_argument("--area-col", default=None, help="Input area column")
    parser.add_argument(
        "--past-missing",
        choices=["current", "zero"],
        default="current",
        help="Assumption when past elevation change rate is unavailable",
    )
    parser.add_argument(
        "--past-error-factor",
        type=float,
        default=2.0,
        help="Multiplier for sigma_dh / dt when past elevation change rate uncertainty is missing",
    )
    parser.add_argument(
        "--dh-error-corr-form",
        choices=["none", "exponential", "gaussian", "spherical"],
        default="none",
        help="Temporal correlation form for elevation change errors",
    )
    parser.add_argument("--dh-error-corr-range", type=float, default=None, help="Correlation range in years")
    parser.add_argument(
        "--no-expand-periods",
        action="store_true",
        help="Do not expand consecutive rows to all contiguous periods",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    Print a single prediction as JSON or write predictions from a CSV file.

    :param argv: Command-line arguments; None reads the process arguments.
    :returns: Zero after a successful prediction.
    """

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.dh is not None and args.dt is None:
        parser.error("--dt is required with --dh")
    if args.regional_output is not None:
        if args.dh is not None:
            parser.error("--regional-output requires glacier observations in a CSV")
        if args.output is not None and Path(args.output).resolve() == Path(args.regional_output).resolve():
            parser.error("Glacier and regional outputs must use different paths")
    model = RhoSurrogate()

    # Predict one period when direct observations are supplied
    if args.dh is not None:
        result = model.predict(
            dh=args.dh,
            sigma_dh=args.sigma_dh,
            dt=args.dt,
            past_dhdt=args.past_dhdt,
            sigma_past_dhdt=args.sigma_past_dhdt,
            area_m2=args.area_m2,
            past_missing=args.past_missing,
            past_error_factor=args.past_error_factor,
        )
        # Undefined density at zero volume change is null in portable JSON; mass change remains finite
        json_result = {
            key: None if isinstance(value, float) and not math.isfinite(value) else value
            for key, value in result.items()
        }
        print(json.dumps(json_result, indent=2, sort_keys=True, allow_nan=False))
        return 0

    # Run a CSV time series prediction otherwise
    if args.input is None or args.output is None:
        raise SystemExit("Provide input/output CSV paths, or use --dh and --dt to predict one period")
    observations = pd.read_csv(args.input)
    corr = make_dh_error_correlation(args.dh_error_corr_form, args.dh_error_corr_range)
    predictions = model.predict_timeseries(
        observations,
        id_col=args.id_col,
        start_col=args.start_col,
        end_col=args.end_col,
        dh_col=args.dh_col,
        sigma_dh_col=args.sigma_dh_col,
        dt_col=args.dt_col,
        past_dhdt_col=args.past_dhdt_col,
        sigma_past_dhdt_col=args.sigma_past_dhdt_col,
        area_col=args.area_col,
        area_m2=args.area_m2,
        expand_periods=not args.no_expand_periods,
        past_missing=args.past_missing,
        past_error_factor=args.past_error_factor,
        dh_error_corr=corr,
    )

    # Validate regional coverage and propagate uncertainty before writing either result
    regional = None
    if args.regional_output is not None:
        regional = model.aggregate_regions(predictions, id_col=args.id_col)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, index=False)
    print(f"Wrote surrogate predictions to {output}")
    if regional is not None:
        regional_output = Path(args.regional_output)
        regional_output.parent.mkdir(parents=True, exist_ok=True)
        regional.to_csv(regional_output, index=False)
        print(f"Wrote regional predictions to {regional_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
