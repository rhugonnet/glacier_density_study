#!/usr/bin/env python3
"""Command-line interface for the effective-density surrogate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .surrogate import RhoSurrogate, make_dh_error_correlation


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser

    :returns: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        prog="glacier-density-surrogate",
        description="Estimate effective density from glacier elevation-change observations.",
    )
    parser.add_argument("input", nargs="?", help="Input CSV with period rows")
    parser.add_argument("output", nargs="?", help="Output CSV for surrogate predictions")
    parser.add_argument("--dh", type=float, help="Single-period elevation change in metres")
    parser.add_argument("--sigma-dh", type=float, default=0.0, help="Single-period elevation-change uncertainty in metres")
    parser.add_argument("--dt", type=float, default=1.0, help="Single-period duration in years")
    parser.add_argument("--past-dh", type=float, default=None, help="Single-period past elevation change in metres")
    parser.add_argument("--sigma-past-dh", type=float, default=None, help="Single-period past elevation-change uncertainty in metres")
    parser.add_argument("--area-m2", type=float, default=None, help="Constant glacier area in square metres")
    parser.add_argument("--start-col", default=None, help="Input start-year column")
    parser.add_argument("--end-col", default=None, help="Input end-year column")
    parser.add_argument("--dt-col", default=None, help="Input period-length column")
    parser.add_argument("--dh-col", default=None, help="Input elevation-change column")
    parser.add_argument("--sigma-dh-col", default=None, help="Input elevation-change uncertainty column")
    parser.add_argument("--past-dh-col", default=None, help="Input past elevation-change column")
    parser.add_argument("--sigma-past-dh-col", default=None, help="Input past elevation-change uncertainty column")
    parser.add_argument("--area-col", default=None, help="Input area column")
    parser.add_argument(
        "--past-missing",
        choices=["current", "zero"],
        default="current",
        help="Assumption when past elevation change is unavailable",
    )
    parser.add_argument(
        "--past-error-factor",
        type=float,
        default=2.0,
        help="Multiplier applied to sigma_dh when defaulting sigma_past_dh",
    )
    parser.add_argument(
        "--dh-error-corr-form",
        choices=["none", "exponential", "gaussian", "spherical"],
        default="none",
        help="Temporal correlation form for elevation-change errors",
    )
    parser.add_argument("--dh-error-corr-range", type=float, default=None, help="Correlation range in years")
    parser.add_argument(
        "--no-expand-periods",
        action="store_true",
        help="Do not expand consecutive rows to all contiguous periods",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI

    :param argv: Optional argument vector
    """
    args = build_parser().parse_args(argv)
    model = RhoSurrogate()

    # Run a single-period prediction when direct values are supplied
    if args.dh is not None:
        result = model.predict(
            dh=args.dh,
            sigma_dh=args.sigma_dh,
            dt=args.dt,
            past_dh=args.past_dh,
            sigma_past_dh=args.sigma_past_dh,
            area_m2=args.area_m2,
            past_missing=args.past_missing,
            past_error_factor=args.past_error_factor,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    # Run a CSV time-series prediction otherwise
    if args.input is None or args.output is None:
        raise SystemExit("Provide input/output CSV paths, or use --dh for a single-period prediction")
    df = pd.read_csv(args.input)
    corr = make_dh_error_correlation(args.dh_error_corr_form, args.dh_error_corr_range)
    out = model.predict_timeseries(
        df,
        start_col=args.start_col,
        end_col=args.end_col,
        dh_col=args.dh_col,
        sigma_dh_col=args.sigma_dh_col,
        dt_col=args.dt_col,
        past_dh_col=args.past_dh_col,
        sigma_past_dh_col=args.sigma_past_dh_col,
        area_col=args.area_col,
        area_m2=args.area_m2,
        expand_periods=not args.no_expand_periods,
        past_missing=args.past_missing,
        past_error_factor=args.past_error_factor,
        dh_error_corr=corr,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(f"[done] Wrote surrogate predictions: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
