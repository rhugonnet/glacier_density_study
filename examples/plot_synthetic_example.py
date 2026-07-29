#!/usr/bin/env python3
"""Plot the README example application of the surrogate model."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from glacier_density_surrogate import RhoSurrogate, make_dh_error_correlation


HERE = Path(__file__).resolve().parent
INPUT_CSV = HERE / "synthetic_elevation_timeseries.csv"
OUTPUT_PNG = HERE / "synthetic_elevation_timeseries_surrogate.png"


def main() -> None:
    """Run the example and write a compact plot."""
    observations = pd.read_csv(INPUT_CSV)
    model = RhoSurrogate()
    predictions = model.predict_timeseries(
        observations,
        dh_error_corr=make_dh_error_correlation("none"),
    )

    annual = predictions[predictions["period_years"].eq(1)].copy()
    annual["mid_year"] = 0.5 * (annual["start"] + annual["end"])

    fig, (ax_dh, ax_rho) = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True)
    ax_dh.axhline(0, color="0.35", lw=0.8)
    ax_dh.bar(
        annual["mid_year"],
        annual["dh_m"],
        width=0.82,
        color="#5DA5DA",
        edgecolor="0.2",
        linewidth=0.4,
    )
    ax_dh.set_ylabel(r"Annual $\Delta h$ (m)")

    ax_rho.axhline(model.rho_ice, color="0.4", lw=0.9, ls="--")
    ax_rho.plot(
        annual["mid_year"],
        annual["mu_rho_independent_kg_m3"],
        color="#4C78A8",
        marker="o",
        label="Without temporal reconciliation",
    )
    ax_rho.plot(
        annual["mid_year"],
        annual["mu_rho_closed_kg_m3"],
        color="#F58518",
        marker="o",
        label="With temporal reconciliation",
    )
    ax_rho.fill_between(
        annual["mid_year"],
        annual["mu_rho_closed_kg_m3"] - annual["sigma_rho_closed_kg_m3"],
        annual["mu_rho_closed_kg_m3"] + annual["sigma_rho_closed_kg_m3"],
        color="#F58518",
        alpha=0.18,
        linewidth=0,
    )
    ax_rho.set_ylabel(r"Effective density (kg m$^{-3}$)")
    ax_rho.set_xlabel("Year")
    ax_rho.legend(frameon=False, loc="best")

    for ax in (ax_dh, ax_rho):
        ax.grid(True, color="0.8", lw=0.5, alpha=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=180)
    print(f"[done] Wrote {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
