#!/usr/bin/env python3
"""Script to generate Table S1 in LaTeX."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parents[1]
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

from study_paths import AGREEMENT_DIAGNOSTICS_DIR, FIGURES_DIR


AGREEMENT_DIR = AGREEMENT_DIAGNOSTICS_DIR
GLACIER_DH_CATEGORY_CSV = AGREEMENT_DIR / "glacier_period_agreement_by_dh_category.csv"
GLACIER_PERIOD_CATEGORY_CSV = AGREEMENT_DIR / "glacier_period_agreement_by_period_category.csv"
REGIONAL_DH_CATEGORY_CSV = AGREEMENT_DIR / "regional_period_agreement_by_dh_category.csv"
REGIONAL_PERIOD_CATEGORY_CSV = AGREEMENT_DIR / "regional_period_agreement_by_period_category.csv"
GLACIER_SUMMARY_CSV = AGREEMENT_DIR / "glacier_period_agreement_summary.csv"
REGIONAL_SUMMARY_CSV = AGREEMENT_DIR / "regional_period_agreement_summary.csv"
OUT_CSV = FIGURES_DIR / "TABLE_supp_01_agreement_categories.csv"
OUT_TEX = FIGURES_DIR / "TABLE_supp_01_agreement_categories.tex"

DH_LABELS = {
    "low_absdh_lt2m": r"$|\Delta h| < 2$ m",
    "mid_absdh_2to10m": r"$2 \le |\Delta h| < 10$ m",
    "high_absdh_ge10m": r"$|\Delta h| \ge 10$ m",
}
PERIOD_LABELS = {
    "short_1to2yr": r"$1 \le \Delta t < 3$ yr",
    "mid_3to7yr": r"$3 \le \Delta t < 8$ yr",
    "long_ge8yr": r"$\Delta t \ge 8$ yr",
}


def ci_weighted_column(table: pd.DataFrame, scale: str) -> str:
    """Select the weighted confidence interval overlap column for one table.

    :param table: Agreement summary table
    :param scale: ``glacier`` or ``regional``
    """
    if scale == "regional" and "ci95_intersection_spatial_weighted_fraction" in table.columns:
        return "ci95_intersection_spatial_weighted_fraction"
    if "ci95_intersection_weighted_fraction" in table.columns:
        return "ci95_intersection_weighted_fraction"
    if "ci95_intersection_independent_weighted_fraction" in table.columns:
        return "ci95_intersection_independent_weighted_fraction"
    raise KeyError(f"No weighted confidence interval overlap column found for {scale} table")


def count_column(table: pd.DataFrame) -> str:
    """Return the count column for one agreement table.

    :param table: Agreement summary table
    """
    for col in ["n_glacier_periods", "n_region_periods"]:
        if col in table.columns:
            return col
    raise KeyError("No agreement count column found")


def read_closed_category_rows(
    path: Path,
    category_col: str,
    labels: dict[str, str],
    group_label: str,
    scale: str,
    scale_label: str,
) -> pd.DataFrame:
    """Read one category summary and return normalized table rows."""
    if not path.exists():
        raise FileNotFoundError(
            "Missing agreement category output. Run "
            "scripts_study_2026/analysis/surrogate_full_model_agreement.py first. "
            f"Missing: {path}"
        )
    table = pd.read_csv(path)
    table = table.loc[table["mode"].astype(str).eq("closed")].copy()
    ci_col = ci_weighted_column(table, scale)
    n_col = count_column(table)
    rows = []
    for order, (category, label) in enumerate(labels.items()):
        match = table.loc[table[category_col].astype(str).eq(category)]
        if match.empty:
            continue
        row = match.iloc[0]
        rows.append(
            {
                "group_order": 0 if category_col == "dh_category" else 1,
                "class_order": order,
                "scale_order": 0 if scale == "glacier" else 1,
                "group": group_label,
                "class_label": label,
                "scale": scale_label,
                "n": int(row[n_col]),
                "variance_explained_percent": 100.0 * float(row["weighted_r2"]),
                "bias_pred_minus_full_kg_m3": float(row["weighted_bias"]),
                "ci95_hits_weighted_percent": 100.0 * float(row[ci_col]),
            }
        )
    return pd.DataFrame(rows)


def read_closed_all_row(path: Path, scale: str, scale_label: str) -> pd.DataFrame:
    """Read one closed-mode all-sample summary row.

    :param path: Overall agreement summary CSV
    :param scale: Stable scale key
    :param scale_label: Human-readable scale label
    """
    if not path.exists():
        raise FileNotFoundError(
            "Missing agreement summary output. Run "
            "scripts_study_2026/analysis/surrogate_full_model_agreement.py first. "
            f"Missing: {path}"
        )
    table = pd.read_csv(path)
    table = table.loc[table["mode"].astype(str).eq("closed")].copy()
    if table.empty:
        return pd.DataFrame()
    ci_col = ci_weighted_column(table, scale)
    n_col = count_column(table)
    row = table.iloc[0]
    return pd.DataFrame(
        [
            {
                "group_order": 2,
                "class_order": 0,
                "scale_order": 0 if scale == "glacier" else 1,
                "group": "All",
                "class_label": "All",
                "scale": scale_label,
                "n": int(row[n_col]),
                "variance_explained_percent": 100.0 * float(row["weighted_r2"]),
                "bias_pred_minus_full_kg_m3": float(row["weighted_bias"]),
                "ci95_hits_weighted_percent": 100.0 * float(row[ci_col]),
            }
        ]
    )


def build_table() -> pd.DataFrame:
    """Build the closed-mode category agreement table."""
    parts = [
        read_closed_category_rows(
            GLACIER_DH_CATEGORY_CSV,
            "dh_category",
            DH_LABELS,
            "Elevation change",
            "glacier",
            "Glacier",
        ),
        read_closed_category_rows(
            REGIONAL_DH_CATEGORY_CSV,
            "dh_category",
            DH_LABELS,
            "Elevation change",
            "regional",
            "Regional",
        ),
        read_closed_category_rows(
            GLACIER_PERIOD_CATEGORY_CSV,
            "period_category",
            PERIOD_LABELS,
            "Period length",
            "glacier",
            "Glacier",
        ),
        read_closed_category_rows(
            REGIONAL_PERIOD_CATEGORY_CSV,
            "period_category",
            PERIOD_LABELS,
            "Period length",
            "regional",
            "Regional",
        ),
        read_closed_all_row(GLACIER_SUMMARY_CSV, "glacier", "Glacier"),
        read_closed_all_row(REGIONAL_SUMMARY_CSV, "regional", "Regional"),
    ]
    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["group_order", "class_order", "scale_order"], kind="stable")
    return out.reset_index(drop=True)


def format_value(value: float, fmt: str) -> str:
    """Format finite numeric values for LaTeX."""
    if not np.isfinite(float(value)):
        return "--"
    return fmt.format(float(value))


def write_latex(table: pd.DataFrame, path: Path) -> None:
    """Write the category agreement table as LaTeX.

    :param table: Normalized category table
    :param path: Output ``.tex`` path
    """
    lines = [
        r"\begin{table}",
        r"\centering",
        r"\caption{Agreement between surrogate and full model effective density estimates by elevation change ($|\Delta h|$) and period length ($\Delta t$) classes. Variance and bias are volume change weighted; CI hits are the percentage of 95\% confidence intervals intersecting the full model value. Note that shorter periods and smaller elevation change both have larger biases, but that those remain small relative to associated uncertainties.}",
        r"\label{tab:closed-agreement-categories}",
        r"\begin{tabular}{lllrrrr}",
        r"\hline",
        r"Grouping variable & Class & Scale & $N$ & Variance explained (\%) & Bias (kg m$^{-3}$) & CI hits (\%) \\",
        r"\hline",
    ]
    last_group = None
    for row in table.itertuples(index=False):
        if last_group is not None and row.group != last_group:
            lines.append(r"\hline")
        last_group = row.group
        lines.append(
            " & ".join(
                [
                    str(row.group),
                    str(row.class_label),
                    str(row.scale),
                    f"{int(row.n):,}",
                    format_value(row.variance_explained_percent, "{:.0f}"),
                    format_value(row.bias_pred_minus_full_kg_m3, "{:.1f}"),
                    format_value(row.ci95_hits_weighted_percent, "{:.0f}"),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def run() -> dict[str, Path]:
    """Generate category agreement CSV and LaTeX outputs."""
    AGREEMENT_DIR.mkdir(parents=True, exist_ok=True)
    table = build_table()
    table.drop(columns=["group_order", "class_order", "scale_order"]).to_csv(OUT_CSV, index=False)
    write_latex(table, OUT_TEX)
    return {"csv": OUT_CSV, "latex": OUT_TEX}


if __name__ == "__main__":
    for name, path in run().items():
        print(f"[done] {name}: {path}")
