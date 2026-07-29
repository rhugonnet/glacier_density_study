#!/usr/bin/env python3
"""
Concatenate full-model effective-density outputs for iteration9, sensmin and sensmax.

For each input scenario, reads the same regional/glacier .dat structure as the
original script, computes all 1- to 20-year period combinations, adds metadata,
and appends a `rho_variant` column identifying the source scenario.
"""

import os
import sys
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd

STUDY_DIR = Path(__file__).resolve().parents[1]
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

from study_paths import PROJECT_DIR

base_dir = str(PROJECT_DIR)

detailed_dir = os.path.join(base_dir, "data", "overview_metadata")
meta_dir = os.path.join(base_dir, "data", "rgi_metadata")

scenario_dirs = {
    "iteration9": os.path.join(
        base_dir,
        "data/rho_dV_final/rho_dV_final/subperiod_grids_iteration9_may2026/iteration9",
    ),
    "sensmin": os.path.join(
        base_dir,
        "data/rho_dV_final/rho_dV_final/subperiod_grids_sensitivity_min/iteration_sensmin9",
    ),
    "sensmax": os.path.join(
        base_dir,
        "data/rho_dV_final/rho_dV_final/subperiod_grids_sensitivity_max/iteration_sensmax9",
    ),
}

region_names = [
    "alaska",
    "westerncanada",
    "arcticcanadaN",
    "arcticcanadaS",
    "greenland",
    "iceland",
    "svalbard",
    "scandinavia",
    "russianarctic",
    "northasia",
    "centraleurope",
    "caucasus",
    "centralasiaN",
    "centralasiaW",
    "centralasiaS",
    "lowlatitudes",
    "southernandes",
    "newzealand",
    "antarctic",
]
region_nbs = np.arange(20)


overwrite_by_propagate_yearly = True

out_csv = os.path.join(
    base_dir,
    "postproc",
    f"rho_dV_may26_final_iteration9_sensmin_sensmax.csv",
)


def read_region_metadata(region_name: str, region_nb: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read area and lon/lat metadata for a region."""
    fn_detailed = os.path.join(
        detailed_dir,
        f"detailed_rgi{str(region_nb).zfill(2)}_{region_name}.dat",
    )
    df_detailed = pd.read_csv(fn_detailed, sep=r"\s+", engine="python", skiprows=1)
    df_detailed["rgiid"] = [
        f"RGI60-{str(region_nb).zfill(2)}.{str(glacier_id).zfill(5)}"
        for glacier_id in df_detailed.ID.values
    ]

    fn_meta = os.path.join(meta_dir, f"thick_{region_name}.dat")
    df_meta = pd.read_csv(
        fn_meta,
        sep=r"\s+",
        usecols=[0, 1, 2],
        names=["ID", "lon", "lat"],
        skiprows=1,
        engine="python",
    )
    df_meta["rgiid"] = [
        f"RGI60-{str(region_nb).zfill(2)}.{str(glacier_id)[:-1].zfill(5)}"
        for glacier_id in df_meta.ID.values
    ]

    return df_detailed, df_meta


def build_period_dataframe(fn: str, region_nb: int) -> pd.DataFrame:
    """Read one glacier .dat file and return all subperiod combinations."""
    df_dat = pd.read_fwf(fn, index_col=0, widths=[4] + [7] * 21)
    bname = os.path.splitext(os.path.basename(fn))[0]

    start_dates, end_dates, rhos, bpers = ([] for _ in range(4))

    for k in np.arange(1, 21):
        sd = np.arange(1999, 2020 - k, 1)
        ed = np.arange(1999 + k, 2020, 1)

        start_dates += list(sd)
        end_dates += list(ed)

        if overwrite_by_propagate_yearly and k != 1:
            yr_b = np.array(bpers[:20]).astype(float)
            yr_rho = np.array(rhos[:20]).astype(float)

            ind_sd = np.arange(0, 21 - k, 1)
            ind_ed = np.arange(k, 21, 1)

            k_years_b = [
                np.sum(yr_b[ind_sd[i] : ind_ed[i]]) / k
                for i in range(len(sd))
            ]
            k_years_rho = [
                np.sum(yr_b[ind_sd[i] : ind_ed[i]])
                / np.sum(yr_b[ind_sd[i] : ind_ed[i]] / yr_rho[ind_sd[i] : ind_ed[i]])
                for i in range(len(sd))
            ]

            bpers += k_years_b
            rhos += k_years_rho
        else:
            bpers += [df_dat.loc[ed[i], str(sd[i])] for i in range(len(sd))]
            rhos += [df_dat.loc[sd[i], str(ed[i])] for i in range(len(sd))]

    return pd.DataFrame(
        data={
            "start_date": start_dates,
            "end_date": end_dates,
            "rho": rhos,
            "b": bpers,
            "rgiid": f"RGI60-{str(region_nb).zfill(2)}.{bname}",
            "region": region_nb,
        }
    )


list_df_all = []

# Loop over input scenarios, then over regions
for rho_variant, in_dir in scenario_dirs.items():
    print(f"Working on rho variant: {rho_variant}")

    for j, region_name in enumerate(region_names):
        region_nb = region_nbs[j] + 1
        print(f"  Working on region: {region_name}")

        region_dir = os.path.join(in_dir, region_name)
        fn_dats = glob(os.path.join(region_dir, "*.dat"))
        fn_dats = [fn for fn in fn_dats if "comp" not in os.path.basename(fn)]

        if len(fn_dats) == 0:
            raise FileNotFoundError(f"No glacier .dat files found in: {region_dir}")

        df_detailed, df_meta = read_region_metadata(region_name, region_nb)

        # Use indexed metadata lookups to avoid repeated dataframe filtering
        area_by_rgiid = df_detailed.set_index("rgiid")["Area(km2)"]
        lat_by_rgiid = df_meta.set_index("rgiid")["lat"]
        lon_by_rgiid = df_meta.set_index("rgiid")["lon"]

        list_df_region = []
        for fn in fn_dats:
            df_out = build_period_dataframe(fn, region_nb)
            rgiid = df_out["rgiid"].iloc[0]

            df_out["area"] = area_by_rgiid.loc[rgiid]
            df_out["lat"] = lat_by_rgiid.loc[rgiid]
            df_out["lon"] = lon_by_rgiid.loc[rgiid]
            df_out["rho_variant"] = rho_variant

            list_df_region.append(df_out)

        list_df_all.append(pd.concat(list_df_region, ignore_index=True))


# Concatenate all regions and all rho variants
df_all = pd.concat(list_df_all, ignore_index=True)

df_all = df_all.sort_values(
    by=["rho_variant", "rgiid", "start_date", "end_date"]
).reset_index(drop=True)

# Convert non-numeric rho entries to NaN, as in the original script but more explicit
df_all["rho"] = pd.to_numeric(df_all["rho"], errors="coerce")

# Keep b numeric as well, without changing valid values
df_all["b"] = pd.to_numeric(df_all["b"], errors="coerce")

os.makedirs(os.path.dirname(out_csv), exist_ok=True)
df_all.to_csv(out_csv, index=False)

print(f"Saved: {out_csv}")
print(f"Rows: {len(df_all):,}")
print(df_all["rho_variant"].value_counts().sort_index())
