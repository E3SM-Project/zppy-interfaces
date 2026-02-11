"""Atmosphere log parser — extracts energy/water conservation diagnostics.

ATM log per-step diagnostic format:

 nstep, te        1   TE_before   TE_after   fixer_frac   ps
 n, dt, W tot mass [kg/m2]        1   3600.0   25.303...
 n, W flux, dWater [kg/m2]        1   flux_val   dwater_val
 n, W flux-dWater [kg/m2]         1   residual
 n, W cflx*dt loss [kg/m2]        1   loss
 n, E d(TE)/dt, RR [W/m2]         1   dTE_dt   RR
 n, E difference [W/m2]           1   diff
 n, E shf loss [W/m2]             1   loss

Date info from ``chem_surfvals_set: ncdate=YYYMMDD`` (once per day).

Note: ``nstep, te`` runs from 0..N+1 while flux diagnostics run n=1..N
(the atm receives and applies fluxes in two consecutive steps).
"""

import gzip
from typing import Dict, List, Tuple

import pandas as pd

from ..schema import (
    COL_COMPONENT,
    COL_PERIOD,
    COL_QUANTITY,
    COL_SOURCE,
    COL_TABLE_TYPE,
    COL_TERM,
    COL_TIME,
    COL_UNITS,
    COL_VALUE,
    COLUMNS,
)
from .base import BaseParser

# Line prefixes for each diagnostic type
_L_NSTEP_TE = "nstep, te "
_L_NCDATE = "chem_surfvals_set: ncdate="
_L_W_TOT = "n, dt, W tot mass [kg/m2]"
_L_W_FLUX = "n, W flux, dWater [kg/m2]"
_L_W_RESID = "n, W flux-dWater [kg/m2]"
_L_W_CFLX = "n, W cflx*dt loss [kg/m2]"
_L_E_DTEDT = "n, E d(TE)/dt, RR [W/m2]"
_L_E_DIFF = "n, E difference [W/m2]"
_L_E_SHF = "n, E shf loss [W/m2]"


def _parse_ncdate(ncdate_int: int) -> Tuple[int, int, int]:
    """Parse ncdate integer YYYMMDD -> (year, month, day).

    Examples: 10101 -> (1, 1, 1), 501231 -> (50, 12, 31).
    """
    day = ncdate_int % 100
    ncdate_int //= 100
    month = ncdate_int % 100
    year = ncdate_int // 100
    return year, month, day


def _open_log(filename: str):
    """Open a log file, handling gzip compression."""
    if filename.endswith(".gz"):
        return gzip.open(filename, "rt")
    return open(filename, "r")


def _gather_energy_data(filename: str) -> Tuple[List[Dict], List[Dict]]:
    """Extract per-step diagnostic data from a single atm log file.

    Returns (nstep_te_rows, flux_diag_rows).
    """
    nstep_te_rows: List[Dict] = []
    flux_diag_rows: List[Dict] = []

    # Current date context (from ncdate lines)
    cur_year, cur_month, cur_day = -1, -1, -1

    # Accumulate flux diagnostic fields for the current step
    cur_flux: Dict = {}

    with _open_log(filename) as f:
        for line in f:
            # --- Date tracking ---
            if _L_NCDATE in line:
                # chem_surfvals_set: ncdate=        10101  co2vmr=...
                parts = line.split("ncdate=")[1].split()
                cur_year, cur_month, cur_day = _parse_ncdate(int(parts[0]))
                continue

            # --- Energy fixer (nstep, te) ---
            if _L_NSTEP_TE in line:
                tokens = line.split()
                # tokens: ['nstep,', 'te', N, te_before, te_after, fixer_frac, ps]
                nstep_te_rows.append(
                    {
                        "nstep": int(tokens[2]),
                        "te_before": float(tokens[3]),
                        "te_after": float(tokens[4]),
                        "fixer_frac": float(tokens[5]),
                        "ps": float(tokens[6]),
                        "year": cur_year,
                        "month": cur_month,
                        "day": cur_day,
                    }
                )
                continue

            # --- Water total mass (starts a new flux diagnostic group) ---
            if _L_W_TOT in line:
                # Flush previous step if any
                if cur_flux:
                    flux_diag_rows.append(cur_flux)
                tokens = line.split()
                cur_flux = {
                    "nstep": int(tokens[6]),
                    "dt": float(tokens[7]),
                    "tw": float(tokens[8]),
                    "year": cur_year,
                    "month": cur_month,
                    "day": cur_day,
                }
                continue

            # --- Water flux, dWater ---
            if _L_W_FLUX in line:
                tokens = line.split()
                cur_flux["w_flux"] = float(tokens[6])
                cur_flux["w_dwater"] = float(tokens[7])
                continue

            # --- Water residual ---
            if _L_W_RESID in line:
                tokens = line.split()
                cur_flux["w_residual"] = float(tokens[5])
                continue

            # --- Water coupling flux loss ---
            if _L_W_CFLX in line:
                tokens = line.split()
                cur_flux["w_cflx_loss"] = float(tokens[6])
                continue

            # --- Energy d(TE)/dt and RR ---
            if _L_E_DTEDT in line:
                tokens = line.split()
                cur_flux["e_dtedt"] = float(tokens[6])
                cur_flux["e_rr"] = float(tokens[7])
                continue

            # --- Energy difference ---
            if _L_E_DIFF in line:
                tokens = line.split()
                cur_flux["e_diff"] = float(tokens[5])
                continue

            # --- Energy shf loss ---
            if _L_E_SHF in line:
                tokens = line.split()
                cur_flux["e_shf_loss"] = float(tokens[6])
                continue

    # Flush last step
    if cur_flux:
        flux_diag_rows.append(cur_flux)

    return nstep_te_rows, flux_diag_rows


class AtmParser(BaseParser):
    """Parse atmosphere log files for energy/water conservation diagnostics."""

    def parse_raw(self, log_files: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Parse raw per-step data from atm log files.

        Returns
        -------
        nstep_te_df : pd.DataFrame
            Energy fixer data per step.
            Columns: nstep, te_before, te_after, fixer_frac, ps,
                     year, month, day.
        flux_diag_df : pd.DataFrame
            Flux diagnostics per step.
            Columns: nstep, dt, tw, w_flux, w_dwater, w_residual,
                     w_cflx_loss, e_dtedt, e_rr, e_diff, e_shf_loss,
                     year, month, day.
        """
        all_nstep: List[Dict] = []
        all_flux: List[Dict] = []

        for fname in sorted(log_files):
            try:
                nstep_rows, flux_rows = _gather_energy_data(fname)
                all_nstep.extend(nstep_rows)
                all_flux.extend(flux_rows)
            except Exception as e:
                print(f"WARNING: Error processing {fname}: {e}")
                continue

        nstep_te_df = pd.DataFrame(all_nstep)
        flux_diag_df = pd.DataFrame(all_flux)
        return nstep_te_df, flux_diag_df

    def parse_files(
        self, log_files: List[str], start_year: int, end_year: int
    ) -> pd.DataFrame:
        """Parse atm logs and return a tidy monthly event table.

        Aggregates per-step data to monthly averages for fluxes and
        beginning/end-of-month values for states.
        """
        nstep_te_df, flux_diag_df = self.parse_raw(log_files)

        if nstep_te_df.empty and flux_diag_df.empty:
            return pd.DataFrame(columns=COLUMNS)

        rows: List[Dict] = []

        # --- Flux diagnostics (monthly aggregation) ---
        if not flux_diag_df.empty:
            flux_df = flux_diag_df[
                (flux_diag_df["year"] >= start_year)
                & (flux_diag_df["year"] <= end_year)
            ]
            if not flux_df.empty:
                for (year, month), grp in flux_df.groupby(["year", "month"]):
                    base_water = {
                        COL_TIME: year,
                        COL_COMPONENT: "atm",
                        COL_QUANTITY: "water",
                        COL_SOURCE: "atm",
                        COL_PERIOD: "monthly",
                    }
                    base_heat = {
                        COL_TIME: year,
                        COL_COMPONENT: "atm",
                        COL_QUANTITY: "heat",
                        COL_SOURCE: "atm",
                        COL_PERIOD: "monthly",
                    }

                    dt = grp["dt"].iloc[0]

                    # Water flux terms (monthly mean rate, kg/m2/s)
                    if "w_flux" in grp.columns:
                        rows.append(
                            {
                                **base_water,
                                COL_TERM: "W flux",
                                COL_VALUE: (grp["w_flux"] / dt).mean(),
                                COL_UNITS: "kg/m2/s",
                                COL_TABLE_TYPE: "flux",
                            }
                        )
                    if "w_dwater" in grp.columns:
                        rows.append(
                            {
                                **base_water,
                                COL_TERM: "dWater",
                                COL_VALUE: (grp["w_dwater"] / dt).mean(),
                                COL_UNITS: "kg/m2/s",
                                COL_TABLE_TYPE: "flux",
                            }
                        )

                    # Water state (begin/end of month)
                    rows.append(
                        {
                            **base_water,
                            COL_TERM: "W tot mass beg",
                            COL_VALUE: grp["tw"].iloc[0],
                            COL_UNITS: "kg/m2",
                            COL_TABLE_TYPE: "state",
                        }
                    )
                    rows.append(
                        {
                            **base_water,
                            COL_TERM: "W tot mass end",
                            COL_VALUE: grp["tw"].iloc[-1],
                            COL_UNITS: "kg/m2",
                            COL_TABLE_TYPE: "state",
                        }
                    )

                    # Water diagnostics (monthly mean per step)
                    if "w_residual" in grp.columns:
                        rows.append(
                            {
                                **base_water,
                                COL_TERM: "W residual",
                                COL_VALUE: grp["w_residual"].mean(),
                                COL_UNITS: "kg/m2",
                                COL_TABLE_TYPE: "diagnostic",
                            }
                        )
                    if "w_cflx_loss" in grp.columns:
                        rows.append(
                            {
                                **base_water,
                                COL_TERM: "W cflx loss",
                                COL_VALUE: grp["w_cflx_loss"].mean(),
                                COL_UNITS: "kg/m2",
                                COL_TABLE_TYPE: "diagnostic",
                            }
                        )

                    # Energy flux terms (monthly mean, W/m2)
                    if "e_dtedt" in grp.columns:
                        rows.append(
                            {
                                **base_heat,
                                COL_TERM: "E d(TE)/dt",
                                COL_VALUE: grp["e_dtedt"].mean(),
                                COL_UNITS: "W/m2",
                                COL_TABLE_TYPE: "flux",
                            }
                        )
                    if "e_rr" in grp.columns:
                        rows.append(
                            {
                                **base_heat,
                                COL_TERM: "E RR",
                                COL_VALUE: grp["e_rr"].mean(),
                                COL_UNITS: "W/m2",
                                COL_TABLE_TYPE: "flux",
                            }
                        )

                    # Energy diagnostics (monthly mean, W/m2)
                    if "e_diff" in grp.columns:
                        rows.append(
                            {
                                **base_heat,
                                COL_TERM: "E difference",
                                COL_VALUE: grp["e_diff"].mean(),
                                COL_UNITS: "W/m2",
                                COL_TABLE_TYPE: "diagnostic",
                            }
                        )
                    if "e_shf_loss" in grp.columns:
                        rows.append(
                            {
                                **base_heat,
                                COL_TERM: "E shf loss",
                                COL_VALUE: grp["e_shf_loss"].mean(),
                                COL_UNITS: "W/m2",
                                COL_TABLE_TYPE: "diagnostic",
                            }
                        )

        # --- Energy fixer state (TE begin/end of month) ---
        if not nstep_te_df.empty:
            te_df = nstep_te_df[
                (nstep_te_df["year"] >= start_year) & (nstep_te_df["year"] <= end_year)
            ]
            if not te_df.empty:
                for (year, month), grp in te_df.groupby(["year", "month"]):
                    base_heat = {
                        COL_TIME: year,
                        COL_COMPONENT: "atm",
                        COL_QUANTITY: "heat",
                        COL_SOURCE: "atm",
                        COL_PERIOD: "monthly",
                    }
                    rows.append(
                        {
                            **base_heat,
                            COL_TERM: "TE beg",
                            COL_VALUE: grp["te_before"].iloc[0],
                            COL_UNITS: "J/m2",
                            COL_TABLE_TYPE: "state",
                        }
                    )
                    rows.append(
                        {
                            **base_heat,
                            COL_TERM: "TE end",
                            COL_VALUE: grp["te_after"].iloc[-1],
                            COL_UNITS: "J/m2",
                            COL_TABLE_TYPE: "state",
                        }
                    )
                    rows.append(
                        {
                            **base_heat,
                            COL_TERM: "E fixer frac",
                            COL_VALUE: grp["fixer_frac"].mean(),
                            COL_UNITS: "1",
                            COL_TABLE_TYPE: "diagnostic",
                        }
                    )

        if not rows:
            return pd.DataFrame(columns=COLUMNS)
        return pd.DataFrame(rows, columns=COLUMNS)
