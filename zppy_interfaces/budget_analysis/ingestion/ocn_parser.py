"""Ocean log parser — extracts mass conservation checks into tidy DataFrames.

Ocean log format (monthly, inside CONSERVATION CHECKS blocks):

 date: 0001-02-01_00:00:00
 Conversion factors:
 Earth area in E3SM:          A =   5.1010114020779156E+14 m^2
 Averaging time interval:    dt = 2676600.0000 s,      30.9792 days
 ...
 MASS CONSERVATION CHECK

 MASS FLUXES
 MPAS-Ocean name          kg/s (F)       coupler name     short name     kg/m^2/s*1e6 (F/A)
 frazilFreshwaterFlux    -1.28194649E+08 o2x_Fioo_frazil  wfreeze             -0.25131222
 ...
 SUM VOLUME FLUXES       -7.63382509E+07                                      -0.14965317

 CHANGE IN MASS: computed from ocean domain
                        kg
 Initial mass           1.36527973E+21
 Final mass             1.36527952E+21
 Mass change           -2.04326962E+14

 MASS CONSERVATION SUMMARY
                        kg              kg/s            kg/m^2/s*1e6
 Mass change           ...
 Net mass flux         ...
 Absolute mass error   ...

 RELATIVE MASS ERROR = ...
"""

import gzip
import re
from typing import Dict, List, Optional, TextIO, Tuple

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


def _parse_date(date_str: str) -> Tuple[int, int]:
    """Parse 'YYYY-MM-DD_HH:MM:SS' -> (year, month).

    The date is printed at the START of the next month,
    so date 0002-01-01 means the check covers Dec of year 1.
    We return the year and month of the COVERED period.
    """
    match = re.match(r"(\d+)-(\d+)-(\d+)", date_str.strip())
    if not match:
        return -1, -1
    y, m = int(match.group(1)), int(match.group(2))
    # Roll back one month
    if m == 1:
        return y - 1, 12
    return y, m - 1


def _parse_mass_fluxes(f: TextIO) -> List[Tuple[str, float]]:
    """Parse MASS FLUXES table, return list of (short_name, kg/m2s*1e6 value)."""
    fluxes: List[Tuple[str, float]] = []

    # Skip blank line + "MASS FLUXES" header + column header line
    line = f.readline()  # blank
    line = f.readline()  # "MASS FLUXES"
    line = f.readline()  # column headers

    line = f.readline()
    while line and line.strip():
        parts = line.split()
        if not parts:
            break
        # Last token is the kg/m^2/s*1e6 value, second-to-last is short_name
        # Format: MPAS_name  kg/s_val  coupler_name  short_name  kg/m2s*1e6_val
        # OR: SUM VOLUME FLUXES  kg/s_val  [empty coupler/short]  kg/m2s*1e6_val
        try:
            val = float(parts[-1])
        except ValueError:
            line = f.readline()
            continue

        # Determine term name
        if "SUM VOLUME FLUXES" in line:
            term = "*SUM*"
        else:
            # short_name is the second-to-last token
            term = parts[-2]

        fluxes.append((term, val))
        line = f.readline()

    return fluxes


def _parse_mass_change(f: TextIO) -> Optional[Dict[str, float]]:
    """Parse CHANGE IN MASS block, return dict of values in kg."""
    result: Dict[str, float] = {}
    # Expect lines like:
    #  Initial mass           1.36527973E+21
    #  Final mass             1.36527952E+21
    #  Mass change           -2.04326962E+14
    for _ in range(4):  # header line + 3 data lines
        line = f.readline()
        if "Initial mass" in line:
            result["initial_mass_kg"] = float(line.split()[-1])
        elif "Final mass" in line:
            result["final_mass_kg"] = float(line.split()[-1])
        elif "Mass change" in line:
            result["mass_change_kg"] = float(line.split()[-1])
    return result if result else None


def _parse_mass_summary(f: TextIO) -> Optional[Dict[str, float]]:
    """Parse MASS CONSERVATION SUMMARY, return kg/m2s*1e6 values.

    Format (immediately after the MASS CONSERVATION SUMMARY header line):
                            kg              kg/s            kg/m^2/s*1e6
     Mass change           ...             ...             -0.14965317
     Net mass flux         ...             ...             -0.14965317
     Absolute mass error   ...             ...              0.00000000
    """
    result: Dict[str, float] = {}
    line = f.readline()  # column headers: kg  kg/s  kg/m^2/s*1e6
    line = f.readline()  # Mass change
    if "Mass change" in line:
        result["mass_change"] = float(line.split()[-1])
    line = f.readline()  # Net mass flux
    if "Net mass flux" in line:
        result["net_mass_flux"] = float(line.split()[-1])
    line = f.readline()  # Absolute mass error
    if "Absolute mass error" in line:
        result["absolute_mass_error"] = float(line.split()[-1])
    return result if result else None


class OcnParser(BaseParser):
    """Parse ocean log files for mass conservation checks."""

    def parse_files(
        self, log_files: List[str], start_year: int, end_year: int
    ) -> pd.DataFrame:
        rows: List[Dict] = []
        for fname in sorted(log_files):
            try:
                with gzip.open(fname, "rt") as f:
                    for line in f:
                        if "CONSERVATION CHECKS" in line and "date:" not in line:
                            # Next line has the date
                            date_line = f.readline()
                            if "date:" not in date_line:
                                continue
                            date_str = date_line.split("date:")[1].strip()
                            year, month = _parse_date(date_str)
                            if year < start_year or year > end_year:
                                continue
                            rows.extend(self._parse_block(f, year, month))
            except Exception as e:
                print(f"WARNING: Error processing {fname}: {e}")
                continue

        if not rows:
            return pd.DataFrame(columns=COLUMNS)
        return pd.DataFrame(rows, columns=COLUMNS)

    def _parse_block(self, f: TextIO, year: int, month: int) -> List[Dict]:
        """Parse one CONSERVATION CHECKS block for mass data."""
        rows: List[Dict] = []

        # Scan for MASS CONSERVATION CHECK within this block
        line = f.readline()
        while line:
            if "MASS CONSERVATION CHECK" in line and "SUMMARY" not in line:
                # Parse flux table
                fluxes = _parse_mass_fluxes(f)
                for term, val in fluxes:
                    rows.append(
                        {
                            COL_TIME: year,
                            COL_COMPONENT: "ocn",
                            COL_QUANTITY: "water",
                            COL_TERM: term,
                            COL_VALUE: val,
                            COL_UNITS: "kg/m2s*1e6",
                            COL_SOURCE: "ocn",
                            COL_PERIOD: "monthly",
                            COL_TABLE_TYPE: "flux",
                        }
                    )

                # Continue reading for SUMMARY (has mass change in kg/m2s*1e6)
                line = f.readline()
                while line:
                    if "MASS CONSERVATION SUMMARY" in line:
                        summary = _parse_mass_summary(f)
                        if summary:
                            if "mass_change" in summary:
                                rows.append(
                                    {
                                        COL_TIME: year,
                                        COL_COMPONENT: "ocn",
                                        COL_QUANTITY: "water",
                                        COL_TERM: "mass_change",
                                        COL_VALUE: summary["mass_change"],
                                        COL_UNITS: "kg/m2s*1e6",
                                        COL_SOURCE: "ocn",
                                        COL_PERIOD: "monthly",
                                        COL_TABLE_TYPE: "flux",
                                    }
                                )
                            if "absolute_mass_error" in summary:
                                rows.append(
                                    {
                                        COL_TIME: year,
                                        COL_COMPONENT: "ocn",
                                        COL_QUANTITY: "water",
                                        COL_TERM: "absolute_mass_error",
                                        COL_VALUE: summary["absolute_mass_error"],
                                        COL_UNITS: "kg/m2s*1e6",
                                        COL_SOURCE: "ocn",
                                        COL_PERIOD: "monthly",
                                        COL_TABLE_TYPE: "diagnostic",
                                    }
                                )
                        break  # Done with this block
                    elif "SALT CONSERVATION" in line:
                        break  # Past mass section
                    line = f.readline()
                break  # Only one MASS CONSERVATION CHECK per block
            elif "===" in line or line.strip() == "":
                pass
            line = f.readline()

        return rows
