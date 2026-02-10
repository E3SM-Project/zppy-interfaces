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
            term = "SUM VOLUME FLUXES"
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


def _parse_heat_fluxes(f: TextIO) -> List[Tuple[str, float]]:
    """Parse HEAT FLUXES sections (explicit + implicit), return (term, W/m2) pairs.

    Reads through explicit and implicit heat flux tables until
    'SUM IMP+EXP  HEAT FLUXES' is found.
    """
    fluxes: List[Tuple[str, float]] = []

    line = f.readline()
    while line:
        stripped = line.strip()
        if not stripped:
            line = f.readline()
            continue

        # Stop after we've captured the combined sum
        if "SUM IMP+EXP" in line and "HEAT FLUXES" in line:
            parts = line.split()
            try:
                val = float(parts[-1])
                fluxes.append(("SUM IMP+EXP HEAT FLUXES", val))
            except ValueError:
                pass
            break

        # Skip header/label lines
        if "HEAT FLUXES" in line or "MPAS-Ocean name" in line:
            line = f.readline()
            continue

        # Parse explicit/implicit individual flux rows and SUM lines
        parts = line.split()
        if not parts:
            line = f.readline()
            continue

        try:
            val = float(parts[-1])
        except ValueError:
            line = f.readline()
            continue

        if "SUM EXPLICIT" in line:
            term = "SUM EXPLICIT HEAT FLUXES"
        elif "SUM IMPLICIT" in line:
            term = "SUM IMPLICIT HEAT FLUXES"
        else:
            # Use short_name (second-to-last) if available, else MPAS name
            term = parts[-2] if len(parts) >= 3 else parts[0]

        fluxes.append((term, val))
        line = f.readline()

    return fluxes


def _parse_energy_summary(f: TextIO) -> Optional[Dict[str, float]]:
    """Parse ENERGY CONSERVATION SUMMARY, return W/m^2 values.

    Format:
                            J               W (J/dt)        W/m^2 (J/dt/A)
     Energy change          ...             ...              7.89999048
     Net energy flux        ...             ...              7.89999048
     Absolute energy error  ...             ...              0.00000000
    """
    result: Dict[str, float] = {}
    line = f.readline()  # column headers
    line = f.readline()  # Energy change
    if "Energy change" in line:
        result["energy_change"] = float(line.split()[-1])
    line = f.readline()  # Net energy flux
    if "Net energy flux" in line:
        result["net_energy_flux"] = float(line.split()[-1])
    line = f.readline()  # Absolute energy error
    if "Absolute energy error" in line:
        result["absolute_energy_error"] = float(line.split()[-1])
    return result if result else None


class OcnParser(BaseParser):
    """Parse ocean log files for mass and energy conservation checks."""

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
        """Parse one CONSERVATION CHECKS block for mass and energy data."""
        rows: List[Dict] = []
        found_mass = False
        found_energy = False

        line = f.readline()
        while line:
            if found_mass and found_energy:
                break
            if "CONSERVATION CHECKS" in line and "date:" not in line:
                break

            if "MASS CONSERVATION CHECK" in line and "SUMMARY" not in line:
                found_mass = True
                rows.extend(self._parse_mass_section(f, year))

            elif "ENERGY CONSERVATION CHECK" in line and "SUMMARY" not in line:
                found_energy = True
                rows.extend(self._parse_energy_section(f, year))

            line = f.readline()

        return rows

    def _parse_mass_section(self, f: TextIO, year: int) -> List[Dict]:
        """Parse MASS CONSERVATION CHECK: fluxes + summary."""
        rows: List[Dict] = []
        base = {
            COL_TIME: year,
            COL_COMPONENT: "ocn",
            COL_QUANTITY: "water",
            COL_UNITS: "kg/m2s*1e6",
            COL_SOURCE: "ocn",
            COL_PERIOD: "monthly",
        }

        for term, val in _parse_mass_fluxes(f):
            rows.append(
                {**base, COL_TERM: term, COL_VALUE: val, COL_TABLE_TYPE: "flux"}
            )

        line = f.readline()
        while line:
            if "MASS CONSERVATION SUMMARY" in line:
                summary = _parse_mass_summary(f)
                if summary:
                    if "mass_change" in summary:
                        rows.append(
                            {
                                **base,
                                COL_TERM: "Mass change",
                                COL_VALUE: summary["mass_change"],
                                COL_TABLE_TYPE: "flux",
                            }
                        )
                    if "absolute_mass_error" in summary:
                        rows.append(
                            {
                                **base,
                                COL_TERM: "Absolute mass error",
                                COL_VALUE: summary["absolute_mass_error"],
                                COL_TABLE_TYPE: "diagnostic",
                            }
                        )
                break
            elif "SALT CONSERVATION" in line:
                break
            line = f.readline()

        return rows

    def _parse_energy_section(self, f: TextIO, year: int) -> List[Dict]:
        """Parse ENERGY CONSERVATION CHECK: fluxes + summary."""
        rows: List[Dict] = []
        base = {
            COL_TIME: year,
            COL_COMPONENT: "ocn",
            COL_QUANTITY: "heat",
            COL_UNITS: "W/m2",
            COL_SOURCE: "ocn",
            COL_PERIOD: "monthly",
        }

        for term, val in _parse_heat_fluxes(f):
            rows.append(
                {**base, COL_TERM: term, COL_VALUE: val, COL_TABLE_TYPE: "flux"}
            )

        line = f.readline()
        while line:
            if "ENERGY CONSERVATION SUMMARY" in line:
                summary = _parse_energy_summary(f)
                if summary:
                    if "energy_change" in summary:
                        rows.append(
                            {
                                **base,
                                COL_TERM: "Energy change",
                                COL_VALUE: summary["energy_change"],
                                COL_TABLE_TYPE: "flux",
                            }
                        )
                    if "absolute_energy_error" in summary:
                        rows.append(
                            {
                                **base,
                                COL_TERM: "Absolute energy error",
                                COL_VALUE: summary["absolute_energy_error"],
                                COL_TABLE_TYPE: "diagnostic",
                            }
                        )
                break
            elif "RELATIVE ENERGY" in line:
                break
            line = f.readline()

        return rows
