"""Ice log parser — extracts mass and energy conservation checks into tidy DataFrames.

Ice log format (monthly, inside Conservation checks blocks):

 ===================================================================================
  Conservation checks: 0001-02-01_00:00:00
 -----------------------------------------------------------------------------------
  Area analysis                                 Global             NH             SH

  Earth radius                   (m) =    6.371229E+06
  Earth area                    (m2) =    5.101011E+14
  Domain area                   (m2) =    3.574961E+14   1.501228E+14   2.073733E+14
  Sea-ice area                  (m2) =    3.038731E+13   1.207167E+13   1.831565E+13
 -----------------------------------------------------------------------------------
  Energy conservation check

  Initial energy ice             (J) =   -2.695578E+22  -8.884723E+21  -1.807106E+22
  Final energy ice               (J) =   -1.949620E+22  -9.304888E+21  -1.019131E+22
  Energy change                  (J) =    7.459584E+21  -4.201653E+20   7.879749E+21
  Energy change flux          (W/m2) =    5.459877E+00  -3.075307E-01   5.767408E+00

  Surface heat flux           (W/m2) =    5.153267E-01  -7.817582E-01   1.297085E+00
     Absorbed shortwave flux  (W/m2) =    2.892564E+00   2.707821E-02   2.865486E+00
     Ocean Shortwave flux     (W/m2) =   -1.960086E-01  -5.682405E-03  -1.903262E-01
  ...
  Net energy change              (J) =    7.449107E+21  -4.186389E+20   7.867746E+21
  Net energy flux             (W/m2) =    5.452209E+00  -3.064135E-01   5.758622E+00
 -----------------------------------------------------------------------------------
  Mass conservation check

  Initial mass ice              (kg) =    7.684570E+16   2.532862E+16   5.151708E+16
  Final mass ice                (kg) =    5.825212E+16   2.706170E+16   3.119042E+16
  Ice mass change               (kg) =   -1.859359E+16   1.733078E+15  -2.032666E+16
  Ice mass change flux      (kg/m2s) =   -1.360916E-05   1.268488E-06  -1.487765E-05
  ...
"""

import gzip
import re
from typing import Dict, List, TextIO, Tuple

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
    Same logic as ocean parser.
    """
    match = re.match(r"(\d+)-(\d+)-(\d+)", date_str.strip())
    if not match:
        return -1, -1
    y, m = int(match.group(1)), int(match.group(2))
    # Roll back one month
    if m == 1:
        return y - 1, 12
    return y, m - 1


def _parse_energy_section(f: TextIO, time: float) -> List[Dict]:
    """Parse Energy conservation check section."""
    rows: List[Dict] = []
    base = {
        COL_TIME: time,
        COL_COMPONENT: "ice",
        COL_QUANTITY: "heat",
        COL_UNITS: "W/m2",
        COL_SOURCE: "ice",
        COL_PERIOD: "monthly",
        COL_TABLE_TYPE: "flux",
    }

    # Read until we find mass section or next conservation block
    while True:
        line = f.readline()
        if not line:  # EOF
            break

        stripped = line.strip()
        if not stripped:
            continue

        # Stop at next section
        if "Mass conservation check" in line:
            # Put this line back by seeking backwards (approximate)
            # We'll let _parse_mass_section handle this line
            break
        if "Conservation checks:" in line:
            break
        if "---" in line and len([c for c in line if c == "-"]) > 10:
            break

        # Parse energy flux terms
        if "(W/m2)" in line:
            # Extract term name and values
            parts = line.split("(W/m2)")
            if len(parts) >= 2:
                term_name = parts[0].strip()
                values_part = parts[1].split("=")
                if len(values_part) >= 2:
                    try:
                        values = values_part[1].strip().split()
                        # Use global value (first column)
                        if values:
                            val = float(values[0])
                            rows.append(
                                {
                                    **base,
                                    COL_TERM: term_name,
                                    COL_VALUE: val,
                                }
                            )
                    except (ValueError, IndexError):
                        pass

    return rows


def _parse_mass_section(f: TextIO, time: float) -> List[Dict]:
    """Parse Mass conservation check section."""
    rows: List[Dict] = []
    base = {
        COL_TIME: time,
        COL_COMPONENT: "ice",
        COL_QUANTITY: "water",
        COL_UNITS: "kg/m2s",
        COL_SOURCE: "ice",
        COL_PERIOD: "monthly",
        COL_TABLE_TYPE: "flux",
    }

    # Read until we find end of mass section
    while True:
        line = f.readline()
        if not line:  # EOF
            break

        stripped = line.strip()
        if not stripped:
            continue

        # Stop at end of section
        if "Conservation checks:" in line:
            break
        if "---" in line and len([c for c in line if c == "-"]) > 10:
            break

        # Parse mass flux terms
        if "(kg/m2s)" in line:
            # Extract term name and values
            parts = line.split("(kg/m2s)")
            if len(parts) >= 2:
                term_name = parts[0].strip()
                values_part = parts[1].split("=")
                if len(values_part) >= 2:
                    try:
                        values = values_part[1].strip().split()
                        # Use global value (first column)
                        if values:
                            val = float(values[0])
                            rows.append(
                                {
                                    **base,
                                    COL_TERM: term_name,
                                    COL_VALUE: val,
                                }
                            )
                    except (ValueError, IndexError):
                        pass

    return rows


class IceParser(BaseParser):
    """Parse ice log files for mass and energy conservation checks."""

    def __init__(self, frequency: str = "annual") -> None:
        super().__init__(frequency=frequency)

    def parse_files(
        self, log_files: List[str], start_year: int, end_year: int
    ) -> pd.DataFrame:
        rows: List[Dict] = []
        for fname in sorted(log_files):
            try:
                with gzip.open(fname, "rt") as f:
                    for line in f:
                        if "Conservation checks:" in line:
                            # Extract date from same line
                            date_match = re.search(
                                r"(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})", line
                            )
                            if not date_match:
                                continue
                            date_str = date_match.group(1)
                            year, month = _parse_date(date_str)
                            if year < start_year or year > end_year:
                                continue
                            if self.frequency == "monthly":
                                time = year + (month - 0.5) / 12.0
                            else:
                                time = float(year)
                            rows.extend(self._parse_block(f, time))
            except Exception as e:
                print(f"WARNING: Error processing {fname}: {e}")
                continue

        if not rows:
            return pd.DataFrame(columns=COLUMNS)
        df = pd.DataFrame(rows, columns=COLUMNS)

        # For annual frequency, aggregate monthly rows to annual means
        if self.frequency == "annual":
            group_keys = [
                COL_TIME,
                COL_COMPONENT,
                COL_QUANTITY,
                COL_TERM,
                COL_UNITS,
                COL_SOURCE,
                COL_TABLE_TYPE,
            ]
            df[COL_PERIOD] = "annual"
            df = df.groupby(group_keys, as_index=False).agg(
                {COL_VALUE: "mean", COL_PERIOD: "first"}
            )

        return df

    def _parse_block(self, f: TextIO, time: float) -> List[Dict]:
        """Parse one Conservation checks block for mass and energy data."""
        rows: List[Dict] = []
        found_mass = False
        found_energy = False

        line = f.readline()
        while line:
            # Stop when both sections are processed
            if found_mass and found_energy:
                break

            # Stop at next conservation block
            if "Conservation checks:" in line:
                break

            # Parse energy section
            if "Energy conservation check" in line:
                found_energy = True
                energy_rows = _parse_energy_section(f, time)
                rows.extend(energy_rows)

            # Parse mass section
            elif "Mass conservation check" in line:
                found_mass = True
                mass_rows = _parse_mass_section(f, time)
                rows.extend(mass_rows)

            line = f.readline()

        return rows
