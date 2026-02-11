"""Land log parser — extracts water flux and state tables into tidy DataFrames.

Land log format examples:

NET WATER FLUXES : period   annual: date =        20101           0
                       Time     |      Time
                     averaged   |    integrated
                   kg/m2s*1e6   |     kg/m2*1e6
--------------------------------|--------------------
            rain     7.168...   |       226051777.94
            ...
           *SUM*    -0.005...   |         -174290.03

WATER STATES (kg/m2*1e6): period   annual: date =        20101           0
                          Canopy     Snow      SFC      Soil Liq   Soil Ice    Aquifer   Grid-level Err |   TOTAL
------...
             beg          ...
             end          ...
    *NET CHANGE*          ...
------...
       *SUM*              ...
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

FLUX_HEADER = "NET WATER FLUXES : period"
STATE_HEADER = "WATER STATES (kg/m2*1e6): period"


def _parse_datestamp(datestamp: str) -> Tuple[int, int]:
    """Convert datestamp to (year, month). Same convention as coupler."""
    mmdd = datestamp[-4:]
    month = int(mmdd[:2])
    year = int(datestamp[:-4]) - 1
    if month == 1:
        month = 12
    else:
        month -= 1
    return year, month


def _make_time(period: str, year: int, month: int) -> float:
    """Encode (year, month) as a float time value."""
    if period == "monthly":
        return year + (month - 0.5) / 12.0
    return float(year)


def _parse_period_and_time(line: str) -> Optional[Tuple[str, float]]:
    """Extract period and time from a header line."""
    period_match = re.search(r"period\s+(\w+):", line)
    date_match = re.search(r"date\s*=\s*(\d+)", line)
    if not period_match or not date_match:
        return None
    period = period_match.group(1)
    year, month = _parse_datestamp(date_match.group(1))
    return period, _make_time(period, year, month)


def _parse_flux_table(f: TextIO, year: float, period: str) -> List[Dict]:
    """Parse a NET WATER FLUXES table.

    Format:
        [2 header lines: Time averaged | Time integrated]
        [1 units line: kg/m2s*1e6 | kg/m2*1e6]
        [separator line: ---...|---...]
        term     rate_value | integrated_value
        ...
        [separator line]
        *SUM*    rate_value | integrated_value
    """
    rows: List[Dict] = []

    # Skip 3 header lines (Time averaged/integrated labels + units)
    for _ in range(3):
        f.readline()

    # Skip separator
    f.readline()

    # Parse data rows until we hit an empty line or non-data line
    line = f.readline()
    while line and line.strip():
        stripped = line.strip()
        # Skip separator lines
        if stripped.startswith("---"):
            line = f.readline()
            continue

        # Split on '|' to separate rate and integrated columns
        parts = stripped.split("|")
        if len(parts) < 2:
            line = f.readline()
            continue

        rate_part = parts[0].strip()
        integrated_part = parts[1].strip()

        # Parse term name and rate value
        rate_tokens = rate_part.split()
        if not rate_tokens:
            line = f.readline()
            continue

        # Find where numeric data starts
        term_parts: List[str] = []
        rate_val = None
        for token in rate_tokens:
            try:
                rate_val = float(token)
                break
            except ValueError:
                term_parts.append(token)

        term = " ".join(term_parts)
        if not term or rate_val is None:
            line = f.readline()
            continue

        # Parse integrated value
        try:
            integrated_val = float(integrated_part)
        except ValueError:
            integrated_val = None

        # Emit flux rate row
        rows.append(
            {
                COL_TIME: year,
                COL_COMPONENT: "lnd",
                COL_QUANTITY: "water",
                COL_TERM: term,
                COL_VALUE: rate_val,
                COL_UNITS: "kg/m2s*1e6",
                COL_SOURCE: "lnd",
                COL_PERIOD: period,
                COL_TABLE_TYPE: "flux",
            }
        )

        # Emit flux integrated row
        if integrated_val is not None:
            rows.append(
                {
                    COL_TIME: year,
                    COL_COMPONENT: "lnd",
                    COL_QUANTITY: "water",
                    COL_TERM: term,
                    COL_VALUE: integrated_val,
                    COL_UNITS: "kg/m2*1e6",
                    COL_SOURCE: "lnd",
                    COL_PERIOD: period,
                    COL_TABLE_TYPE: "flux_integrated",
                }
            )

        line = f.readline()

    return rows


def _parse_state_table(f: TextIO, year: float, period: str) -> List[Dict]:
    """Parse a WATER STATES table.

    Format:
        [column header line: Canopy  Snow  SFC  Soil Liq  Soil Ice  Aquifer  Grid-level Err | TOTAL]
        [separator line]
        beg          val1  val2  ...  | total
        end          val1  val2  ...  | total
        *NET CHANGE* val1  val2  ...  | total
        [separator line]
        *SUM*        ...              | total
    """
    rows: List[Dict] = []

    # Column header line
    col_line = f.readline()
    # Split on '|' — left side has pool names, right has TOTAL
    col_parts = col_line.split("|")
    left_header = col_parts[0].strip() if col_parts else ""
    # Parse pool names from left header (separated by 2+ spaces)
    pool_names = [p.strip() for p in re.split(r"\s{2,}", left_header) if p.strip()]

    # Skip separator
    f.readline()

    # Parse data rows
    line = f.readline()
    while line and line.strip():
        stripped = line.strip()
        if stripped.startswith("---"):
            line = f.readline()
            continue

        # Split on '|'
        parts = stripped.split("|")
        left_part = parts[0].strip()
        right_part = parts[1].strip() if len(parts) > 1 else ""

        tokens = left_part.split()
        if not tokens:
            line = f.readline()
            continue

        # Find row label and values
        label_parts: List[str] = []
        data_start = 0
        for j, token in enumerate(tokens):
            try:
                float(token)
                data_start = j
                break
            except ValueError:
                label_parts.append(token)

        row_label = " ".join(label_parts)
        values = tokens[data_start:]

        # Parse TOTAL from right side of '|'
        total_val = None
        if right_part:
            try:
                total_val = float(right_part)
            except ValueError:
                pass

        # Emit per-pool values
        for i, val_str in enumerate(values):
            if i < len(pool_names):
                try:
                    val = float(val_str)
                except ValueError:
                    continue
                rows.append(
                    {
                        COL_TIME: year,
                        COL_COMPONENT: "lnd",
                        COL_QUANTITY: "water",
                        COL_TERM: f"{row_label}_{pool_names[i]}",
                        COL_VALUE: val,
                        COL_UNITS: "kg/m2*1e6",
                        COL_SOURCE: "lnd",
                        COL_PERIOD: period,
                        COL_TABLE_TYPE: "state",
                    }
                )

        # Emit TOTAL
        if total_val is not None:
            rows.append(
                {
                    COL_TIME: year,
                    COL_COMPONENT: "lnd",
                    COL_QUANTITY: "water",
                    COL_TERM: f"{row_label}_TOTAL",
                    COL_VALUE: total_val,
                    COL_UNITS: "kg/m2*1e6",
                    COL_SOURCE: "lnd",
                    COL_PERIOD: period,
                    COL_TABLE_TYPE: "state",
                }
            )

        line = f.readline()

    return rows


class LndParser(BaseParser):
    """Parse land log files for water flux and state tables."""

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
                        stripped = line.strip()

                        if stripped.startswith(FLUX_HEADER):
                            result = _parse_period_and_time(stripped)
                            if result is None:
                                continue
                            period, time = result
                            if period != self.frequency:
                                continue
                            if start_year <= time <= end_year + 1:
                                rows.extend(_parse_flux_table(f, time, period))

                        elif stripped.startswith(STATE_HEADER):
                            result = _parse_period_and_time(stripped)
                            if result is None:
                                continue
                            period, time = result
                            if period != self.frequency:
                                continue
                            if start_year <= time <= end_year + 1:
                                rows.extend(_parse_state_table(f, time, period))

            except Exception as e:
                print(f"WARNING: Error processing {fname}: {e}")
                continue

        if not rows:
            return pd.DataFrame(columns=COLUMNS)
        return pd.DataFrame(rows, columns=COLUMNS)
