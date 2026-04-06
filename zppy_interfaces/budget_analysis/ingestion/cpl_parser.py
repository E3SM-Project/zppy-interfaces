"""Coupler log parser — extracts budget tables into tidy DataFrames."""

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

# Header patterns for each budget quantity.
HEADER_PATTERNS: Dict[str, str] = {
    "water": "(seq_diag_print_mct) NET WATER BUDGET (kg/m2s*1e6):",
    "heat": "(seq_diag_print_mct) NET HEAT BUDGET (W/m2):",
    "carbon": "(seq_diagBGC_print_mct) NET CARBON BUDGET (kg-C/m2s*1e10):",
}

UNITS: Dict[str, str] = {
    "water": "kg/m2s*1e6",
    "heat": "W/m2",
    "carbon": "kg-C/m2s*1e10",
}


def _normalize_component_name(name: str) -> str:
    """Normalize component names: 'ice nh' -> 'ice_nh'."""
    return name.strip().replace(" ", "_")


def _parse_datestamp(datestamp: str, period: str = "monthly") -> Tuple[int, int]:
    """Convert coupler datestamp to (year, month).

    For monthly data: '10201' -> year 1, month 1 (roll back one month)
    For annual data: '20101' -> year 1 (annual summary for year 1, output at start of year 2)
    """
    mmdd = datestamp[-4:]
    month = int(mmdd[:2])
    year = int(datestamp[:-4])

    if period == "annual":
        # Annual data: date represents start of year after the summary year
        # e.g., '20101' = annual summary for year 1, output at start of year 2
        return year - 1, 12  # Return summary year with month 12 for annual data
    else:
        # Monthly data: roll back one month (date is start of *next* period)
        if month == 1:
            month = 12
            year -= 1  # Roll back year when going from Jan to Dec
        else:
            month -= 1
        return year, month


def _make_time(period: str, year: int, month: int) -> float:
    """Encode (year, month) as a float time value.

    Annual: integer year.  Monthly: year + (month - 0.5) / 12.
    """
    if period == "monthly":
        return year + (month - 0.5) / 12.0
    return float(year)


def _parse_header_line(line: str, pattern: str) -> Optional[Tuple[str, float]]:
    """Extract period and time from a budget header line.

    Returns (period, time) or None on failure.
    """
    if not line.startswith(pattern):
        return None

    remainder = line[len(pattern) :]
    period_match = re.search(r"period\s*=\s*(\w+)", remainder)
    date_match = re.search(r"date\s*=\s*(\d+)", remainder)
    if not period_match or not date_match:
        return None

    period = period_match.group(1)
    year, month = _parse_datestamp(date_match.group(1), period)
    return period, _make_time(period, year, month)


def _parse_table(f: TextIO, year: float, quantity: str, period: str) -> List[Dict]:
    """Parse one budget table after the header line was consumed."""
    rows: List[Dict] = []
    units = UNITS[quantity]

    # First line after header: column names separated by 2+ spaces
    col_line = f.readline().strip()
    col_names = [
        _normalize_component_name(c) for c in re.split(r"\s{2,}", col_line) if c
    ]

    # Data rows until blank line
    line = f.readline()
    while line and line.strip():
        parts = line.split()
        # Find where numeric data starts (handles multi-word term names)
        term_parts: List[str] = []
        data_start = 0
        for j, part in enumerate(parts):
            try:
                float(part)
                data_start = j
                break
            except ValueError:
                term_parts.append(part)
        term = " ".join(term_parts)
        values = parts[data_start:]

        for i, val_str in enumerate(values):
            if i < len(col_names):
                rows.append(
                    {
                        COL_TIME: year,
                        COL_COMPONENT: col_names[i],
                        COL_QUANTITY: quantity,
                        COL_TERM: term,
                        COL_VALUE: float(val_str),
                        COL_UNITS: units,
                        COL_SOURCE: "cpl",
                        COL_PERIOD: period,
                        COL_TABLE_TYPE: "flux",
                    }
                )

        line = f.readline()

    return rows


class CplParser(BaseParser):
    """Parse coupler log budget tables into a tidy event table."""

    def __init__(
        self,
        quantities: Optional[List[str]] = None,
        frequency: str = "annual",
    ):
        super().__init__(frequency=frequency)
        self.quantities = quantities or ["water", "heat"]

    def parse_files(
        self, log_files: List[str], start_year: int, end_year: int
    ) -> pd.DataFrame:
        rows: List[Dict] = []
        for fname in sorted(log_files):
            try:
                with gzip.open(fname, "rt") as f:
                    for line in f:
                        for quantity in self.quantities:
                            pattern = HEADER_PATTERNS.get(quantity)
                            if not pattern:
                                continue
                            result = _parse_header_line(line, pattern)
                            if result is None:
                                continue
                            period, time = result
                            if period != self.frequency:
                                continue
                            # Extract year from time for consistent filtering
                            year = int(time)
                            if start_year <= year <= end_year:
                                rows.extend(_parse_table(f, time, quantity, period))
            except Exception as e:
                print(f"WARNING: Error processing {fname}: {e}")
                continue

        if not rows:
            return pd.DataFrame(columns=COLUMNS)
        return pd.DataFrame(rows, columns=COLUMNS)
