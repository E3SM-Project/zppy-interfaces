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


def _parse_datestamp(datestamp: str) -> int:
    """Convert coupler datestamp to year.

    The date is reported at the start of the next period.
    E.g. '20101' -> strip last 4 chars -> '2' -> minus 1 -> year 1.
    """
    return int(datestamp[:-4]) - 1


def _parse_header_line(line: str, pattern: str) -> Optional[Tuple[str, int]]:
    """Extract period and year from a budget header line.

    Returns (period, year) or None on failure.
    """
    if not line.startswith(pattern):
        return None

    remainder = line[len(pattern) :]
    period_match = re.search(r"period\s*=\s*(\w+)", remainder)
    date_match = re.search(r"date\s*=\s*(\d+)", remainder)
    if not period_match or not date_match:
        return None

    period = period_match.group(1)
    year = _parse_datestamp(date_match.group(1))
    return period, year


def _parse_table(f: TextIO, year: int, quantity: str, period: str) -> List[Dict]:
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

    def __init__(self, quantities: Optional[List[str]] = None):
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
                            period, year = result
                            if start_year <= year <= end_year:
                                rows.extend(_parse_table(f, year, quantity, period))
            except Exception as e:
                print(f"WARNING: Error processing {fname}: {e}")
                continue

        if not rows:
            return pd.DataFrame(columns=COLUMNS)
        return pd.DataFrame(rows, columns=COLUMNS)
