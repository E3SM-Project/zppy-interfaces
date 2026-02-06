"""E3SM coupler log budget data parsing."""

import gzip
import re
from typing import Dict, List, Optional, TextIO

import numpy as np
import numpy.ma as ma


class Budget:
    """
    Class to parse and store E3SM coupler budget data from log files.

    Parses budget tables from E3SM coupler log files and stores data
    in structured arrays for analysis and visualization.
    """

    def __init__(self, header: str, years: np.ndarray):
        """
        Initialize Budget object.

        Args:
            header: Budget header string to search for in log files
            years: Array of years to process
        """
        # Identifying header
        self.header = header

        # Years to save
        self.years = years
        # Dictionary to look up year indices
        self.iyear = {key: i for i, key in enumerate(self.years)}

        # Extract units
        self.units = re.findall(r"\(([^)]+)", self.header)[1]

        # To be defined later
        self.cols: Optional[List[str]] = None
        self.rows: Optional[List[str]] = None
        self.icol: Optional[Dict[str, int]] = None
        self.irow: Optional[Dict[str, int]] = None
        self.data: Optional[ma.MaskedArray] = None

    def parse(self, f: TextIO, datestamp: str) -> None:
        """
        Parse budget data from file for given datestamp.

        Args:
            f: Open file object
            datestamp: Date stamp string from log file
        """
        # Check if year is within range
        year = int(datestamp[:-4]) - 1
        if year not in self.iyear:
            return

        # Read table header, extract column names
        tmp = f.readline().strip()
        cols = re.split(r"\s{2,}", tmp)

        # Store or check column names
        if self.cols is None:
            self.cols = cols
            self.icol = {key: i for i, key in enumerate(cols)}
        elif self.cols != cols:
            print("ERROR: cols mismatched")

        # Read table rows
        lines = []
        tmp = f.readline()
        while tmp.strip():
            parts = tmp.split()
            # Handle multi-word row names (e.g., "surface co2", "black carbon")
            # Find where the numeric data starts
            row_name_parts = []
            data_start = 0
            for j, part in enumerate(parts):
                try:
                    float(part)
                    data_start = j
                    break
                except ValueError:
                    row_name_parts.append(part)
            row_name = " ".join(row_name_parts)
            data_values = parts[data_start:]
            lines.append((row_name, data_values))
            tmp = f.readline()

        # Store or check row names
        rows = [line[0] for line in lines]
        if self.rows is None:
            self.rows = rows
            self.irow = {key: i for i, key in enumerate(rows)}
        elif self.rows != rows:
            print("ERROR: rows mismatched")

        # Store in 3d array
        if self.data is None:
            self.data = ma.masked_all((len(self.years), len(self.rows), len(self.cols)))
        iyear = self.iyear[year]

        for i, (row_name, data_values) in enumerate(lines):
            try:
                # Convert string values to float
                numeric_values = [float(v) for v in data_values]
                self.data[iyear, i, :] = numeric_values
            except (ValueError, TypeError) as e:
                print(f"ERROR converting row '{row_name}' values {data_values}: {e}")
                # Keep as masked values if conversion fails

        return


def parse_budget_types(budget_types_str: str) -> list[str]:
    """Parse comma-separated budget types string."""
    return [bt.strip() for bt in budget_types_str.split(",") if bt.strip()]


def initialize_budgets(budget_types: list[str], years: np.ndarray) -> dict[str, Budget]:
    """Initialize budget objects for specified types and years."""
    budget_headers = {
        "area": "(seq_diag_print_mct) NET AREA BUDGET (m2/m2): period =   annual: date =",
        "water": "(seq_diag_print_mct) NET WATER BUDGET (kg/m2s*1e6): period =   annual: date =",
        "heat": "(seq_diag_print_mct) NET HEAT BUDGET (W/m2): period =   annual: date =",
        "carbon": "(seq_diagBGC_print_mct) NET CARBON BUDGET (kg-C/m2s*1e10): period =   annual: date =",
    }

    budgets = {}
    for budget_type in budget_types:
        if budget_type in budget_headers:
            budgets[budget_type] = Budget(budget_headers[budget_type], years)
        else:
            print(f"WARNING: Unknown budget type {budget_type}, skipping")

    return budgets


def process_log_files(log_files: list[str], budgets: dict[str, Budget]) -> None:
    """Process coupler log files and extract budget data."""
    for fname in log_files:
        print(f"Processing {fname}")
        try:
            with gzip.open(fname, "rt") as f:
                line = f.readline()
                while line != "":
                    for budget_type, budget_obj in budgets.items():
                        if line.startswith(budget_obj.header):
                            datestamp = line.replace(budget_obj.header, "").split()[0]
                            budget_obj.parse(f, datestamp)
                    line = f.readline()
        except Exception as e:
            print(f"ERROR processing {fname}: {e}")
            continue
