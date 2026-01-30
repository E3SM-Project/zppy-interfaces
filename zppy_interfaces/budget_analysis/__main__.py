#!/usr/bin/env python3

"""
E3SM Water and Energy Budget Analysis CLI Tool

Analyzes E3SM coupler log files to extract water and energy budget data
and generates interactive HTML plots and ASCII summary tables.

This tool is designed to be called by zppy or used standalone for
budget conservation analysis of E3SM simulations.
"""

import argparse
import glob
import os
import sys

import numpy as np

from .parser import initialize_budgets, parse_budget_types, process_log_files
from .plotting import generate_ascii_output, generate_html_plots


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze E3SM water and energy budgets from coupler log files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --log_path /path/to/case/archive/logs --start_year 114 --end_year 206
  %(prog)s --log_path /path/to/case/archive/logs --start_year 114 --end_year 150 --budget_types water,heat
        """,
    )

    parser.add_argument(
        "--log_path",
        required=True,
        help="Path to directory containing coupler log files (cpl.log.*.gz)",
    )
    parser.add_argument(
        "--start_year", type=int, required=True, help="Starting year for analysis"
    )
    parser.add_argument(
        "--end_year", type=int, required=True, help="Ending year for analysis"
    )
    parser.add_argument(
        "--budget_types",
        default="water,heat",
        help="Comma-separated list of budget types to analyze (water,heat)",
    )
    parser.add_argument(
        "--output_dir",
        default=".",
        help="Output directory for results (default: current directory)",
    )
    parser.add_argument(
        "--output_html",
        action="store_true",
        default=True,
        help="Generate HTML plots (default: True)",
    )
    parser.add_argument(
        "--output_ascii",
        action="store_true",
        default=True,
        help="Generate ASCII summary tables (default: True)",
    )

    args = parser.parse_args()

    # Validate inputs
    if args.start_year > args.end_year:
        print("ERROR: start_year must be <= end_year")
        return 1

    # Use provided log path
    log_path = args.log_path

    if not os.path.exists(log_path):
        print(f"ERROR: Log path does not exist: {log_path}")
        return 1

    # Parse budget types
    budget_types = parse_budget_types(args.budget_types)
    valid_types = ["area", "water", "heat"]
    for bt in budget_types:
        if bt not in valid_types:
            print(f"ERROR: Invalid budget type '{bt}'. Valid types: {valid_types}")
            return 1

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    print("E3SM Budget Analysis Tool")
    print("=========================")
    print(f"Years: {args.start_year} to {args.end_year}")
    print(f"Budget types: {budget_types}")
    print(f"Log path: {log_path}")
    print(f"Output directory: {args.output_dir}")

    # Set up years array
    years = np.arange(args.start_year, args.end_year + 1)

    # Initialize budget objects
    budgets = initialize_budgets(budget_types, years)

    # Find and process log files
    log_pattern = os.path.join(log_path, "cpl.log.*.gz")
    log_files = sorted(glob.glob(log_pattern))

    if not log_files:
        print(f"ERROR: No coupler log files found at {log_pattern}")
        return 1

    print(f"Found {len(log_files)} coupler log files")

    # Process log files
    process_log_files(log_files, budgets)

    # Generate outputs
    print("\nGenerating output files...")

    # Generate ASCII summaries if requested
    if args.output_ascii:
        for budget_type, budget_obj in budgets.items():
            generate_ascii_output(budget_obj, budget_type, args.output_dir)

    # Generate HTML plots if requested
    if args.output_html:
        generate_html_plots(budgets, budget_types, args.output_dir)

    print("\nBudget analysis completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
