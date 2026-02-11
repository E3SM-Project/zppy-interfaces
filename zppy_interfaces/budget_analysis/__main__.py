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
from .plotting import generate_html_plots


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
        help="Comma-separated list of budget types to analyze (water,heat,carbon)",
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
        "--mode",
        choices=["legacy", "whole-model"],
        default="legacy",
        help="'legacy' (coupler-only cumulative) or 'whole-model' (multi-source budget checks)",
    )
    parser.add_argument(
        "--frequency",
        choices=["monthly", "annual"],
        default="annual",
        help="Temporal frequency for budget data: 'monthly' or 'annual' (default: annual)",
    )

    args = parser.parse_args()

    # Validate inputs
    if args.start_year > args.end_year:
        print("ERROR: start_year must be <= end_year")
        return 1

    log_path = args.log_path
    if not os.path.exists(log_path):
        print(f"ERROR: Log path does not exist: {log_path}")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "whole-model":
        return _run_whole_model(args)

    return _run_legacy(args)


def _run_legacy(args) -> int:
    """Original coupler-only cumulative budget pipeline."""
    budget_types = parse_budget_types(args.budget_types)
    valid_types = ["area", "water", "heat", "carbon"]
    for bt in budget_types:
        if bt not in valid_types:
            print(f"ERROR: Invalid budget type '{bt}'. Valid types: {valid_types}")
            return 1

    print("E3SM Budget Analysis Tool (legacy mode)")
    print("========================================")
    print(f"Years: {args.start_year} to {args.end_year}")
    print(f"Budget types: {budget_types}")
    print(f"Log path: {args.log_path}")

    years = np.arange(args.start_year, args.end_year + 1)
    budgets = initialize_budgets(budget_types, years)

    log_files = sorted(glob.glob(os.path.join(args.log_path, "cpl.log.*.gz")))
    if not log_files:
        print("ERROR: No coupler log files found")
        return 1

    print(f"Found {len(log_files)} coupler log files")
    process_log_files(log_files, budgets)

    if args.output_html:
        generate_html_plots(budgets, budget_types, args.output_dir)

    print("\nBudget analysis completed successfully!")
    return 0


def _run_whole_model(args) -> int:
    """Whole-model budget pipeline: ingest -> normalize -> check -> visualize."""
    import pandas as pd

    from .checks import (
        DEFAULT_CARBON_CHECKS,
        DEFAULT_HEAT_CHECKS,
        DEFAULT_WATER_CHECKS,
        run_checks,
    )
    from .ingestion.atm_parser import AtmParser
    from .ingestion.cpl_parser import CplParser
    from .ingestion.lnd_parser import LndParser
    from .ingestion.ocn_parser import OcnParser
    from .normalization import normalize
    from .viz import generate_budget_report, generate_landing_page

    budget_types = parse_budget_types(args.budget_types)

    print("E3SM Budget Analysis Tool (whole-model mode)")
    print("=============================================")
    print(f"Years: {args.start_year} to {args.end_year}")
    print(f"Budget types: {budget_types}")
    print(f"Frequency: {args.frequency}")
    print(f"Log path: {args.log_path}")

    # Ingest
    print("\nIngesting log files...")
    cpl_files = sorted(glob.glob(os.path.join(args.log_path, "cpl.log.*.gz")))
    lnd_files = sorted(glob.glob(os.path.join(args.log_path, "lnd.log.*.gz")))
    ocn_files = sorted(glob.glob(os.path.join(args.log_path, "ocn.log.*.gz")))
    atm_files = sorted(glob.glob(os.path.join(args.log_path, "atm.log.*")))

    if not cpl_files:
        print("ERROR: No coupler log files found")
        return 1
    print(f"  {len(cpl_files)} coupler log files")
    print(f"  {len(lnd_files)} land log files")
    print(f"  {len(ocn_files)} ocean log files")
    print(f"  {len(atm_files)} atmosphere log files")

    freq = args.frequency
    frames = []
    frames.append(
        CplParser(quantities=budget_types, frequency=freq).parse_files(
            cpl_files, args.start_year, args.end_year
        )
    )
    if lnd_files:
        frames.append(
            LndParser(frequency=freq).parse_files(
                lnd_files, args.start_year, args.end_year
            )
        )
    if ocn_files:
        frames.append(
            OcnParser(frequency=freq).parse_files(
                ocn_files, args.start_year, args.end_year
            )
        )
    if atm_files:
        frames.append(
            AtmParser(frequency=freq).parse_files(
                atm_files, args.start_year, args.end_year
            )
        )
    events = pd.concat(frames, ignore_index=True)
    print(f"  {len(events)} total event rows")

    # Normalize
    print("\nNormalizing...")
    events = normalize(events)

    # Check and visualize per quantity
    checks_map = {
        "water": DEFAULT_WATER_CHECKS,
        "heat": DEFAULT_HEAT_CHECKS,
        "carbon": DEFAULT_CARBON_CHECKS,
    }

    report_paths = {}
    for quantity in budget_types:
        checks = checks_map.get(quantity)
        if checks is None:
            print(f"\n  WARNING: No checks defined for '{quantity}', skipping")
            continue

        print(f"\nRunning {quantity} budget checks...")
        results = run_checks(events, checks)

        print(f"\nGenerating {quantity} report...")
        html_path = generate_budget_report(
            results, events, args.output_dir, quantity=quantity
        )
        if html_path:
            report_paths[quantity] = html_path

    # Landing page
    if len(report_paths) > 0:
        index_path = generate_landing_page(args.output_dir, report_paths)
        print(f"\nBudget analysis completed! Landing page: {index_path}")
    else:
        print("\nNo reports generated.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
