import argparse
import os
import shutil
import sys

import coupled_global
import ocean_month

from typing import Dict

def main():
    args: argparse.Namespace = _get_args()

    if _str2bool(args.use_ocn):
        print("Create ocean time series")
        os.chdir(args.global_ts_dir)
        os.makedirs(f"{args.case_dir}/post/ocn/glb/ts/monthly/{args.ts_num_years}yr", exist_ok=True)
        input: str = f"{args.input}/{args.input_subdir}"
        ocean_month.ocean_month(input, args.case_dir, int(args.start_yr), int(args.end_yr), int(args.ts_num_years))

        print("Copy moc file")
        os.chdir(f"{args.case_dir}/post/analysis/mpas_analysis/cache/timeseries/moc")
        shutil.copy(args.moc_file, f"../../../../../ocn/glb/ts/monthly/{args.ts_num_years}yr/")

    print("Update time series figures")
    os.chdir(args.global_ts_dir)
    parameters: Dict[str, str] = {
        "case_dir": args.case_dir,
        "experiment_name": args.experiment_name,
        "figstr": args.figstr,
        "start_yr": args.start_yr,
        "end_yr": args.end_yr,
        "color": args.color,
        "ts_num_years": args.ts_num_years,
        "plots_original": args.plots_original,
        "atmosphere_only": args.atmosphere_only.lower(),
        "plots_atm": args.plots_atm,
        "plots_ice": args.plots_ice,
        "plots_lnd": args.plots_lnd,
        "plots_ocn": args.plots_ocn,
        "regions": args.regions,
    }
    coupled_global.coupled_global(parameters)


def _get_args() -> argparse.Namespace:
    # Parser
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        usage="zppy-interfaces global-time-series <args>", description="Generate Global Time Series plots"
    )

    # For ocean_month
    parser.add_argument("use_ocn", type=str, help="Use ocean")
    parser.add_argument("global_ts_dir", type=str, help="Global time series directory")
    parser.add_argument("input", type=str, help="Input directory")
    parser.add_argument("input_subdir", type=str, help="Input subdirectory")
    parser.add_argument("moc_file", type=str, help="MOC file")

    # For coupled_global
    parser.add_argument("case_dir", type=str, help="Case directory")
    parser.add_argument("experiment_name", type=str, help="Experiment name")
    parser.add_argument("figstr", type=str, help="Figure string")
    parser.add_argument("color", type=str, help="Color")
    parser.add_argument("ts_num_years", type=str, help="Time series number of years")
    parser.add_argument("plots_original", type=str, help="Plots original")
    parser.add_argument("atmosphere_only", type=str, help="Atmosphere only")  
    parser.add_argument("plots_atm", type=str, help="Plots atmosphere")
    parser.add_argument("plots_ice", type=str, help="Plots ice")
    parser.add_argument("plots_lnd", type=str, help="Plots land")
    parser.add_argument("plots_ocn", type=str, help="Plots ocean")
    parser.add_argument("regions", type=str, help="Regions")

    # For both
    parser.add_argument("start_yr", type=str, help="Start year")
    parser.add_argument("end_yr", type=str, help="End year")  

    # Now that we're inside a subcommand, ignore the first two argvs
    # (zppy-interfaces global-time-series)
    args: argparse.Namespace = parser.parse_args(sys.argv[2:])

    return args

def _str2bool(s: str) -> bool:
    return s.lower() == "true"
