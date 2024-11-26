import argparse
import os
import shutil
import sys

from zppy_interfaces.global_time_series.coupled_global import coupled_global
from zppy_interfaces.global_time_series.ocean_month import ocean_month
from zppy_interfaces.global_time_series.utils import Parameters
from zppy_interfaces.multi_utils.logger import _setup_custom_logger

logger = _setup_custom_logger(__name__)


def main(parameters=None):
    if not parameters:
        parameters = _get_args()

    if parameters.use_ocn:
        logger.info("Create ocean time series")
        # NOTE: MODIFIES THE CASE DIRECTORY (parameters.case_dir) post subdirectory
        # Creates the directory post/ocn
        os.makedirs(
            f"{parameters.case_dir}/post/ocn/glb/ts/monthly/{parameters.ts_num_years_str}yr",
            exist_ok=True,
        )
        input: str = f"{parameters.input}/{parameters.input_subdir}"
        # NOTE: MODIFIES THE CASE DIRECTORY (parameters.case_dir) post subdirectory
        # Modifies post/ocn (which we just created in the first place)
        ocean_month(
            input,
            parameters.case_dir,
            parameters.year1,
            parameters.year2,
            int(parameters.ts_num_years_str),
        )

        logger.info("Copy moc file")
        # NOTE: MODIFIES THE CASE DIRECTORY (parameters.case_dir) post subdirectory
        # Copies files to post/ocn (which we just created in the first place)
        shutil.copy(
            f"{parameters.case_dir}/post/analysis/mpas_analysis/cache/timeseries/moc/{parameters.moc_file}",
            f"{parameters.case_dir}/post/ocn/glb/ts/monthly/{parameters.ts_num_years_str}yr/",
        )

    logger.info("Update time series figures")
    # NOTE: PRODUCES OUTPUT IN THE CURRENT DIRECTORY (not necessarily the case directory)
    # Creates the directory parameters.results_dir
    coupled_global(parameters)


def _get_args() -> Parameters:
    # Parser
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        usage="zi-global-time-series <args>",
        description="Generate Global Time Series plots",
    )

    # For ocean_month
    parser.add_argument("--use_ocn", type=str, help="Use ocean")
    parser.add_argument("--input", type=str, help="Input directory")
    parser.add_argument("--input_subdir", type=str, help="Input subdirectory")
    parser.add_argument("--moc_file", type=str, help="MOC file")

    # For coupled_global
    parser.add_argument("--case_dir", type=str, help="Case directory")
    parser.add_argument("--experiment_name", type=str, help="Experiment name")
    parser.add_argument("--figstr", type=str, help="Figure string")
    parser.add_argument("--color", type=str, help="Color")
    parser.add_argument("--ts_num_years", type=str, help="Time series number of years")
    parser.add_argument("--plots_original", type=str, help="Plots original")
    parser.add_argument("--atmosphere_only", type=str, help="Atmosphere only")
    parser.add_argument("--plots_atm", type=str, help="Plots atmosphere")
    parser.add_argument("--plots_ice", type=str, help="Plots ice")
    parser.add_argument("--plots_lnd", type=str, help="Plots land")
    parser.add_argument("--plots_ocn", type=str, help="Plots ocean")
    parser.add_argument("--nrows", type=str, help="Number of rows in pdf")
    parser.add_argument("--ncols", type=str, help="Number of columns in pdf")
    parser.add_argument("--results_dir", type=str, help="Results directory")
    parser.add_argument("--regions", type=str, help="Regions")

    # For both
    parser.add_argument("--start_yr", type=str, help="Start year")
    parser.add_argument("--end_yr", type=str, help="End year")

    # Ignore the first arg
    # (zi-global-time-series)
    args: argparse.Namespace = parser.parse_args(sys.argv[1:])

    return Parameters(vars(args))
