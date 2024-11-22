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
        os.makedirs(
            f"{parameters.case_dir}/post/ocn/glb/ts/monthly/{parameters.ts_num_years_str}yr",
            exist_ok=True,
        )
        input: str = f"{parameters.input}/{parameters.input_subdir}"
        # NOTE: MODIFIES THE CASE DIRECTORY (parameters.case_dir) post subdirectory
        ocean_month(
            input,
            parameters.case_dir,
            parameters.year1,
            parameters.year2,
            int(parameters.ts_num_years_str),
        )

        logger.info("Copy moc file")
        # NOTE: MODIFIES THE CASE DIRECTORY (parameters.case_dir) post subdirectory
        shutil.copy(
            f"{parameters.case_dir}/post/analysis/mpas_analysis/cache/timeseries/moc/{parameters.moc_file}",
            f"{parameters.case_dir}/post/ocn/glb/ts/monthly/{parameters.ts_num_years_str}yr/",
        )

    logger.info("Update time series figures")
    # NOTE: PRODUCES OUTPUT IN THE CURRENT DIRECTORY
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


# Run with `python __main__.py`
if __name__ == "__main__":
    parameters: Parameters = Parameters(
        {
            "use_ocn": "True",
            "input": "/lcrc/group/e3sm2/ac.wlin/E3SMv3/v3.LR.historical_0051",
            "input_subdir": "archive/ocn/hist",
            "moc_file": "mocTimeSeries_1985-1995.nc",
            "case_dir": "/lcrc/group/e3sm/ac.forsyth2/zppy_min_case_global_time_series_original_8_output/test-642-working-env-20241121/v3.LR.historical_0051",
            "experiment_name": "v3.LR.historical_0051",
            "figstr": "v3.LR.historical_0051",
            "color": "Blue",
            "ts_num_years": "5",
            "plots_original": "net_toa_flux_restom,global_surface_air_temperature,toa_radiation,net_atm_energy_imbalance,change_ohc,max_moc,change_sea_level,net_atm_water_imbalance",
            "atmosphere_only": "False",
            "plots_atm": "None",
            "plots_ice": "None",
            "plots_lnd": "None",
            "plots_ocn": "None",
            "nrows": "4",
            "ncols": "2",
            "results_dir": "global_time_series_1985-1995_results",
            "regions": "glb,n,s",
            "start_yr": "1985",
            "end_yr": "1995",
        }
    )
    main(parameters)
