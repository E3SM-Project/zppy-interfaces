import argparse
import sys

import zppy_interfaces.global_time_series.classic.driver as classic_driver
import zppy_interfaces.global_time_series.viewer.driver as viewer_driver
from zppy_interfaces.global_time_series.utils import Parameters
from zppy_interfaces.multi_utils.logger import _setup_child_logger, _setup_root_logger

# Set up the root logger and module level logger. The module level logger is
# a child of the root logger.
_setup_root_logger()
logger = _setup_child_logger(__name__)


def main(parameters=None):
    if not parameters:
        parameters = _get_args()
    # Determine if we want the Classic PDF or the Viewer
    if parameters.make_viewer:
        viewer_driver.run(parameters)
    else:
        classic_driver.run(parameters)


def _get_args() -> Parameters:
    # Parser
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        usage="zi-global-time-series <args>",
        description="Generate Global Time Series plots",
    )

    # Used by both Classic PDF and Viewer
    # For determining which output type to produce
    parser.add_argument("--make_viewer", type=str, help="Make viewer")
    # For coupled_global
    parser.add_argument("--case_dir", type=str, help="Case directory")
    parser.add_argument("--experiment_name", type=str, help="Experiment name")
    parser.add_argument("--figstr", type=str, help="Figure string")
    parser.add_argument("--color", type=str, help="Color")
    parser.add_argument("--ts_num_years", type=str, help="Time series number of years")
    parser.add_argument("--results_dir", type=str, help="Results directory")
    parser.add_argument("--regions", type=str, help="Regions")
    # For both ocean_month and coupled_global
    parser.add_argument("--start_yr", type=str, help="Start year")
    parser.add_argument("--end_yr", type=str, help="End year")

    # Classic PDF only
    # For ocean_month
    parser.add_argument("--subsection", type=str, help="Subtask name")
    parser.add_argument("--use_ocn", type=str, help="Use ocean")
    parser.add_argument("--input", type=str, help="Input directory")
    parser.add_argument("--input_subdir", type=str, help="Input subdirectory")
    parser.add_argument("--moc_file", type=str, help="MOC file")
    # For coupled_global
    parser.add_argument("--plots_original", type=str, help="Plots original")
    parser.add_argument("--nrows", type=str, help="Number of rows in pdf")
    parser.add_argument("--ncols", type=str, help="Number of columns in pdf")

    # Viewer only
    # For coupled_global
    parser.add_argument("--plots_atm", type=str, help="Plots atmosphere")
    parser.add_argument("--plots_ice", type=str, help="Plots ice")
    parser.add_argument("--plots_lnd", type=str, help="Plots land")
    parser.add_argument("--plots_ocn", type=str, help="Plots ocean")

    # Ignore the first arg
    # (zi-global-time-series)
    args: argparse.Namespace = parser.parse_args(sys.argv[1:])

    return Parameters(vars(args))
