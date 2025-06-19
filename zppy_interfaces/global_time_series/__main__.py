import argparse
import sys

from zppy_interfaces.global_time_series.coupled_global.driver import run_coupled_global
from zppy_interfaces.global_time_series.create_ocean_ts import create_ocean_ts
from zppy_interfaces.global_time_series.utils import Parameters
from zppy_interfaces.multi_utils.logger import _setup_child_logger, _setup_root_logger

# Set up the root logger and module level logger. The module level logger is
# a child of the root logger.
_setup_root_logger()
logger = _setup_child_logger(__name__)


def main(parameters=None):
    if not parameters:
        parameters = _get_args()
    """
    Determine if we want the Classic PDF or the Viewer
    There are several cases to consider. In markdown table format:

    | case | make_viewer = | plots_original non-empty? | `plots_<component>` non-empty? | results page shows | plots_original | `plots_<component>` |
    | --- | --- | --- | --- | --- | --- | --- |
    | 1 | T | T | T | viewer list HTML    | `<rgn>_original` PDF & PNG | each component gets a Viewer, with rows=vars, cols=rgns |
    | 2 | T | T | F | viewer list HTML    | `<rgn>_original` PDF & PNG | no Viewers |
    | 3 | T | F | T | viewer list HTML    | no original PDF/PNG links | each component gets a Viewer, with rows=vars, cols=rgns |
    | 4 | T | F | F | viewer list HTML    | no original PDF/PNG links | no Viewers |
    | 5 | F | T | T | no-frills file list | `<rgn>_original` PDF & PNG | each component gets a cumulative PDF |
    | 6 | F | T | F | no-frills file list | `<rgn>_original` PDF & PNG | no component PDFs |
    | 7 | F | F | T | no-frills file list | no classic plots | each component gets a cumulative PDF |
    | 8 | F | F | F | no-frills file list | no classic plots | no component PDFs |

    examples/post.v3.LR.historical_zppy_v3.cfg has: make_viewer = True, plots_original = 8 plots, plots_lnd = variable list
    That is: | T | T | T |, or case 1 in the table above.

    By default: make_viewer = False, plots_original = 8 plots, plots_<component> = ""
    That is: | F | T | F |, or case 6 in the table above.

    We can simplify the above table to:

    | make_viewer= | plots_original | `plots_<component>` |
    | --- | --- | --- |
    | True | `<rgn>_original` PDF & PNG | each component gets a Viewer, with rows=vars, cols=rgns |
    | False | `<rgn>_original` PDF & PNG | each component gets a cumulative PDF |
    """
    if parameters.use_ocn:
        # From zppy's default.ini:
        # Remove the 3 ocean plots (change_ohc,max_moc,change_sea_level) if you don't have ocean data.
        # plots_original = string(default="net_toa_flux_restom,global_surface_air_temperature,toa_radiation,net_atm_energy_imbalance,change_ohc,max_moc,change_sea_level,net_atm_water_imbalance")
        if set(["change_ohc", "max_moc", "change_sea_level"]) & set(
            parameters.plots_original
        ):
            create_ocean_ts(parameters)
    logger.info("Update time series figures")
    # NOTE: PRODUCES OUTPUT IN THE CURRENT DIRECTORY (not necessarily the case directory)
    # Creates the directory parameters.results_dir
    run_coupled_global(parameters)
    # TODO: Add tests for all of the above cases on the zppy side


def _get_args() -> Parameters:
    # Parser
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        usage="zi-global-time-series <args>",
        description="Generate Global Time Series plots",
    )

    # Used in all cases
    # > For determining which output type to produce
    parser.add_argument("--make_viewer", type=str, help="Make viewer")
    # > For coupled_global
    parser.add_argument("--case_dir", type=str, help="Case directory")
    parser.add_argument("--experiment_name", type=str, help="Experiment name")
    parser.add_argument("--figstr", type=str, help="Figure string")
    parser.add_argument("--color", type=str, help="Color")
    parser.add_argument("--ts_num_years", type=str, help="Time series number of years")
    parser.add_argument("--results_dir", type=str, help="Results directory")
    parser.add_argument("--regions", type=str, help="Regions")
    # > For both ocean_month and coupled_global
    parser.add_argument("--start_yr", type=str, help="Start year")
    parser.add_argument("--end_yr", type=str, help="End year")

    # For plots_original
    # > For ocean_month
    parser.add_argument("--subsection", type=str, help="Subtask name")
    parser.add_argument("--use_ocn", type=str, help="Use ocean")
    parser.add_argument("--input", type=str, help="Input directory")
    parser.add_argument("--input_subdir", type=str, help="Input subdirectory")
    parser.add_argument("--moc_file", type=str, help="MOC file")
    # > For coupled_global
    parser.add_argument("--plots_original", type=str, help="Plots original")

    # For plots_component
    # > For coupled_global
    parser.add_argument("--plots_atm", type=str, help="Plots atmosphere")
    parser.add_argument("--plots_ice", type=str, help="Plots ice")
    parser.add_argument("--plots_lnd", type=str, help="Plots land")
    parser.add_argument("--plots_ocn", type=str, help="Plots ocean")

    # Used for mode_pdf, regardless of plot type
    # > For coupled_global
    parser.add_argument("--ncols", type=str, help="Number of columns in pdf")
    parser.add_argument("--nrows", type=str, help="Number of rows in pdf")

    # Ignore the first arg
    # (zi-global-time-series)
    args: argparse.Namespace = parser.parse_args(sys.argv[1:])

    return Parameters(vars(args))
