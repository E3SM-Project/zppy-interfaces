import argparse
import os
import shutil
import sys

import coupled_global
import ocean_month

from typing import Any, Dict, List

def param_get_list(param_value: str) -> List[str]:
    if param_value == "None":
        return []
    else:
        return param_value.split(",")


def get_region(rgn: str) -> str:
    if rgn.lower() in ["glb", "global"]:
        rgn = "glb"
    elif rgn.lower() in ["n", "north", "northern"]:
        rgn = "n"
    elif rgn.lower() in ["s", "south", "southern"]:
        rgn = "s"
    else:
        raise ValueError(f"Invalid rgn={rgn}")
    return rgn


class Parameters(object):
    def __init__(self, args: Dict[str, str]):

        # For ocean_month
        self.use_ocn: bool = _str2bool(args["use_ocn"])
        #self.global_ts_dir: str = args["global_ts_dir"]
        self.input: str = args["input"]
        self.input_subdir: str = args["input_subdir"]
        self.moc_file: str = args["moc_file"]

        # For coupled_global
        self.case_dir: str = args["case_dir"]
        self.experiment_name: str = args["experiment_name"]
        self.figstr: str = args["figstr"]
        self.color: str = args["color"]
        self.ts_num_years_str: str = args["ts_num_years"]
        self.plots_original: List[str] = param_get_list(args["plots_original"])
        self.atmosphere_only: bool = _str2bool(args["atmosphere_only"])
        self.plots_atm: List[str] = param_get_list(args["plots_atm"])
        self.plots_ice: List[str] = param_get_list(args["plots_ice"])
        self.plots_lnd: List[str] = param_get_list(args["plots_lnd"])
        self.plots_ocn: List[str] = param_get_list(args["plots_ocn"])
        # These regions are used often as strings,
        # so making an Enum Region={GLOBAL, NORTH, SOUTH} would be limiting.
        self.regions: List[str] = list(
            map(lambda rgn: get_region(rgn), args["regions"].split(","))
        )

        # For both
        self.year1: int = int(args["start_yr"])
        self.year2: int = int(args["end_yr"])

def main(parameters = None):
    if not parameters:
        parameters = _get_args()

    if parameters.use_ocn:
        print("Create ocean time series")
        # NOTE: MODIFIES THE CASE DIRECTORY (parameters.case_dir)
        os.makedirs(f"{parameters.case_dir}/post/ocn/glb/ts/monthly/{parameters.ts_num_years_str}yr", exist_ok=True)
        input: str = f"{parameters.input}/{parameters.input_subdir}"
        # NOTE: MODIFIES THE CASE DIRECTORY (parameters.case_dir)
        ocean_month.ocean_month(input, parameters.case_dir, parameters.year1, parameters.year2, int(parameters.ts_num_years_str))

        print("Copy moc file")
        # NOTE: MODIFIES THE CASE DIRECTORY (parameters.case_dir)
        shutil.copy(f"{parameters.case_dir}/post/analysis/mpas_analysis/cache/timeseries/moc/{parameters.moc_file}", f"{parameters.case_dir}/post/ocn/glb/ts/monthly/{parameters.ts_num_years_str}yr/")

    print("Update time series figures")
    # NOTE: PRODUCES OUTPUT IN THE CURRENT DIRECTORY
    coupled_global.coupled_global(parameters)


# TODO: replace command line arguments with _get_cfg_parameters, like https://github.com/E3SM-Project/e3sm_diags/blob/main/e3sm_diags/parser/core_parser.py#L809
def _get_args() -> Parameters:
    # Parser
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        usage="zppy-interfaces global-time-series <args>", description="Generate Global Time Series plots"
    )

    # For ocean_month
    parser.add_argument("use_ocn", type=str, help="Use ocean")
    #parser.add_argument("global_ts_dir", type=str, help="Global time series directory")
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

    return Parameters(vars(args))

def _str2bool(s: str) -> bool:
    return s.lower() == "true"

# Run with `python global_time_series.py`
if __name__ == "__main__":
    # Run off results from `zppy -c tests/integration/generated/test_min_case_global_time_series_setup_only_chrysalis.cfg`

    # TODO: fix readTS errors
    # TODO: change to setup_only once that job finishes
    case_dir = "/lcrc/group/e3sm/ac.forsyth2/zppy_min_case_global_time_series_original_8_output/test-642-2024-1104/v3.LR.historical_0051"
    parameters: Parameters = Parameters({
        "use_ocn": "True",
        #"global_ts_dir": dir,
        "input": "/lcrc/group/e3sm2/ac.wlin/E3SMv3/v3.LR.historical_0051",
        "input_subdir": "archive/atm/hist",
        "moc_file": "mocTimeSeries_1985-1995.nc",
        "case_dir": case_dir,
        "experiment_name": "v3.LR.historical_0051",
        "figstr": "v3.LR.historical_0051",
        "color": "Blue",
        "ts_num_years": "5",
        "plots_original": "net_toa_flux_restom,global_surface_air_temperature,toa_radiation,net_atm_energy_imbalance,change_ohc,max_moc,change_sea_level,net_atm_water_imbalance",
        "atmosphere_only": "True",
        "plots_atm": "None",
        "plots_ice": "None",
        "plots_lnd": "None",
        "plots_ocn": "None",
        "regions": "glb,n,s",
        "start_yr": "1985",
        "end_yr": "1994",
    })
    main(parameters)