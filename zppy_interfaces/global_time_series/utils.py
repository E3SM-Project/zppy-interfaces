from typing import Dict, List


# Parameters ##################################################################
class Parameters(object):
    def __init__(self, args: Dict[str, str]):

        # For ocean_month
        self.use_ocn: bool = _str2bool(args["use_ocn"])
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
        self.nrows: int = int(args["nrows"])
        self.ncols: int = int(args["ncols"])
        self.results_dir: str = args["results_dir"]
        # These regions are used often as strings,
        # so making an Enum Region={GLOBAL, NORTH, SOUTH} would be limiting.
        self.regions: List[str] = list(
            map(lambda rgn: get_region(rgn), args["regions"].split(","))
        )

        # For both
        self.year1: int = int(args["start_yr"])
        self.year2: int = int(args["end_yr"])


def _str2bool(s: str) -> bool:
    return s.lower() == "true"


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


###############################################################################
