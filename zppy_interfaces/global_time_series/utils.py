from typing import Dict, List

from zppy_interfaces.multi_utils.logger import _setup_child_logger

logger = _setup_child_logger(__name__)


# Parameters ##################################################################
class Parameters(object):
    def __init__(self, args: Dict[str, str]):
        # Used by both Classic PDF and Viewer
        # For determining which output type to produce
        self.make_viewer: bool = _str2bool(args["make_viewer"])
        # For coupled_global
        self.case_dir: str = args["case_dir"]
        self.experiment_name: str = args["experiment_name"]
        self.figstr: str = args["figstr"]
        self.color: str = args["color"]
        self.ts_num_years_str: str = args["ts_num_years"]
        self.results_dir: str = args["results_dir"]
        # These regions are used often as strings,
        # so making an Enum Region={GLOBAL, NORTH, SOUTH} would be limiting.
        self.regions: List[str] = list(
            map(lambda rgn: get_region(rgn), args["regions"].split(","))
        )
        # For both ocean_month and coupled_global
        self.year1: int = int(args["start_yr"])
        self.year2: int = int(args["end_yr"])

        # Used by Classic PDF only
        # For ocean_month
        self.use_ocn: bool = _str2bool(args["use_ocn"])
        self.input: str = args["input"]
        self.input_subdir: str = args["input_subdir"]
        self.moc_file: str
        if args["moc_file"] == "None":
            self.moc_file = ""
        else:
            self.moc_file = args["moc_file"]
        # For coupled_global
        self.plots_original: List[str] = param_get_list(args["plots_original"])
        self.nrows: int = int(args["nrows"])
        self.ncols: int = int(args["ncols"])

        # Used by Viewer only
        # For coupled_global
        self.plots_atm: List[str] = param_get_list(args["plots_atm"])
        self.plots_ice: List[str] = param_get_list(args["plots_ice"])
        self.plots_lnd: List[str] = param_get_list(args["plots_lnd"])
        self.plots_ocn: List[str] = param_get_list(args["plots_ocn"])

        # Input validation
        # Note: use_ocn should be True if ocean plots are requested in plots_original
        # (change_ohc, max_moc, change_sea_level) or plots_ocn is non-empty.
        # This follows zppy's logic where use_ocn is auto-determined from plot content.
        if self.plots_original and self.use_ocn and (not self.moc_file):
            raise ValueError(
                "moc_file must be set for ocean plots in the original 8-plot set."
            )
        if not (
            self.plots_original
            or self.plots_atm
            or self.plots_ice
            or self.plots_lnd
            or self.plots_ocn
        ):
            raise ValueError("No plots are specified, so nothing will be generated.")


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
