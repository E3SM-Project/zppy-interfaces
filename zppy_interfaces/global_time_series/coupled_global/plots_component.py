from typing import Any, Dict, List

import matplotlib as mpl
import numpy as np

from zppy_interfaces.global_time_series.coupled_global.plotting import plot
from zppy_interfaces.global_time_series.coupled_global.utils import (
    RequestedVariables,
    get_data_dir,
    set_var,
)
from zppy_interfaces.global_time_series.utils import Parameters
from zppy_interfaces.multi_utils.logger import _setup_custom_logger

mpl.use("Agg")

logger = _setup_custom_logger(__name__)

# This file is for handling the component plots
# Hence, "plots_component"

# Used by driver.run ##########################################################


def process_data(
    parameters: Parameters, requested_variables: RequestedVariables
) -> List[Dict[str, Any]]:
    exps: List[Dict[str, Any]] = _get_exps(parameters)
    valid_vars: List[str] = []
    invalid_vars: List[str] = []
    exp: Dict[str, Any]
    for exp in exps:
        exp["annual"] = {}

        requested_variables.vars_atm = set_var(
            exp,
            "atmos",
            requested_variables.vars_atm,
            valid_vars,
            invalid_vars,
            parameters,
        )
        requested_variables.vars_ice = set_var(
            exp,
            "ice",
            requested_variables.vars_ice,
            valid_vars,
            invalid_vars,
            parameters,
        )
        requested_variables.vars_land = set_var(
            exp,
            "land",
            requested_variables.vars_land,
            valid_vars,
            invalid_vars,
            parameters,
        )
        requested_variables.vars_ocn = set_var(
            exp,
            "ocean",
            requested_variables.vars_ocn,
            valid_vars,
            invalid_vars,
            parameters,
        )

    logger.info(
        f"globalAnnual was computed successfully for these variables: {valid_vars}"
    )
    if invalid_vars:
        logger.error(
            f"globalAnnual could not be computed for these variables: {invalid_vars}"
        )
    return exps


def _get_exps(parameters: Parameters) -> List[Dict[str, Any]]:
    # Experiments
    exps: List[Dict[str, Any]] = [
        {
            "atmos": get_data_dir(parameters, "atm", parameters.plots_atm != []),
            "ice": get_data_dir(parameters, "ice", parameters.plots_ice != []),
            "land": get_data_dir(parameters, "lnd", parameters.plots_lnd != []),
            "ocean": get_data_dir(parameters, "ocn", parameters.plots_ocn != []),
            "name": parameters.experiment_name,
            "yoffset": 0.0,
            "yr": ([parameters.year1, parameters.year2],),
            "color": f"{parameters.color}",
        }
    ]
    return exps


# Plotting ####################################################################
# Used by mix_viewer_component.produce_pngs_for_viewer
# Used by mode_pdf.assemble_cumulative_pdf


def plot_generic(ax, xlim, exps, var_name, rgn):
    logger.info(f"plot_generic for {var_name}, rgn={rgn}")
    param_dict = {
        "2nd_var": False,
        "axhline_y": 0,
        "check_exp_ocean": False,
        "check_exp_vol": False,
        "check_exp_year": True,
        "default_ylim": [],
        "do_add_line": True,
        "do_add_trend": True,
        "format": "%4.2f",
        "glb_only": False,
        "lw": 1.0,
        "ohc": False,
        "set_axhline": False,
        "set_legend": True,
        "shorten_year": False,
        "title": var_name,
        "use_getmoc": False,
        "var": lambda exp: np.array(exp["annual"][var_name][rgn][0]),
        "verbose": False,
        "vol": False,
        "ylabel": lambda exp: np.array(exp["annual"][var_name][rgn][1]),
    }
    plot(ax, xlim, exps, param_dict, rgn)
