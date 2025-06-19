from typing import Any, Dict, List

import matplotlib as mpl

from zppy_interfaces.global_time_series.coupled_global.utils import (
    DatasetWrapper,
    RequestedVariables,
    Variable,
    get_data_dir,
    set_var,
)
from zppy_interfaces.global_time_series.utils import Parameters
from zppy_interfaces.multi_utils.logger import _setup_custom_logger

mpl.use("Agg")

logger = _setup_custom_logger(__name__)

# This file is for handling the original plots
# Hence, "plots_original"

# Used by driver.run ##########################################################


def process_data(
    parameters: Parameters, requested_variables: RequestedVariables
) -> List[Dict[str, Any]]:
    exps: List[Dict[str, Any]] = _get_exps(parameters)
    valid_vars: List[str] = []
    invalid_vars: List[str] = []
    exp: Dict[str, Any]

    logger.info("Processing data for Classic PDF.")
    for exp in exps:
        exp["annual"] = {}

        logger.info("Setting requested variables")
        requested_variables.vars_original = set_var(
            exp,
            "atmos",
            requested_variables.vars_original,
            valid_vars,
            invalid_vars,
        )
        # Optionally read ohc
        logger.info("Reading ohc")
        if exp["moc"] != "":
            ohc_variable = Variable("ohc")
            dataset_wrapper = DatasetWrapper(exp["moc"])
            exp["annual"]["ohc"], _ = dataset_wrapper.globalAnnual(ohc_variable)
            # anomalies with respect to first year
            exp["annual"]["ohc"][:] = exp["annual"]["ohc"][:] - exp["annual"]["ohc"][0]

        logger.info("Reading vol")
        if exp["vol"] != "":
            vol_variable = Variable("volume")
            dataset_wrapper = DatasetWrapper(exp["vol"])
            exp["annual"]["volume"], _ = dataset_wrapper.globalAnnual(vol_variable)
            # annomalies with respect to first year
            exp["annual"]["volume"][:] = (
                exp["annual"]["volume"][:] - exp["annual"]["volume"][0]
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
    atm_set_intersection: set = set(
        [
            "net_toa_flux_restom",
            "global_surface_air_temperature",
            "toa_radiation",
            "net_atm_energy_imbalance",
            "net_atm_water_imbalance",
        ]
    ) & set(parameters.plots_original)
    # Use set intersection: check if any of these 3 plots were requested
    ocn_set_intersection: set = set(
        ["change_ohc", "max_moc", "change_sea_level"]
    ) & set(parameters.plots_original)
    ocean_dir = get_data_dir(parameters, "ocn", ocn_set_intersection != set())
    ocean_month_dir = get_data_dir(
        parameters,
        f"{parameters.subtask_name}/ocn",
        ocn_set_intersection != set(),
    )
    exps: List[Dict[str, Any]] = [
        {
            "atmos": get_data_dir(parameters, "atm", atm_set_intersection != set()),
            "ocean": ocean_dir,
            "moc": ocean_month_dir,
            "vol": ocean_month_dir,
            "name": parameters.experiment_name,
            "yoffset": 0.0,
            "yr": ([parameters.year1, parameters.year2],),
            "color": f"{parameters.color}",
        }
    ]
    return exps
