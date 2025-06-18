from typing import Any, Dict, List

from zppy_interfaces.global_time_series.classic.coupled_global_plotting import (
    make_plot_pdfs,
)
from zppy_interfaces.global_time_series.coupled_global_utils import (
    DatasetWrapper,
    Variable,
    get_data_dir,
    set_var,
)
from zppy_interfaces.global_time_series.utils import Parameters
from zppy_interfaces.multi_utils.logger import _setup_custom_logger

logger = _setup_custom_logger(__name__)


# Classes #####################################################################
class RequestedVariables(object):
    def __init__(self, parameters: Parameters):
        self.vars_original: List[Variable] = get_vars_original(
            parameters.plots_original
        )


def get_vars_original(plots_original: List[str]) -> List[Variable]:
    # NOTE: These are ALL atmosphere variables
    vars_original: List[Variable] = []
    if ("net_toa_flux_restom" in plots_original) or (
        "net_atm_energy_imbalance" in plots_original
    ):
        vars_original.append(Variable("RESTOM"))
    if "net_atm_energy_imbalance" in plots_original:
        vars_original.append(Variable("RESSURF"))
    if "global_surface_air_temperature" in plots_original:
        vars_original.append(Variable("TREFHT"))
    if "toa_radiation" in plots_original:
        vars_original.append(Variable("FSNTOA"))
        vars_original.append(Variable("FLUT"))
    if "net_atm_water_imbalance" in plots_original:
        vars_original.append(Variable("PRECC"))
        vars_original.append(Variable("PRECL"))
        vars_original.append(Variable("QFLX"))
    return vars_original


# Main functionality ##########################################################


def run_coupled_global(parameters: Parameters) -> None:
    requested_variables = RequestedVariables(parameters)
    run(parameters, requested_variables)


def run(parameters: Parameters, requested_variables: RequestedVariables):
    # Experiments
    exps: List[Dict[str, Any]] = process_data(parameters, requested_variables)

    xlim: List[float] = [float(parameters.year1), float(parameters.year2)]

    # Note: we use `parameters.plots_original` rather than `requested_variables.vars_original`
    # because the "original" plots are expecting plot names that are not variable names.
    # The model components however are expecting plot names to be variable names.
    for rgn in parameters.regions:
        valid_plots: List[str] = []
        invalid_plots: List[str] = []
        make_plot_pdfs(
            parameters,
            rgn,
            "original",
            xlim,
            exps,
            parameters.plots_original,
            valid_plots,
            invalid_plots,
        )
        logger.info(f"These {rgn} region plots generated successfully: {valid_plots}")
        if invalid_plots:
            logger.error(
                f"These {rgn} region plots could not be generated successfully: {invalid_plots}"
            )


def process_data(
    parameters: Parameters, requested_variables: RequestedVariables
) -> List[Dict[str, Any]]:
    exps: List[Dict[str, Any]] = get_exps(parameters)
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


def get_exps(parameters: Parameters) -> List[Dict[str, Any]]:
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
