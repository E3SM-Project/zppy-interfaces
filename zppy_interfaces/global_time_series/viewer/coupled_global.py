import csv
import importlib.resources as imp_res
from typing import Any, Dict, List, Tuple

from zppy_interfaces.global_time_series.coupled_global_utils import (
    Metric,
    Variable,
    get_data_dir,
    set_var,
)
from zppy_interfaces.global_time_series.utils import Parameters
from zppy_interfaces.global_time_series.viewer.coupled_global_plotting import (
    make_plot_pdfs,
)
from zppy_interfaces.global_time_series.viewer.coupled_global_viewer import (
    create_viewer,
    create_viewer_index,
)
from zppy_interfaces.multi_utils.logger import _setup_custom_logger

logger = _setup_custom_logger(__name__)


# Classes #####################################################################
class RequestedVariables(object):
    def __init__(self, parameters: Parameters):
        # Land variables are constructed differently
        self.vars_land: List[Variable] = construct_land_variables(parameters.plots_lnd)

        # Other variables use the generic constructor
        self.vars_atm: List[Variable] = construct_generic_variables(
            parameters.plots_atm
        )
        self.vars_ice: List[Variable] = construct_generic_variables(
            parameters.plots_ice
        )
        self.vars_ocn: List[Variable] = construct_generic_variables(
            parameters.plots_ocn
        )


def construct_land_variables(requested_vars: List[str]) -> List[Variable]:
    var_list: List[Variable] = []
    header = True
    csv_filename = str(
        imp_res.files("zppy_interfaces.global_time_series") / "zppy_land_fields.csv"
    )
    with open(csv_filename, newline="") as csv_file:
        logger.debug("Reading zppy_land_fields.csv")
        var_reader = csv.reader(csv_file)
        for row in var_reader:
            # logger.debug(f"row={row}")
            # Skip the header row
            if header:
                header = False
            else:
                # If set to "all" then we want all variables.
                # Design note: we can't simply run all variables if requested_vars is empty because
                # that would actually mean the user doesn't want to make *any* land plots.
                if (requested_vars == ["all"]) or (row[0] in requested_vars):
                    row_elements_strip_whitespace: List[str] = list(
                        map(lambda x: x.strip(), row)
                    )
                    var_list.append(land_csv_row_to_var(row_elements_strip_whitespace))
    return var_list


def land_csv_row_to_var(csv_row: List[str]) -> Variable:
    # “A” or “T” for global average over land area or global total, respectively
    metric: Metric
    if csv_row[1] == "A":
        metric = Metric.AVERAGE
    elif csv_row[1] == "T":
        metric = Metric.TOTAL
    else:
        raise ValueError(f"Invalid metric={csv_row[1]}")
    return Variable(
        variable_name=csv_row[0],
        metric=metric,
        scale_factor=float(csv_row[2]),
        original_units=csv_row[3],
        final_units=csv_row[4],
        group=csv_row[5],
        long_name=csv_row[6],
    )


def construct_generic_variables(requested_vars: List[str]) -> List[Variable]:
    var_list: List[Variable] = []
    for var_name in requested_vars:
        var_list.append(Variable(var_name))
    return var_list


# Main functionality ##########################################################


def run_coupled_global(parameters: Parameters) -> None:
    requested_variables = RequestedVariables(parameters)
    run(parameters, requested_variables)
    title_and_url_list: List[Tuple[str, str]] = []
    for component in [
        "atm",
        "ice",
        "lnd",
        "ocn",
    ]:  # Don't create viewer for original component
        vars_list: List[Variable] = get_vars(requested_variables, component)
        if vars_list:
            url = create_viewer(parameters, vars_list, component)
            logger.info(f"Viewer URL for {component}: {url}")
            title_and_url_list.append((component, url))

    index_url: str = create_viewer_index(parameters.results_dir, title_and_url_list)
    logger.info(f"Viewer index URL: {index_url}")


def run(parameters: Parameters, requested_variables: RequestedVariables):
    # Experiments
    exps: List[Dict[str, Any]] = process_data(parameters, requested_variables)

    xlim: List[float] = [float(parameters.year1), float(parameters.year2)]

    # Use list of tuples rather than a dict, to keep order
    mapping: List[Tuple[str, List[str]]] = [
        ("atm", list(map(lambda v: v.variable_name, requested_variables.vars_atm))),
        ("ice", list(map(lambda v: v.variable_name, requested_variables.vars_ice))),
        ("lnd", list(map(lambda v: v.variable_name, requested_variables.vars_land))),
        ("ocn", list(map(lambda v: v.variable_name, requested_variables.vars_ocn))),
    ]
    for rgn in parameters.regions:
        valid_plots: List[str] = []
        invalid_plots: List[str] = []
        for component, plot_list in mapping:
            make_plot_pdfs(
                parameters,
                rgn,
                component,
                xlim,
                exps,
                plot_list,
                valid_plots,
                invalid_plots,
            )
        logger.info(f"These {rgn} region plots generated successfully: {valid_plots}")
        if invalid_plots:
            logger.error(
                f"These {rgn} region plots could not be generated successfully: {invalid_plots}"
            )


def get_vars(requested_variables: RequestedVariables, component: str) -> List[Variable]:
    vars: List[Variable]
    if component == "atm":
        vars = requested_variables.vars_atm
    elif component == "ice":
        vars = requested_variables.vars_ice
    elif component == "lnd":
        vars = requested_variables.vars_land
    elif component == "ocn":
        vars = requested_variables.vars_ocn
    else:
        raise ValueError(f"Invalid component={component}")
    return vars


def process_data(
    parameters: Parameters, requested_variables: RequestedVariables
) -> List[Dict[str, Any]]:
    exps: List[Dict[str, Any]] = get_exps(parameters)
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


def get_exps(parameters: Parameters) -> List[Dict[str, Any]]:
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
