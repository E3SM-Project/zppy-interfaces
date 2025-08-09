# Script to plot some global atmosphere and ocean time series
import csv
import importlib.resources as imp_res
from typing import Any, Dict, List, Tuple

import cftime
import numpy as np
import xarray

from zppy_interfaces.global_time_series.coupled_global_dataset_wrapper import (
    DatasetWrapper,
)
from zppy_interfaces.global_time_series.coupled_global_plotting import make_plot_pdfs
from zppy_interfaces.global_time_series.coupled_global_utils import Metric, Variable
from zppy_interfaces.global_time_series.coupled_global_viewer import (
    create_viewer,
    create_viewer_index,
)
from zppy_interfaces.global_time_series.utils import Parameters
from zppy_interfaces.multi_utils.logger import _setup_child_logger

logger = _setup_child_logger(__name__)


# Useful helper functions and classes #########################################


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


def construct_generic_variables(requested_vars: List[str]) -> List[Variable]:
    var_list: List[Variable] = []
    for var_name in requested_vars:
        var_list.append(Variable(var_name))
    return var_list


class RequestedVariables(object):
    def __init__(self, parameters: Parameters):
        self.vars_original: List[Variable] = get_vars_original(
            parameters.plots_original
        )
        self.vars_land: List[Variable] = construct_land_variables(parameters.plots_lnd)

        # Use generic constructor
        self.vars_atm: List[Variable] = construct_generic_variables(
            parameters.plots_atm
        )
        self.vars_ice: List[Variable] = construct_generic_variables(
            parameters.plots_ice
        )
        self.vars_ocn: List[Variable] = construct_generic_variables(
            parameters.plots_ocn
        )


# Setup #######################################################################
def get_data_dir(parameters: Parameters, component: str, conditional: bool) -> str:
    return (
        f"{parameters.case_dir}/post/{component}/glb/ts/monthly/{parameters.ts_num_years_str}yr/"
        if conditional
        else ""
    )


def get_exps(parameters: Parameters) -> List[Dict[str, Any]]:
    # Experiments
    use_atmos: bool = (parameters.plots_atm != []) or (parameters.plots_original != [])
    # Use set intersection: check if any of these 3 plots were requested
    set_intersection: set = set(["change_ohc", "max_moc", "change_sea_level"]) & set(
        parameters.plots_original
    )
    has_original_ocn_plots: bool = set_intersection != set()
    use_ocn: bool = (parameters.plots_ocn != []) or has_original_ocn_plots
    ocean_dir = get_data_dir(parameters, "ocn", use_ocn)
    exps: List[Dict[str, Any]] = [
        {
            "atmos": get_data_dir(parameters, "atm", use_atmos),
            "ice": get_data_dir(parameters, "ice", parameters.plots_ice != []),
            "land": get_data_dir(parameters, "lnd", parameters.plots_lnd != []),
            "ocean": ocean_dir,
            "moc": ocean_dir,
            "vol": ocean_dir,
            "name": parameters.experiment_name,
            "yoffset": 0.0,
            "yr": ([parameters.year1, parameters.year2],),
            "color": f"{parameters.color}",
        }
    ]
    return exps


def set_var(
    exp: Dict[str, Any],
    exp_key: str,
    var_list: List[Variable],
    valid_vars: List[str],
    invalid_vars: List[str],
    rgn: str,
) -> List[Variable]:
    new_var_list: List[Variable] = []
    if exp[exp_key] != "":
        try:
            dataset_wrapper: DatasetWrapper = DatasetWrapper(exp[exp_key], var_list)
        except Exception as e:
            logger.critical(e)
            logger.critical(
                f"DatasetWrapper object could not be created for {exp_key}={exp[exp_key]}"
            )
            raise e
        for var in var_list:
            var_str: str = var.variable_name
            try:
                data_array: xarray.core.dataarray.DataArray
                units: str
                data_array, units = dataset_wrapper.globalAnnual(var)
                valid_vars.append(str(var_str))  # Append the name
                new_var_list.append(var)  # Append the variable itself
            except Exception as e:
                logger.error(e)
                logger.error(f"globalAnnual failed for {var_str}")
                invalid_vars.append(str(var_str))
                continue
            if data_array.sizes["rgn"] > 1:
                # number of years x 3 regions = data_array.shape
                # 3 regions = global, northern hemisphere, southern hemisphere
                # We get here if we used the updated `ts` task
                # (using `rgn_avg` rather than `glb_avg`).
                if rgn == "glb":
                    n = 0
                elif rgn == "n":
                    n = 1
                elif rgn == "s":
                    n = 2
                else:
                    raise RuntimeError(f"Invalid rgn={rgn}")
                data_array = data_array.isel(rgn=n)  # Just use nth region
            elif rgn != "glb":
                # data_array only has one dimension -- glb.
                # Therefore it is not possible to get n or s plots.
                raise RuntimeError(
                    f"var={var_str} only has global data. Cannot process rgn={rgn}"
                )
            exp["annual"][var_str] = (data_array, units)
            if "year" not in exp["annual"]:
                years: np.ndarray[cftime.DatetimeNoLeap] = data_array.coords[
                    "time"
                ].values
                exp["annual"]["year"] = [x.year for x in years]
        del dataset_wrapper
    return new_var_list


def process_data(
    parameters: Parameters, requested_variables: RequestedVariables, rgn: str
) -> List[Dict[str, Any]]:
    exps: List[Dict[str, Any]] = get_exps(parameters)
    valid_vars: List[str] = []
    invalid_vars: List[str] = []
    exp: Dict[str, Any]
    for exp in exps:
        exp["annual"] = {}

        requested_variables.vars_original = set_var(
            exp,
            "atmos",
            requested_variables.vars_original,
            valid_vars,
            invalid_vars,
            rgn,
        )
        requested_variables.vars_atm = set_var(
            exp, "atmos", requested_variables.vars_atm, valid_vars, invalid_vars, rgn
        )
        requested_variables.vars_ice = set_var(
            exp, "ice", requested_variables.vars_ice, valid_vars, invalid_vars, rgn
        )
        requested_variables.vars_land = set_var(
            exp,
            "land",
            requested_variables.vars_land,
            valid_vars,
            invalid_vars,
            rgn,
        )
        requested_variables.vars_ocn = set_var(
            exp, "ocean", requested_variables.vars_ocn, valid_vars, invalid_vars, rgn
        )

        # Optionally read ohc
        if exp["ocean"] != "":
            ohc_variable = Variable("ohc")
            dataset_wrapper = DatasetWrapper(exp["ocean"], [ohc_variable])
            exp["annual"]["ohc"], _ = dataset_wrapper.globalAnnual(ohc_variable)
            # anomalies with respect to first year
            exp["annual"]["ohc"][:] = exp["annual"]["ohc"][:] - exp["annual"]["ohc"][0]

        if exp["vol"] != "":
            vol_variable = Variable("volume")
            dataset_wrapper = DatasetWrapper(exp["vol"], [vol_variable])
            exp["annual"]["volume"], _ = dataset_wrapper.globalAnnual(vol_variable)
            # annomalies with respect to first year
            exp["annual"]["volume"][:] = (
                exp["annual"]["volume"][:] - exp["annual"]["volume"][0]
            )

    logger.info(
        f"{rgn} region globalAnnual was computed successfully for these variables: {valid_vars}"
    )
    logger.error(
        f"{rgn} region globalAnnual could not be computed for these variables: {invalid_vars}"
    )
    return exps


# Run coupled_global ##########################################################
def run(parameters: Parameters, requested_variables: RequestedVariables, rgn: str):
    # Experiments
    exps: List[Dict[str, Any]] = process_data(parameters, requested_variables, rgn)

    xlim: List[float] = [float(parameters.year1), float(parameters.year2)]

    valid_plots: List[str] = []
    invalid_plots: List[str] = []

    # Use list of tuples rather than a dict, to keep order
    # Note: we use `parameters.plots_original` rather than `requested_variables.vars_original`
    # because the "original" plots are expecting plot names that are not variable names.
    # The model components however are expecting plot names to be variable names.
    mapping: List[Tuple[str, List[str]]] = [
        ("original", parameters.plots_original),
        ("atm", list(map(lambda v: v.variable_name, requested_variables.vars_atm))),
        ("ice", list(map(lambda v: v.variable_name, requested_variables.vars_ice))),
        ("lnd", list(map(lambda v: v.variable_name, requested_variables.vars_land))),
        ("ocn", list(map(lambda v: v.variable_name, requested_variables.vars_ocn))),
    ]
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
    logger.error(
        f"These {rgn} region plots could not be generated successfully: {invalid_plots}"
    )


def get_vars(requested_variables: RequestedVariables, component: str) -> List[Variable]:
    vars: List[Variable]
    if component == "original":
        vars = requested_variables.vars_original
    elif component == "atm":
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


def coupled_global(parameters: Parameters) -> None:
    requested_variables = RequestedVariables(parameters)
    for rgn in parameters.regions:
        run(parameters, requested_variables, rgn)
    if parameters.make_viewer:
        # In this case, we don't want the summary PDF.
        # Rather, we want to construct a viewer similar to E3SM Diags.
        title_and_url_list: List[Tuple[str, str]] = []
        for component in [
            "atm",
            "ice",
            "lnd",
            "ocn",
        ]:  # Don't create viewer for original component
            vars = get_vars(requested_variables, component)
            if vars:
                url = create_viewer(parameters, vars, component)
                logger.info(f"Viewer URL for {component}: {url}")
                title_and_url_list.append((component, url))
        # Special case for original plots: always use user-provided dimensions.
        vars = get_vars(requested_variables, "original")
        if vars:
            logger.info("Using user provided dimensions for original plots PDF")
            title_and_url_list.append(
                (
                    "original",
                    f"{parameters.figstr}_glb_original.pdf",
                )
            )

        index_url: str = create_viewer_index(parameters.results_dir, title_and_url_list)
        logger.info(f"Viewer index URL: {index_url}")
