import csv
import importlib.resources as imp_res
import multiprocessing as mp
import os
import os.path
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import cftime
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray
import xcdat

from zppy_interfaces.global_time_series.utils import Parameters
from zppy_interfaces.multi_utils.logger import _setup_child_logger

# Set matplotlib backend and xarray options
matplotlib.use("Agg")  # Use non-interactive backend
xarray.set_options(use_new_combine_kwarg_defaults=True)

logger = _setup_child_logger(__name__)


class Metric(Enum):
    AVERAGE = 1
    TOTAL = 2


class Variable(object):
    def __init__(
        self,
        variable_name,
        metric=Metric.AVERAGE,
        scale_factor=1.0,
        original_units="",
        final_units="",
        group="All Variables",
        long_name="",
    ):
        # The name of the EAM/ELM/etc. variable on the monthly h0 history file
        self.variable_name: str = variable_name

        # These fields are used for computation
        # Global average over land area or global total
        self.metric: Metric = metric
        # The factor that should convert from original_units to final_units, after standard processing with nco
        self.scale_factor: float = scale_factor
        # Test string for the units as given on the history file (included here for possible testing)
        self.original_units: str = original_units
        # The units that should be reported in time series plots, based on metric and scale_factor
        self.final_units: str = final_units

        # These fields are used for plotting
        # A name used to cluster variables together, to be separated in groups within the output web pages
        self.group: str = group
        # Descriptive text to add to the plot page to help users identify the variable
        self.long_name: str = long_name


class RequestedVariables(object):
    def __init__(self, parameters: Parameters):
        # Original plots
        self.vars_original: List[Variable] = get_vars_original(
            parameters.plots_original
        )
        # Component plots
        # > Land variables are constructed differently
        self.vars_land: List[Variable] = construct_land_variables(parameters.plots_lnd)
        # > Other variables use the generic constructor
        self.vars_atm: List[Variable] = construct_generic_variables(
            parameters.plots_atm
        )
        self.vars_ice: List[Variable] = construct_generic_variables(
            parameters.plots_ice
        )
        self.vars_ocn: List[Variable] = construct_generic_variables(
            parameters.plots_ocn
        )


# For plots_original


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


# For plots_components


def get_variable_files(
    var_name: str, directory: str, parameters: Parameters
) -> List[str]:
    """Get list of NetCDF files for a single variable."""
    file_path_list: List[str] = []
    num_years: int = int(parameters.ts_num_years_str)
    y1: int = parameters.year1
    y2: int = parameters.year1 + num_years - 1

    while y2 <= parameters.year2:
        file_path: str = f"{directory}{var_name}_{y1:04d}01_{y2:04d}12.nc"
        if os.path.exists(file_path):
            file_path_list.append(file_path)
        y1 += num_years
        y2 += num_years

    return file_path_list


def apply_scaling(
    data_array: xarray.core.dataarray.DataArray,
    metric: Metric,
    dataset: xarray.core.dataset.Dataset,
) -> xarray.core.dataarray.DataArray:
    """Apply area scaling for TOTAL metrics."""
    if metric != Metric.TOTAL:
        return data_array

    # Calculate land areas for scaling
    keys = list(dataset.keys())
    if "valid_area_per_gridcell" in keys:
        land_area_per_gridcell = dataset["valid_area_per_gridcell"]
        total_land_area = land_area_per_gridcell.sum()
        north_land_area = land_area_per_gridcell.where(
            land_area_per_gridcell.lat >= 0
        ).sum()
        south_land_area = land_area_per_gridcell.where(
            land_area_per_gridcell.lat < 0
        ).sum()
    else:
        area = dataset["area"]
        landfrac = dataset["landfrac"]
        total_land_area = (area * landfrac).sum()
        north_area = area.where(area.lat >= 0)
        north_landfrac = landfrac.where(landfrac.lat >= 0)
        north_land_area = (north_area * north_landfrac).sum()
        south_area = area.where(area.lat < 0)
        south_landfrac = landfrac.where(landfrac.lat < 0)
        south_land_area = (south_area * south_landfrac).sum()

    # Apply scaling
    data_array[:, 0] *= total_land_area
    data_array[:, 1] *= north_land_area
    data_array[:, 2] *= south_land_area

    return data_array


def process_variable(
    var: Variable, directory: str, parameters: Parameters
) -> Tuple[xarray.core.dataarray.DataArray, str]:
    """Process a single variable independently - load, compute, cleanup."""
    try:
        # 1. Get file paths for this variable
        file_paths = get_variable_files(var.variable_name, directory, parameters)
        if not file_paths:
            raise ValueError(f"No data files found for variable {var.variable_name}")

        # 2. Load only this variable's data
        dataset = xcdat.open_mfdataset(file_paths, center_times=True)

        try:
            # 3. Compute annual average
            annual_dataset = dataset.temporal.group_average(var.variable_name, "year")
            data_array = annual_dataset.data_vars[var.variable_name]

            # 4. Apply area scaling if needed
            data_array = apply_scaling(data_array, var.metric, dataset)

            # 5. Apply unit scaling
            units = data_array.units
            if (
                (units != "1")
                and (var.original_units != "")
                and var.original_units != units
            ):
                raise ValueError(f"Units don't match: {units} vs {var.original_units}")
            if (var.scale_factor != 1) and (var.final_units != ""):
                data_array *= var.scale_factor
                units = var.final_units

            return data_array, units

        finally:
            # 6. Always cleanup immediately
            dataset.close()

    except Exception as e:
        logger.error(f"Failed to process variable {var.variable_name}: {e}")
        raise


def process_variable_worker(args):
    """
    Worker function for multiprocessing - unpacks arguments and processes single variable.

    Args:
        args: Tuple of (var, directory, parameters)

    Returns:
        Tuple of (var_name, data_array, units, success_flag, error_msg)
    """
    var, directory, parameters = args
    var_name = var.variable_name

    try:
        data_array, units = process_variable(var, directory, parameters)
        return (var_name, data_array, units, True, None)
    except Exception as e:
        return (var_name, None, None, False, str(e))


def process_and_plot_worker(args):
    """
    Combined worker: process variable and generate plots immediately.

    Args:
        args: Tuple of (var, directory, parameters, plot_config)

    Returns:
        Tuple of (var_name, success_flag, error_msg, plot_info, data_array, units)
    """
    var, directory, parameters, plot_config = args
    var_name = var.variable_name

    try:
        # Process the variable first
        data_array, units = process_variable(var, directory, parameters)

        # Check if we got valid data
        if data_array is None:
            return (
                var_name,
                False,
                "No data returned from processing",
                None,
                None,
                None,
            )

        # Generate plots only if processing succeeded
        component_name = plot_config.get("component", "lnd")
        plot_info = generate_variable_plots(
            var_name, data_array, units, parameters, plot_config, component_name
        )

        # Return data for populating exp["annual"] - don't delete here
        return (var_name, True, None, plot_info, data_array, units)

    except Exception as e:
        # Processing failed - don't attempt plotting
        return (var_name, False, str(e), None, None, None)


def generate_variable_plots(
    var_name: str,
    data_array: xarray.core.dataarray.DataArray,
    units: str,
    parameters: Parameters,
    plot_config: Dict[str, Any],
    component: str,
) -> Dict[str, Any]:
    """
    Generate PNG plots for a single variable immediately after processing.

    Args:
        var_name: Variable name
        data_array: Processed data array
        units: Data units
        parameters: Processing parameters
        plot_config: Plotting configuration

    Returns:
        Dictionary with plot file paths and metadata
    """
    plot_info: Dict[str, Any] = {"var_name": var_name, "plots": []}

    # Validate input data
    if data_array is None:
        logger.error(f"Cannot plot {var_name}: data_array is None")
        return plot_info

    if data_array.size == 0:
        logger.error(f"Cannot plot {var_name}: data_array is empty")
        return plot_info

    # Create temporary exp structure for plotting compatibility
    temp_exp = {
        "annual": {var_name: {"glb": (data_array.isel(rgn=0), units)}},
        "color": plot_config.get("color", "blue"),
        "name": plot_config.get("name", "data"),
        "yoffset": plot_config.get("yoffset", 0),
        "yr": ([parameters.year1, parameters.year2],),
    }

    if data_array.sizes["rgn"] > 1:
        temp_exp["annual"][var_name]["n"] = (data_array.isel(rgn=1), units)
        temp_exp["annual"][var_name]["s"] = (data_array.isel(rgn=2), units)

    # Add year data
    years = data_array.coords["time"].values
    temp_exp["annual"]["year"] = [x.year for x in years]

    # Generate plots for each region
    regions = ["glb"]
    if data_array.sizes["rgn"] > 1:
        regions.extend(["n", "s"])

    for rgn in regions:
        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Use existing plot_generic function with proper xlim
            xlim = [parameters.year1, parameters.year2]
            from zppy_interfaces.global_time_series.coupled_global.plots_component import (
                plot_generic,
            )

            plot_generic(ax, xlim, [temp_exp], var_name, rgn)

            # Ensure results directory exists before saving plot
            os.makedirs(parameters.results_dir, exist_ok=True)

            # Save plot
            plot_filename = f"{parameters.figstr}_{rgn}_{component}_{var_name}.png"
            plot_path = f"{parameters.results_dir}/{plot_filename}"
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            plot_info["plots"].append(
                {"region": rgn, "filename": plot_filename, "path": plot_path}
            )

            logger.debug(f"Generated plot: {plot_filename}")

        except Exception as e:
            logger.error(f"Failed to generate plot for {var_name}_{rgn}: {e}")

    return plot_info


def process_variables_parallel(
    var_list: List[Variable],
    directory: str,
    parameters: Parameters,
    num_processes: Optional[int] = None,
) -> Dict[str, Tuple[xarray.core.dataarray.DataArray, str]]:
    """
    Process multiple variables in parallel using multiprocessing.Pool.

    Args:
        var_list: List of variables to process
        directory: Data directory path
        parameters: Processing parameters
        num_processes: Number of parallel processes (default: CPU count)

    Returns:
        Dictionary mapping variable names to (data_array, units) tuples

    Raises:
        Exception: If no variables processed successfully
    """
    if num_processes is None:
        num_processes = min(16, len(var_list))

    logger.info(
        f"Starting parallel processing of {len(var_list)} variables with {num_processes} processes"
    )

    # Prepare arguments for worker processes
    worker_args = [(var, directory, parameters) for var in var_list]

    results = {}
    failed_vars = []

    try:
        with mp.Pool(processes=num_processes) as pool:
            # Process all variables in parallel
            worker_results = pool.map(process_variable_worker, worker_args)

        # Collect results
        for var_name, data_array, units, success, error_msg in worker_results:
            if success:
                results[var_name] = (data_array, units)
                logger.info(f"✓ Completed processing variable: {var_name}")
            else:
                failed_vars.append(var_name)
                logger.error(f"✗ Failed processing variable {var_name}: {error_msg}")

    except Exception as e:
        logger.error(f"Parallel processing failed: {e}")
        raise

    if failed_vars:
        logger.warning(f"Failed to process {len(failed_vars)} variables: {failed_vars}")

    if not results:
        raise Exception("No variables processed successfully")

    logger.info(
        f"Parallel processing complete. {len(results)}/{len(var_list)} variables processed successfully"
    )
    return results


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
                    var_list.append(_land_csv_row_to_var(row_elements_strip_whitespace))
    return var_list


def _land_csv_row_to_var(csv_row: List[str]) -> Variable:
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


class DatasetWrapper(object):
    def __init__(
        self,
        directory: str,
        var_list: Optional[List[Variable]] = None,
        parameters: Optional[Parameters] = None,
    ):

        self.directory: str = directory
        self.var_list: Optional[List[Variable]] = var_list

        # `directory` will be of the form `{case_dir}/post/<component>/glb/ts/monthly/{ts_num_years_str}yr/`
        self.dataset: xarray.core.dataset.Dataset
        if var_list and parameters:
            file_path_list: List[str] = []
            for var in var_list:
                num_years: int = int(parameters.ts_num_years_str)
                y1: int = parameters.year1
                y2: int = parameters.year1 + num_years - 1
                while y2 <= parameters.year2:
                    # `var.variable_name` will be of the form `FSNS`, `FLNS`, etc.
                    file_path: str = (
                        f"{directory}{var.variable_name}_{y1:04d}01_{y2:04d}12.nc"
                    )
                    if os.path.exists(file_path):
                        file_path_list.append(file_path)
                    else:
                        logger.info(f"{file_path} does not exist.")
                    y1 += num_years
                    y2 += num_years
            self.dataset = xcdat.open_mfdataset(file_path_list, center_times=True)
        else:
            self.dataset = xcdat.open_mfdataset(f"{directory}*.nc", center_times=True)
        self.area_tuple: Optional[Tuple[Any, Any, Any]] = None

    def set_area_tuple(self):
        keys = list(self.dataset.keys())
        if "valid_area_per_gridcell" in keys:
            logger.debug("Setting area_tuple, using valid_area_per_gridcell")
            land_area_per_gridcell = self.dataset["valid_area_per_gridcell"]
            # land_area_per_gridcell.shape = (360, 720)
            logger.debug(f"land_area_per_gridcell.shape={land_area_per_gridcell.shape}")
            total_land_area = land_area_per_gridcell.sum()  # Sum over all dimensions
            # Account for hemispheric plots:
            north_land_area = land_area_per_gridcell.where(
                land_area_per_gridcell.lat >= 0
            ).sum()
            south_land_area = land_area_per_gridcell.where(
                land_area_per_gridcell.lat < 0
            ).sum()
        else:
            logger.debug("Setting area_tuple, using area and landfrac")
            area: xarray.core.dataarray.DataArray = self.dataset["area"]
            landfrac: xarray.core.dataarray.DataArray = self.dataset["landfrac"]

            # area.shape = (180, 360)
            logger.debug(f"area.shape={area.shape}")
            # landfrac.shape = (180, 360)
            logger.debug(f"landfrac.shape={landfrac.shape}")

            total_land_area = (area * landfrac).sum()  # Sum over all dimensions

            # Account for hemispheric plots:
            north_area = area.where(area.lat >= 0)
            north_landfrac = landfrac.where(landfrac.lat >= 0)
            north_land_area = (north_area * north_landfrac).sum()

            south_area = area.where(area.lat < 0)
            south_landfrac = landfrac.where(landfrac.lat < 0)
            south_land_area = (south_area * south_landfrac).sum()

        logger.debug(f"total_land_area.shape={total_land_area.shape}")
        logger.debug(f"north_land_area.shape={north_land_area.shape}")
        logger.debug(f"south_land_area.shape={south_land_area.shape}")

        # logger.debug(f"total_land_area={total_land_area.item()}")
        # logger.debug(f"north_land_area={north_land_area.item()}")
        # logger.debug(f"south_land_area={south_land_area.item()}")

        self.area_tuple = (total_land_area, north_land_area, south_land_area)
        # logger.debug(f"For Metric.TOTAL, data_array's glb,n,s will be scaled respectively by {self.area_tuple}")

    def __del__(self):

        try:
            self.dataset.close()
        except AttributeError:
            raise AttributeError(
                "DatasetWrapper.dataset was not set. This could be because of a failure producing the dataset."
            )

    def globalAnnualHelper(
        self,
        var: str,
        metric: Metric,
        scale_factor: float,
        original_units: str,
        final_units: str,
    ) -> Tuple[xarray.core.dataarray.DataArray, str]:

        data_array: xarray.core.dataarray.DataArray
        units: str = ""

        # Constants, from AMWG diagnostics
        Lv = 2.501e6
        Lf = 3.337e5

        if (not self.var_list) and (
            var in ["RESTOM", "RESTOA", "LHFLX", "RESSURF", "PREC"]
        ):
            # We've loaded ALL variables.
            # That means we can attempt derivations from other variables
            # not explicitally requested.
            if var == "RESTOM":
                FSNT, _ = self.globalAnnualHelper(
                    "FSNT", metric, scale_factor, original_units, final_units
                )
                FLNT, _ = self.globalAnnualHelper(
                    "FLNT", metric, scale_factor, original_units, final_units
                )
                data_array = FSNT - FLNT
            elif var == "RESTOA":
                logger.warning("NOT READY")
                FSNTOA, _ = self.globalAnnualHelper(
                    "FSNTOA", metric, scale_factor, original_units, final_units
                )
                FLUT, _ = self.globalAnnualHelper(
                    "FLUT", metric, scale_factor, original_units, final_units
                )
                data_array = FSNTOA - FLUT
            elif var == "LHFLX":
                QFLX, _ = self.globalAnnualHelper(
                    "QFLX", metric, scale_factor, original_units, final_units
                )
                PRECC, _ = self.globalAnnualHelper(
                    "PRECC", metric, scale_factor, original_units, final_units
                )
                PRECL, _ = self.globalAnnualHelper(
                    "PRECL", metric, scale_factor, original_units, final_units
                )
                PRECSC, _ = self.globalAnnualHelper(
                    "PRECSC", metric, scale_factor, original_units, final_units
                )
                PRECSL, _ = self.globalAnnualHelper(
                    "PRECSL", metric, scale_factor, original_units, final_units
                )
                data_array = (Lv + Lf) * QFLX - Lf * 1.0e3 * (
                    PRECC + PRECL - PRECSC - PRECSL
                )
            elif var == "RESSURF":
                FSNS, _ = self.globalAnnualHelper(
                    "FSNS", metric, scale_factor, original_units, final_units
                )
                FLNS, _ = self.globalAnnualHelper(
                    "FLNS", metric, scale_factor, original_units, final_units
                )
                SHFLX, _ = self.globalAnnualHelper(
                    "SHFLX", metric, scale_factor, original_units, final_units
                )
                LHFLX, _ = self.globalAnnualHelper(
                    "LHFLX", metric, scale_factor, original_units, final_units
                )
                data_array = FSNS - FLNS - SHFLX - LHFLX
            elif var == "PREC":
                PRECC, _ = self.globalAnnualHelper(
                    "PRECC", metric, scale_factor, original_units, final_units
                )
                PRECL, _ = self.globalAnnualHelper(
                    "PRECL", metric, scale_factor, original_units, final_units
                )
                data_array = 1.0e3 * (PRECC + PRECL)
            else:
                raise ValueError(f"Invalid var={var}")
        else:
            # Non-derived variables
            annual_average_dataset_for_var: xarray.core.dataset.Dataset = (
                self.dataset.temporal.group_average(var, "year")
            )
            data_array = annual_average_dataset_for_var.data_vars[var]
            if metric == Metric.TOTAL:
                if not self.area_tuple:
                    self.set_area_tuple()
                # Appease the type checker (avoid `Value of type "Optional[Any]" is not indexable`)
                if not self.area_tuple:
                    raise ValueError("area_tuple still not set")
                # data_array.shape = (number of years, number of regions)
                # We want to keep those dimensions, but with these values:
                # (glb*total_land_area, n*north_land_area, s*south_land_area)
                try:
                    data_array[:, 0] *= self.area_tuple[0]
                    data_array[:, 1] *= self.area_tuple[1]
                    data_array[:, 2] *= self.area_tuple[2]
                except Exception as e:
                    logger.error(f"Error while scaling data_array: {e}")
                    raise e
            units = data_array.units
            # `units` will be "1" if it's a dimensionless quantity
            if (units != "1") and (original_units != "") and original_units != units:
                raise ValueError(
                    f"Units don't match up: Have {units} but expected {original_units}. This renders the supplied scale_factor ({scale_factor}) unusable."
                )
            if (scale_factor != 1) and (final_units != ""):
                data_array *= scale_factor
                units = final_units
        return data_array, units

    def globalAnnual(
        self, var: Variable
    ) -> Tuple[xarray.core.dataarray.DataArray, str]:
        return self.globalAnnualHelper(
            var.variable_name,
            var.metric,
            var.scale_factor,
            var.original_units,
            var.final_units,
        )


# Helper functions ############################################################
def get_data_dir(parameters: Parameters, component: str, conditional: bool) -> str:
    return (
        f"{parameters.case_dir}/post/{component}/glb/ts/monthly/{parameters.ts_num_years_str}yr/"
        if conditional
        else ""
    )


def set_var_parallel_with_plots(
    exp: Dict[str, Any],
    exp_key: str,
    var_list: List[Variable],
    valid_vars: List[str],
    invalid_vars: List[str],
    parameters: Parameters,
    num_processes: Optional[int] = None,
) -> List[Variable]:
    """Combined parallel processing + plotting version."""
    new_var_list: List[Variable] = []
    if var_list == []:
        return new_var_list

    if exp[exp_key] != "":
        directory = exp[exp_key]

        # Map exp_key to component name for filename
        component_map = {
            "atmos": "atm",
            "ice": "ice",
            "land": "lnd",
            "ocean": "ocn",
        }
        component_name = component_map.get(exp_key, exp_key)

        # Combined processing + plotting
        plot_config = {
            "color": exp.get("color", "blue"),
            "name": exp.get("name", "data"),
            "yoffset": exp.get("yoffset", 0),
            "component": component_name,
        }

        if num_processes is None:
            num_processes = min(16, len(var_list))

        logger.info(f"Processing {len(var_list)} variables")

        worker_results = []
        for i, var in enumerate(var_list):
            logger.info(f"Processing {i + 1}/{len(var_list)}: {var.variable_name}")
            try:
                result = process_and_plot_worker(
                    (var, directory, parameters, plot_config)
                )
                worker_results.append(result)
                if not result[1]:
                    logger.error(f"Failed {var.variable_name}: {result[2]}")
            except Exception as e:
                logger.error(f"Exception processing {var.variable_name}: {e}")
                worker_results.append(
                    (var.variable_name, False, str(e), None, None, None)
                )

        # Process results
        for (
            var_name,
            success,
            error_msg,
            plot_info,
            data_array,
            units,
        ) in worker_results:
            if success:
                valid_vars.append(var_name)
                var_obj = next(v for v in var_list if v.variable_name == var_name)
                new_var_list.append(var_obj)

                exp["annual"][var_name] = {"glb": (data_array.isel(rgn=0), units)}
                if data_array.sizes["rgn"] > 1:
                    exp["annual"][var_name]["n"] = (
                        data_array.isel(rgn=1),
                        units,
                    )
                    exp["annual"][var_name]["s"] = (
                        data_array.isel(rgn=2),
                        units,
                    )
                if "year" not in exp["annual"]:
                    years = data_array.coords["time"].values
                    exp["annual"]["year"] = [x.year for x in years]

                del data_array
            else:
                invalid_vars.append(var_name)

    return new_var_list


def set_var_parallel(
    exp: Dict[str, Any],
    exp_key: str,
    var_list: List[Variable],
    valid_vars: List[str],
    invalid_vars: List[str],
    parameters: Parameters,
    num_processes: Optional[int] = None,
) -> List[Variable]:
    """Parallel version of set_var for component plots."""
    new_var_list: List[Variable] = []
    if var_list == []:
        return new_var_list

    if exp[exp_key] != "":
        directory = exp[exp_key]

        if len(var_list) > 1:
            # Use parallel processing
            results = process_variables_parallel(
                var_list, directory, parameters, num_processes
            )

            # Process results
            for var in var_list:
                var_str = var.variable_name
                if var_str in results:
                    data_array, units = results[var_str]
                    valid_vars.append(var_str)
                    new_var_list.append(var)

                    exp["annual"][var_str] = {"glb": (data_array.isel(rgn=0), units)}
                    if data_array.sizes["rgn"] > 1:
                        exp["annual"][var_str]["n"] = (data_array.isel(rgn=1), units)
                        exp["annual"][var_str]["s"] = (data_array.isel(rgn=2), units)
                    if "year" not in exp["annual"]:
                        years: np.ndarray[cftime.DatetimeNoLeap] = data_array.coords[
                            "time"
                        ].values
                        exp["annual"]["year"] = [x.year for x in years]
                else:
                    invalid_vars.append(var_str)
        else:
            # Single variable - use sequential
            return set_var(exp, exp_key, var_list, valid_vars, invalid_vars, parameters)

    return new_var_list


def set_var(
    exp: Dict[str, Any],
    exp_key: str,
    var_list: List[Variable],
    valid_vars: List[str],
    invalid_vars: List[str],
    parameters: Optional[Parameters] = None,
) -> List[Variable]:
    new_var_list: List[Variable] = []
    if parameters and (var_list == []):
        return new_var_list

    if exp[exp_key] != "":
        directory = exp[exp_key]

        for var in var_list:
            var_str: str = var.variable_name
            try:
                if parameters:
                    # Use simplified approach with lazy loading
                    data_array, units = process_variable(var, directory, parameters)
                else:
                    # Legacy fallback: use DatasetWrapper for backward compatibility
                    dataset_wrapper = DatasetWrapper(directory)
                    data_array, units = dataset_wrapper.globalAnnual(var)
                    del dataset_wrapper

                valid_vars.append(str(var_str))
                new_var_list.append(var)
            except Exception as e:
                logger.error(e)
                logger.error(f"Processing failed for {var_str}")
                invalid_vars.append(str(var_str))
                continue

            exp["annual"][var_str] = {"glb": (data_array.isel(rgn=0), units)}
            if data_array.sizes["rgn"] > 1:
                exp["annual"][var_str]["n"] = (data_array.isel(rgn=1), units)
                exp["annual"][var_str]["s"] = (data_array.isel(rgn=2), units)
            if "year" not in exp["annual"]:
                years: np.ndarray[cftime.DatetimeNoLeap] = data_array.coords[
                    "time"
                ].values
                exp["annual"]["year"] = [x.year for x in years]

    return new_var_list
