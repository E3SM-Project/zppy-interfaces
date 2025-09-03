import csv
import importlib.resources as imp_res
import os.path
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import cftime
import numpy as np
import xarray
import xcdat

from zppy_interfaces.global_time_series.utils import Parameters
from zppy_interfaces.multi_utils.logger import _setup_child_logger

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
        # If we want to load specific variables,
        # but none are specified,
        # then we can just immediately return.
        return new_var_list
    if exp[exp_key] != "":
        try:
            dataset_wrapper: DatasetWrapper
            if parameters:
                # If this is passed in, then we want to load specific vars.
                dataset_wrapper = DatasetWrapper(exp[exp_key], var_list, parameters)
            else:
                dataset_wrapper = DatasetWrapper(exp[exp_key])
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
            exp["annual"][var_str] = {"glb": (data_array.isel(rgn=0), units)}
            if data_array.sizes["rgn"] > 1:
                # data_array.shape => number of years x 3 regions
                # 3 regions = global, northern hemisphere, southern hemisphere
                # We get here if we used the updated `ts` task
                # (using `rgn_avg` rather than `glb_avg`).
                exp["annual"][var_str]["n"] = (data_array.isel(rgn=1), units)
                exp["annual"][var_str]["s"] = (data_array.isel(rgn=2), units)
            if "year" not in exp["annual"]:
                years: np.ndarray[cftime.DatetimeNoLeap] = data_array.coords[
                    "time"
                ].values
                exp["annual"]["year"] = [x.year for x in years]
        del dataset_wrapper
    return new_var_list
