from enum import Enum
from typing import Any, Dict, List

import cftime
import numpy as np
import xarray

from zppy_interfaces.global_time_series.coupled_global_dataset_wrapper import (
    DatasetWrapper,
)
from zppy_interfaces.global_time_series.utils import Parameters
from zppy_interfaces.multi_utils.logger import _setup_custom_logger

logger = _setup_custom_logger(__name__)


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
    rgn: str,
    load_all_vars: bool = False,
) -> List[Variable]:
    new_var_list: List[Variable] = []
    if exp[exp_key] != "":
        try:
            dataset_wrapper: DatasetWrapper
            if load_all_vars:
                dataset_wrapper = DatasetWrapper(exp[exp_key])
            else:
                dataset_wrapper = DatasetWrapper(exp[exp_key], var_list)
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
