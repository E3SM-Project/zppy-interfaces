from typing import Tuple

import xarray
import xcdat

from zppy_interfaces.global_time_series.coupled_global_utils import Metric, Variable
from zppy_interfaces.multi_utils.logger import _setup_custom_logger

logger = _setup_custom_logger(__name__)


class DatasetWrapper(object):
    def __init__(self, directory, extra_vars_directory=None):

        self.directory: str = directory

        # `directory` will be of the form `{case_dir}/post/<component>/glb/ts/monthly/{ts_num_years_str}yr/`
        self.dataset: xarray.core.dataset.Dataset = xcdat.open_mfdataset(
            f"{directory}*.nc", center_times=True
        )

        self.extra_var_dataset: xarray.core.dataset.Dataset
        if extra_vars_directory:
            # If an extra_vars_directory is provided, we'll use that to open a new dataset
            self.extra_var_dataset = xcdat.open_mfdataset(
                f"{extra_vars_directory}*.nc",
                center_times=True,
            )
        else:
            # Otherwise, we'll use the same dataset.
            self.extra_var_dataset = self.dataset

    def __del__(self):

        self.dataset.close()

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

        # Is this a derived variable?
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
            # Non-derived variables
            annual_average_dataset_for_var: xarray.core.dataset.Dataset = (
                self.dataset.temporal.group_average(var, "year")
            )
            data_array = annual_average_dataset_for_var.data_vars[var]
            if metric == Metric.TOTAL:
                keys = list(self.extra_var_dataset.keys())
                logger.debug(f"self.extra_var_dataset.keys()={keys}")
                if "area_times_landfrac" in keys:
                    total_land_area = self.extra_var_dataset["area_times_landfrac"]
                else:
                    area: xarray.core.dataarray.DataArray = self.extra_var_dataset[
                        "area"
                    ]
                    landfrac: xarray.core.dataarray.DataArray = self.extra_var_dataset[
                        "landfrac"
                    ]
                    # area.shape() = (180, 360)
                    # landfrac.shape() =(180, 360)
                    total_land_area = (area * landfrac).sum()  # Sum over all dimensions
                # data_array.shape = (number of years, number of regions)
                data_array *= total_land_area
                logger.info(
                    f"for Metric.TOTAL, data_array has been scaled by total_land_area={total_land_area}"
                )
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
