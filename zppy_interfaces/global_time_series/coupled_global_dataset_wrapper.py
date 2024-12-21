from typing import Tuple

import xarray
import xcdat

from zppy_interfaces.global_time_series.coupled_global_utils import Metric, Variable
from zppy_interfaces.multi_utils.logger import _setup_custom_logger

logger = _setup_custom_logger(__name__)


class DatasetWrapper(object):
    def __init__(self, directory):

        self.directory: str = directory

        # `directory` will be of the form `{case_dir}/post/<component>/glb/ts/monthly/{ts_num_years_str}yr/`
        self.dataset: xarray.core.dataset.Dataset = xcdat.open_mfdataset(
            f"{directory}*.nc", center_times=True
        )

        self.area_tuple = None

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
