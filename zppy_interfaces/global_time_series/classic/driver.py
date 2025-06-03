import os
import shutil

from zppy_interfaces.global_time_series.classic.coupled_global import run_coupled_global
from zppy_interfaces.global_time_series.classic.ocean_month import ocean_month
from zppy_interfaces.global_time_series.utils import Parameters
from zppy_interfaces.multi_utils.logger import _setup_custom_logger

logger = _setup_custom_logger(__name__)


def run(parameters: Parameters):
    # From zppy's default.ini:
    # Remove the 3 ocean plots (change_ohc,max_moc,change_sea_level) if you don't have ocean data.
    # plots_original = string(default="net_toa_flux_restom,global_surface_air_temperature,toa_radiation,net_atm_energy_imbalance,change_ohc,max_moc,change_sea_level,net_atm_water_imbalance")
    if parameters.use_ocn:
        if set(["change_ohc", "max_moc", "change_sea_level"]) & set(
            parameters.plots_original
        ):
            logger.info("Create ocean time series")
            # NOTE: MODIFIES THE CASE DIRECTORY (parameters.case_dir) post subdirectory
            # Creates the directory post/<subtask>ocn
            os.makedirs(
                f"{parameters.case_dir}/post/{parameters.subtask_name}/ocn/glb/ts/monthly/{parameters.ts_num_years_str}yr",
                exist_ok=True,
            )
            input_dir: str = f"{parameters.input}/{parameters.input_subdir}"
            # NOTE: MODIFIES THE CASE DIRECTORY (parameters.case_dir) post subdirectory
            # Modifies post/ocn (which we just created in the first place)
            ocean_month(
                input_dir,
                parameters.subtask_name,
                parameters.case_dir,
                parameters.year1,
                parameters.year2,
                int(parameters.ts_num_years_str),
            )

            src: str = (
                f"{parameters.case_dir}/post/analysis/mpas_analysis/cache/timeseries/moc/{parameters.moc_file}"
            )
            dst: str = (
                f"{parameters.case_dir}/post/{parameters.subtask_name}/ocn/glb/ts/monthly/{parameters.ts_num_years_str}yr/"
            )
            logger.info(f"Copy moc file from {src} to {dst}")
            # NOTE: MODIFIES THE CASE DIRECTORY (parameters.case_dir) post subdirectory
            # Copies files to post/<subtask>/ocn (which we just created in the first place)
            shutil.copy(
                src,
                dst,
            )
        else:
            logger.info(
                "use_ocn is set unecessarily. ocn plots have not been requested"
            )
    logger.info("Update time series figures")
    # NOTE: PRODUCES OUTPUT IN THE CURRENT DIRECTORY (not necessarily the case directory)
    # Creates the directory parameters.results_dir
    run_coupled_global(parameters)
