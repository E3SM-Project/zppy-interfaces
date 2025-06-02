from zppy_interfaces.global_time_series.utils import Parameters
from zppy_interfaces.global_time_series.viewer.coupled_global import run_coupled_global
from zppy_interfaces.multi_utils.logger import _setup_custom_logger

logger = _setup_custom_logger(__name__)


def run(parameters: Parameters):
    logger.info("Update time series figures")
    # NOTE: PRODUCES OUTPUT IN THE CURRENT DIRECTORY (not necessarily the case directory)
    # Creates the directory parameters.results_dir
    run_coupled_global(parameters)
