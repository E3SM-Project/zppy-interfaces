from zppy_interfaces.global_time_series.utils import Parameters
from zppy_interfaces.multi_utils.logger import _setup_custom_logger
from zppy_interfaces.multi_utils.viewer import OutputViewer

logger = _setup_custom_logger(__name__)

# This file is for making Viewers for the original plots.
# Hence, "mix_viewer_original"

# Used by mode_viewer.produce_viewer ##########################################


def create_viewer_for_original(parameters: Parameters) -> str:
    component: str = "original"
    logger.info(f"Creating viewer for {component}")
    index_name = f"zppy global time-series plot: {parameters.experiment_name} {component} component ({parameters.year1}-{parameters.year2})"
    viewer = OutputViewer(path=parameters.results_dir, index_name=index_name)
    viewer.add_page(f"table_{component}", parameters.regions)
    viewer.add_group("Original Plots")
    # PDFs -- these don't show up when clicked on, but they CAN be downloaded
    row_title: str = "Original Plots, PDFs"
    viewer.add_row(row_title)
    for rgn in parameters.regions:
        viewer.add_col(
            f"{parameters.figstr}_{rgn}_{component}.pdf",
            is_file=True,
            title=f"{rgn}_{component}",
        )
    # PNGs -- these show up when clicked on
    row_title = "Original Plots, PNGs"
    viewer.add_row(row_title)
    for rgn in parameters.regions:
        viewer.add_col(
            f"{parameters.figstr}_{rgn}_{component}.png",
            is_file=True,
            title=f"{rgn}_{component}",
        )
    url = viewer.generate_page()
    viewer.generate_viewer()
    return url
