import math
import os
import traceback
from typing import List

import matplotlib as mpl
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt

from zppy_interfaces.global_time_series.coupled_global.mix_pdf_original import (
    PLOT_DICT,
    get_required_vars,
)
from zppy_interfaces.global_time_series.coupled_global.plots_component import (
    plot_generic,
)
from zppy_interfaces.global_time_series.utils import Parameters
from zppy_interfaces.multi_utils.logger import _setup_custom_logger

mpl.use("Agg")

logger = _setup_custom_logger(__name__)

# This file is for making cumulative PDFs
# Hence, "mode_pdf"

# Used by driver.run ##########################################################


def assemble_cumulative_pdf(
    parameters: Parameters,
    rgn: str,
    component: str,
    xlim,
    exps,
    plot_list: List[str],
    valid_plots: List[str],
    invalid_plots: List[str],
):
    logger.info(f"Assembling Cumulative PDF for rgn={rgn}, component={component}")
    num_plots = len(plot_list)
    if num_plots == 0:
        return

    plots_per_page = parameters.nrows * parameters.ncols
    num_pages = math.ceil(num_plots / plots_per_page)

    counter = 0
    os.makedirs(parameters.results_dir, exist_ok=True)
    # https://stackoverflow.com/questions/58738992/save-multiple-figures-with-subplots-into-a-pdf-with-multiple-pages
    pdf = matplotlib.backends.backend_pdf.PdfPages(
        f"{parameters.results_dir}/{parameters.figstr}_{rgn}_{component}.pdf"
    )
    for page in range(num_pages):
        fig = plt.figure(1, figsize=[13.5, 16.5])
        logger.info(f"Figure size={fig.get_size_inches() * fig.dpi}")
        fig.suptitle(f"{parameters.figstr}_{rgn}_{component}")
        for j in range(plots_per_page):
            logger.info(
                f"Plotting plot {j} on page {page}. This is plot {counter} in total."
            )
            # The final page doesn't need to be filled out with plots.
            if counter >= num_plots:
                break
            ax = plt.subplot(
                parameters.nrows,
                parameters.ncols,
                j + 1,
            )
            plot_name = plot_list[counter]
            if component == "original":
                try:
                    plot_function = PLOT_DICT[plot_name]
                except KeyError:
                    raise KeyError(f"Invalid plot name: {plot_name}")
                try:
                    plot_function(ax, xlim, exps, rgn)
                    valid_plots.append(plot_name)
                except Exception:
                    traceback.print_exc()
                    required_vars: List[str] = get_required_vars(plot_name)
                    logger.error(
                        f"Failed plot_function for {plot_name}. Check that {required_vars} are available."
                    )
                    invalid_plots.append(plot_name)
            else:
                try:
                    plot_generic(ax, xlim, exps, plot_name, rgn)
                    valid_plots.append(plot_name)
                except Exception:
                    traceback.print_exc()
                    logger.error(
                        f"plot_generic failed. Invalid plot={plot_name}, rgn={rgn}"
                    )
                    invalid_plots.append(plot_name)
            counter += 1

        fig.tight_layout()
        pdf.savefig(1)
        # Also save PNGs
        if num_pages > 1:
            fig.savefig(
                f"{parameters.results_dir}/{parameters.figstr}_{rgn}_{component}_{page}.png",
                dpi=150,
            )
        else:
            fig.savefig(
                f"{parameters.results_dir}/{parameters.figstr}_{rgn}_{component}.png",
                dpi=150,
            )
        plt.close(fig)
    pdf.close()
