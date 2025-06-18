import os
import traceback

import matplotlib as mpl
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np

from zppy_interfaces.global_time_series.coupled_global_plotting import plot
from zppy_interfaces.global_time_series.utils import Parameters
from zppy_interfaces.multi_utils.logger import _setup_custom_logger

mpl.use("Agg")

logger = _setup_custom_logger(__name__)


# FIXME: C901 'make_plot_pdfs' is too complex (20)
def make_plot_pdfs(  # noqa: C901
    parameters: Parameters,
    rgn,
    component,
    xlim,
    exps,
    plot_list,
    valid_plots,
    invalid_plots,
):
    logger.info(
        f"Global Time Series Viewer: make_plot_pdfs for rgn={rgn}, component={component}"
    )
    num_plots = len(plot_list)
    if num_plots == 0:
        return
    nrows = 1
    ncols = 1

    os.makedirs(parameters.results_dir, exist_ok=True)
    for i in range(num_plots):
        fig = plt.figure(1, figsize=[13.5 / 2, 16.5 / 4])
        logger.info(f"Figure size={fig.get_size_inches() * fig.dpi}")
        plot_name = plot_list[i]
        fig.suptitle(plot_name)
        ax = plt.subplot(
            nrows,
            ncols,
            1,
        )
        try:
            plot_generic(ax, xlim, exps, plot_name, rgn)
            valid_plots.append(plot_name)
        except Exception:
            traceback.print_exc()
            logger.error(f"plot_generic failed. Invalid plot={plot_name}, rgn={rgn}")
            invalid_plots.append(plot_name)

        fig.tight_layout()
        # Save individual PNGs
        fig.savefig(
            f"{parameters.results_dir}/{parameters.figstr}_{rgn}_{component}_{plot_name}.png",
            dpi=150,
        )
        plt.close(fig)


# Generic plot function
def plot_generic(ax, xlim, exps, var_name, rgn):
    logger.info(f"plot_generic for {var_name}, rgn={rgn}")
    param_dict = {
        "2nd_var": False,
        "axhline_y": 0,
        "check_exp_ocean": False,
        "check_exp_vol": False,
        "check_exp_year": True,
        "default_ylim": [],
        "do_add_line": True,
        "do_add_trend": True,
        "format": "%4.2f",
        "glb_only": False,
        "lw": 1.0,
        "ohc": False,
        "set_axhline": False,
        "set_legend": True,
        "shorten_year": False,
        "title": var_name,
        "use_getmoc": False,
        "var": lambda exp: np.array(exp["annual"][var_name][rgn][0]),
        "verbose": False,
        "vol": False,
        "ylabel": lambda exp: np.array(exp["annual"][var_name][rgn][1]),
    }
    plot(ax, xlim, exps, param_dict, rgn)
