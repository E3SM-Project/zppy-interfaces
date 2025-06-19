import os
import traceback
from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt

from zppy_interfaces.global_time_series.coupled_global.plots_component import (
    plot_generic,
)
from zppy_interfaces.global_time_series.coupled_global.utils import (
    Metric,
    RequestedVariables,
    Variable,
)
from zppy_interfaces.global_time_series.utils import Parameters
from zppy_interfaces.multi_utils.logger import _setup_child_logger
from zppy_interfaces.multi_utils.viewer import OutputViewer

mpl.use("Agg")

logger = _setup_child_logger(__name__)

# This file is for making Viewers for the component plots.
# Hence, "mix_viewer_component"

# Class #######################################################################


class VariableGroup(object):
    def __init__(self, name: str, variables: List[Variable]):
        self.group_name = name
        self.variables = variables


# Used by coupled_global.run ##################################################


def produce_pngs_for_viewer(
    parameters: Parameters,
    rgn: str,
    component: str,
    xlim,
    exps,
    plot_list: List[str],
    valid_plots: List[str],
    invalid_plots: List[str],
):
    logger.info(f"Assembling Viewer for rgn={rgn}, component={component}")
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


# Used by mode_viewer.produce_viewer ##########################################


def get_vars_component(
    requested_variables: RequestedVariables, component: str
) -> List[Variable]:
    vars: List[Variable]
    if component == "atm":
        vars = requested_variables.vars_atm
    elif component == "ice":
        vars = requested_variables.vars_ice
    elif component == "lnd":
        vars = requested_variables.vars_land
    elif component == "ocn":
        vars = requested_variables.vars_ocn
    else:
        raise ValueError(f"Invalid component={component}")
    return vars


def create_viewer_for_component(
    parameters: Parameters, vars: List[Variable], component: str
) -> str:
    logger.info(f"Creating viewer for {component}")
    if not vars:
        raise RuntimeError("No vars specified for viewer.")
    index_name = f"zppy global time-series plot: {parameters.experiment_name} {component} component ({parameters.year1}-{parameters.year2})"
    viewer = OutputViewer(path=parameters.results_dir, index_name=index_name)
    viewer.add_page(f"table_{component}", parameters.regions)
    groups: List[VariableGroup] = _get_variable_groups(vars)
    if not groups:
        raise RuntimeError("No groups specified for viewer.")
    for group in groups:
        logger.info(f"Adding group {group.group_name}")
        # Only groups that have at least one variable will be returned by `_get_variable_groups`
        # So, we know this group will be non-empty and should therefore be added to the viewer.
        viewer.add_group(group.group_name)
        for var in group.variables:
            plot_name: str = var.variable_name
            row_title: str
            if var.metric == Metric.AVERAGE:
                metric_name = "AVERAGE"
            elif var.metric == Metric.TOTAL:
                metric_name = "TOTAL"
            else:
                # This shouldn't be possible
                raise ValueError(f"Invalid Enum option for metric={var.metric}")
            if var.long_name != "":
                row_title = f"{plot_name}: {var.long_name}, metric={metric_name}"
            else:
                row_title = f"{plot_name}, metric={metric_name}"
            viewer.add_row(row_title)
            for rgn in parameters.regions:
                viewer.add_col(
                    f"{parameters.figstr}_{rgn}_{component}_{plot_name}.png",
                    is_file=True,
                    title=f"{rgn}_{component}_{plot_name}",
                )

    url = viewer.generate_page()
    viewer.generate_viewer()
    # Example links:
    # Viewer is expecting the actual images to be in the directory above `table`.
    # table/index.html links to previews with: ../v3.LR.historical_0051_glb_lnd_FSH.png
    # Viewer is expecting individual image html pages to be under both group and var subdirectories.
    # table/energy-flux/fsh-sensible-heat/glb_lnd_fsh.html links to: ../../../v3.LR.historical_0051_glb_lnd_FSH.png
    return url


def _get_variable_groups(variables: List[Variable]) -> List[VariableGroup]:
    group_names: List[str] = []
    groups: List[VariableGroup] = []
    for v in variables:
        g: str = v.group
        if g not in group_names:
            # A new group!
            group_names.append(g)
            groups.append(VariableGroup(g, [v]))
        else:
            # Add a new variable to this existing group
            for group in groups:
                if g == group.group_name:
                    group.variables.append(v)
    return groups
