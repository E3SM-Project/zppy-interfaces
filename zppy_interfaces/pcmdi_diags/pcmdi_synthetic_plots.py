import argparse
import json
import os
import shutil
import sys
from typing import Dict, List

from zppy_interfaces.multi_utils.logger import _setup_child_logger, _setup_root_logger
from zppy_interfaces.pcmdi_diags.synthetic_plots.synthetic_metrics_plotter import (
    SyntheticMetricsPlotter,
)
from zppy_interfaces.pcmdi_diags.viewer import (
    collect_config,
    generate_data_html,
    generate_methodology_html,
    generate_viewer_html,
)

# Set up the root logger and module level logger. The module level logger is
# a child of the root logger.
_setup_root_logger()
logger = _setup_child_logger(__name__)


# Classes #####################################################################
class SyntheticPlotsParameters(object):
    def __init__(self, args: Dict[str, str]):
        self.figure_sets: List[str] = args["figure_sets"].split(",")
        self.figure_sets_period: List[str] = args["figure_sets_period"].split(",")
        self.figure_format: str = args["figure_format"]
        self.www: str = args["www"]
        self.results_dir: str = args["results_dir"]
        self.case: str = args["case"]
        self.model_name: str = args["model_name"]
        self.model_tableID: str = args["model_tableID"]
        self.web_dir: str = args["web_dir"]
        self.cmip_clim_dir: str = args["cmip_clim_dir"]
        self.cmip_clim_set: str = args["cmip_clim_set"]
        self.cmip_movs_dir: str = args["cmip_movs_dir"]
        self.cmip_movs_set: str = args["cmip_movs_set"]
        self.atm_modes: str = args["atm_modes"]
        self.cpl_modes: str = args["cpl_modes"]
        self.cmip_enso_dir: str = args["cmip_enso_dir"]
        self.cmip_enso_set: str = args["cmip_enso_set"]
        self.groups: List[str] = args["sub_sets"].split(",")
        self.pcmdi_webtitle: str = args["pcmdi_webtitle"]
        self.pcmdi_version: str = args["pcmdi_version"]
        self.run_type: str = args["run_type"]
        self.pcmdi_external_prefix: str = args["pcmdi_external_prefix"]
        self.pcmdi_viewer_template: str = args["pcmdi_viewer_template"]


# Functions ###################################################################
def main():
    args: Dict[str, str] = _get_args()
    parameters = SyntheticPlotsParameters(args)

    #########################################
    # plot synthetic figures for pcmdi metrics
    #########################################
    logger.info("generate synthetic metrics plot ...")
    test_input_path = os.path.join(
        parameters.www,
        "put_model_here",
        "pcmdi_diags",
        parameters.results_dir,
        "metrics_data",
        "%(group_type)",
    )
    metric_dict = json.load(open("synthetic_metrics_list.json"))
    plotter = SyntheticMetricsPlotter(
        case_name=parameters.case,
        test_name=parameters.model_name,
        table_id=parameters.model_tableID,
        figure_format=parameters.figure_format,
        figure_sets=parameters.figure_sets,
        metric_dict=metric_dict,
        save_data=True,
        base_test_input_path=test_input_path,
        results_dir=os.path.join(parameters.web_dir, parameters.results_dir),
        cmip_clim_dir=parameters.cmip_clim_dir,
        cmip_clim_set=parameters.cmip_clim_set,
        cmip_movs_dir=parameters.cmip_movs_dir,
        cmip_movs_set=parameters.cmip_movs_set,
        atm_modes=parameters.atm_modes,
        cpl_modes=parameters.cpl_modes,
        cmip_enso_dir=parameters.cmip_enso_dir,
        cmip_enso_set=parameters.cmip_enso_set,
    )
    # Generate Summary Metrics plots
    # e.g., "climatology,enso,variability"
    logger.info(f"Generating groups={parameters.groups}")
    plotter.generate(parameters.groups)
    logger.info("Generating viewer page for diagnostics...")
    subtitle = parameters.run_type.replace("_", " ").capitalize()
    # figure_sets_period is a list like figure_sets
    figure_sets_period: List[str] = (
        parameters.figure_sets_period
        if isinstance(parameters.figure_sets_period, list)
        else []
    )
    # Validate and unpack periods
    if len(figure_sets_period) == 3:
        clim_period, emov_period, enso_period = [p.strip() for p in figure_sets_period]
    else:
        raise ValueError(
            f"Expected 3 periods (climatology, EMoV, ENSO), "
            f"but got {len(figure_sets_period)}: {figure_sets_period}"
        )
    # Set up paths
    obs_dir = os.path.join(
        parameters.pcmdi_external_prefix, "observations", "Atm", "time-series"
    )
    pmp_dir = os.path.join(parameters.pcmdi_external_prefix, "pcmdi_data")
    out_dir = os.path.join(parameters.web_dir, parameters.results_dir, "viewer")
    os.makedirs(out_dir, exist_ok=True)
    # Copy logo
    web_logo_src = os.path.join(
        parameters.pcmdi_external_prefix,
        parameters.pcmdi_viewer_template,
        "e3sm_pmp_logo.png",
    )
    web_logo_dst = os.path.join(out_dir, "e3sm_pmp_logo.png")
    shutil.copy(web_logo_src, web_logo_dst)
    # Build config
    config = collect_config(
        title=parameters.pcmdi_webtitle,
        subtitle=subtitle,
        version=parameters.pcmdi_version,
        case_id=parameters.case,
        diag_dir=parameters.web_dir,
        obs_dir=obs_dir,
        pmp_dir=pmp_dir,
        out_dir=out_dir,
        clim_period=clim_period,
        emov_period=emov_period,
        enso_period=enso_period,
    )
    # Render viewer
    generate_methodology_html(config)
    generate_data_html(config)
    generate_viewer_html(config)


def _get_args() -> Dict[str, str]:
    # Parser
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        usage="zi-pcmdi-synthetic-plots <args>"
    )

    # For SyntheticPlotsParameters
    parser.add_argument("--synthetic_sets", type=str)
    parser.add_argument("--figure_format", type=str)
    parser.add_argument("--www", type=str)
    parser.add_argument("--results_dir", type=str)
    parser.add_argument("--case", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model_tableID", type=str)
    parser.add_argument("--web_dir", type=str)
    parser.add_argument("--cmip_clim_dir", type=str)
    parser.add_argument("--cmip_clim_set", type=str)
    parser.add_argument("--cmip_movs_dir", type=str)
    parser.add_argument("--cmip_movs_set", type=str)
    parser.add_argument("--atm_modes", type=str)
    parser.add_argument("--cpl_modes", type=str)
    parser.add_argument("--cmip_enso_dir", type=str)
    parser.add_argument("--cmip_enso_set", type=str)
    parser.add_argument("--figure_sets", type=str)
    parser.add_argument("--pcmdi_webtitle", type=str)
    parser.add_argument("--pcmdi_version", type=str)
    parser.add_argument("--run_type", type=str)
    parser.add_argument("--figure_sets_period", type=str)
    parser.add_argument("--pcmdi_external_prefix", type=str)
    parser.add_argument("--pcmdi_viewer_template", type=str)

    # Ignore the first arg
    # (zi-pcmdi-synthetic-plots)
    args: argparse.Namespace = parser.parse_args(sys.argv[1:])

    return vars(args)
