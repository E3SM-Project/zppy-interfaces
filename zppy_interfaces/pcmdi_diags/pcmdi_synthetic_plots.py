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
        self.figure_format: str = args["figure_format"]
        self.www: str = args["www"]
        self.save_all_data: bool = str(args["save_all_data"]).lower() in (
            "true",
            "1",
            "yes",
        )
        self.results_dir: str = args["results_dir"]
        self.case: str = args["case"]
        self.model_name: str = args["model_name"]
        self.model_tableID: str = args["model_tableID"]
        self.web_dir: str = args["web_dir"]
        self.clim_viewer: bool = str(args["clim_viewer"]).lower() in (
            "true",
            "1",
            "yes",
        )
        self.clim_vars: List[str] = args["clim_vars"].split(",")
        self.clim_years: str = args["clim_years"]
        self.clim_regions: List[str] = args["clim_regions"].split(",")
        self.cmip_clim_dir: str = args["cmip_clim_dir"]
        self.cmip_clim_set: str = args["cmip_clim_set"]
        self.mova_viewer: bool = str(args["mova_viewer"]).lower() in (
            "true",
            "1",
            "yes",
        )
        self.mova_modes: List[str] = args["mova_modes"].split(",")
        self.mova_vars: List[str] = args["mova_vars"].split(",")
        self.mova_years: str = args["mova_years"]
        self.movc_viewer: bool = str(args["movc_viewer"]).lower() in (
            "true",
            "1",
            "yes",
        )
        self.movc_modes: List[str] = args["movc_modes"].split(",")
        self.movc_vars: List[str] = args["movc_vars"].split(",")
        self.movc_years: str = args["movc_years"]
        self.cmip_movs_dir: str = args["cmip_movs_dir"]
        self.cmip_movs_set: str = args["cmip_movs_set"]
        self.enso_viewer: bool = str(args["enso_viewer"]).lower() in (
            "true",
            "1",
            "yes",
        )
        self.enso_vars: List[str] = args["enso_vars"].split(",")
        self.enso_years: str = args["enso_years"]
        self.cmip_enso_dir: str = args["cmip_enso_dir"]
        self.cmip_enso_set: str = args["cmip_enso_set"]
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
        # Core
        case_name=parameters.case,
        test_name=parameters.model_name,
        table_id=parameters.model_tableID,
        figure_format=parameters.figure_format,
        metric_dict=metric_dict,
        save_data=parameters.save_all_data,
        base_test_input_path=test_input_path,
        results_dir=os.path.join(parameters.web_dir, parameters.results_dir),
        # Mean climate
        clim_viewer=parameters.clim_viewer,
        clim_vars=parameters.clim_vars,
        clim_regions=parameters.clim_regions,
        cmip_clim_dir=parameters.cmip_clim_dir,
        cmip_clim_set=parameters.cmip_clim_set,
        # MOVA
        mova_viewer=parameters.mova_viewer,
        mova_modes=parameters.mova_modes,
        # MOVC
        movc_viewer=parameters.movc_viewer,
        movc_modes=parameters.movc_modes,
        cmip_movs_dir=parameters.cmip_movs_dir,
        cmip_movs_set=parameters.cmip_movs_set,
        # ENSO
        enso_viewer=parameters.enso_viewer,
        cmip_enso_dir=parameters.cmip_enso_dir,
        cmip_enso_set=parameters.cmip_enso_set,
    )

    # Generate Summary Metrics plots
    # e.g., "climatology,enso,variability"
    figure_sets = []
    if parameters.clim_viewer:
        figure_sets.append("climatology")
    if parameters.mova_viewer:
        figure_sets.append("variability(ATM)")
    if parameters.movc_viewer:
        figure_sets.append("variability(CPL)")
    if parameters.enso_viewer:
        figure_sets.append("enso")

    logger.info(f"Generating groups={figure_sets}")
    # This calls the _handle_{figure_set} functions
    # Those call the {figure_set}_plot_driver functions
    plotter.generate()

    logger.info("Generating viewer page for diagnostics...")
    subtitle = parameters.run_type.replace("_", " ").capitalize()

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
        clim_viewer=parameters.clim_viewer,
        clim_period=parameters.clim_years,
        clim_regions=parameters.clim_regions,
        clim_vars=parameters.clim_vars,
        mova_viewer=parameters.mova_viewer,
        mova_modes=parameters.mova_modes,
        mova_period=parameters.mova_years,
        movc_viewer=parameters.movc_viewer,
        movc_modes=parameters.movc_modes,
        movc_period=parameters.movc_years,
        enso_viewer=parameters.enso_viewer,
        enso_period=parameters.enso_years,
    )
    # Render viewer
    generate_methodology_html(config)
    generate_data_html(config)
    generate_viewer_html(config)


def str2bool(v):
    if isinstance(v, bool):
        return v
    val = str(v).lower()
    if val in ("yes", "true", "t", "1", "y", "on"):
        return True
    elif val in ("no", "false", "f", "0", "n", "off"):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


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
    parser.add_argument("--clim_viewer", type=str2bool)
    parser.add_argument("--clim_vars", type=str)
    parser.add_argument("--clim_years", type=str)
    parser.add_argument("--clim_regions", type=str)
    parser.add_argument("--cmip_clim_dir", type=str)
    parser.add_argument("--cmip_clim_set", type=str)
    parser.add_argument("--mova_viewer", type=str2bool)
    parser.add_argument("--mova_modes", type=str)
    parser.add_argument("--mova_vars", type=str)
    parser.add_argument("--mova_years", type=str)
    parser.add_argument("--movc_viewer", type=str2bool)
    parser.add_argument("--movc_modes", type=str)
    parser.add_argument("--movc_vars", type=str)
    parser.add_argument("--movc_years", type=str)
    parser.add_argument("--cmip_movs_dir", type=str)
    parser.add_argument("--cmip_movs_set", type=str)
    parser.add_argument("--enso_viewer", type=str2bool)
    parser.add_argument("--enso_vars", type=str)
    parser.add_argument("--enso_years", type=str)
    parser.add_argument("--cmip_enso_dir", type=str)
    parser.add_argument("--cmip_enso_set", type=str)
    parser.add_argument("--pcmdi_webtitle", type=str)
    parser.add_argument("--pcmdi_version", type=str)
    parser.add_argument("--run_type", type=str)
    parser.add_argument("--pcmdi_external_prefix", type=str)
    parser.add_argument("--pcmdi_viewer_template", type=str)
    parser.add_argument("--save_all_data", type=str2bool)
    parser.add_argument("--debug", type=str)

    # Ignore the first arg
    # (zi-pcmdi-synthetic-plots)
    args: argparse.Namespace = parser.parse_args(sys.argv[1:])

    if args.debug and args.debug.lower() == "true":
        logger.setLevel("DEBUG")
        logger.debug("Debug logging enabled")

    return vars(args)
