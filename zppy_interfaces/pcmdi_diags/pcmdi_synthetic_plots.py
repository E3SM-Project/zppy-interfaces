import argparse
import json
import os
import shutil
import sys
from typing import Any, Dict, List, Optional

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
    def __init__(self, args: Dict[str, Any]):
        self.save_all_data: bool = str2bool(
            args.get("save_all_data")
            if args.get("save_all_data") is not None
            else False
        )
        self.clim_viewer: bool = str2bool(
            args.get("clim_viewer") if args.get("clim_viewer") is not None else False
        )
        self.mova_viewer: bool = str2bool(
            args.get("mova_viewer") if args.get("mova_viewer") is not None else False
        )
        self.movc_viewer: bool = str2bool(
            args.get("movc_viewer") if args.get("movc_viewer") is not None else False
        )
        self.enso_viewer: bool = str2bool(
            args.get("enso_viewer") if args.get("enso_viewer") is not None else False
        )

        if not any(
            [self.clim_viewer, self.mova_viewer, self.movc_viewer, self.enso_viewer]
        ):
            raise ValueError("At least one diagnostics viewer must be enabled.")

        self.figure_format: str = _required_value(args, "figure_format")
        self.www: str = _required_value(args, "www")
        self.results_dir: str = _required_value(args, "results_dir")
        self.case: str = _required_value(args, "case")
        self.model_name: str = _required_value(args, "model_name")
        self.model_tableID: str = _required_value(args, "model_tableID")
        self.web_dir: str = _required_value(args, "web_dir")
        self.pcmdi_webtitle: str = _required_value(args, "pcmdi_webtitle")
        self.pcmdi_version: str = _required_value(args, "pcmdi_version")
        self.run_type: str = _required_value(args, "run_type")
        self.pcmdi_external_prefix: str = _required_value(args, "pcmdi_external_prefix")
        self.pcmdi_viewer_template: str = _required_value(args, "pcmdi_viewer_template")

        self.clim_vars = _optional_list(args.get("clim_vars"))
        self.clim_regions = _optional_list(args.get("clim_regions"))
        self.clim_years = _viewer_value(args, "clim_years", self.clim_viewer)
        self.cmip_clim_dir = _viewer_value(args, "cmip_clim_dir", self.clim_viewer)
        self.cmip_clim_set = _viewer_value(args, "cmip_clim_set", self.clim_viewer)

        self.mova_modes = _optional_list(args.get("mova_modes"))
        if self.mova_viewer and self.mova_modes is None:
            self.mova_modes = ["NAM", "PNA", "NPO", "NAO", "SAM", "PSA1", "PSA2"]
        self.mova_vars = _optional_list(args.get("mova_vars"))
        self.mova_years = _viewer_value(args, "mova_years", self.mova_viewer)

        self.movc_modes = _optional_list(args.get("movc_modes"))
        if self.movc_viewer and self.movc_modes is None:
            self.movc_modes = ["PDO", "NPGO", "AMO"]
        self.movc_vars = _optional_list(args.get("movc_vars"))
        self.movc_years = _viewer_value(args, "movc_years", self.movc_viewer)
        movs_enabled = self.mova_viewer or self.movc_viewer
        self.cmip_movs_dir = _viewer_value(args, "cmip_movs_dir", movs_enabled)
        self.cmip_movs_set = _viewer_value(args, "cmip_movs_set", movs_enabled)

        self.enso_vars = _optional_list(args.get("enso_vars"))
        self.enso_years = _viewer_value(args, "enso_years", self.enso_viewer)
        self.cmip_enso_dir = _viewer_value(args, "cmip_enso_dir", self.enso_viewer)
        self.cmip_enso_set = _viewer_value(args, "cmip_enso_set", self.enso_viewer)


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
    with open("synthetic_metrics_list.json") as _f:
        metric_dict = json.load(_f)
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
    if not os.path.exists(web_logo_src):
        logger.warning(f"Logo file not found, skipping copy: {web_logo_src}")
    else:
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
        clim_period=parameters.clim_years or "",
        clim_regions=parameters.clim_regions,
        clim_vars=parameters.clim_vars,
        mova_viewer=parameters.mova_viewer,
        mova_modes=parameters.mova_modes,
        mova_vars=parameters.mova_vars,
        mova_period=parameters.mova_years or "",
        movc_viewer=parameters.movc_viewer,
        movc_modes=parameters.movc_modes,
        movc_vars=parameters.movc_vars,
        movc_period=parameters.movc_years or "",
        enso_viewer=parameters.enso_viewer,
        enso_vars=parameters.enso_vars,
        enso_period=parameters.enso_years or "",
    )
    # Render viewer
    generate_methodology_html(config)
    generate_data_html(config)
    generate_viewer_html(config)


def _required_value(args: Dict[str, Any], key: str) -> str:
    value = args.get(key)
    if value is None or not str(value).strip():
        raise ValueError(f"--{key} is required but was not provided.")
    return str(value).strip()


def _viewer_value(
    args: Dict[str, Any], key: str, viewer_enabled: bool
) -> Optional[str]:
    if viewer_enabled:
        return _required_value(args, key)
    value = args.get(key)
    return str(value).strip() if value is not None and str(value).strip() else None


def _optional_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    raw_values = value if isinstance(value, (list, tuple)) else str(value).split(",")
    values = [str(item).strip() for item in raw_values if str(item).strip()]
    return values or None


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
    parser.add_argument("--debug", type=str2bool, default=False)

    # Ignore the first arg
    # (zi-pcmdi-synthetic-plots)
    args: argparse.Namespace = parser.parse_args(sys.argv[1:])

    if args.debug:
        logger.setLevel("DEBUG")
        logger.debug("Debug logging enabled")

    return vars(args)
