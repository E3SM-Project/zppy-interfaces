import argparse
import glob
import json
import os
import re
import sys
import time
from collections import OrderedDict
from typing import Dict, List

from zppy_interfaces.multi_utils.logger import _setup_custom_logger
from zppy_interfaces.pcmdi_diags.pcmdi_setup import CoreOutput, CoreParameters, set_up
from zppy_interfaces.pcmdi_diags.utils import run_parallel_jobs, run_serial_jobs

logger = _setup_custom_logger(__name__)


# Classes #####################################################################
class ENSOParameters(object):
    def __init__(self, args: Dict[str, str]):
        self.enso_groups: str = args["enso_groups"]


class EnsoDiagnosticsCollector:
    def __init__(
        self, fig_format, refname, model_name_parts, case_id, input_dir, output_dir
    ):
        self.fig_format = fig_format
        self.refname = refname
        self.mip, self.exp, self.model, self.relm = model_name_parts
        self.case_id = case_id
        self.model_name = f"{self.mip}.{self.exp}.{self.model}_{self.relm}"
        self.input_dir = input_dir.replace("%(metric_type)", "enso_metric")
        self.output_dir = output_dir
        self.diag_metric = "enso_metric"
        self.fig_sets = OrderedDict([("ENSO_metric", ["graphics", "*"])])

    def collect_figures(self, groups) -> bool:
        logger.info("Entering EnsoDiagnosticsCollector.collect_figures")
        success: bool = True
        for fset, (subdir, pattern) in self.fig_sets.items():
            logger.info(f"Processing {fset}, ({subdir}, {pattern})")
            fdir = self.input_dir.replace("%(output_type)", subdir)
            logger.info(f"Processing fdir={fdir}")
            found_groups: List[str] = os.listdir(fdir)
            if sorted(groups) != sorted(found_groups):
                logger.error(
                    f"Groups mismatch: expected {sorted(groups)}, found {sorted(found_groups)} in {fdir}"
                )
                success = False
                continue
            for group in groups:
                logger.info(f"Processing group={group}")
                template = os.path.join(fdir, group, f"{pattern}.{self.fig_format}")
                logger.info(
                    f"template={template}, pattern.fig_format={pattern}.{self.fig_format}"
                )
                fpaths = sorted(glob.glob(template))

                if not fpaths:
                    logger.error(
                        f"fpaths={fpaths}, self.input_dir={self.input_dir}, template={os.path.abspath(template)}, files in template={os.listdir(os.path.join(fdir, group))}"
                    )
                    success = False
                for fpath in fpaths:
                    logger.info(f"Processing fpath={fpath}")
                    tail = fpath.split("/")[-1].split(f"{self.model}_{self.relm}")[-1]
                    outpath = os.path.join(
                        self.output_dir.replace("%(group_type)", fset), group
                    )
                    logger.info(f"outpath={outpath}")
                    os.makedirs(outpath, exist_ok=True)
                    outfile = f"{group}{tail}"
                    os.rename(fpath, os.path.join(outpath, outfile))
        return success

    def collect_metrics(self) -> bool:
        logger.info("Entering EnsoDiagnosticsCollector.collect_metrics")
        success: bool = True
        inpath = self.input_dir.replace("%(output_type)", "metrics_results")
        fpaths = sorted(glob.glob(os.path.join(inpath, "*/*.json")))

        if not fpaths:
            logger.error(
                f"fpaths={fpaths}, self.input_dir={self.input_dir}, inpath={os.path.abspath(inpath)}, files in inpath={os.listdir(inpath)}"
            )
            success = False
        for fpath in fpaths:
            logger.info(f"Processing fpath={fpath}")
            refmode = fpath.split("/")[-2]
            reffile = fpath.split("/")[-1]
            outpath = os.path.join(
                self.output_dir.replace("%(group_type)", "metrics_data"),
                self.diag_metric,
                refmode,
            )
            logger.info(f"outpath={outpath}")
            os.makedirs(outpath, exist_ok=True)

            base_filename = (
                f"{refmode}.{self.model_name}.vs.{self.refname}.{self.case_id}.json"
            )
            outfile = (
                base_filename.replace(".json", ".diveDown.json")
                if "diveDown" in reffile
                else base_filename
            )
            os.rename(fpath, os.path.join(outpath, outfile))
        return success

    def collect_diags(self) -> bool:
        logger.info("Entering EnsoDiagnosticsCollector.collect_diags")
        success: bool = True
        inpath = self.input_dir.replace("%(output_type)", "diagnostic_results")
        fpaths = sorted(glob.glob(os.path.join(inpath, "*/*/*/*/*/*.nc")))

        if not fpaths:
            logger.error(
                f"fpaths={fpaths}, self.input_dir={self.input_dir}, inpath={os.path.abspath(inpath)}, files in inpath={os.listdir(inpath)}"
            )
            success = False
        for fpath in fpaths:
            logger.info(f"Processing fpath={fpath}")
            refmode = fpath.split("/")[-2]
            reffile = fpath.split("/")[-1]
            outpath = os.path.join(
                self.output_dir.replace("%(group_type)", "metrics_data"),
                self.diag_metric,
                refmode,
            )
            logger.info(f"outpath={outpath}")
            os.makedirs(outpath, exist_ok=True)

            os.rename(fpath, os.path.join(outpath, reffile))
        return success

    def run(self, groups):
        logger.info("Entering EnsoDiagnosticsCollector.run")
        figures_success: bool = self.collect_figures(groups)
        metrics_success: bool = self.collect_metrics()
        diags_success: bool = self.collect_diags()
        if figures_success and metrics_success and diags_success:
            logger.info("Completing EnsoDiagnosticsCollector.run")
        else:
            raise RuntimeError(
                "EnsoDiagnosticsCollector.run failed: "
                f"figures_success={figures_success}, metrics_success={metrics_success}, diags_success={diags_success}"
            )


# Functions ###################################################################
def main():
    args: Dict[str, str] = _get_args()
    core_parameters = CoreParameters(args)
    enso_parameters = ENSOParameters(args)
    core_output: CoreOutput = set_up(core_parameters)

    #############################################
    # call enso_driver.py to process diagnostics
    #############################################
    build_enso_obsvar_catalog(core_output.obs_dic, core_parameters.variables)
    build_enso_obsvar_landmask(core_output.obs_dic, core_parameters.variables)
    # now start enso driver
    lstcmd = generate_enso_cmds(enso_parameters.enso_groups, core_parameters.case_id)
    logger.info(
        f"input_template={core_output.input_template}; if the directories based on this template are empty, lstcmd={lstcmd} failed to produce output."
    )
    if (len(lstcmd) > 0) and core_parameters.multiprocessing:
        logger.info(f"Running parallel jobs for {lstcmd}")
        try:
            results = run_parallel_jobs(lstcmd, core_parameters.num_workers)
            for i, (stdout, stderr, return_code) in enumerate(results):
                logger.info(f"Command {i+1} finished:")
                logger.info(f"STDOUT: {stdout}")
                logger.info(f"STDERR: {stderr}")
                logger.info(f"Return code: {return_code}")
        except RuntimeError as e:
            logger.error(f"Execution failed: {e}")
    elif (len(lstcmd) > 0) and not core_parameters.multiprocessing:
        logger.info(f"Running serial jobs for {lstcmd}")
        try:
            results = run_serial_jobs(lstcmd)
            for i, (stdout, stderr, return_code) in enumerate(results):
                logger.info(f"Command {i+1} finished:")
                logger.info(f"STDOUT: {stdout}")
                logger.info(f"STDERR: {stderr}")
                logger.info(f"Return code: {return_code}")
        except RuntimeError as e:
            logger.error(f"Execution failed: {e}")
    else:
        logger.info("no jobs to run...")
    logger.info("successfully finish all jobs....")
    # time delay to ensure process completely finished
    time.sleep(5)
    # Initialize and run collector
    obs_dict = json.load(open("obs_catalogue.json"))
    obs_name = list(obs_dict.keys())[0]
    collector = EnsoDiagnosticsCollector(
        fig_format=core_parameters.figure_format,
        refname=obs_name,
        model_name_parts=core_parameters.model_name.split("."),
        case_id=core_parameters.case_id,
        input_dir=core_output.input_template,
        output_dir=core_output.out_path,
    )
    enso_groups: List[str] = enso_parameters.enso_groups.split(",")
    collector.run(enso_groups)


def _get_args() -> Dict[str, str]:
    # Parser
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        usage="zi-pcmdi-enso <args>"
    )

    # For CoreParameters
    parser.add_argument("--num_workers", type=str)
    parser.add_argument("--multiprocessing", type=str)
    parser.add_argument("--subsection", type=str)
    parser.add_argument("--climo_ts_dir_primary", type=str)  # needs ts_dir_primary
    parser.add_argument("--climo_ts_dir_ref", type=str)  # needs ts_dir_ref
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model_tableID", type=str)
    parser.add_argument("--figure_format", type=str)
    parser.add_argument("--run_type", type=str)
    parser.add_argument("--obs_sets", type=str)  # run_type == "model_vs_obs" only
    parser.add_argument(
        "--model_name_ref", type=str
    )  # run_type == "model_vs_model" only
    parser.add_argument("--vars", type=str)
    parser.add_argument("--tableID_ref", type=str)  # run_type == "model_vs_model" only
    parser.add_argument("--generate_sftlf", type=str)
    parser.add_argument("--case_id", type=str)
    parser.add_argument("--results_dir", type=str)

    # For ENSOParameters
    parser.add_argument("--enso_groups", type=str)

    # Ignore the first arg
    # (zi-pcmdi-enso)
    args: argparse.Namespace = parser.parse_args(sys.argv[1:])

    return vars(args)


def build_enso_obsvar_catalog(
    obs_dic: Dict, variables: List[str], output_file: str = "obs_catalogue.json"
) -> None:
    """
    Organize observational data for the ENSO driver based on the variable list.

    Parameters:
        obs_dic (dict): Dictionary mapping variable names to their observation sets and data files.
        variables (list): List of variable names to process.
        output_file (str): Output JSON file path to save the observation catalogue.
    """
    refr_dic: OrderedDict = OrderedDict()

    for var in variables:
        vkey = re.split(r"[_-]", var)[0] if "_" in var or "-" in var else var

        if vkey not in obs_dic:
            raise KeyError(
                f"Variable key '{vkey}' not found in observation dictionary. Available keys are {obs_dic.keys()}"
            )

        refset = obs_dic[vkey]["set"]
        refname = obs_dic[vkey].get(refset)

        if not refname:
            raise KeyError(
                f"Reference name not found for variable '{vkey}' and set '{refset}'."
            )

        refr_dic.setdefault(refname, {})[vkey] = obs_dic[vkey][refname]

    with open(output_file, "w") as f:
        json.dump(refr_dic, f, indent=4, sort_keys=False, separators=(",", ": "))

    logger.info(f"[INFO] Observation catalogue written to: {output_file}")


def build_enso_obsvar_landmask(
    obs_dic: Dict,
    variables: List[str],
    output_file: str = "obs_landmask.json",
    mask_dir: str = "fixed",
) -> None:
    """
    Organize observational land/sea mask mapping for ENSO diagnostics.

    Parameters:
        obs_dic (dict): Dictionary mapping variables to observation metadata.
        variables (list): List of variable names used in ENSO analysis.
        output_file (str): Path to output the landmask JSON.
        mask_dir (str): Directory prefix where the landmask files are located.
    """
    relf_dic: OrderedDict = OrderedDict()

    for var in variables:
        vkey = re.split(r"[_-]", var)[0] if "_" in var or "-" in var else var

        if vkey not in obs_dic:
            raise KeyError(
                f"Variable key '{vkey}' not found in observation dictionary."
            )

        refset = obs_dic[vkey]["set"]
        refname = obs_dic[vkey].get(refset)

        if not refname:
            raise KeyError(
                f"Reference name not found for variable '{vkey}' and set '{refset}'."
            )

        relf_dic.setdefault(refname, os.path.join(mask_dir, f"sftlf.{refname}.nc"))

    with open(output_file, "w") as f:
        json.dump(relf_dic, f, indent=4, sort_keys=False, separators=(",", ": "))

    logger.info(f"[INFO] Landmask mapping written to: {output_file}")


def generate_enso_cmds(
    enso_groups_str,
    case_id,
    param_file="parameterfile.py",
    driver_script="enso_driver.py",
):
    """
    Generate ENSO driver command-line strings for given metric groups.

    Parameters:
        enso_groups_str: Comma-separated list of ENSO metric groups.
        case_id: Case identifier.
        param_file: Parameter file used by the driver script.
        driver_script: ENSO driver script filename.

    Returns:
        cmds: List of shell command strings to run.
    """
    enso_groups = enso_groups_str.split(",")
    commands = [
        "{} -p {} --metricsCollection {} --case_id {}".format(
            driver_script, param_file, group, case_id
        )
        for group in enso_groups
    ]
    return commands
