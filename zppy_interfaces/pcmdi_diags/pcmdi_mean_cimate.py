import argparse
import glob
import json
import os
import re
import sys
import time
from collections import OrderedDict
from typing import Dict, List

from zppy_interfaces.multi_utils.logger import _setup_child_logger, _setup_root_logger
from zppy_interfaces.pcmdi_diags.pcmdi_setup import CoreOutput, CoreParameters, set_up
from zppy_interfaces.pcmdi_diags.utils import run_parallel_jobs, run_serial_jobs

# Set up the root logger and module level logger. The module level logger is
# a child of the root logger.
_setup_root_logger()
logger = _setup_child_logger(__name__)


# Classes #####################################################################
class MeanClimateParameters(object):
    def __init__(self, args: Dict[str, str]):
        self.regions: List[str] = args["regions"].split(",")


class MeanClimateMetricsCollector:
    def __init__(
        self,
        regions,
        variables,
        fig_format,
        model_info,
        case_id,
        input_template,
        output_dir,
    ):
        self.regions = regions
        self.variables = variables
        self.fig_format = fig_format
        self.mip, self.exp, self.model, self.relm = model_info
        self.case_id = case_id
        self.input_template = input_template
        self.output_dir = output_dir
        self.diag_metric = "mean_climate"
        self.seasons = ["DJF", "MAM", "JJA", "SON", "AC"]
        self.model_name = f"{self.mip}.{self.exp}.{self.model}_{self.relm}"

    def collect(self):
        self._collect_figures()
        self._collect_metrics()
        self._collect_diags()

    def _collect_figures(self):
        fig_sets = OrderedDict()
        fig_sets["CLIM_patttern"] = ["graphics", "*"]

        for fset, (fig_type, prefix) in fig_sets.items():
            for region in self.regions:
                for season in self.seasons:
                    for var in self.variables:
                        indir = self.input_template.replace(
                            "%(metric_type)", self.diag_metric
                        )
                        indir = indir.replace("%(output_type)", fig_type)
                        search_path = os.path.join(
                            indir, var, f"{prefix}{region}_{season}*.{self.fig_format}"
                        )
                        fpaths = sorted(glob.glob(search_path))

                        for fpath in fpaths:
                            refname = os.path.basename(fpath).split("_")[0]
                            filname = f"{refname}_{region}_{season}.{self.fig_format}"
                            outpath = os.path.join(
                                self.output_dir.replace("%(group_type)", fset),
                                region,
                                season,
                            )
                            os.makedirs(outpath, exist_ok=True)
                            outfile = os.path.join(outpath, filname)
                            os.rename(fpath, outfile)

    def _collect_diags(self):
        inpath = self.input_template.replace("%(metric_type)", self.diag_metric)
        inpath = inpath.replace("%(output_type)", "diagnostic_results")
        outpath = os.path.join(
            self.output_dir.replace("%(group_type)", "metrics_data"), self.diag_metric
        )

        os.makedirs(outpath, exist_ok=True)
        fpaths = sorted(glob.glob(os.path.join(inpath, "*/*/*/*/*/*/*.nc")))

        for fpath in fpaths:
            filname = fpath.split("/")[-1]
            outfile = os.path.join(outpath, filname)
            os.rename(fpath, outfile)

    def _collect_metrics(self):
        inpath = self.input_template.replace("%(metric_type)", self.diag_metric)
        inpath = inpath.replace("%(output_type)", "metrics_results")
        outpath = os.path.join(
            self.output_dir.replace("%(group_type)", "metrics_data"), self.diag_metric
        )

        os.makedirs(outpath, exist_ok=True)
        fpaths = sorted(glob.glob(os.path.join(inpath, "*.json")))

        for fpath in fpaths:
            refname = os.path.basename(fpath).split("_")[:2]
            filname = f"{refname[0]}.{refname[1]}.{self.model_name}.{self.case_id}.json"
            outfile = os.path.join(outpath, filname)
            os.rename(fpath, outfile)


# Functions ###################################################################
def main():
    args: Dict[str, str] = _get_args()
    core_parameters = CoreParameters(args)
    mean_climate_parameters = MeanClimateParameters(args)
    core_output: CoreOutput = set_up(core_parameters)

    # assign region to each variable
    save_variable_regions(core_parameters.variables, mean_climate_parameters.regions)
    # generate the command list
    lstcmd = generate_mean_clim_cmds(
        variables=core_parameters.variables,
        obs_dic=core_output.obs_dic,
        case_id=core_parameters.case_id,
    )
    ####################################################
    # call pcmdi mean climate diagnostics
    ####################################################
    if (len(lstcmd) > 0) and core_output.multiprocessing:
        try:
            results = run_parallel_jobs(lstcmd, core_parameters.num_workers)
            for i, (stdout, stderr, return_code) in enumerate(results):
                print(f"\nCommand {i + 1} finished:")
                print(f"STDOUT: {stdout}")
                print(f"STDERR: {stderr}")
                print(f"Return code: {return_code}")
        except RuntimeError as e:
            print(f"Execution failed: {e}")
    elif len(lstcmd) > 0:
        try:
            results = run_serial_jobs(lstcmd)
            for i, (stdout, stderr, return_code) in enumerate(results):
                print(f"\nCommand {i + 1} finished:")
                print(f"STDOUT: {stdout}")
                print(f"STDERR: {stderr}")
                print(f"Return code: {return_code}")
        except RuntimeError as e:
            print(f"Execution failed: {e}")
    else:
        print("no jobs to run,continue....")
    print("successfully finish all jobs....")
    # time delay to ensure process completely finished
    time.sleep(5)
    # orgnize diagnostic output
    model_info_str: List[str] = core_parameters.model_name.split(".")
    if len(model_info_str) == 4:
        # (mip, exp, model, relm)
        # model_info_tuple: Tuple[str, str, str, str] =
        model_info_tuple = tuple(model_info_str)
    else:
        raise ValueError(
            f"(mip, exp, model, relm) cannot be parsed from {core_parameters.model_name}"
        )
    collector = MeanClimateMetricsCollector(
        regions=mean_climate_parameters.regions,
        variables=core_parameters.variables,
        fig_format=core_parameters.figure_format,
        model_info=model_info_tuple,
        case_id=core_parameters.case_id,
        input_template=core_output.input_template,
        output_dir=core_output.out_path,
    )
    collector.collect()


def _get_args() -> Dict[str, str]:
    # Parser
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        usage="zi-pcmdi-mean-climate <args>"
    )

    # For CoreParameters
    parser.add_argument("--num_workers", type=str)
    parser.add_argument("--multiprocessing", type=str)
    parser.add_argument("--subsection", type=str)
    parser.add_argument("--climo_ts_dir_primary", type=str)  # needs climo_dir_primary
    parser.add_argument("--climo_ts_dir_ref", type=str)  # needs climo_dir_ref
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
    parser.add_argument("--debug", type=str)

    # For MeanClimateParameters
    parser.add_argument("--regions", type=str)

    # Ignore the first arg
    # (zi-pcmdi-mean-climate)
    args: argparse.Namespace = parser.parse_args(sys.argv[1:])

    if args.debug and args.debug.lower() == "true":
        logger.setLevel("DEBUG")
        logger.debug("Debug logging enabled")

    return vars(args)


def save_variable_regions(variables, regions, output_path="regions.json"):
    """
    Maps each variable (simplified key) to a list of regions and saves to JSON.
    """
    region_map = OrderedDict()
    for var in variables:
        var_key = re.split(r"[_-]", var)[0] if "_" in var or "-" in var else var
        region_map[var_key] = regions

    with open(output_path, "w") as f:
        json.dump(region_map, f, sort_keys=False, indent=4, separators=(",", ": "))
    return region_map


def generate_mean_clim_cmds(variables, obs_dic, case_id):
    """
    Generates a list of shell commands for mean climate diagnostics.
    """
    commands = []
    for var in variables:
        var_key = re.split(r"[_-]", var)[0] if "_" in var or "-" in var else var
        if var_key in obs_dic:
            refset = obs_dic[var_key]["set"]
            cmd = " ".join(
                [
                    "mean_climate_driver.py",
                    "-p parameterfile.py",
                    "--vars",
                    var,
                    "-r",
                    refset,
                    "--case_id",
                    case_id,
                ]
            )
            commands.append(cmd)
    return commands
