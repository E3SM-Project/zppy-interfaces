import argparse
import glob
import os
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
class VariabilityModesParameters(object):
    def __init__(self, args: Dict[str, str]):
        self.var_modes: List[str] = args["var_modes"].split(",")
        # self.vars is distinct from the list version in CoreParameters
        self.vars: str = args["vars"]


class VariabilityMetricsCollector:
    def __init__(
        self, modes, fig_format, mip, exp, model, relm, case_id, input_dir, output_dir
    ):
        self.modes = modes
        self.fig_format = fig_format
        self.mip = mip
        self.exp = exp
        self.model = model
        self.relm = relm
        self.case_id = case_id
        self.input_dir = input_dir.replace("%(metric_type)", "variability_modes")
        self.output_dir = output_dir
        self.model_name = f"{mip}.{exp}.{model}_{relm}"
        self.seasons = ["DJF", "MAM", "JJA", "SON", "yearly", "monthly"]
        self.fig_sets = OrderedDict(
            {
                "MOV_eoftest": ["diagnostic_results", "EG_Spec*"],
                "MOV_compose": ["graphics", "*compare_obs"],
                "MOV_telecon": ["graphics", "*teleconnection"],
                "MOV_pattern": ["graphics", "*"],
            }
        )

    def collect(self):
        self._collect_figures()
        self._collect_metrics()
        self._collect_diags()

    def _collect_figures(self):
        for fig_set, (out_type, pattern_base) in self.fig_sets.items():
            for mode in self.modes:
                for season in self.seasons:
                    indir = self.input_dir.replace("%(output_type)", out_type)
                    template = (
                        f"{pattern_base}_{mode}_{season}*.{self.fig_format}"
                        if fig_set == "MOV_eoftest"
                        else f"{mode}_*_{season}_{pattern_base}.{self.fig_format}"
                    )
                    search_path = os.path.join(indir, mode, "*", template)
                    matched_files = sorted(glob.glob(search_path))

                    for fpath in matched_files:
                        filename = os.path.basename(fpath)
                        outfile = self._classify_output_name(
                            fig_set, mode, season, filename
                        )
                        outdir = os.path.join(
                            self.output_dir.replace("%(group_type)", "MOV_metric"),
                            fig_set,
                            season,
                        )
                        os.makedirs(outdir, exist_ok=True)
                        os.rename(fpath, os.path.join(outdir, outfile))

    def _classify_output_name(self, fig_set, mode, season, filename):
        suffix = "unknown"
        if "North_test" in filename:
            suffix = "EG_Spec"
        elif "_cbf_" in filename:
            suffix = "cbf"
        elif "EOF1" in filename:
            suffix = "eof1"
        elif "EOF2" in filename:
            suffix = "eof2"
        elif "EOF3" in filename:
            suffix = "eof3"
        return f"{fig_set}_{mode}_{season}_{suffix}.{self.fig_format}"

    def _collect_metrics(self):
        metrics_dir = self.input_dir.replace("%(output_type)", "metrics_results")
        json_files = sorted(glob.glob(os.path.join(metrics_dir, "*/*/*.json")))

        for fpath in json_files:
            refmode = fpath.split("/")[-3]
            refname = fpath.split("/")[-2]
            reffile = fpath.split("/")[-1]

            eof_lookup = {"PSA1": "EOF2", "NPO": "EOF2", "NPGO": "EOF2", "PSA2": "EOF3"}
            refeof = eof_lookup.get(refmode, "EOF1")

            outdir = os.path.join(
                self.output_dir.replace("%(group_type)", "metrics_data"),
                "variability_modes",
                refmode,
                refname,
            )
            os.makedirs(outdir, exist_ok=True)

            base_name = f"var_mode_{refmode}.{refeof}.{self.model_name}.vs.{refname}.{self.case_id}"
            if "diveDown" in reffile:
                outfile = os.path.join(outdir, f"{base_name}.diveDown.json")
            else:
                outfile = os.path.join(outdir, f"{base_name}.json")

            os.rename(fpath, outfile)

    def _collect_diags(self):
        diags_dir = self.input_dir.replace("%(output_type)", "diagnostic_results")
        json_files = sorted(glob.glob(os.path.join(diags_dir, "*/*/*.nc")))

        for fpath in json_files:
            refmode = fpath.split("/")[-3]
            refname = fpath.split("/")[-2]
            reffile = fpath.split("/")[-1]

            outdir = os.path.join(
                self.output_dir.replace("%(group_type)", "metrics_data"),
                "variability_modes",
                refmode,
                refname,
            )
            os.makedirs(outdir, exist_ok=True)

            outfile = os.path.join(outdir, reffile)

            os.rename(fpath, outfile)


# Functions ###################################################################
def main():
    args: Dict[str, str] = _get_args()
    core_parameters = CoreParameters(args)
    variability_modes_parameters = VariabilityModesParameters(args)
    core_output: CoreOutput = set_up(core_parameters)

    ##########################################
    # call pcmdi mode variability diagnostics
    ##########################################
    # from configuration file
    varOBS = variability_modes_parameters.vars
    if varOBS not in core_output.obs_dic:
        raise KeyError(
            f"VarOBS '{varOBS}' not found in obs_dic. Available keys are {core_output.obs_dic.keys()}"
        )
    refset = core_output.obs_dic[varOBS]["set"]
    refname = core_output.obs_dic[varOBS][refset]
    refpath = core_output.obs_dic[varOBS][refname]["file_path"]
    reftyrs = int(str(core_output.obs_dic[varOBS][refname]["yymms"])[0:4])
    reftyre = int(str(core_output.obs_dic[varOBS][refname]["yymme"])[0:4])
    # Call the function
    lstcmd = generate_varmode_cmds(
        modes=variability_modes_parameters.var_modes,
        varOBS=varOBS,
        reftyrs=reftyrs,
        reftyre=reftyre,
        refname=refname,
        refpath=refpath,
        case_id=core_parameters.case_id,
    )
    if (len(lstcmd) > 0) and core_parameters.multiprocessing:
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
        print("no jobs to run,continue...")
    print("successfully finish all jobs....")
    # time delay to ensure process completely finished
    time.sleep(5)
    # Create the collector instance
    split_name: List[str] = core_parameters.model_name.split(".")
    collector = VariabilityMetricsCollector(
        modes=variability_modes_parameters.var_modes,
        fig_format=core_parameters.figure_format,
        mip=split_name[0],
        exp=split_name[1],
        model=split_name[2],
        relm=split_name[3],
        case_id=core_parameters.case_id,
        input_dir=core_output.input_template,
        output_dir=core_output.out_path,
    )
    # Run the collection process
    collector.collect()


def _get_args() -> Dict[str, str]:
    # Parser
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        usage="zi-pcmdi-variability-modes <args>"
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
    parser.add_argument("--debug", type=str)

    # For VariabilityModesParameters
    parser.add_argument("--var_modes", type=str)  # use either atm_mdoes or cpl_modes

    # Ignore the first arg
    # (zi-pcmdi-variability-modes)
    args: argparse.Namespace = parser.parse_args(sys.argv[1:])

    if args.debug and args.debug.lower() == "true":
        logger.setLevel("DEBUG")
        logger.debug("Debug logging enabled")

    return vars(args)


def generate_varmode_cmds(modes, varOBS, reftyrs, reftyre, refname, refpath, case_id):
    """Generates a list of command strings for variability modes processing."""

    # EOF mode overrides for specific variability modes (default is 1)
    eofn_map = {"NPO": 2, "NPGO": 2, "PSA1": 2, "PSA2": 3}

    commands = []

    for var_mode in modes:
        var_mode = var_mode.strip()
        # Use specified EOF number if in map, otherwise default to 1
        eofn = eofn_map.get(var_mode, 1)
        cmd = (
            f"variability_modes_driver.py -p parameterfile.py "
            f"--variability_mode {var_mode} "
            f"--eofn_mod {eofn} "
            f"--eofn_obs {eofn} "
            f"--varOBS {varOBS} "
            f"--osyear {reftyrs} "
            f"--oeyear {reftyre} "
            f"--reference_data_name {refname} "
            f"--reference_data_path {refpath} "
            f"--case_id {case_id}"
        )
        commands.append(cmd)

    return commands
