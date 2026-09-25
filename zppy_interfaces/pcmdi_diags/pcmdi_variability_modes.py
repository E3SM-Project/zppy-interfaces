import argparse
import glob
import os
import shlex
import shutil
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


def _normalize_modes(value: str) -> List[str]:
    modes = []
    for raw_mode in value.split(","):
        mode = raw_mode.strip().upper()
        if mode and mode not in modes:
            modes.append(mode)
    if not modes:
        raise ValueError("--var_modes must contain at least one mode.")
    return modes


# Classes #####################################################################
class VariabilityModesParameters(object):
    def __init__(self, args: Dict[str, str]):
        var_modes = args.get("var_modes")
        if not var_modes:
            raise ValueError("--var_modes is required but was not provided.")
        self.var_modes = _normalize_modes(var_modes)
        # self.vars is distinct from the list version in CoreParameters
        vars_arg = args.get("vars")
        if not vars_arg:
            raise ValueError("--vars is required but was not provided.")
        variables = [var.strip() for var in vars_arg.split(",") if var.strip()]
        if len(variables) != 1:
            raise ValueError(
                "Variability modes requires exactly one variable from --vars; "
                f"got {variables}."
            )
        self.vars = variables[0]


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
        collected_count = 0
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

                    if not matched_files:
                        logger.warning(
                            f"No figures found for fig_set={fig_set}, mode={mode}, "
                            f"season={season}: {search_path}"
                        )
                    for fpath in matched_files:
                        collected_count += 1
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
                        _move_output(fpath, os.path.join(outdir, outfile))

        if collected_count == 0:
            raise FileNotFoundError(
                f"No variability-mode figures found under {self.input_dir}."
            )

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
        if suffix == "unknown":
            raise ValueError(
                f"Could not classify output name for file '{filename}' "
                f"(fig_set={fig_set}, mode={mode}, season={season})."
            )
        return f"{fig_set}_{mode}_{season}_{suffix}.{self.fig_format}"

    def _collect_metrics(self):
        metrics_dir = self.input_dir.replace("%(output_type)", "metrics_results")
        json_files = sorted(glob.glob(os.path.join(metrics_dir, "*/*/*.json")))
        if not json_files:
            raise FileNotFoundError(
                f"No variability-mode metrics JSON files found in {metrics_dir}."
            )

        for fpath in json_files:
            refname = os.path.basename(os.path.dirname(fpath))
            refmode = os.path.basename(os.path.dirname(os.path.dirname(fpath)))
            reffile = os.path.basename(fpath)

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

            _move_output(fpath, outfile)

    def _collect_diags(self):
        diags_dir = self.input_dir.replace("%(output_type)", "diagnostic_results")
        diagnostic_files = sorted(glob.glob(os.path.join(diags_dir, "*/*/*.nc")))
        if not diagnostic_files:
            raise FileNotFoundError(
                f"No variability-mode diagnostic NetCDF files found in {diags_dir}."
            )

        for fpath in diagnostic_files:
            refname = os.path.basename(os.path.dirname(fpath))
            refmode = os.path.basename(os.path.dirname(os.path.dirname(fpath)))
            reffile = os.path.basename(fpath)

            outdir = os.path.join(
                self.output_dir.replace("%(group_type)", "metrics_data"),
                "variability_modes",
                refmode,
                refname,
            )
            os.makedirs(outdir, exist_ok=True)

            outfile = os.path.join(outdir, reffile)

            _move_output(fpath, outfile)


# Functions ###################################################################
def _move_output(source: str, destination: str) -> None:
    """Move an output across filesystems without overwriting directories."""
    if os.path.isdir(destination):
        raise IsADirectoryError(f"Destination is a directory: {destination}")
    if os.path.exists(destination):
        logger.warning(f"Destination already exists, replacing: {destination}")
        os.remove(destination)
    shutil.move(source, destination)


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
    if (len(lstcmd) > 0) and core_output.multiprocessing:
        try:
            results = run_parallel_jobs(lstcmd, core_parameters.num_workers)
            for i, (stdout, stderr, return_code) in enumerate(results):
                logger.info(f"Command {i + 1} finished:")
                logger.info(f"STDOUT: {stdout}")
                logger.info(f"STDERR: {stderr}")
                logger.info(f"Return code: {return_code}")
        except RuntimeError as e:
            logger.error(f"Execution failed: {e}")
            raise
    elif len(lstcmd) > 0:
        try:
            results = run_serial_jobs(lstcmd)
            for i, (stdout, stderr, return_code) in enumerate(results):
                logger.info(f"Command {i + 1} finished:")
                logger.info(f"STDOUT: {stdout}")
                logger.info(f"STDERR: {stderr}")
                logger.info(f"Return code: {return_code}")
        except RuntimeError as e:
            logger.error(f"Execution failed: {e}")
            raise
    else:
        raise RuntimeError("No variability-mode diagnostic commands were generated.")
    logger.info("successfully finished all jobs.")
    # time delay to ensure process completely finished
    time.sleep(5)
    # Create the collector instance
    split_name: List[str] = core_parameters.model_name.split(".")
    if len(split_name) != 4:
        raise ValueError(
            f"model_name must have 4 dot-separated parts (mip.exp.model.relm), "
            f"got {len(split_name)}: {core_parameters.model_name}"
        )
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

    normalized_modes = _normalize_modes(",".join(str(mode) for mode in modes))
    for var_mode in normalized_modes:
        # Use specified EOF number if in map, otherwise default to 1
        eofn = eofn_map.get(var_mode, 1)
        command_parts = [
            "variability_modes_driver.py",
            "-p",
            "parameterfile.py",
            "--variability_mode",
            var_mode,
            "--eofn_mod",
            eofn,
            "--eofn_obs",
            eofn,
            "--varOBS",
            varOBS,
            "--osyear",
            reftyrs,
            "--oeyear",
            reftyre,
            "--reference_data_name",
            refname,
            "--reference_data_path",
            refpath,
            "--case_id",
            case_id,
        ]
        cmd = " ".join(shlex.quote(str(part)) for part in command_parts)
        commands.append(cmd)

    return commands
