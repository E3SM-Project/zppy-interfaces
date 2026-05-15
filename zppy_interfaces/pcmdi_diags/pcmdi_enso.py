import argparse
import glob
import json
import os
import re
import shutil
import sys
import time
from collections import OrderedDict
from typing import Dict, List

from zppy_interfaces.multi_utils.logger import _setup_child_logger, _setup_root_logger
from zppy_interfaces.pcmdi_diags.pcmdi_setup import CoreOutput, CoreParameters, set_up
from zppy_interfaces.pcmdi_diags.utils import (
    ALT_OBS_MAP,
    run_parallel_jobs,
    run_serial_jobs,
)

# Set up the root logger and module level logger. The module level logger is
# a child of the root logger.
_setup_root_logger()
logger = _setup_child_logger(__name__)


# Classes #####################################################################
class ENSOParameters(object):
    def __init__(self, args: Dict[str, str]):
        enso_groups = args.get("enso_groups")
        if not enso_groups:
            raise ValueError("--enso_groups is required but was not provided.")
        self.enso_groups: str = enso_groups


class EnsoDiagnosticsCollector:
    def __init__(
        self, fig_format, refname, model_name_parts, case_id, input_dir, output_dir
    ):
        self.fig_format = fig_format
        self.refname = refname
        if len(model_name_parts) != 4:
            raise ValueError(
                f"model_name must have 4 dot-separated parts (mip.exp.model.relm), "
                f"got {len(model_name_parts)}: {model_name_parts}"
            )
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

            if not os.path.isdir(fdir):
                logger.error(f"Expected output directory does not exist: {fdir}")
                success = False
                continue

            found_groups: List[str] = [
                name
                for name in os.listdir(fdir)
                if os.path.isdir(os.path.join(fdir, name))
            ]
            missing_groups = sorted(set(groups) - set(found_groups))

            if missing_groups:
                logger.error(
                    f"Missing expected group directories: {missing_groups}; "
                    f"found {sorted(found_groups)} in {fdir}"
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
                    group_dir = os.path.join(fdir, group)
                    dir_contents = (
                        os.listdir(group_dir)
                        if os.path.isdir(group_dir)
                        else "<directory missing>"
                    )
                    logger.error(
                        f"No figures found. input_dir={self.input_dir}, "
                        f"template={os.path.abspath(template)}, "
                        f"files in group dir={dir_contents}"
                    )
                    success = False

                for fpath in fpaths:
                    logger.info(f"Processing fpath={fpath}")
                    fname = os.path.basename(fpath)
                    marker = f"{self.model}_{self.relm}"

                    if marker in fname:
                        tail = fname.split(marker, 1)[-1]
                        outfile = f"{group}{tail}"
                    else:
                        logger.warning(
                            f"Expected '{marker}' in filename '{fname}'; "
                            "using full filename as output."
                        )
                        outfile = fname

                    outpath = os.path.join(
                        self.output_dir.replace("%(group_type)", fset), group
                    )
                    logger.info(f"outpath={outpath}")
                    os.makedirs(outpath, exist_ok=True)

                    dest = os.path.join(outpath, outfile)
                    if os.path.isdir(dest):
                        raise IsADirectoryError(f"Destination is a directory: {dest}")
                    if os.path.exists(dest):
                        logger.warning(f"Destination already exists, replacing: {dest}")
                        os.remove(dest)
                    shutil.move(fpath, dest)

        return success

    def collect_metrics(self) -> bool:
        logger.info("Entering EnsoDiagnosticsCollector.collect_metrics")
        success: bool = True
        inpath = self.input_dir.replace("%(output_type)", "metrics_results")
        fpaths = sorted(glob.glob(os.path.join(inpath, "*/*.json")))

        if not fpaths:
            dir_contents = (
                os.listdir(inpath) if os.path.isdir(inpath) else "<directory missing>"
            )
            logger.error(
                f"No metrics JSON files found. input_dir={self.input_dir}, "
                f"inpath={os.path.abspath(inpath)}, contents={dir_contents}"
            )
            success = False

        for fpath in fpaths:
            logger.info(f"Processing fpath={fpath}")
            refmode = os.path.basename(os.path.dirname(fpath))
            reffile = os.path.basename(fpath)
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
            dest = os.path.join(outpath, outfile)

            with open(fpath) as _f:
                data = json.load(_f)

            try:
                for _model, _members in data["RESULTS"]["model"].items():
                    for _member, _entry in _members.items():
                        value_block = _entry.get("value", {})
                        incomplete = [
                            m for m, v in value_block.items() if not v.get("metric")
                        ]
                        if incomplete:
                            logger.warning(
                                f"{reffile}: dropping {len(incomplete)} incomplete "
                                f"metric(s) with empty 'metric' dict: {incomplete}"
                            )
                            for m in incomplete:
                                del value_block[m]
            except (KeyError, AttributeError) as e:
                logger.warning(f"Could not prune incomplete metrics in {fpath}: {e}")

            if os.path.isdir(dest):
                raise IsADirectoryError(f"Destination is a directory: {dest}")

            if os.path.exists(dest):
                logger.warning(f"Destination already exists, replacing: {dest}")

            tmp_dest = f"{dest}.tmp"
            try:
                with open(tmp_dest, "w") as _f:
                    json.dump(
                        data,
                        _f,
                        indent=4,
                        separators=(",", ": "),
                        sort_keys=False,
                    )
                os.replace(tmp_dest, dest)
            except Exception:
                if os.path.exists(tmp_dest):
                    os.remove(tmp_dest)
                raise

            os.remove(fpath)

        return success

    def collect_diags(self) -> bool:
        logger.info("Entering EnsoDiagnosticsCollector.collect_diags")
        success: bool = True
        inpath = self.input_dir.replace("%(output_type)", "diagnostic_results")
        fpaths = sorted(glob.glob(os.path.join(inpath, "*/*.nc")))

        if not fpaths:
            dir_contents = (
                os.listdir(inpath) if os.path.isdir(inpath) else "<directory missing>"
            )
            logger.error(
                f"No diagnostic NetCDF files found. input_dir={self.input_dir}, "
                f"inpath={os.path.abspath(inpath)}, contents={dir_contents}"
            )
            success = False
        for fpath in fpaths:
            logger.info(f"Processing fpath={fpath}")
            refmode = os.path.basename(os.path.dirname(fpath))
            reffile = os.path.basename(fpath)
            outpath = os.path.join(
                self.output_dir.replace("%(group_type)", "metrics_data"),
                self.diag_metric,
                refmode,
            )
            logger.info(f"outpath={outpath}")
            os.makedirs(outpath, exist_ok=True)

            dest = os.path.join(outpath, reffile)
            if os.path.isdir(dest):
                raise IsADirectoryError(f"Destination is a directory: {dest}")
            if os.path.exists(dest):
                logger.warning(f"Destination already exists, replacing: {dest}")
                os.remove(dest)
            shutil.move(fpath, dest)

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
    # logger.error("zi-pcmdi-enso is not yet supported. Exiting.")
    # sys.exit(1)
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
    check_enso_input()
    normalize_enso_model_catalogue(core_parameters.variables)
    lstcmd = generate_enso_cmds(enso_parameters.enso_groups, core_parameters.case_id)
    logger.info(
        f"input_template={core_output.input_template}; If directories derived from this template are empty, it may indicate that lstcmd did not produce output."
    )

    if (len(lstcmd) > 0) and core_parameters.multiprocessing:
        logger.info(f"Running parallel jobs for {lstcmd}")
        try:
            results = run_parallel_jobs(lstcmd, core_parameters.num_workers)
            check_enso_output(results)
        except RuntimeError as e:
            logger.error(f"Execution failed: {e}")
            raise
    elif (len(lstcmd) > 0) and not core_parameters.multiprocessing:
        logger.info(f"Running serial jobs for {lstcmd}")
        try:
            results = run_serial_jobs(lstcmd)
            check_enso_output(results)
        except RuntimeError as e:
            logger.error(f"Execution failed: {e}")
            raise
    else:
        logger.info("no jobs to run...")
    logger.info("successfully finish all jobs....")
    # time delay to ensure process completely finished
    time.sleep(5)
    # Initialize and run collector
    with open("obs_catalogue.json") as _f:
        obs_dict = json.load(_f)
    if not obs_dict:
        raise ValueError(
            "obs_catalogue.json is empty; cannot determine observation name."
        )
    obs_name = list(obs_dict.keys())[0]
    collector = EnsoDiagnosticsCollector(
        fig_format=core_parameters.figure_format,
        refname=obs_name,
        model_name_parts=core_parameters.model_name.split("."),
        case_id=core_parameters.case_id,
        input_dir=core_output.input_template,
        output_dir=core_output.out_path,
    )
    enso_groups: List[str] = [
        group.strip()
        for group in enso_parameters.enso_groups.split(",")
        if group.strip()
    ]
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
    parser.add_argument("--debug", type=str)

    # For ENSOParameters
    parser.add_argument("--enso_groups", type=str)

    # Ignore the first arg
    # (zi-pcmdi-enso)
    args: argparse.Namespace = parser.parse_args(sys.argv[1:])

    if args.debug and args.debug.lower() == "true":
        logger.setLevel("DEBUG")
        logger.debug("Debug logging enabled")

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


def normalize_enso_model_catalogue(
    variables: List[str],
    catalogue_file: str = "pcmdi_diags/ts_enso_catalogue.json",
) -> None:
    """
    Ensure the ENSO model catalogue exposes both logical diagnostic names
    and available source-variable names when aliases are used.

    Example:
        EAM/EAMxx may provide SST as source variable 'ts', while ENSO
        diagnostics may expect logical variable 'sst'. If either 'ts' or
        'sst' appears in the variable list/catalogue, make sure the logical
        'sst' catalogue entry exists and points to the available source data.
    """
    if not os.path.exists(catalogue_file):
        logger.warning(
            f"ENSO model catalogue not found, skipping normalization: {catalogue_file}"
        )
        return

    with open(catalogue_file) as f:
        catalogue = json.load(f, object_pairs_hook=OrderedDict)

    changed = False

    # Normalize requested variables to simple variable keys.
    requested_vars = {
        re.split(r"[_-]", var)[0] if "_" in var or "-" in var else var
        for var in variables
    }

    for logical_var, source_var in ALT_OBS_MAP.items():
        # Only act when this alias is relevant to the requested variables
        # or already present in the catalogue.
        alias_is_relevant = (
            logical_var in requested_vars
            or source_var in requested_vars
            or logical_var in catalogue
            or source_var in catalogue
        )

        if not alias_is_relevant:
            continue

        # Case 1:
        # The catalogue already has logical_var, e.g. "sst".
        # Make sure it has the expected metadata.
        if logical_var in catalogue:
            refset = catalogue[logical_var].get("set")
            model_name = catalogue[logical_var].get(refset)

            if refset and model_name and model_name in catalogue[logical_var]:
                entry = catalogue[logical_var][model_name]

                if entry.get("var_name") != logical_var:
                    entry["var_name"] = logical_var
                    changed = True

                if source_var in catalogue and entry.get("var_in_file") not in {
                    logical_var,
                    source_var,
                }:
                    entry["var_in_file"] = source_var
                    changed = True

            continue

        # Case 2:
        # The catalogue lacks logical_var, e.g. no "sst",
        # but has source_var, e.g. "ts". Add logical_var from source_var.
        if source_var not in catalogue:
            logger.warning(
                f"Cannot add logical ENSO variable '{logical_var}' to catalogue "
                f"because source variable '{source_var}' is missing from "
                f"{catalogue_file}."
            )
            continue

        logger.info(
            f"Adding logical ENSO catalogue entry '{logical_var}' using source "
            f"variable '{source_var}'."
        )

        catalogue[logical_var] = catalogue[source_var].copy()

        refset = catalogue[logical_var].get("set")
        model_name = catalogue[logical_var].get(refset)

        if refset and model_name and model_name in catalogue[logical_var]:
            catalogue[logical_var][model_name] = catalogue[logical_var][
                model_name
            ].copy()

            entry = catalogue[logical_var][model_name]
            entry["var_in_file"] = source_var
            entry["var_name"] = logical_var

            old_file_path = entry.get("file_path", "")
            old_template = entry.get("template", "")

            # Prefer the symlinked logical filename, e.g. *.sst.*.nc,
            # because check_enso_input() creates this link before normalization.
            entry["file_path"] = old_file_path.replace(
                f".{source_var}.", f".{logical_var}."
            )
            entry["template"] = old_template.replace(
                f".{source_var}.", f".{logical_var}."
            )

        changed = True

    if changed:
        with open(catalogue_file, "w") as f:
            json.dump(
                catalogue,
                f,
                indent=4,
                sort_keys=False,
                separators=(",", ": "),
            )

        logger.info(f"Normalized ENSO model catalogue: {catalogue_file}")


def check_enso_input():
    current_dir: str = os.path.abspath(os.getcwd())
    ts_dir: str = os.path.join(current_dir, "ts")

    if not os.path.exists(ts_dir):
        raise FileNotFoundError(f"{ts_dir} (input for enso_driver) does not exist.")

    if not os.listdir(ts_dir):
        raise FileNotFoundError(f"{ts_dir} is empty.")

    for obs_var_name, cmip_var_name in ALT_OBS_MAP.items():
        logger.info(
            f"Symlinking cmip-standard {cmip_var_name} to observational "
            f"variable name {obs_var_name}, if present"
        )

        found_nc_file = (
            glob.glob(os.path.join(ts_dir, f"*.{cmip_var_name}.*.nc"))
            + glob.glob(os.path.join(ts_dir, f"{cmip_var_name}_*.nc"))
        )

        if found_nc_file:
            source_file = found_nc_file[0]
            source_basename = os.path.basename(source_file)

            if f".{cmip_var_name}." in source_basename:
                link_name = source_file.replace(
                    f".{cmip_var_name}.", f".{obs_var_name}."
                )
            elif source_basename.startswith(f"{cmip_var_name}_"):
                link_name = os.path.join(
                    ts_dir,
                    source_basename.replace(
                        f"{cmip_var_name}_", f"{obs_var_name}_", 1
                    ),
                )
            else:
                logger.warning(
                    f"Could not infer symlink name for source file: {source_file}"
                )
                link_name = None

            if link_name:
                if not os.path.exists(link_name):
                    os.symlink(source_file, link_name)
                else:
                    logger.info(f"Symlink already exists, skipping: {link_name}")

        found_txt_file = glob.glob(os.path.join(ts_dir, f"{cmip_var_name}_files.txt"))
        if found_txt_file:
            source_file = found_txt_file[0]
            link_name = os.path.join(ts_dir, f"{obs_var_name}_files.txt")

            if not os.path.exists(link_name):
                os.symlink(source_file, link_name)
            else:
                logger.info(f"Symlink already exists, skipping: {link_name}")


def generate_enso_cmds(
    enso_groups_str,
    case_id,
    param_file="parameterfile.py",
    driver_script="enso_driver.py",
):
    enso_groups = [
        group.strip() for group in enso_groups_str.split(",") if group.strip()
    ]
    commands = [
        "{} -p {} --metricsCollection {} --case_id {}".format(
            driver_script, param_file, group, case_id
        )
        for group in enso_groups
    ]

    current_dir: str = os.path.abspath(os.getcwd())
    logger.info(f"Commands will be run from current_dir={current_dir}")
    dir_contents: List[str] = os.listdir(current_dir)

    if param_file not in dir_contents:
        logger.error(
            f"Parameter file '{param_file}' not found in current directory: {current_dir}"
        )
        raise FileNotFoundError(f"Parameter file '{param_file}' not found.")

    return commands


def check_enso_output(results):
    logger.info("Checking ENSO output.")
    success: bool = True

    for i, (stdout, stderr, return_code) in enumerate(results):
        logger.info(f"Command {i + 1} finished:")
        logger.info(f"STDOUT: {stdout}")
        logger.info(f"STDERR: {stderr}")
        logger.info(f"Return code: {return_code}")

        if return_code != 0:
            logger.error(f"Command {i + 1} failed with return code {return_code}.")
            success = False

        if not check_vars(stdout):
            logger.error(f"Command {i + 1} failed to produce expected variables.")
            success = False

        if not check_output_dirs(stdout):
            logger.error(
                f"Command {i + 1} failed to produce expected output directories."
            )
            success = False

    if not success:
        raise RuntimeError("ENSO output check failed.")

    logger.info("ENSO output check passed.")


def check_vars(stdout: str) -> bool:
    """
    Check if the output from an enso_driver.py command contains expected variables.

    Parameters:
        stdout (str): Standard output from the command execution.

    Returns:
        bool: True if expected variables are found, or if only optional
        process-level variables are missing. False otherwise.
    """
    match_object = re.search(r"list_variables\s*[:=]\s*\[(.*?)\]", stdout, re.DOTALL)

    # Allow ENSO diagnostics to continue if only selected variables are missing.
    # ssh and thf are rarely available process-level variables and are mainly used
    # by selected ENSO process diagnostics, not the core ENSO performance diagnostics.
    #
    # v4 note: EAMxx does not always output taux and tauy by default, so we also
    # treat them as soft-missing variables here to avoid stopping the full ENSO
    # workflow when surface wind stress diagnostics are unavailable.
    optional_missing = {"ssh", "thf", "taux", "tauy"}

    if not match_object:
        logger.error("No variable list found in stdout.")
        return False

    variables_content = match_object.group(1)

    requested_variables: List[str] = []
    for var in variables_content.split(","):
        clean_var = re.sub(r"['\"\s]", "", var.strip())
        if clean_var:
            requested_variables.append(clean_var)

    current_dir: str = os.path.abspath(os.getcwd())
    ts_dir = os.path.join(current_dir, "ts")

    variables_missing_data: List[str] = []

    for var in requested_variables:
        # Support both common filename conventions:
        #   ts/*.<var>.*.nc
        #   ts/<var>_*.nc
        found_nc_file = (
            glob.glob(os.path.join(ts_dir, f"*.{var}.*.nc"))
            + glob.glob(os.path.join(ts_dir, f"{var}_*.nc"))
        )
        found_txt_file = glob.glob(os.path.join(ts_dir, f"{var}_files.txt"))

        if (not found_nc_file) or (not found_txt_file):
            variables_missing_data.append(var)

            # Check whether an alternative/source variable exists.
            # This may indicate that variable derivation/mapping was not applied.
            if var in ALT_OBS_MAP:
                alt_var = ALT_OBS_MAP[var]
                found_nc_file_alt = (
                    glob.glob(os.path.join(ts_dir, f"*.{alt_var}.*.nc"))
                    + glob.glob(os.path.join(ts_dir, f"{alt_var}_*.nc"))
                )
                found_txt_file_alt = glob.glob(
                    os.path.join(ts_dir, f"{alt_var}_files.txt")
                )
                if found_nc_file_alt or found_txt_file_alt:
                    logger.warning(
                        f"Found alternative variable '{alt_var}' for expected variable "
                        f"'{var}' in {ts_dir}. "
                        f"NetCDF found: {bool(found_nc_file_alt)}; "
                        f"txt found: {bool(found_txt_file_alt)}. "
                        "This indicates that the variable derivation/mapping may not "
                        "have been applied correctly."
                    )

    if variables_missing_data:
        if set(variables_missing_data) <= optional_missing:
            logger.warning(
                f"Optional process-level variables missing: {variables_missing_data} "
                f"in directory {ts_dir}; continuing ENSO diagnostics."
            )
            return True

        logger.error(
            f"Variables missing data: {variables_missing_data} in directory {ts_dir}"
        )

        if os.path.isdir(ts_dir):
            logger.error(f"Full contents of {ts_dir}: {os.listdir(ts_dir)}")
        else:
            logger.error(f"Directory does not exist: {ts_dir}")

        return False

    logger.info(
        f"All requested variables {requested_variables} found in directory {ts_dir}"
    )
    return True


def check_output_dirs(stdout: str) -> bool:
    current_dir: str = os.path.abspath(os.getcwd())
    success: bool = True

    for output_type in ["graphics", "diagnostic_results", "metrics_results"]:
        match_object = re.search(
            rf"output directory for {re.escape(output_type)}:(.*)",
            stdout,
        )

        if not match_object:
            logger.warning(
                f"No output directory line found for {output_type} in stdout."
            )
            continue

        subdir = match_object.group(1).strip()
        combined_dir = os.path.join(current_dir, subdir)

        if not os.path.exists(combined_dir):
            logger.error(
                f"{output_type} output directory does not exist: {combined_dir}"
            )
            success = False
            continue

        if not os.path.isdir(combined_dir):
            logger.error(
                f"{output_type} output path exists but is not a directory: {combined_dir}"
            )
            success = False
            continue

        if not os.listdir(combined_dir):
            logger.error(f"{output_type} output directory is empty: {combined_dir}")
            success = False

    return success
