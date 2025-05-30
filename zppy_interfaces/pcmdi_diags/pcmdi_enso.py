import argparse
import json
import sys
import time
from typing import Dict, List

from pcmdi_zppy_util import (
    EnsoDiagnosticsCollector,
    build_enso_obsvar_catalog,
    build_enso_obsvar_landmask,
    generate_enso_cmds,
    run_parallel_jobs,
    run_serial_jobs,
)

from zppy_interfaces.pcmdi_diags.pcmdi_common import CoreOutput, CoreParameters, set_up


class ENSOParameters(object):
    def __init__(self, args: Dict[str, str]):
        self.enso_groups: str = args["enso_groups"]
        self.figure_format: str = args["figure_format"]


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
    if (len(lstcmd) > 0) and core_parameters.multiprocessing:
        try:
            results = run_parallel_jobs(lstcmd, core_parameters.num_workers)
            for i, (stdout, stderr, return_code) in enumerate(results):
                print(f"\nCommand {i+1} finished:")
                print(f"STDOUT: {stdout}")
                print(f"STDERR: {stderr}")
                print(f"Return code: {return_code}")
        except RuntimeError as e:
            print(f"Execution failed: {e}")
    elif (len(lstcmd) > 0) and not core_parameters.multiprocessing:
        try:
            results = run_serial_jobs(lstcmd)
            for i, (stdout, stderr, return_code) in enumerate(results):
                print(f"\nCommand {i+1} finished:")
                print(f"STDOUT: {stdout}")
                print(f"STDERR: {stderr}")
                print(f"Return code: {return_code}")
        except RuntimeError as e:
            print(f"Execution failed: {e}")
    else:
        print("no jobs to run...")
    print("successfully finish all jobs....")
    # time delay to ensure process completely finished
    time.sleep(5)
    # Initialize and run collector
    obs_dict = json.load(open("obs_catalogue.json"))
    obs_name = list(obs_dict.keys())[0]
    collector = EnsoDiagnosticsCollector(
        fig_format=enso_parameters.figure_format,
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
    parser.add_argument("--figure_format", type=str)

    # Ignore the first arg
    # (zi-pcmdi-enso)
    args: argparse.Namespace = parser.parse_args(sys.argv[1:])

    return vars(args)
