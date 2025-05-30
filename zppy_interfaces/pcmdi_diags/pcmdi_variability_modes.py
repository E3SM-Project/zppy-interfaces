import argparse
import sys
import time
from typing import Dict, List

from pcmdi_zppy_util import (
    VariabilityMetricsCollector,
    generate_varmode_cmds,
    run_parallel_jobs,
    run_serial_jobs,
)

from zppy_interfaces.pcmdi_diags.pcmdi_common import CoreOutput, CoreParameters, set_up


class VariabilityModesParameters(object):
    def __init__(self, args: Dict[str, str]):
        self.var_modes: List[str] = args["var_modes"].split(",")
        # self.vars is distinct from the list version in CoreParameters
        self.vars: str = args["vars"]
        self.figure_format: str = args["figure_format"]


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
                print(f"\nCommand {i+1} finished:")
                print(f"STDOUT: {stdout}")
                print(f"STDERR: {stderr}")
                print(f"Return code: {return_code}")
        except RuntimeError as e:
            print(f"Execution failed: {e}")
    elif len(lstcmd) > 0:
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
        print("no jobs to run,continue...")
    print("successfully finish all jobs....")
    # time delay to ensure process completely finished
    time.sleep(5)
    # Create the collector instance
    split_name: List[str] = core_parameters.model_name.split(".")
    collector = VariabilityMetricsCollector(
        modes=variability_modes_parameters.var_modes,
        fig_format=variability_modes_parameters.figure_format,
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

    # For VariabilityModesParameters
    parser.add_argument("--var_modes", type=str)  # use either atm_mdoes or cpl_modes
    parser.add_argument("--figure_format", type=str)

    # Ignore the first arg
    # (zi-pcmdi-variability-modes)
    args: argparse.Namespace = parser.parse_args(sys.argv[1:])

    return vars(args)
