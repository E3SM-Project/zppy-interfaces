import argparse
import sys
import time
from typing import Dict, List

from pcmdi_zppy_util import (
    MeanClimateMetricsCollector,
    generate_mean_clim_cmds,
    run_parallel_jobs,
    run_serial_jobs,
    save_variable_regions,
)

from zppy_interfaces.pcmdi_diags.pcmdi_common import CoreOutput, CoreParameters, set_up


class MeanClimateParameters(object):
    def __init__(self, args: Dict[str, str]):
        self.regions: List[str] = args["region"].split(",")


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
        print("no jobs to run,continue....")
    print("successfully finish all jobs....")
    # time delay to ensure process completely finished
    time.sleep(5)
    # orgnize diagnostic output
    collector = MeanClimateMetricsCollector(
        regions=mean_climate_parameters.regions,
        variables=core_parameters.variables,
        fig_format="{{figure_format}}",
        model_info=tuple("${model_name}".split(".")),  # (mip, exp, model, relm)
        case_id="${case_id}",
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

    # For MeanClimateParameters
    parser.add_argument("--regions", type=str)

    # Ignore the first arg
    # (zi-pcmdi-mean-climate)
    args: argparse.Namespace = parser.parse_args(sys.argv[1:])

    return vars(args)
