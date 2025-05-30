import glob
import os
import re
from typing import Dict, List

from pcmdi_zppy_util import (
    DataCatalogueBuilder,
    LandSeaMaskGenerator,
    derive_missing_variable,
)


class CoreParameters(object):
    def __init__(self, args: Dict[str, str]):
        self.num_workers: int = int(args["num_workers"])
        self.multiprocessing: bool = args["multiprocessing"].lower() == "true"
        self.subsection: str = args["subsection"]
        self.test_data_path: str = args["climo_ts_dir_primary"]
        self.reference_data_path: str = args["climo_ts_dir_ref"]
        self.model_name: str = args["model_name"]
        self.run_type: str = args["run_type"]
        self.obs_sets: str = args["obs_sets"]  # run_type == "model_vs_obs" only
        self.model_name_ref: str = args[
            "model_name_ref"
        ]  # run_type == "model_vs_model" only
        self.variables: List[str] = args["vars"].split(",")
        self.tableID_ref: str = args["tableID_ref"]  # run_type == "model_vs_model" only
        # Whether to generate the land/sea mask
        self.generate_flag: str = args["generate_sftlf"]
        self.case_id: str = args["case_id"]
        self.results_dir: str = args["results_dir"]


class CoreOutput(object):
    def __init__(self, multiprocessing, obs_dic, input_template, out_path):
        self.multiprocessing = multiprocessing
        self.obs_dic = obs_dic
        self.input_template = input_template
        self.out_path = out_path


def set_up(parameters: CoreParameters) -> CoreOutput:
    # Determine multiprocessing usage
    multiprocessing: bool = (
        parameters.multiprocessing if parameters.num_workers >= 2 else False
    )
    # Dataset identifiers
    test_data_set: List[str] = [parameters.model_name.split(".")[1]]
    reference_data_set: List[str]
    if parameters.run_type == "model_vs_obs":
        reference_data_set = parameters.obs_sets.split(",")
    elif parameters.run_type == "model_vs_model":
        reference_data_set = [parameters.model_name_ref.split(".")[1]]
    else:
        raise ValueError(f"Invalid run_type={parameters.run_type}")
    ###############################################################
    # Check and process derived quantities; these quantities are
    # likely not included as default in e3sm_to_cmip module
    ###############################################################
    for var in parameters.variables:
        varin = re.split(r"[_-]", var)[0] if "_" in var or "-" in var else var
        test_fpaths = sorted(
            glob.glob(os.path.join(parameters.test_data_path, f"*.{var}.*.nc"))
        )
        if not test_fpaths:
            derive_missing_variable(
                varin, parameters.test_data_path, "${model_name}.${tableID}"
            )
            if parameters.run_type == "model_vs_model":
                ref_fpaths = sorted(
                    glob.glob(
                        os.path.join(parameters.reference_data_path, f"*.{var}.*.nc")
                    )
                )
                if not ref_fpaths:
                    derive_missing_variable(
                        varin,
                        parameters.reference_data_path,
                        f"{parameters.model_name_ref}.{parameters.tableID_ref}",
                    )
    #######################################################
    # collect and document data info in a dictionary
    # for convenience of pcmdi processing
    #######################################################
    builder = DataCatalogueBuilder(
        parameters.test_data_path,
        test_data_set,
        parameters.reference_data_path,
        reference_data_set,
        parameters.variables,
        "{{subsection}}",
        "pcmdi_diags",
    )
    _, obs_dic = builder.build_catalogues()
    ##########################################################
    # land/sea mask is needed in PCMDI diagnostics, check and
    # generate it here as these data are not always available
    # for model or observations
    ##########################################################
    # Instantiate and run
    mask_generator = LandSeaMaskGenerator(
        test_path=parameters.test_data_path,
        ref_path=parameters.reference_data_path,
        subsection=parameters.subsection,
        fixed_dir="fixed",
    )
    mask_generator.run(parameters.generate_flag)
    # Diagnostic input file templates
    input_template = os.path.join(
        "pcmdi_diags",
        "%(output_type)",
        "%(metric_type)",
        parameters.model_name.split(".")[0],
        parameters.model_name.split(".")[1],
        parameters.case_id,
    )
    # Diagnostic output path templates
    out_path = os.path.join(parameters.results_dir, "%(group_type)")
    return CoreOutput(multiprocessing, obs_dic, input_template, out_path)
