import glob
import json
import os
import re
from collections import OrderedDict
from datetime import datetime
from typing import Dict, List, Tuple

import xarray as xr
from pcmdi_metrics.io import xcdat_open
from pcmdi_metrics.utils import create_land_sea_mask

from zppy_interfaces.multi_utils.logger import _setup_custom_logger

logger = _setup_custom_logger(__name__)


# Classes #####################################################################
class CoreParameters(object):
    def __init__(self, args: Dict[str, str]):
        self.num_workers: int = int(args["num_workers"])
        self.multiprocessing: bool = args["multiprocessing"].lower() == "true"
        self.subsection: str = args["subsection"]
        self.test_data_path: str = args["climo_ts_dir_primary"]
        self.reference_data_path: str = args["climo_ts_dir_ref"]
        self.model_name: str = args["model_name"]
        self.model_tableID: str = args["model_tableID"]
        self.figure_format: str = args["figure_format"]
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


class DataCatalogueBuilder:
    def __init__(
        self,
        test_path: str,
        test_set: List[str],
        ref_path: str,
        ref_set: List[str],
        variables: List[str],
        label,
        output_dir,
    ):
        self.test_path: str = test_path
        self.test_set: List[str] = test_set
        self.ref_path: str = ref_path
        self.ref_set: List[str] = ref_set
        self.variables: List[str] = variables
        self.label = label
        self.output_dir = output_dir

        self.test_info: OrderedDict = OrderedDict()
        self.ref_info: OrderedDict = OrderedDict()

    def build_catalogues(self) -> Tuple[OrderedDict, OrderedDict]:
        if not self.variables:
            logger.info("DataCatalogueBuilder's variable list is empty")
        else:
            logger.info(f"DataCatalogueBuilder's vars: {self.variables}")
        for idx, var in enumerate(self.variables):
            logger.info(f"Building catalogue for {var}")
            varin = self._get_base_varname(var)
            logger.info(f"Looking for {varin}, the base var name of {var}")
            logger.info(f"Finding test files in {self.test_path}")
            test_files = sorted(
                glob.glob(os.path.join(self.test_path, f"*.{varin}.*.nc"))
            )
            logger.info(f"Finding ref files in {self.ref_path}")
            ref_files = sorted(
                glob.glob(os.path.join(self.ref_path, f"*.{varin}.*.nc"))
            )

            if (
                test_files
                and ref_files
                and os.path.exists(test_files[0])
                and os.path.exists(ref_files[0])
            ):
                logger.info(
                    f"Extracting & assigining metadata for {varin}, the base var name of {var}"
                )
                for fileset, info_dict, dataset, dataset_set in [
                    (test_files[0], self.test_info, self.variables, self.test_set),
                    (ref_files[0], self.ref_info, self.variables, self.ref_set),
                ]:
                    metadata = self._extract_metadata(fileset, varin, var)
                    self._assign_metadata(
                        info_dict, varin, dataset, dataset_set, idx, metadata
                    )
            else:
                logger.info(
                    f"NOT extracting & assigining metadata for {varin}, the base var name of {var}."
                )
                logger.info(f"test_files={test_files}")
                logger.info(f"ref_files={ref_files}")
                if test_files:
                    logger.info(f"test_files[0]={test_files[0]}")
                if ref_files:
                    logger.info(f"ref_files[0]={ref_files[0]}")

        # `odict_keys([])` evaluates as False/None would.
        if self.test_info.keys():
            self._save_catalogue(self.test_path, self.test_info)
        else:
            logger.info(f"test_info has no data to dump to {self.test_path}")
        if self.ref_info.keys():
            self._save_catalogue(self.ref_path, self.ref_info)
        else:
            logger.info(f"ref_info has no data to dump to {self.ref_path}")

        return self.test_info, self.ref_info

    def _get_base_varname(self, var):
        return re.split("_|-", var)[0] if ("_" in var or "-" in var) else var

    def _extract_metadata(self, filepath, varin, var):
        filename = os.path.basename(filepath)
        parts = filename.split(".")
        yymm_range = parts[6].split("-")
        return {
            "mip": parts[0],
            "exp": parts[1],
            "model": parts[2],
            "realization": parts[3],
            "tableID": parts[4],
            "yymms": yymm_range[0],
            "yymme": yymm_range[1],
            "var_in_file": varin,
            "var_name": var,
            "file_path": filepath,
            "template": filename,
        }

    def _assign_metadata(
        self, target_dict: OrderedDict, varin, dataset, dataset_names, idx, metadata
    ):
        if varin not in target_dict:
            target_dict[varin] = {}
        kset = (
            dataset_names[0]
            if len(dataset_names) != len(dataset)
            else dataset_names[idx]
        )
        model = metadata["model"]

        target_dict[varin]["set"] = kset
        target_dict[varin][kset] = model
        target_dict[varin][model] = metadata

    def _save_catalogue(self, source_path: str, data_dict: OrderedDict):
        filename = f"{source_path}_{self.label}_catalogue.json"
        filepath = os.path.join(self.output_dir, filename)
        logger.info(
            f"Saving catalogue {filepath}, absolute path {os.path.abspath(filepath)}"
        )
        with open(filepath, "w") as f:
            json.dump(data_dict, f, indent=4, sort_keys=False, separators=(",", ": "))


class LandSeaMaskGenerator:
    def __init__(self, test_path, ref_path, subsection, fixed_dir="fixed"):
        self.test_path = test_path
        self.ref_path = ref_path
        self.subsection = subsection
        self.fixed_dir = fixed_dir

    def run(self, enable_flag):
        if self._parse_flag(enable_flag):
            for group_path in [self.test_path, self.ref_path]:
                self._process_group(group_path)

    def _parse_flag(self, flag):
        return str(flag).lower() in ["true", "y", "yes"]

    def _process_group(self, group):
        catalog_path = os.path.join(
            "pcmdi_diags", f"{group}_{self.subsection}_catalogue.json"
        )

        if not os.path.exists(catalog_path):
            print(
                f"Warning: Catalogue not found at {catalog_path}, absolute path {os.path.abspath(catalog_path)}"
            )
            return

        with open(catalog_path) as f:
            data_catalog = json.load(f)

        for var, meta in data_catalog.items():
            dataset = meta["set"]
            model = meta[dataset]
            input_file = meta[model]["file_path"]
            output_file = os.path.join(self.fixed_dir, f"sftlf.{model}.nc")

            if not os.path.exists(self.fixed_dir):
                os.makedirs(self.fixed_dir)

            if not os.path.exists(output_file):
                self._generate_mask(input_file, output_file, model)

    def _generate_mask(self, input_path, output_path, model_name):
        ds = xcdat_open(input_path, decode_times=True)
        ds = ds.bounds.add_missing_bounds()

        try:
            mask = create_land_sea_mask(ds, method="regionmask")
            print("Land mask estimated using regionmask method.")
        except Exception:
            mask = create_land_sea_mask(ds, method="pcmdi")
            print("Land mask estimated using PCMDI method.")

        mask = mask * 100.0
        mask.attrs.update(
            {"long_name": "land_area_fraction", "units": "%", "id": "sftlf"}
        )

        mask_ds = mask.to_dataset(name="sftlf").compute()
        mask_ds = mask_ds.bounds.add_missing_bounds()
        mask_ds = mask_ds.fillna(1.0e20)

        mask_ds.attrs.update(
            {
                "model": model_name,
                "associated_files": input_path,
                "history": f"File processed: {datetime.now().strftime('%Y%m%d')}",
            }
        )

        comp = dict(_FillValue=1.0e20, zlib=True, complevel=5)
        encoding = {
            v: comp for v in set(mask_ds.data_vars.keys()) | set(mask_ds.coords.keys())
        }

        mask_ds.to_netcdf(output_path, encoding=encoding)

        del ds, mask_ds, mask


# Functions ###################################################################


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
                varin,
                parameters.test_data_path,
                f"{parameters.model_name}.{parameters.model_tableID}",
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
        parameters.subsection,
        "pcmdi_diags",
    )
    _, obs_dic = builder.build_catalogues()
    if not obs_dic.keys():
        raise ValueError("obs_dic has no keys!")
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
    logger.info(f"out_path={out_path}")
    return CoreOutput(multiprocessing, obs_dic, input_template, out_path)


def derive_missing_variable(varin, path, model_id):
    """
    Derive variable with existing variables, preserving coordinates and attributes.

    Args:
        varin (str): Name of the derived variable (e.g., 'rstcre').
        path (str): Directory to look for/create the file.
        model_id (str): Identifier for constructing output filenames.
    """
    derived_var_map = {
        "rstcre": {"rsutcs": 1, "rsut": -1},
        "rltcre": {"rlutcs": 1, "rlut": -1},
    }

    if varin not in derived_var_map:
        return  # Nothing to derive

    var_dic = derived_var_map[varin]
    derived_data = None
    base_ds = None
    output_file = None

    for i, (src_var, scale) in enumerate(var_dic.items()):
        fpaths = sorted(glob.glob(os.path.join(path, f"*.{src_var}.*.nc")))
        if not fpaths:
            raise FileNotFoundError(
                f"No file found for source variable '{src_var}' in {path}"
            )
        fpath = fpaths[0]
        ds = xcdat_open(fpath)
        data = ds[src_var] * scale

        if i == 0:
            base_ds = ds.copy(deep=True)
            derived_data = data.copy(deep=True)
            template = os.path.basename(fpath)
            output_file = os.path.join(
                path, template.replace(f".{src_var}.", f".{varin}.")
            )
        else:
            derived_data = derived_data + data

    if base_ds is not None and derived_data is not None:
        derived_da = xr.DataArray(
            data=derived_data.data,
            coords=derived_data.coords,
            dims=derived_data.dims,
            attrs=derived_data.attrs,
        )

        out_ds = base_ds.drop_vars(list(var_dic.keys()), errors="ignore")
        out_ds[varin] = derived_da

        # Optional: annotate global attributes
        out_ds.attrs.update(
            {
                "derived_variable": varin,
                "derived_from": ", ".join(var_dic.keys()),
                "model_id": model_id,
            }
        )

        out_ds.to_netcdf(output_file)
        print(f"Derived variable '{varin}' written to {output_file}")

    return
