import glob
import json
import os
import re
import shutil
import time
from collections import OrderedDict
from collections.abc import MutableMapping
from copy import deepcopy
from datetime import datetime
from subprocess import PIPE, Popen
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import xarray as xr
from pcmdi_metrics.enso.lib import enso_portrait_plot
from pcmdi_metrics.graphics import (
    Metrics,
    normalize_by_median,
    parallel_coordinate_plot,
    portrait_plot,
)
from pcmdi_metrics.io import xcdat_open
from pcmdi_metrics.utils import create_land_sea_mask, sort_human


def count_child_processes(process=None):
    """
    Count the number of child processes for a given process.

    Parameters:
    - process (psutil.Process, optional): The process to check. If None, uses the current process.

    Returns:
    - int: Number of child processes.
    """
    if process is None:
        process = psutil.Process()

    children = process.children()
    return len(children)


def run_parallel_jobs(cmds: List[str], num_workers: int) -> List[Tuple[str, str, int]]:
    """
    Execute shell commands in parallel batches.

    Parameters:
    - cmds: List of command strings to run.
    - num_workers: Maximum number of subprocesses to run concurrently.

    Returns:
    - List of tuples: (stdout, stderr, return_code) for each command.
    """
    results = []
    procs = []

    for i, cmd in enumerate(cmds):
        proc = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True, text=True)
        procs.append((cmd, proc))

        # Run the batch if full or if it's the last command
        if len(procs) >= num_workers or i == len(cmds) - 1:
            print(f"Running {count_child_processes()} subprocesses...")
            for cmd, proc in procs:
                stdout, stderr = proc.communicate()
                return_code = proc.returncode

                if return_code != 0:
                    print(f"ERROR: Process failed: '{cmd}'\nError: {stderr.strip()}")
                    raise RuntimeError(f"Subprocess failed: {cmd}")

                results.append((stdout.strip(), stderr.strip(), return_code))

            time.sleep(0.25)  # Throttle before starting the next batch
            procs = []

    return results


def run_serial_jobs(cmds: List[str]) -> List[Tuple[str, str, int]]:
    """
    Execute shell commands one at a time (serially).

    Parameters:
    - cmds: List of command strings to run.

    Returns:
    - List of tuples: (stdout, stderr, return_code) for each command.
    """
    results = []

    for i, cmd in enumerate(cmds):
        print(f"Running [{i+1}/{len(cmds)}]: {cmd}")
        proc = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True, text=True)
        stdout, stderr = proc.communicate()
        return_code = proc.returncode

        if return_code != 0:
            print(f"ERROR: Process failed: '{cmd}'\nError: {stderr.strip()}")
            raise RuntimeError(f"Subprocess failed: {cmd}")

        results.append((stdout.strip(), stderr.strip(), return_code))

    return results


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


def get_highlight_models(all_models, model_name):
    """
    Prioritize models containing 'e3sm' and then any additional specified models.

    Parameters:
        data_dict (dict): Dictionary with a 'model' key containing a list of model names.
        model_name (list): List of models to also highlight (after e3sm models).

    Returns:
        list: Ordered list of unique models to highlight.
    """
    highlight_model1 = []

    # First, collect all models that contain "e3sm" (case-insensitive)
    e3sm_models = [m for m in all_models if "e3sm" in m.lower()]

    # Then collect models in model_name that are not already in e3sm_models
    additional_models = [
        m for m in all_models if m in model_name and m not in e3sm_models
    ]

    # Combine both lists
    highlight_model1 = e3sm_models + additional_models

    return highlight_model1


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
                f"Variable key '{vkey}' not found in observation dictionary."
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

    print(f"[INFO] Observation catalogue written to: {output_file}")


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

    print(f"[INFO] Landmask mapping written to: {output_file}")


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


class ObservationLinker:
    def __init__(
        self,
        model_name,
        variables,
        obs_sets,
        ts_dir_ref_source,
        obstmp_dir,
        obs_alias_file,
        altobs_dic,
    ):
        self.model_name = model_name
        self.variables = variables
        self.obs_sets = obs_sets
        self.ts_dir_ref_source = ts_dir_ref_source
        self.obstmp_dir = obstmp_dir
        self.obs_dic = json.load(open(obs_alias_file))
        self.altobs_dic = altobs_dic

    def _resolve_obs_file(self, varin, obsid):
        if varin not in self.obs_dic or obsid not in self.obs_dic[varin]:
            print(f"[Warning] No alias found for variable '{varin}' in obsid '{obsid}'")
            return None, None

        obsname = self.obs_dic[varin][obsid]
        obsstr = (
            obsname.replace("_", "*").replace("-", "*")
            if "ceres_ebaf" in obsname
            else obsname
        )
        pattern = os.path.join(self.ts_dir_ref_source, obsstr, f"{varin}_*.nc")
        fpaths = sorted(glob.glob(pattern))

        if fpaths and os.path.exists(fpaths[0]):
            return fpaths[0], varin

        # Try altobs mapping
        if varin in self.altobs_dic:
            alt_var = self.altobs_dic[varin]
            pattern_alt = os.path.join(
                self.ts_dir_ref_source, obsstr, f"{alt_var}_*.nc"
            )
            fpaths = sorted(glob.glob(pattern_alt))
            if fpaths and os.path.exists(fpaths[0]):
                return fpaths[0], alt_var

        print(f"[Warning] Observation file not found for {varin} ({obsid})")
        return None, None

    def link_obs_data(self):
        for i, vv in enumerate(self.variables):
            varin = re.split(r"_|-", vv)[0] if "_" in vv or "-" in vv else vv
            if len(self.obs_sets) > 1 and len(self.obs_sets) == len(self.variables):
                obsid = self.obs_sets[i]
            else:
                obsid = self.obs_sets[0]

            filepath, resolved_var = self._resolve_obs_file(varin, obsid)
            if filepath:
                template = os.path.basename(filepath)
                parts = template.replace(".nc", "").split("_")
                if len(parts) < 3:
                    print(f"[Error] Unexpected filename format: {template}")
                    continue
                yms, yme = parts[-2][:6], parts[-1][:6]
                obsname = self.obs_dic[varin][obsid].replace(".", "_")
                out = os.path.join(
                    self.obstmp_dir,
                    f"{self.model_name.replace('%(model)', obsname)}.{varin}.{yms}-{yme}.nc",
                )

                if not os.path.exists(out):
                    os.makedirs(os.path.dirname(out), exist_ok=True)
                    if resolved_var == varin:
                        os.symlink(filepath, out)
                        print(f"[Info] Linked {resolved_var} → {out}")
                    else:
                        ds = xcdat_open(filepath)
                        ds = ds.rename({resolved_var: varin})
                        ds.to_netcdf(out)
                        print(
                            f"[Info] Renamed and saved {resolved_var} as {varin} → {out}"
                        )
                else:
                    print(f"[Info] Skipping existing file: {out}")

    def derive_var(self, vout, var_dic):
        template = None
        out = None
        ds_out = None

        for i, (var, scale) in enumerate(var_dic.items()):
            fpaths = sorted(glob.glob(os.path.join(self.obstmp_dir, f"*.{var}.*.nc")))
            if not fpaths:
                print(
                    f"[Warning] No file found for base variable '{var}' needed to derive '{vout}'"
                )
                continue

            ds = xcdat_open(fpaths[0])
            if i == 0:
                template = os.path.basename(fpaths[0])
                out = os.path.join(
                    self.obstmp_dir, template.replace(f".{var}.", f".{vout}.")
                )
                shutil.copy(fpaths[0], out)
                ds_out = ds.rename_vars({var: vout})
                ds_out[vout] = ds_out[vout] * scale
            else:
                ds_other = xcdat_open(fpaths[0])
                if ds_out:
                    ds_out[vout] = ds_out[vout] + ds_other[var] * scale
                else:
                    raise ValueError("ds_out is None")

        if template and ds_out:
            ds_out.to_netcdf(out)
            print(f"[Info] Derived variable '{vout}' written to {out}")

    def process_derived_variables(self):
        for vv in self.variables:
            if vv in ["rltcre", "rstcre"]:
                fpaths = sorted(glob.glob(os.path.join(self.obstmp_dir, f"*{vv}_*.nc")))
                if not fpaths:
                    if vv == "rstcre":
                        self.derive_var("rstcre", {"rsutcs": 1, "rsut": -1})
                    elif vv == "rltcre":
                        self.derive_var("rltcre", {"rlutcs": 1, "rlut": -1})


class DataCatalogueBuilder:
    def __init__(
        self, test_path, test_set, ref_path, ref_set, variables, label, output_dir
    ):
        self.test_path = test_path
        self.test_set = test_set
        self.ref_path = ref_path
        self.ref_set = ref_set
        self.variables = variables
        self.label = label
        self.output_dir = output_dir

        self.test_info = OrderedDict()
        self.ref_info = OrderedDict()

    def build_catalogues(self):
        for idx, var in enumerate(self.variables):
            varin = self._get_base_varname(var)
            test_files = sorted(
                glob.glob(os.path.join(self.test_path, f"*.{varin}.*.nc"))
            )
            ref_files = sorted(
                glob.glob(os.path.join(self.ref_path, f"*.{varin}.*.nc"))
            )

            if (
                test_files
                and ref_files
                and os.path.exists(test_files[0])
                and os.path.exists(ref_files[0])
            ):
                for j, (fileset, info_dict, dataset, dataset_set) in enumerate(
                    [
                        (test_files[0], self.test_info, self.variables, self.test_set),
                        (ref_files[0], self.ref_info, self.variables, self.ref_set),
                    ]
                ):
                    metadata = self._extract_metadata(fileset, varin, var)
                    self._assign_metadata(
                        info_dict, varin, var, dataset, dataset_set, idx, metadata
                    )

        self._save_catalogue(self.test_path, self.test_info)
        self._save_catalogue(self.ref_path, self.ref_info)

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
        self, target_dict, varin, var, dataset, dataset_names, idx, metadata
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

    def _save_catalogue(self, source_path, data_dict):
        filename = f"{source_path}_{self.label}_catalogue.json"
        filepath = os.path.join(self.output_dir, filename)
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
            print(f"Warning: Catalogue not found at {catalog_path}")
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

    def collect_figures(self, groups):
        for fset, (subdir, pattern) in self.fig_sets.items():
            for group in groups:
                fdir = self.input_dir.replace("%(output_type)", subdir)
                template = os.path.join(fdir, group, f"{pattern}.{self.fig_format}")
                fpaths = sorted(glob.glob(template))

                for fpath in fpaths:
                    tail = fpath.split("/")[-1].split(f"{self.model}_{self.relm}")[-1]
                    outpath = os.path.join(
                        self.output_dir.replace("%(group_type)", fset), group
                    )
                    os.makedirs(outpath, exist_ok=True)
                    outfile = f"{group}{tail}"
                    os.rename(fpath, os.path.join(outpath, outfile))

    def collect_metrics(self):
        inpath = self.input_dir.replace("%(output_type)", "metrics_results")
        fpaths = sorted(glob.glob(os.path.join(inpath, "*/*.json")))

        for fpath in fpaths:
            refmode = fpath.split("/")[-2]
            reffile = fpath.split("/")[-1]
            outpath = os.path.join(
                self.output_dir.replace("%(group_type)", "metrics_data"),
                self.diag_metric,
                refmode,
            )
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

    def collect_diags(self):
        inpath = self.input_dir.replace("%(output_type)", "diagnostic_results")
        fpaths = sorted(glob.glob(os.path.join(inpath, "*/*/*/*/*/*.nc")))

        for fpath in fpaths:
            refmode = fpath.split("/")[-2]
            reffile = fpath.split("/")[-1]
            outpath = os.path.join(
                self.output_dir.replace("%(group_type)", "metrics_data"),
                self.diag_metric,
                refmode,
            )
            os.makedirs(outpath, exist_ok=True)

            os.rename(fpath, os.path.join(outpath, reffile))

    def run(self, groups):
        self.collect_figures(groups)
        self.collect_metrics()
        self.collect_diags()


class SyntheticMetricsPlotter:
    def __init__(
        self,
        case_name,
        test_name,
        table_id,
        figure_format,
        figure_sets,
        metric_dict,
        save_data,
        base_test_input_path,
        results_dir=None,
        cmip_clim_dir=None,
        cmip_clim_set=None,
        cmip_movs_dir=None,
        cmip_movs_set=None,
        atm_modes=None,
        cpl_modes=None,
        cmip_enso_dir=None,
        cmip_enso_set=None,
    ):
        self.case_name = case_name
        self.test_name = test_name
        self.table_id = table_id
        self.figure_format = figure_format
        self.figure_sets = figure_sets
        self.metric_dict = metric_dict
        self.save_data = save_data
        self.base_test_input_path = base_test_input_path
        self.results_dir = results_dir or "."
        self.cmip_clim_dir = cmip_clim_dir
        self.cmip_clim_set = cmip_clim_set
        self.cmip_movs_dir = cmip_movs_dir
        self.cmip_movs_set = cmip_movs_set
        self.atm_modes = atm_modes.split(",") if atm_modes else []
        self.cpl_modes = cpl_modes.split(",") if cpl_modes else []
        self.cmip_enso_dir = cmip_enso_dir
        self.cmip_enso_set = cmip_enso_set

        self.parameter = self._initialize_parameter()

    def _initialize_parameter(self):
        parsed_test_names = []
        parsed_model_names = []

        for raw_test, raw_case in zip(
            self.test_name.split(","), self.case_name.split(",")
        ):
            parts = raw_test.strip().split(".")
            if len(parts) != 4:
                raise ValueError(
                    f"Invalid test format '{raw_test}'. Expected 'a.b.c.d'"
                )

            # Construct strings
            test_id = f"{parts[0]}.{parts[1]}.{parts[2]}_{parts[3]}"
            parsed_test_names.append(test_id)

            parsed_model_names.append(raw_case)

        return OrderedDict(
            {
                "save_data": self.save_data,
                "out_dir": os.path.join(self.results_dir, "ERROR_metric"),
                "test_name": parsed_test_names,
                "model_name": parsed_model_names,
                "tableID": [self.table_id],
            }
        )

    def generate(self, metric_sets):
        print("Generating synthetic metrics plots ...")
        for metric in metric_sets:
            print(f"Processing metric: {metric}")
            self.parameter["test_path"] = self.base_test_input_path.replace(
                "%(group_type)", metric
            )
            self.parameter["diag_vars"] = self.metric_dict[metric]

            if metric == "mean_climate":
                self._handle_mean_climate(metric)

            elif metric == "variability_modes":
                self._handle_variability_modes(metric)

            elif metric == "enso_metric":
                self._handle_enso_metric(metric)

    def _handle_mean_climate(self, metric):
        self.parameter.update(
            {"cmip_path": self.cmip_clim_dir, "cmip_name": self.cmip_clim_set}
        )

        # Instantiate the collector
        collector = ClimMetricsReader(self.parameter)
        # Collect and merge metrics
        merge_lib = collector.collect()

        # merge_lib = collect_clim_metrics(self.parameter)
        for stat, vars_ in self.metric_dict[metric].items():
            mean_climate_plot_driver(
                metric,
                stat,
                merge_lib.regions,
                self.parameter["model_name"],
                vars_,
                merge_lib.df_dict[stat],
                merge_lib.var_list,
                merge_lib.var_unit_list,
                self.parameter["save_data"],
                self.parameter["out_dir"],
                self.figure_format,
            )

    def _handle_variability_modes(self, metric):
        self.parameter.update(
            {
                "cmip_path": self.cmip_movs_dir,
                "cmip_name": self.cmip_movs_set,
                "movs_mode": self.atm_modes + self.cpl_modes,
            }
        )

        reader = MoVsMetricsReader(self.parameter)
        merge_lib, mode_season_list = reader.collect_metrics()

        for stat, vars_ in self.metric_dict[metric].items():
            variability_modes_plot_driver(
                metric,
                stat,
                self.parameter["model_name"],
                vars_,
                merge_lib[stat],
                mode_season_list,
                self.parameter["save_data"],
                self.parameter["out_dir"],
                self.figure_format,
            )

    def _handle_enso_metric(self, metric):
        self.parameter.update(
            {
                "cmip_path": self.cmip_enso_dir,
                "cmip_name": self.cmip_enso_set,
            }
        )
        for stat in self.metric_dict[metric]:
            # Step 1: Load metrics JSON paths using the reader
            reader = EnsoMetricsReader(self.parameter, metric, stat)
            dict_json_path = reader.run()
            # Step 2: generate figures
            enso_plot_driver(
                metric, stat, dict_json_path, self.parameter, self.figure_format
            )


def shift_row_to_bottom(df, index_to_shift):
    """
    Moves the specified row to the bottom of the DataFrame and resets the index.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        index_to_shift (int): The index of the row to move to the bottom.

    Returns:
        pd.DataFrame: A new DataFrame with the row moved to the bottom and index reset.
    """
    if index_to_shift not in df.index:
        raise IndexError(f"Index {index_to_shift} not found in DataFrame.")

    df_top = df.drop(index=index_to_shift)
    df_bottom = df.loc[[index_to_shift]]

    new_df = pd.concat([df_top, df_bottom], ignore_index=True)
    return new_df


def check_badvals(data_lib):
    """
    Replaces known bad values in the data library with NaN.

    Parameters:
        data_lib (Metrics): Metrics object containing diagnostic DataFrames.

    Returns:
        Metrics: Updated metrics with bad values replaced by NaN.
    """
    # Define known bad values (model → variable with suspect data)
    bad_model_vars = {"E3SM-1-0": "ta-850", "E3SM-1-1-ECA": "ta-850", "CIESM": "pr"}

    for stat in data_lib.df_dict:
        for season in data_lib.df_dict[stat]:
            for region in data_lib.df_dict[stat][season]:
                df = pd.DataFrame(data_lib.df_dict[stat][season][region])

                for model, bad_var in bad_model_vars.items():
                    if bad_var in df.columns:
                        # Find all rows matching this model
                        bad_idx = df[df["model"] == model].index
                        df.loc[bad_idx, bad_var] = np.nan

                # Save cleaned DataFrame back
                data_lib.df_dict[stat][season][region] = df

    return data_lib


def find_latest_file_list(
    path: str,
    file_pattern: str,
    var_pattern=r"\.(\w+)\.\d{8}\.nc$",
    time_pattern=r"\.(\d{8})\.nc$",
) -> List[str]:
    """
    Find the latest NetCDF file for each variable in the directory based on timestamps in filenames.

    Args:
        path (str): Directory to search.
        file_pattern (str): Regex to search file lists.
        var_pattern (str): Regex to extract variable name.
        time_pattern (str): Regex to extract date.

    Returns:
        List[str]: List of file paths, one for each variable (latest by timestamp).
    """
    latest_files: Dict[str, Tuple[datetime, str]] = {}
    files = glob.glob(os.path.join(path, file_pattern))

    for f in files:
        fname = os.path.basename(f)
        var_match = re.search(var_pattern, fname)
        time_match = re.search(time_pattern, fname)

        if var_match and time_match:
            var = var_match.group(1)
            try:
                timestamp = datetime.strptime(time_match.group(1), "%Y%m%d")
            except ValueError:
                continue

            if var not in latest_files or timestamp > latest_files[var][0]:
                latest_files[var] = (timestamp, f)

    return [file for _, file in latest_files.values()]


class ClimMetricsMerger:
    def __init__(self, model_lib=None, cmip_lib=None, model_names=None):
        self.model_lib = model_lib or {}
        self.cmip_lib = cmip_lib or {}
        self.model_names = model_names or []
        self.merged_lib = None

    def merge(self):
        self._normalize_references()
        self._filter_regions()
        self._merge_and_standardize_units()
        self._highlight_and_sort_models()
        return self.merged_lib

    def _normalize_references(self):
        if hasattr(self.model_lib, "references") and isinstance(
            self.model_lib.references, dict
        ):
            self.model_lib.references = self._check_references(
                self.model_lib.references
            )
        if hasattr(self.cmip_lib, "references") and isinstance(
            self.cmip_lib.references, dict
        ):
            self.cmip_lib.references = self._check_references(self.cmip_lib.references)

    def _check_references(
        self,
        data_dict: MutableMapping[str, Optional[List[str]]],
        reference_alias: Optional[Dict[str, str]] = None,
    ) -> MutableMapping[str, Optional[List[str]]]:
        if reference_alias is None:
            reference_alias = {
                "ceres_ebaf_toa_v4.1": "ceres_ebaf_v4_1",
                "ceres_ebaf_toa_v4.0": "ceres_ebaf_v4_0",
                "ceres_ebaf_toa_v2.8": "ceres_ebaf_v2_8",
                "ceres_ebaf_surface_v4.1": "ceres_ebaf_v4_1",
                "ceres_ebaf_surface_v4.0": "ceres_ebaf_v4_0",
                "ceres_ebaf_surface_v2.8": "ceres_ebaf_v2_8",
                "CERES-EBAF-4-1": "ceres_ebaf_v4_1",
                "CERES-EBAF-4-0": "ceres_ebaf_v4_0",
                "CERES-EBAF-2-8": "ceres_ebaf_v2_8",
                "GPCP_v2.3": "GPCP_v2_3",
                "GPCP_v2.2": "GPCP_v2_2",
                "GPCP_v3.2": "GPCP_v3_2",
                "GPCP-2-3": "GPCP_v2_3",
                "GPCP-2-2": "GPCP_v2_2",
                "GPCP-3-2": "GPCP_v3_2",
                "NOAA_20C": "NOAA-20C",
                "ERA-INT": "ERA-Interim",
                "ERA-5": "ERA5",
            }

        for key, values in data_dict.items():
            if isinstance(values, list):
                data_dict[key] = [reference_alias.get(val, val) for val in values]
            elif values is not None:
                data_dict[key] = reference_alias.get(values, values)
            else:
                raise ValueError("values is None")

        return data_dict

    def _filter_regions(self):
        self.model_lib, self.cmip_lib = self._check_regions(
            self.model_lib, self.cmip_lib
        )

    def _check_regions(self, data_lib, refr_lib):
        shared_regions = [
            region for region in data_lib.regions if region in refr_lib.regions
        ]

        for lib in [refr_lib, data_lib]:
            for stat in lib.df_dict:
                for season in lib.df_dict[stat]:
                    lib.df_dict[stat][season] = {
                        region: lib.df_dict[stat][season][region]
                        for region in shared_regions
                        if region in lib.df_dict[stat][season]
                    }

        data_lib.regions = shared_regions
        refr_lib.regions = shared_regions

        return data_lib, refr_lib

    @staticmethod
    def _prune_empty_dfs(lib):
        for stat in lib.df_dict:
            for season in lib.df_dict[stat]:
                lib.df_dict[stat][season] = {
                    region: df
                    for region, df in lib.df_dict[stat][season].items()
                    if not df.empty and not df.isna().all().all()
                }
        return lib

    @staticmethod
    def _safe_merge_libs(lib1, lib2):
        """
        Merge two data libraries with nested dicts of DataFrames,
        gracefully handling missing or inconsistent keys, while
        avoiding FutureWarning due to all-NA/empty entries.
        """
        merged = deepcopy(lib1)  # Avoid modifying original

        for stat in lib2.df_dict:
            if stat not in merged.df_dict:
                merged.df_dict[stat] = {}

            for season in lib2.df_dict[stat]:
                if season not in merged.df_dict[stat]:
                    merged.df_dict[stat][season] = {}

                for region, df2 in lib2.df_dict[stat][season].items():
                    df1 = merged.df_dict[stat][season].get(region)

                    # Collect and clean valid DataFrames
                    valid_dfs = []
                    for df in (df1, df2):
                        if (
                            isinstance(df, pd.DataFrame)
                            and not df.empty
                            and not df.isna().all().all()
                        ):
                            # Drop columns that are entirely NaN
                            df_clean = df.dropna(axis=1, how="all")
                            if not df_clean.empty and df_clean.shape[1] > 0:
                                valid_dfs.append(df_clean)

                    if valid_dfs:
                        merged_df = pd.concat(valid_dfs, ignore_index=True, sort=False)
                    else:
                        merged_df = pd.DataFrame()

                    merged.df_dict[stat][season][region] = merged_df

        return merged

    def _merge_and_standardize_units(self):
        # Prune empty or fully-NaN DataFrames from the model library
        cleaned_model_lib = self._prune_empty_dfs(self.model_lib)

        # Safe merge with fallback for missing stats/seasons/regions
        self.merged_lib = self._safe_merge_libs(self.cmip_lib, cleaned_model_lib)

        # Standardize units after merging
        self.merged_lib = self._check_units(self.merged_lib)

    def _check_units(self, data_lib, verbose=False):
        units_all = {
            "prw": "[kg m$^{-2}$]",
            "pr": "[mm d$^{-1}$]",
            "prsn": "[mm d$^{-1}$]",
            "prc": "[mm d$^{-1}$]",
            "hfls": "[W m$^{-2}$]",
            "hfss": "[W m$^{-2}$]",
            "clivi": "[kg $m^{-2}$]",
            "clwvi": "[kg $m^{-2}$]",
            "psl": "[Pa]",
            "rlds": "[W m$^{-2}$]",
            "rldscs": "[W $m^{-2}$]",
            "evspsbl": "[kg m$^{-2} s^{-1}$]",
            "rtmt": "[W m$^{-2}$]",
            "rsdt": "[W m$^{-2}$]",
            "rlus": "[W m$^{-2}$]",
            "rluscs": "[W m$^{-2}$]",
            "rlut": "[W m$^{-2}$]",
            "rlutcs": "[W m$^{-2}$]",
            "rsds": "[W m$^{-2}$]",
            "rsdscs": "[W m$^{-2}$]",
            "rstcre": "[W m$^{-2}$]",
            "rltcre": "[W m$^{-2}$]",
            "rsus": "[W m$^{-2}$]",
            "rsuscs": "[W m$^{-2}$]",
            "rsut": "[W m$^{-2}$]",
            "rsutcs": "[W m$^{-2}$]",
            "ts": "[K]",
            "tas": "[K]",
            "tauu": "[Pa]",
            "tauv": "[Pa]",
            "zg-500": "[m]",
            "ta-200": "[K]",
            "sfcWind": "[m s$^{-1}$]",
            "ta-850": "[K]",
            "ua-200": "[m s$^{-1}$]",
            "ua-850": "[m s$^{-1}$]",
            "va-200": "[m s$^{-1}$]",
            "va-850": "[m s$^{-1}$]",
            "uas": "[m s$^{-1}$]",
            "vas": "[m s$^{-1}$]",
            "tasmin": "[K]",
            "tasmax": "[K]",
            "clt": "[%]",
        }

        # Identify common variables and handle aliases like 'rt' or 'rmt'
        common_vars = [var for var in data_lib.var_list if var in units_all]
        if "rtmt" not in common_vars and any(
            var in data_lib.var_list for var in ["rt", "rmt"]
        ):
            common_vars.append("rtmt")

        # Collect units for these variables
        common_unts = [units_all[var] for var in common_vars if var in units_all]

        # Filter and correct reference list
        new_var_ref_dict = {}
        for var, ref in data_lib.var_ref_dict.items():
            if var in common_vars:
                new_var_ref_dict[var] = ref
            elif var in ["rt", "rmt"]:
                new_var_ref_dict["rtmt"] = ref
                if verbose:
                    print(f"Alias {var} mapped to 'rtmt' in references.")

        data_lib.var_ref_dict = self._check_references(new_var_ref_dict)

        # Clean DataFrames
        for stat, seasons in data_lib.df_dict.items():
            for season, regions in seasons.items():
                for region, df in regions.items():
                    df = df.copy()
                    # Handle aliases
                    if "rt" in df.columns:
                        df["rtmt"] = df["rt"]
                    elif "rmt" in df.columns:
                        df["rtmt"] = df["rmt"]

                    # Drop irrelevant variables
                    drop_cols = [
                        var for var in df.columns[3:] if var not in common_vars
                    ]
                    if drop_cols and verbose:
                        print(
                            f"Dropping variables in {stat}/{season}/{region}: {drop_cols}"
                        )
                    df = df.drop(columns=drop_cols)
                    data_lib.df_dict[stat][season][region] = df

        data_lib.var_list = common_vars
        data_lib.var_unit_list = common_unts

        return data_lib

    def _highlight_and_sort_models(self):
        if self.merged_lib:
            for stat, seasons in self.merged_lib.df_dict.items():
                for season, regions in seasons.items():
                    for region, df in regions.items():
                        df = pd.DataFrame(df)
                        highlight_models = get_highlight_models(
                            df.get("model", []), self.model_names
                        )
                        for model in highlight_models:
                            for idx in df[df["model"] == model].index:
                                df = shift_row_to_bottom(df, idx)
                        self.merged_lib.df_dict[stat][season][region] = df.fillna(
                            np.nan
                        )
        else:
            raise ValueError("merged_lib is None")


class ClimMetricsReader:
    def __init__(self, parameter, unit_check=True):
        """
        Initialize the climate metrics collector.

        Args:
            parameter (dict): Contains path, model info, and identifiers.
            unit_check (bool): Whether to apply unit consistency check.
        """
        self.parameter = parameter
        self.unit_check = unit_check
        self.cmip_lib = None
        self.all_lib = None
        self.all_names = []

        self.var_pattern = re.compile(r"^([A-Za-z0-9\-]+)\.")
        self.time_pattern = re.compile(r"\.v(\d{8})\.json$")

    def _load_clim_metrics_from_files(self, file_paths):
        """
        Loads and processes synthetic climate metric data from JSON files.

        Parameters:
            file_paths (list): List of file paths to load.

        Returns:
            Metrics: Processed Metrics object.
        """
        lib = Metrics(file_paths)
        lib = check_badvals(lib)
        if self.unit_check:
            a = ClimMetricsMerger()
            lib = a._check_units(lib)
        return lib

    def _load_cmip_metrics(self):
        cmip_id_parts = self.parameter["cmip_name"].split(".")
        cmip_dir = os.path.join(
            self.parameter["cmip_path"],
            cmip_id_parts[0],
            cmip_id_parts[1],
            cmip_id_parts[2],
        )

        cmip_files = sorted(
            glob.glob(os.path.join(cmip_dir, f"*.{cmip_id_parts[2]}.json"))
        )
        if not cmip_files:
            raise FileNotFoundError(f"No CMIP metrics found in: {cmip_dir}")

        print(f"Loading CMIP metrics from {len(cmip_files)} files...")
        self.cmip_lib = self._load_clim_metrics_from_files(cmip_files)

    def _process_test_model(self, test_name, model_name):
        test_key = test_name.split(".")[1]
        test_path = self.parameter["test_path"].replace("%(model_name)", model_name)

        model_files = find_latest_file_list(
            path=test_path,
            file_pattern="*.v*.json",
            var_pattern=self.var_pattern,
            time_pattern=self.time_pattern,
        )

        if not model_files or not os.path.exists(model_files[0]):
            raise FileNotFoundError(
                f"No synthetic mean climate metrics found for model: {model_name}"
            )

        print(
            f"Reading metrics for model: {model_name} from {len(model_files)} file(s)..."
        )

        valid_model_files = []

        for file_path in model_files:
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)

                results = data.get("RESULTS", {})
                modified = False

                for model, model_data in results.items():
                    if test_key in model_data:
                        model_data["default"] = model_data.pop(test_key)
                        modified = True

                if modified:
                    with open(file_path, "w") as f:
                        json.dump(data, f, indent=2)
                    print(f"Updated file: {file_path}")

                valid_model_files.append(file_path)

            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Warning: Could not load {file_path}: {e}")

        # Load metrics from valid files
        model_lib = self._load_clim_metrics_from_files(valid_model_files)

        # Standardize model name in metric DataFrames
        for stat, seasons in model_lib.df_dict.items():
            for season, regions in seasons.items():
                for region, df in regions.items():
                    df = pd.DataFrame(df)
                    if "model" in df.columns:
                        df["model"] = model_name
                    model_lib.df_dict[stat][season][region] = df

        return model_lib

    def collect(self):
        self._load_cmip_metrics()

        for i, (test_name, model_name) in enumerate(
            zip(self.parameter["test_name"], self.parameter["model_name"])
        ):
            model_lib = self._process_test_model(test_name, model_name)
            self.all_lib = (
                model_lib.copy()
                if self.all_lib is None
                else self.all_lib.merge(model_lib)
            )
            self.all_names.append(model_name)

        print("Merging model metrics with CMIP reference metrics...")
        merger = ClimMetricsMerger(
            model_lib=self.all_lib, cmip_lib=self.cmip_lib, model_names=self.all_names
        )
        merged_metrics = merger.merge()  # Returns a new merged metrics library

        return merged_metrics


class MoVsMetricsReader:
    def __init__(self, parameter):
        self.parameter = parameter
        self.cmip_group, self.cmip_model, self.cmip_version = self.parameter[
            "cmip_name"
        ].split(".")
        self.movs_mode = parameter["movs_mode"]
        self.var_pattern = re.compile(r"var_mode_(\w+)\.EOF\d+\..*\.json$")
        self.time_pattern = re.compile(r"\.v(\d{8})\.json$")

    def collect_metrics(self):
        cmip_files = self._get_cmip_files()
        if not cmip_files or not os.path.exists(cmip_files[0]):
            raise FileNotFoundError(
                "ERROR: No Synthetic MoVs Metrics Data For CMIP, Aborting."
            )

        print("Found Synthetic MoVs Metrics Data For CMIP, Reading...")
        cmip_lib = self._load_movs_files(cmip_files)

        merge_lib = {}
        for stat, diag_vars in self.parameter["diag_vars"].items():
            merge_df, mode_season_list = self._movs_dict_to_df(cmip_lib, stat)

            for i, model_name in enumerate(self.parameter["model_name"]):
                model_path = self.parameter["test_path"].replace(
                    "%(model_name)", model_name
                )
                model_files = find_latest_file_list(
                    path=f"{model_path}/*/*",
                    file_pattern="var_mode_*.json",
                    var_pattern=self.var_pattern,
                    time_pattern=self.time_pattern,
                )
                if not model_files or not os.path.exists(model_files[0]):
                    raise FileNotFoundError(
                        f"No Synthetic MoVs Metrics Data For {model_name}, Aborting."
                    )

                print(f"Found Synthetic MoVs Metrics for {model_name}, Reading...")
                model_lib = self._load_movs_files(model_files)

                # Normalize model name key to match targets
                model_lib = {
                    mode: {model_name: next(iter(model_data.values()))}
                    for mode, model_data in model_lib.items()
                }

                # Convert dictionary to DataFrame
                model_df, _ = self._movs_dict_to_df(model_lib, stat)

                # Append to the merged DataFrame
                merge_df = pd.concat([merge_df, model_df], ignore_index=True)

            # Highlight and reorder models if applicable
            highlight_models = get_highlight_models(
                merge_df.get("model", []), self.parameter["model_name"]
            )
            for model in merge_df["model"].tolist():
                if model in highlight_models:
                    for idx in merge_df[merge_df["model"] == model].index:
                        merge_df = shift_row_to_bottom(merge_df, idx)

            merge_lib[stat] = merge_df

        return merge_lib, mode_season_list

    def _get_cmip_files(self):
        return glob.glob(
            os.path.join(
                self.parameter["cmip_path"],
                self.cmip_group,
                self.cmip_model,
                self.cmip_version,
                "*/*/var_mode_*.json",
            )
        )

    def _load_movs_files(self, file_lists):
        json_lib = {}
        for mode in self.movs_mode:
            eof = {"PSA1": "EOF2", "NPO": "EOF2", "NPGO": "EOF2", "PSA2": "EOF3"}.get(
                mode, "EOF1"
            )
            for json_file in file_lists:
                if mode in json_file and eof in json_file:
                    try:
                        with open(json_file, "r") as fj:
                            data = json.load(fj)
                            json_lib[mode] = data.get("RESULTS", {})
                    except (FileNotFoundError, json.JSONDecodeError) as e:
                        print(f"Warning: Could not load {json_file}: {e}")
                    break
        return json_lib

    def _movs_dict_to_df(self, movs_dict, stat):
        models = sorted(movs_dict.get("NAM", {}).keys())
        df = pd.DataFrame({"model": models, "num_runs": np.nan})
        mode_season_list = []

        for mode in self.movs_mode:
            seasons = (
                ["monthly"]
                if mode in ["PDO", "NPGO"]
                else ["yearly"] if mode == "AMO" else ["DJF", "MAM", "JJA", "SON"]
            )

            for season in seasons:
                col_name = f"{mode}_{season}"
                df[col_name] = np.nan
                mode_season_list.append(col_name)

                for idx, model in enumerate(models):
                    value = np.nan
                    num_runs = 0

                    if mode in movs_dict and model in movs_dict[mode]:
                        runs = sort_human(list(movs_dict[mode][model].keys()))
                        stat_values = []

                        for run in runs:
                            try:
                                run_stat = movs_dict[mode][model][run][
                                    "defaultReference"
                                ][mode][season]["cbf"][stat]
                                stat_values.append(run_stat)
                            except KeyError:
                                continue

                        if stat_values:
                            value = np.mean(stat_values)
                            num_runs = len(stat_values)

                    df.at[idx, col_name] = value
                    if np.isnan(df.at[idx, "num_runs"]):
                        df.at[idx, "num_runs"] = num_runs
                    elif num_runs > 0:
                        df.at[idx, "num_runs"] = max(df.at[idx, "num_runs"], num_runs)

        return df, mode_season_list


def archive_data(
    region, stat, season, data_dict, model_name, var_names, var_units, outdir
):
    """
    Archive processed data into a CSV file with variable units in column headers if available.

    Parameters:
        region (str): Region name.
        stat (str): Statistic type (e.g., mean, std).
        season (str): Season name.
        data_dict (dict or DataFrame): Data to archive.
        model_name (str): Model identifier.
        var_names (list): List of variable names.
        var_units (list): List of variable units (optional, same order as var_names).
        outdir (str): Directory to save the CSV file.
    """
    df = pd.DataFrame(data_dict)

    # Determine the index of the first variable column (assumes first 3 are metadata)
    metadata_cols = df.columns[:3].tolist()
    variable_cols = df.columns[3:]

    filtered_cols = []
    new_column_names = df.columns.tolist()

    for var in variable_cols:
        if var in var_names:
            filtered_cols.append(var)
            if var_units:
                idx = df.columns.get_loc(var)
                unit_label = var_units[var_names.index(var)]
                new_column_names[idx] = f"{var} ({unit_label})"

    # Subset dataframe and rename columns if units provided
    df = df[metadata_cols + filtered_cols]
    df.columns = new_column_names[: len(df.columns)]

    # Ensure output directory exists
    os.makedirs(outdir, exist_ok=True)

    # Construct and save the output filename
    outfile = f"{stat}_{region}_{season}_{model_name}.csv"
    df.to_csv(os.path.join(outdir, outfile), index=False)

    return


def drop_vars(data_dict, var_names, var_units=None):
    """
    Drop variables (columns) from data_dict where more than 90% of the values are NaN.

    Parameters:
        data_dict (pd.DataFrame): Data containing variable columns.
        var_names (list): List of variable names matching data_dict columns.
        var_units (list, optional): List of units for variables. Must match var_names in order.

    Returns:
        Tuple of (filtered_data_dict, updated_var_names, updated_var_units)
    """
    protected_columns = {"model", "run", "model_run", "num_runs"}
    columns_to_drop = []

    for column in data_dict.columns:
        if column in protected_columns:
            continue
        nan_ratio = data_dict[column].isna().mean()
        if nan_ratio > 0.9:
            columns_to_drop.append(column)

    # Drop columns from DataFrame
    data_dict = data_dict.drop(columns=columns_to_drop)

    # Update var_names and var_units if applicable
    updated_var_names = [v for v in var_names if v not in columns_to_drop]
    updated_var_units = None
    if var_units is not None:
        # Keep units only for remaining variables
        name_to_unit = dict(zip(var_names, var_units))
        updated_var_units = [
            name_to_unit[v] for v in updated_var_names if v in name_to_unit
        ]

    return data_dict, updated_var_names, updated_var_units


class EnsoMetricsReader:
    def __init__(self, parameter, metric, stat):
        self.parameter = parameter
        self.metric = metric
        self.stat = stat
        self.metric_dict = self.parameter["diag_vars"][stat]
        self.metrics_collections = self.metric_dict["collection"]
        self.mips = [self.parameter["cmip_name"].split(".")[0]] + self.parameter[
            "model_name"
        ]
        self.dict_json_path = {}

        self.var_pattern = re.compile(r"\.(\w+)\..*\.v(\d{8})\.json$")
        self.time_pattern = re.compile(r"\.v(\d{8})\.json$")

    def run(self):
        """Collect paths to ENSO metrics JSON files and return the mapping."""
        for mip in self.mips:
            self.dict_json_path[mip] = {}
            for metrics_collection in self.metrics_collections:
                if "cmip" in mip:
                    self.dict_json_path[mip][metrics_collection] = (
                        self._get_cmip_json_path(mip, metrics_collection)
                    )
                else:
                    self.dict_json_path[mip][metrics_collection] = (
                        self._get_test_json_path(mip, metrics_collection)
                    )

            if len(self.dict_json_path[mip]) < 1:
                raise FileNotFoundError(
                    f"No Synthetic ENSO Metrics Data for {mip}, aborting..."
                )

        return self.dict_json_path

    def _get_cmip_json_path(self, mip, metrics_collection):
        path = os.path.join(
            self.parameter["cmip_path"],
            self.parameter["cmip_name"].split(".")[0],
            self.parameter["cmip_name"].split(".")[1],
            self.parameter["cmip_name"].split(".")[2],
            metrics_collection,
            f"{mip.lower()}_{self.parameter['cmip_name'].split('.')[1]}_{metrics_collection}_*.json",
        )
        matches = glob.glob(path)
        if not matches:
            raise FileNotFoundError(
                f"CMIP metrics file not found for {mip} and {metrics_collection}"
            )
        return matches[0]

    def _get_test_json_path(self, mip, metrics_collection):
        for i, model_name in enumerate(self.parameter["model_name"]):
            model_path = self.parameter["test_path"].replace(
                "%(model_name)", model_name
            )
            model_files = find_latest_file_list(
                path=f"{model_path}/{metrics_collection}",
                file_pattern="*.json",
                var_pattern=self.var_pattern,
                time_pattern=self.time_pattern,
            )
            print(f"{model_path}/{metrics_collection}")
            if not model_files or not os.path.exists(model_files[0]):
                raise FileNotFoundError(
                    f"No Synthetic ENSO Metrics Data For {mip} {model_name}, Aborting."
                )

            for json_path in model_files:
                with open(json_path) as ff:
                    data_json = json.load(ff)

            old_key = list(data_json["RESULTS"]["model"].keys())[0]

            data_json["RESULTS"]["model"][mip] = data_json["RESULTS"]["model"].pop(
                old_key
            )

            with open(json_path, "w", encoding="utf8") as ff:
                json.dump(
                    data_json, ff, indent=4, separators=(",", ": "), sort_keys=True
                )

        return json_path


def enso_plot_driver(metric, stat, dict_json_path, parameter, fig_format):
    """
    Driver function to plot ENSO metrics based on specified type (e.g., portrait).
    """
    metric_dict = parameter["diag_vars"][stat]
    metrics_collections = metric_dict["collection"]
    mips = [parameter["cmip_name"].split(".")[0]] + parameter["model_name"]

    for mtype in metric_dict["type"]:
        if mtype == "portrait":
            print(f"Processing Portrait Plots for {metric} {stat}...")

            list_project = mips
            list_obs: List[object] = (
                []
            )  # fill in if observational references are needed
            outdir = os.path.join(parameter["out_dir"], metric)
            os.makedirs(outdir, exist_ok=True)

            outfile = f"{metric}_{stat}_portrait.{fig_format}"
            figure_name = os.path.join(outdir, outfile)

            fig, ref_info_dict = enso_portrait_plot(
                metrics_collections,
                list_project,
                list_obs,
                dict_json_path,
                figure_name=figure_name,
                reduced_set=True,
            )

    return


def variability_modes_plot_driver(
    metric,
    stat,
    model_name,
    metric_dict,
    df_dict,
    mode_season_list,
    save_data,
    out_path,
    fig_format,
):
    """Driver Function for the modes variability metrics plot"""
    season = "mon"
    if len(model_name) > 1:
        mout_name = model_name[0].split("_")[0]
    else:
        mout_name = model_name[0]

    for mtype in metric_dict["type"]:
        if mtype == "portrait":
            print("Processing Portrait  Plots for {} {}....".format(metric, stat))
            if stat not in ["stdv_pc_ratio_to_obs"]:
                data_nor = normalize_by_median(
                    df_dict[mode_season_list].to_numpy().T, axis=1
                )
            else:
                data_nor = df_dict[mode_season_list].to_numpy().T
            if save_data:
                df_dict[mode_season_list] = data_nor.T
                outdir = os.path.join(out_path, metric)
                archive_data(
                    metric,
                    stat,
                    season,
                    df_dict,
                    mout_name,
                    mode_season_list,
                    None,
                    outdir,
                )
            run_list = df_dict["model"].to_list()
            stat_name = metric_dict["name"]
            portrait_metric_plot(
                metric,
                stat,
                season,
                data_nor,
                stat_name,
                model_name,
                mode_season_list,
                run_list,
                out_path,
                fig_format,
            )
        elif mtype == "parcoord":
            print(
                "Processing Parallel Coordinate Plots for {} {}....".format(
                    metric, stat
                )
            )
            # drop data if all is NaNs
            data_dict, var_names, var_units = drop_vars(
                df_dict.copy(), mode_season_list.copy(), None
            )
            if save_data:
                outdir = os.path.join(out_path, metric)
                archive_data(
                    metric,
                    stat,
                    season,
                    data_dict,
                    mout_name,
                    mode_season_list,
                    None,
                    outdir,
                )
            run_list = data_dict["model"].to_list()
            stat_name = metric_dict["name"]
            parcoord_metric_plot(
                metric,
                stat,
                season,
                data_dict,
                stat_name,
                model_name,
                var_names,
                var_units,
                run_list,
                out_path,
                fig_format,
            )

    return


def mean_climate_plot_driver(
    metric,
    stat,
    regions,
    model_name,
    metric_dict,
    df_dict,
    var_list,
    var_unit_list,
    save_data,
    out_path,
    fig_format,
):
    """Driver Function for the mean climate metrics plot"""
    if len(model_name) > 1:
        mout_name = model_name[0].split("_")[0]
    else:
        mout_name = model_name[0]

    for region in regions:
        for mtype in metric_dict["type"]:
            if region in metric_dict["region"]:
                do_plot = True
            else:
                do_plot = False
            if do_plot and mtype == "portrait":
                print(
                    "Processing Portrait  Plots for {} {} {}....".format(
                        metric, region, stat
                    )
                )
                var_names = sorted(var_list.copy())
                # label information
                var_units = []
                for i, var in enumerate(var_names):
                    index = var_list.index(var)
                    var_units.append(var_unit_list[index])
                data_nor = dict()
                for season in metric_dict["season"]:
                    data_dict = df_dict[season][region].copy()
                    if stat == "cor_xy":
                        data_nor[season] = data_dict[var_names].to_numpy().T
                    else:
                        data_nor[season] = normalize_by_median(
                            data_dict[var_names].to_numpy().T, axis=1
                        )
                    if save_data:
                        outdir = os.path.join(out_path, metric, region)
                        outdic = data_dict.drop(columns=["model_run"]).copy()
                        outdic[var_names] = data_nor[season].T
                        archive_data(
                            region,
                            stat,
                            season,
                            data_dict,
                            mout_name,
                            var_names,
                            var_units,
                            outdir,
                        )
                run_list = data_dict["model"].to_list()
                stat_name = metric_dict["name"]
                outdir = os.path.join(out_path, metric)
                portrait_metric_plot(
                    region,
                    stat,
                    metric,
                    data_nor,
                    stat_name,
                    model_name,
                    var_names,
                    run_list,
                    outdir,
                    fig_format,
                )
            elif do_plot and mtype == "parcoord":
                print(
                    "Processing Parallel Coordinate Plots for {} {} {}....".format(
                        metric, region, stat
                    )
                )
                for season in metric_dict["season"]:
                    if season in df_dict.keys():
                        # drop data if all is NaNs
                        data_dict, var_names, var_units = drop_vars(
                            df_dict[season][region].copy(),
                            var_list.copy(),
                            var_unit_list.copy(),
                        )
                        if save_data:
                            outdir = os.path.join(out_path, metric, region)
                            outdic = data_dict.drop(columns=["model_run"]).copy()
                            archive_data(
                                region,
                                stat,
                                season,
                                outdic,
                                mout_name,
                                var_list,
                                var_unit_list,
                                outdir,
                            )
                        run_list = data_dict["model"].to_list()
                        stat_name = metric_dict["name"]
                        outdir = os.path.join(out_path, metric)
                        parcoord_metric_plot(
                            region,
                            stat,
                            metric,
                            data_dict,
                            stat_name,
                            model_name,
                            var_names,
                            var_units,
                            run_list,
                            outdir,
                            fig_format,
                        )
    return


def parcoord_metric_plot(
    region,
    stat,
    group,
    data_dict,
    stat_name,
    model_name,
    var_names,
    var_units,
    model_list,
    out_path,
    fig_format,
):
    """Function for parallel coordinate plots"""
    fontsize = 18
    figsize = (40, 18)
    legend_ncol = int(7 * figsize[0] / 40.0)
    legend_posistion = (0.50, -0.14)
    # hide markers for CMIP models
    identify_all_models = False
    # colors for highlight lines
    xcolors = [
        "#e41a1c",
        "#ff7f00",
        "#4daf4a",
        "#f781bf",
        "#a65628",
        "#984ea3",
        "#377eb8",
        "#dede00",
    ]

    # ensemble mean for E3SM group
    highlight_model1 = get_highlight_models(data_dict.get("model", []), model_name)
    irow_str = data_dict[data_dict["model"] == highlight_model1[0]].index[0]
    irow_end = data_dict[data_dict["model"] == highlight_model1[-1]].index[0] + 1
    data_dict.loc["E3SM MMM"] = data_dict[irow_str:irow_end].mean(
        numeric_only=True, skipna=True
    )
    data_dict.at["E3SM MMM", "model"] = "E3SM (MMM)"

    # ensemble mean for CMIP group
    irow_sub = data_dict[data_dict["model"] == highlight_model1[0]].index[0]
    data_dict.loc["CMIP MMM"] = data_dict[:irow_sub].mean(
        numeric_only=True, skipna=True
    )
    data_dict.at["CMIP MMM", "model"] = "CMIP (MMM)"
    data_dict.loc["E3SM MMM"] = data_dict[irow_sub:].mean(
        numeric_only=True, skipna=True
    )
    data_dict.at["E3SM MMM", "model"] = "E3SM (MMM)"

    model_list = data_dict["model"].to_list()
    highlight_model2 = highlight_model1 + ["CMIP (MMM)", "E3SM (MMM)"]

    # colors for highlight lines
    lncolors = []
    for i, model in enumerate(highlight_model2):
        if model == "CMIP (MMM)":
            lncolors.append("#000000")
        elif model == "E3SM (MMM)":
            lncolors.append("#5b5b5b")  # ("#999999")
        else:
            lncolors.append(xcolors[i])

    var_name1 = sorted(var_names.copy())
    # label information
    var_labels = []
    for i, var in enumerate(var_name1):
        index = var_names.index(var)
        if var_units is not None:
            var_labels.append(var_names[index] + "\n" + var_units[index])
        else:
            var_labels.append(var_names[index])

    # final plot data
    data_var = data_dict[var_name1].to_numpy()

    xlabel = "Metric"
    ylabel = "{} ({})".format(stat_name, stat.upper())
    color_map = "tab20_r"

    if "mean_climate" in [group, region]:
        title = "Model Performance of Annual Climatology ({}, {})".format(
            stat.upper(), region.upper()
        )
    elif "variability_modes" in [group, region]:
        title = "Model Performance of Modes Variability ({})".format(stat.upper())
    elif "enso" in [group, region]:
        title = "Model Performance of ENSO ({})".format(stat.upper())

    fig, ax = parallel_coordinate_plot(
        data_var,
        var_labels,
        model_list,
        model_names2=highlight_model1,
        group1_name="CMIP6",
        group2_name="E3SM",
        models_to_highlight=highlight_model2,
        models_to_highlight_colors=lncolors,
        models_to_highlight_labels=highlight_model2,
        identify_all_models=identify_all_models,
        vertical_center="median",
        vertical_center_line=True,
        title=title,
        figsize=figsize,
        colormap=color_map,
        show_boxplot=False,
        show_violin=True,
        violin_colors=("lightgrey", "pink"),
        legend_ncol=legend_ncol,
        legend_bbox_to_anchor=legend_posistion,
        legend_fontsize=fontsize * 0.85,
        xtick_labelsize=fontsize * 0.95,
        ytick_labelsize=fontsize * 0.95,
        logo_rect=[0, 0, 0, 0],
        logo_off=True,
    )

    ax.set_xlabel(xlabel, fontsize=fontsize * 1.1)
    ax.set_ylabel(ylabel, fontsize=fontsize * 1.1)
    ax.set_title(title, fontsize=fontsize * 1.1)

    # Save figure as an image file
    outdir = os.path.join(out_path, region)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outfile = "{}_{}_parcoord_{}.{}".format(stat, region, group, fig_format)
    fig.savefig(os.path.join(outdir, outfile), facecolor="w", bbox_inches="tight")
    plt.close(fig)

    return


def portrait_metric_plot(
    region,
    stat,
    group,
    data_dict,
    stat_name,
    model_name,
    var_list,
    model_list,
    out_path,
    fig_format,
):
    # process figure
    fontsize = 20
    figsize = (40, 18)
    legend_box_xy = (1.08, 1.18)
    legend_box_size = 4
    legend_lw = 1.5
    shrink = 0.8
    legend_fontsize = fontsize * 0.8

    if group == "mean_climate":
        # data for final plot
        data_all_nor = np.stack(
            [data_dict["djf"], data_dict["mam"], data_dict["jja"], data_dict["son"]]
        )
        legend_on = True
        legend_labels = ["DJF", "MAM", "JJA", "SON"]
    else:
        data_all_nor = data_dict
        legend_on = False
        legend_labels = []

    highlight_models = get_highlight_models(model_list, model_name)
    lable_colors = []
    for model in model_list:
        if model in model_name:
            lable_colors.append("#FC5A50")
        elif "e3sm" in model.lower():
            lable_colors.append("#5170d7")
        else:
            lable_colors.append("#000000")

    var_range: Tuple[float, float]
    if stat in ["cor_xy"]:
        var_range = (0, 1.0)
        cmap_color = "viridis"
        cmap_bounds = np.linspace(0, 1, 21)
    elif stat in ["stdv_pc_ratio_to_obs"]:
        var_range = (0.5, 1.5)
        cmap_color = "jet"
        cmap_bounds = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
        cmap_bounds = [r / 10 for r in range(5, 16, 1)]
    else:
        var_range = (-0.5, 0.5)
        cmap_color = "RdYlBu_r"
        cmap_bounds = np.linspace(-0.5, 0.5, 11)

    fig, ax, cbar = portrait_plot(
        data_all_nor,
        xaxis_labels=model_list,
        yaxis_labels=var_list,
        cbar_label=stat_name,
        cbar_label_fontsize=fontsize * 1.0,
        cbar_tick_fontsize=fontsize,
        box_as_square=True,
        vrange=var_range,
        figsize=figsize,
        cmap=cmap_color,
        cmap_bounds=cmap_bounds,
        cbar_kw={"extend": "both", "shrink": shrink},
        missing_color="white",
        legend_on=legend_on,
        legend_labels=legend_labels,
        legend_box_xy=legend_box_xy,
        legend_box_size=legend_box_size,
        legend_lw=legend_lw,
        legend_fontsize=legend_fontsize,
        logo_rect=[0, 0, 0, 0],
        logo_off=True,
    )

    ax.axvline(x=len(model_list) - len(highlight_models), color="k", linewidth=3)
    ax.set_xticklabels(model_list, rotation=45, va="bottom", ha="left")
    ax.set_yticklabels(var_list, rotation=0, va="center", ha="right")
    for xtick, color in zip(ax.get_xticklabels(), lable_colors):
        xtick.set_color(color)
    ax.yaxis.label.set_color(lable_colors[0])

    # Save figure as an image file
    outdir = os.path.join(out_path, region)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outfile = "{}_{}_portrait_{}.{}".format(stat, region, group, fig_format)
    fig.savefig(os.path.join(outdir, outfile), facecolor="w", bbox_inches="tight")
    plt.close(fig)

    return
