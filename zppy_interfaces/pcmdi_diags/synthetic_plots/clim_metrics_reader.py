import glob
import json
import os
import re

import numpy as np
import pandas as pd
from pcmdi_metrics.graphics import Metrics

from zppy_interfaces.multi_utils.logger import _setup_child_logger
from zppy_interfaces.pcmdi_diags.synthetic_plots.clim_metrics_merger import (
    ClimMetricsMerger,
)
from zppy_interfaces.pcmdi_diags.synthetic_plots.utils import find_latest_file_list

logger = _setup_child_logger(__name__)


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
        logger.info("file_paths=")
        for i, fp in enumerate(file_paths):
            logger.info(f"{i}. {fp}")
        """
        FAILURE:

        list(results_dict_var["RESULTS"][model_list[0]]["default"][run_list[0]].keys())
        IndexError: list index out of range

        cat /lcrc/group/e3sm/public_html/diagnostic_output/ac.forsyth2/zppy_pr719_output/unique_id_21/v3.LR.amip_0101/pcmdi_diags/model_vs_obs/metrics_data/mean_climate/rlus.2.5x2.5.e3sm.amip.v3-LR_0101.v20250725.json

        "RESULTS": {
            "v3-LR": {
                "default": {
                    "source": "ceres_ebaf_v4_1"
                }
            }
        },

        SYNTHETIC PLOTS ERROR #1: "source" is supposed to be a dictionary itself, even though mean_climate job completed successfully!
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

        logger.info(f"Loading CMIP metrics from {len(cmip_files)} files...")
        self.cmip_lib = self._load_clim_metrics_from_files(cmip_files)

    def _process_test_model(self, test_name, model_name):
        test_key = test_name.split(".")[1]
        test_path = self.parameter["test_path"].replace("put_model_here", model_name)

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

        logger.info(
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
                    logger.info(f"Updated file: {file_path}")

                valid_model_files.append(file_path)

            except (FileNotFoundError, json.JSONDecodeError) as e:
                logger.info(f"Warning: Could not load {file_path}: {e}")

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
        if self.cmip_lib and hasattr(self.cmip_lib, "var_list"):
            logger.info(
                f"ClimMetricsReader.cmip_lib.vars_list: {self.cmip_lib.var_list}"
            )

        for i, (test_name, model_name) in enumerate(
            zip(self.parameter["test_name"], self.parameter["model_name"])
        ):
            logger.info(
                f"Processing model {i + 1}: test_name={test_name}, model_name={model_name}"
            )
            model_lib = self._process_test_model(test_name, model_name)
            if hasattr(model_lib, "var_list"):
                logger.info(f"model_lib.vars_list: {model_lib.var_list}")
            self.all_lib = (
                model_lib.copy()
                if self.all_lib is None
                else self.all_lib.merge(model_lib)
            )
            self.all_names.append(model_name)

        logger.info("Merging model metrics with CMIP reference metrics...")
        if self.all_lib and hasattr(self.all_lib, "var_list"):
            logger.info(f"ClimMetricsReader.all_lib.vars_list: {self.all_lib.var_list}")
        merger = ClimMetricsMerger(
            model_lib=self.all_lib, cmip_lib=self.cmip_lib, model_names=self.all_names
        )
        merged_metrics = merger.merge()  # Returns a new merged metrics library

        return merged_metrics


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
