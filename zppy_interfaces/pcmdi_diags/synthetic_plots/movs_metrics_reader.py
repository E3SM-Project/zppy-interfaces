import glob
import json
import os
import re

import numpy as np
import pandas as pd
from pcmdi_metrics.utils import sort_human

from zppy_interfaces.multi_utils.logger import _setup_child_logger
from zppy_interfaces.pcmdi_diags.synthetic_plots.utils import (
    find_latest_file_list,
    get_highlight_models,
    shift_row_to_bottom,
)

logger = _setup_child_logger(__name__)


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

        logger.info("Found Synthetic MoVs Metrics Data For CMIP, Reading...")
        cmip_lib = self._load_movs_files(cmip_files)

        merge_lib = {}
        for stat, diag_vars in self.parameter["diag_vars"].items():
            merge_df, mode_season_list = self._movs_dict_to_df(cmip_lib, stat)

            for i, model_name in enumerate(self.parameter["model_name"]):
                model_path = self.parameter["test_path"].replace(
                    "put_model_here", model_name
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

                logger.info(
                    f"Found Synthetic MoVs Metrics for {model_name}, Reading..."
                )
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
                        logger.info(f"Warning: Could not load {json_file}: {e}")
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
