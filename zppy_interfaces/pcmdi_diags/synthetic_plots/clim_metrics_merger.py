from collections.abc import MutableMapping
from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from zppy_interfaces.multi_utils.logger import _setup_child_logger
from zppy_interfaces.pcmdi_diags.synthetic_plots.utils import (
    get_highlight_models,
    shift_row_to_bottom,
)

logger = _setup_child_logger(__name__)


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
        if hasattr(cleaned_model_lib, "var_list"):
            logger.debug(f"cleaned_model_lib.var_list: {cleaned_model_lib.var_list}")

        # Safe merge with fallback for missing stats/seasons/regions
        self.merged_lib = self._safe_merge_libs(self.cmip_lib, cleaned_model_lib)
        if hasattr(self.merged_lib, "var_list"):
            logger.debug(f"merged_lib.var_list: {self.merged_lib.var_list}")
        if hasattr(cleaned_model_lib, "var_list") and hasattr(
            self.cmip_lib, "var_list"
        ):
            var_set_cleaned_model_lib = set(cleaned_model_lib.var_list)
            var_set_cmip_lib = set(self.cmip_lib.var_list)
            logger.debug(
                f"Var list sizes - cleaned_model_lib: {len(var_set_cleaned_model_lib)}, cmip_lib: {len(var_set_cmip_lib)}"
            )
            logger.debug(
                f"Var list differences - in cleaned_model_lib not in cmip_lib: {var_set_cleaned_model_lib - var_set_cmip_lib}"
            )
            logger.debug(
                f"Var list differences - in cmip_lib not in cleaned_model_lib: {var_set_cmip_lib - var_set_cleaned_model_lib}"
            )

        # Standardize units after merging
        self.merged_lib = self._check_units(self.merged_lib)
        if hasattr(self.merged_lib, "var_list"):
            logger.debug(
                f"Post-unit-check merged_lib.var_list: {self.merged_lib.var_list}"
            )

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
                    logger.info(f"Alias {var} mapped to 'rtmt' in references.")

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
                        logger.info(
                            f"Dropping variables in {stat}/{season}/{region}: {drop_cols}"
                        )
                    df = df.drop(columns=drop_cols)
                    data_lib.df_dict[stat][season][region] = df

        logger.debug(f"Setting data_lib.var_list={common_vars}")
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
