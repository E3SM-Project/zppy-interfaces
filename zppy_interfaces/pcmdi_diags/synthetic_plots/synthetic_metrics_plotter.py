import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pcmdi_metrics.enso.lib import enso_portrait_plot
from pcmdi_metrics.graphics import (
    normalize_by_median,
    parallel_coordinate_plot,
    portrait_plot,
)

from zppy_interfaces.multi_utils.logger import _setup_child_logger
from zppy_interfaces.pcmdi_diags.synthetic_plots.clim_metrics_reader import (
    ClimMetricsReader,
)
from zppy_interfaces.pcmdi_diags.synthetic_plots.enso_metrics_reader import (
    EnsoMetricsReader,
)
from zppy_interfaces.pcmdi_diags.synthetic_plots.movs_metrics_reader import (
    MoVsMetricsReader,
)
from zppy_interfaces.pcmdi_diags.synthetic_plots.utils import get_highlight_models

logger = _setup_child_logger(__name__)


class SyntheticMetricsPlotter:
    def __init__(
        self,
        case_name: str,
        test_name: str,
        table_id: str,
        figure_format: str,
        metric_dict: Dict[str, Any],
        save_data: bool,
        base_test_input_path: str,
        results_dir: Optional[str] = None,
        # Mean-climate viewer
        clim_viewer: bool = True,
        clim_vars: Optional[Union[List[str], str]] = None,
        clim_regions: Optional[Union[List[str], str]] = None,
        cmip_clim_dir: Optional[str] = None,
        cmip_clim_set: Optional[str] = None,
        # Atmosphere modes (MOVA)
        mova_viewer: bool = True,
        mova_modes: Optional[Union[List[str], str]] = None,
        # Coupled modes (MOVC)
        movc_viewer: bool = True,
        movc_modes: Optional[Union[List[str], str]] = None,
        cmip_movs_dir: Optional[str] = None,
        cmip_movs_set: Optional[str] = None,
        # ENSO viewer
        enso_viewer: bool = True,
        cmip_enso_dir: Optional[str] = None,
        cmip_enso_set: Optional[str] = None,
    ):
        # Core
        self.case_name = case_name
        self.test_name = test_name
        self.table_id = table_id
        self.figure_format = figure_format
        self.metric_dict = metric_dict
        self.save_data = bool(save_data)
        self.base_test_input_path = base_test_input_path
        self.results_dir = results_dir or "."

        # Mean climate
        self.clim_viewer = bool(clim_viewer)
        self.clim_vars = self._to_list(clim_vars)  # [] => all available
        self.clim_regions = self._to_list(clim_regions)  # [] => all regions
        self.cmip_clim_dir = cmip_clim_dir
        self.cmip_clim_set = cmip_clim_set

        # MOVA
        self.mova_viewer = bool(mova_viewer)
        self.mova_modes = self._to_list(mova_modes) if self.mova_viewer else []
        self.movc_viewer = bool(movc_viewer)
        self.movc_modes = self._to_list(movc_modes) if self.movc_viewer else []
        self.movs_viewer = self.mova_viewer or self.movc_viewer

        self.cmip_movs_dir = cmip_movs_dir
        self.cmip_movs_set = cmip_movs_set

        # ENSO
        self.enso_viewer = bool(enso_viewer)
        self.cmip_enso_dir = cmip_enso_dir
        self.cmip_enso_set = cmip_enso_set

        # Final bundle for downstream readers/builders
        self.parameter: Dict[str, Any] = self._initialize_parameter()

    # ---------- helpers ----------
    @staticmethod
    def _to_list(value: Optional[Union[List[str], str]]) -> List[str]:
        """Accept None | list[str] | comma/space-separated str -> List[str]."""
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        s = str(value).strip()
        if not s:
            return []
        parts = s.split(",") if "," in s else s.split()
        return [p.strip() for p in parts if p.strip()]

    def _initialize_parameter(self):
        # Parse comma-separated lists
        tests = [t.strip() for t in str(self.test_name).split(",") if t.strip()]
        cases = [c.strip() for c in str(self.case_name).split(",") if c.strip()]

        if len(tests) != len(cases):
            raise ValueError(
                f"test_name count ({len(tests)}) != case_name count ({len(cases)}). "
                "They must align positionally."
            )

        parsed_test_names = []
        parsed_model_names = []

        for raw_test, raw_case in zip(tests, cases):
            parts = raw_test.split(".")
            if len(parts) != 4:
                raise ValueError(
                    f"Invalid test format '{raw_test}'. Expected 'a.b.c.d'"
                )
            # Re-map 'a.b.c.d' -> 'a.b.c_d' (your original behavior)
            test_id = f"{parts[0]}.{parts[1]}.{parts[2]}_{parts[3]}"
            parsed_test_names.append(test_id)
            parsed_model_names.append(raw_case)

        # Keep the exact keys your pipeline expects
        param = OrderedDict(
            {
                "save_data": self.save_data,
                "out_dir": os.path.join(self.results_dir, "ERROR_metric"),
                "test_name": parsed_test_names,
                "model_name": parsed_model_names,
                "tableID": [self.table_id],
            }
        )

        return param

    def generate(self) -> None:
        logger.info("Generating synthetic metrics plots ...")
        tasks = [
            (self.clim_viewer, "mean_climate", self._handle_mean_climate),
            (self.movs_viewer, "variability_modes", self._handle_variability_modes),
            (self.enso_viewer, "enso_metric", self._handle_enso_metric),
        ]

        at_least_one_success = False
        failures = []

        for enabled, metric, handler in tasks:
            self.parameter["test_path"] = self.base_test_input_path.replace(
                "%(group_type)", metric
            )
            self.parameter["diag_vars"] = self.metric_dict[metric]
            if not enabled:
                continue
            logger.info("Processing metric: %s", metric)
            try:
                handler(metric)
                at_least_one_success = True
            except Exception as e:
                logger.error("Failed to handle metric=%s: %s", metric, e, exc_info=True)
                failures.append(metric)

        if not at_least_one_success:
            raise RuntimeError("No synthetic metrics plots could be generated.")

        if failures:
            logger.warning("Completed with partial failures: %s", ", ".join(failures))

    def _handle_mean_climate(self, metric: str) -> None:
        logger.info("Handling mean climate…")
        self.parameter.update(
            {"cmip_path": self.cmip_clim_dir, "cmip_name": self.cmip_clim_set}
        )

        collector = ClimMetricsReader(self.parameter)
        merge_lib = collector.collect()

        # Variables (preserve original behavior unless filters are provided)
        var_list = list(merge_lib.var_list)
        var_unit_list = list(merge_lib.var_unit_list)
        if self.clim_vars is not None:
            name_to_unit = dict(zip(merge_lib.var_list, merge_lib.var_unit_list))
            missing = [v for v in self.clim_vars if v not in name_to_unit]
            if missing:
                logger.warning(
                    f"[mean_climate] Requested variables not found and will be skipped: {missing}"
                )
            var_list = [v for v in self.clim_vars if v in name_to_unit]
            var_unit_list = [name_to_unit[v] for v in var_list]

        # Regions (preserve order)
        regions = list(merge_lib.regions)
        if self.clim_regions is not None:
            missing_r = [r for r in self.clim_regions if r not in merge_lib.regions]
            if missing_r:
                logger.warning(
                    f"[mean_climate] Requested regions not found and will be skipped: {missing_r}"
                )
            regions = [r for r in self.clim_regions if r in merge_lib.regions]

        # Use the same `metric` variable as before (assuming it's defined in scope)
        for stat, vars_ in self.metric_dict[metric].items():
            logger.debug(f"[mean_climate] Running plot driver: stat={stat}")
            # Keep the exact positional calling convention you had before
            mean_climate_plot_driver(
                metric,
                stat,
                regions,
                self.parameter["model_name"],
                vars_,
                merge_lib.df_dict[stat],
                var_list,
                var_unit_list,
                self.parameter["save_data"],
                self.parameter["out_dir"],
                self.figure_format,
            )

    def _handle_variability_modes(self, metric: str) -> None:
        logger.info("Handling modes variability …")

        # Combine atmospheric and coupled modes (already lists)
        modes_list = (self.mova_modes or []) + (self.movc_modes or [])

        if not modes_list:
            logger.warning(
                "[variability_modes] No modes specified; skipping variability mode plots."
            )
            return

        # Update parameters for reader
        self.parameter.update(
            {
                "cmip_path": self.cmip_movs_dir,
                "cmip_name": self.cmip_movs_set,
                "movs_mode": modes_list,
            }
        )

        # Collect metrics
        reader = MoVsMetricsReader(self.parameter)
        merge_lib, mode_season_list = reader.collect_metrics()

        # Ensure metric exists in dictionary
        if metric not in self.metric_dict:
            logger.error(
                f"[variability_modes] Metric '{metric}' not found in metric_dict keys={list(self.metric_dict.keys())}"
            )
            return

        # Loop through stats and plot
        for stat, vars_ in self.metric_dict[metric].items():
            if stat not in merge_lib:
                logger.warning(
                    f"[variability_modes] stat='{stat}' not found in merge_lib; available={list(merge_lib.keys())}"
                )
                continue

            logger.debug(f"[variability_modes] Running plot driver for stat={stat}")
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

    def _handle_enso_metric(self, metric: str) -> None:
        logger.info("Handling ENSO metrics…")

        # Update paths
        self.parameter.update(
            {
                "cmip_path": self.cmip_enso_dir,
                "cmip_name": self.cmip_enso_set,
            }
        )

        # --- Build enso_mips: [<MIP tag from cmip_name>] + model_name(s) ---
        cmip_name = self.parameter.get("cmip_name", "")
        mip_tag = (
            cmip_name.split(".")[0]
            if isinstance(cmip_name, str) and cmip_name
            else None
        )

        model_name = self.parameter.get("model_name", [])
        if isinstance(model_name, str):
            model_name = [model_name]
        elif isinstance(model_name, tuple):
            model_name = list(model_name)
        elif not isinstance(model_name, list):
            logger.warning(
                f"[enso] Unexpected model_name type: {type(model_name).__name__}; coercing to list if possible."
            )
            model_name = list(model_name) if model_name is not None else []

        enso_mips = ([mip_tag] if mip_tag else []) + model_name
        if not enso_mips:
            logger.warning(
                "[enso] No MIP/model names resolved for ENSO; continuing with empty list."
            )

        # --- Collections (optional config) ---
        enso_collections = self.metric_dict.get("collection", [])
        if not isinstance(enso_collections, (list, tuple)):
            logger.warning(
                f"[enso] 'collection' should be list/tuple; got {type(enso_collections).__name__}. Using empty list."
            )
            enso_collections = []

        # --- Validate metric entry ---
        if metric not in self.metric_dict or not isinstance(
            self.metric_dict[metric], dict
        ):
            logger.error(
                f"[enso] metric_dict['{metric}'] missing or not a dict. Available: {list(self.metric_dict.keys())}"
            )
            return

        diag_vars_all = self.parameter.get("diag_vars", {})
        if not isinstance(diag_vars_all, dict):
            logger.error(
                f"[enso] parameter['diag_vars'] must be a dict; got {type(diag_vars_all).__name__}."
            )
            return

        # --- Main loop over stats ---
        for stat in self.metric_dict[metric].keys():
            metric_dict = diag_vars_all.get(stat, {})
            if not metric_dict:
                logger.warning(
                    f"[enso] No variables configured for stat='{stat}'. Skipping."
                )
                continue

            logger.debug(
                f"[enso] stat='{stat}', enso_mips={enso_mips}, collections={enso_collections}"
            )
            try:
                reader = EnsoMetricsReader(
                    self.parameter, stat, metric_dict, enso_mips, enso_collections
                )
                dict_json_path = reader.run()
            except Exception as e:
                logger.exception(f"[enso] Reader failed for stat='{stat}': {e}")
                continue

            if not dict_json_path:
                logger.warning(
                    f"[enso] Reader returned empty path for stat='{stat}'. Skipping plot."
                )
                continue

            try:
                enso_plot_driver(
                    metric, stat, dict_json_path, self.parameter, self.figure_format
                )
                logger.debug(f"[enso] Plotted stat='{stat}' successfully.")
            except Exception as e:
                logger.exception(f"[enso] Plot driver failed for stat='{stat}': {e}")


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
                logger.info(
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
                        logger.debug(
                            f"var_names={var_names} derived from var_list={var_list}."
                        )
                        logger.debug(f"Available columns: {data_dict.columns.tolist()}")
                        try:
                            data_nor[season] = normalize_by_median(
                                data_dict[var_names].to_numpy().T, axis=1
                            )
                        except KeyError as e:
                            logger.error(f"KeyError on var_names={var_names}")
                            raise e
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
                logger.info(
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
            logger.info("Processing Portrait  Plots for {} {}....".format(metric, stat))
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
            logger.info(
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


def enso_plot_driver(metric, stat, dict_json_path, parameter, fig_format):
    """
    Driver function to plot ENSO metrics based on specified type (e.g., portrait).
    """
    metric_dict = parameter["diag_vars"][stat]
    metrics_collections = metric_dict["collection"]
    mips = [parameter["cmip_name"].split(".")[0]] + parameter["model_name"]

    for mtype in metric_dict["type"]:
        if mtype == "portrait":
            logger.info(f"Processing Portrait Plots for {metric} {stat}...")

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
    base_fontsize=20,
    base_figsize=(40, 18),
    base_legend_lw=1.5,
    box_as_square=True,
    missing_color="white",
    logo_rect=[0, 0, 0, 0],
    logo_off=True,
):
    # === Figure scaling setup ===
    fscale = len(var_list) / 30.0
    fscale = max(0.5, min(fscale, 1.5))  # clamp to avoid extremes

    # Apply scaled parameters
    fontsize = base_fontsize
    figsize = (base_figsize[0], base_figsize[1] * fscale)
    legend_box_xy = (1.08, 1.20)
    legend_box_size = 4 * fscale
    legend_lw = base_legend_lw * fscale
    shrink = 0.8 * fscale
    legend_fontsize = fontsize * 0.8

    # --- SMALL GUARD A: basic inputs present ---
    if not var_list:
        logger.warning("[Portrait]: No variables to plot (var_list empty); returning.")
        return
    if not model_list:
        logger.warning("[Portrait]: No models to plot (model_list empty); returning.")
        return

    if group == "mean_climate":
        # --- SMALL GUARD B: seasonal arrays exist & stack cleanly ---
        required = ["djf", "mam", "jja", "son"]
        missing = [k for k in required if (k not in data_dict or data_dict[k] is None)]
        if missing:
            logger.warning(
                "[Portrait]: Missing seasonal arrays for %s; returning. Missing=%s",
                group,
                missing,
            )
            return
        try:
            arrs = [np.asarray(data_dict[k]) for k in required]
            if any(a.size == 0 for a in arrs):
                logger.warning(
                    "[Portrait]: One or more seasonal arrays are empty; returning."
                )
                return
            data_all_nor = np.stack(arrs)
        except Exception as e:
            logger.warning(
                "[Portrait]: Failed to stack seasonal arrays: %s; returning.", e
            )
            return

        legend_on = True
        legend_labels = ["DJF", "MAM", "JJA", "SON"]
    else:
        # --- SMALL GUARD C: non-seasonal data present ---
        data_all_nor = np.asarray(data_dict)
        if data_all_nor.size == 0:
            logger.warning("[Portrait]: Input data array is empty; returning.")
            return
        legend_on = False
        legend_labels = []

    # --- SMALL GUARD D: minimal shape sanity (avoid cryptic errors downstream) ---
    if data_all_nor.ndim < 2:
        logger.warning(
            "[Portrait]: Data has ndim=%d (<2); returning.", data_all_nor.ndim
        )
        return

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
        cmap_bounds = [r / 10 for r in range(5, 16, 1)]
    else:
        var_range = (-0.5, 0.5)
        cmap_color = "RdYlBu_r"
        cmap_bounds = np.linspace(-0.5, 0.5, 11)

    fig, ax, cbar = portrait_plot(
        data_all_nor,
        xaxis_labels=model_list,
        yaxis_labels=var_list,
        cbar_label=stat,
        cbar_label_fontsize=fontsize * 0.95,
        cbar_tick_fontsize=fontsize * 0.95,
        box_as_square=box_as_square,
        vrange=var_range,
        figsize=figsize,
        cmap=cmap_color,
        cmap_bounds=cmap_bounds,
        cbar_kw={"extend": "both", "shrink": shrink},
        missing_color=missing_color,
        legend_on=legend_on,
        legend_labels=legend_labels,
        legend_box_xy=legend_box_xy,
        legend_box_size=legend_box_size,
        legend_lw=legend_lw,
        legend_fontsize=legend_fontsize,
        logo_rect=logo_rect,
        logo_off=logo_off,
    )

    ax.axvline(x=len(model_list) - len(highlight_models), color="k", linewidth=3)
    ax.set_xticklabels(model_list, rotation=45, va="bottom", ha="left")
    ax.set_yticklabels(var_list, rotation=0, va="center", ha="right")
    for xtick, color in zip(ax.get_xticklabels(), lable_colors):
        xtick.set_color(color)
    ax.yaxis.label.set_color(lable_colors[0])

    # Add title
    fig.suptitle(
        f"{region} — {group} ({stat_name})",
        fontsize=fontsize * 1.1,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # leave top 5 % free for title

    # Save figure as an image file
    outdir = os.path.join(out_path, region)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outfile = "{}_{}_portrait_{}.{}".format(stat, region, group, fig_format)
    fig.savefig(os.path.join(outdir, outfile), facecolor="w", bbox_inches="tight")
    plt.close(fig)

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
    base_fontsize=20,
    base_figsize=(60, 20),
    base_legend_lw=1.5,
    color_map="tab20_r",
    xcolors=None,
    group1_name="CMIP",
    mean1_name="CMIP (Mean)",
    group2_name="E3SM",
    mean2_name="E3SM (Mean)",
    identify_all_models=True,
    vertical_center="median",
    vertical_center_line=True,
    show_boxplot=False,
    show_violin=True,
    violin_colors=("lightgrey", "pink"),
    logo_rect=[0, 0, 0, 0],
    logo_off=True,
):
    """Function for parallel coordinate plots"""
    # === Figure scaling setup ===
    fscale = len(var_names) / 30.0
    fscale = max(0.6, min(fscale, 1.5))  # clamp to avoid extremes

    fontsize = base_fontsize
    figsize = (base_figsize[0] * fscale, base_figsize[1] * fscale)

    legend_ncol = int(7 * figsize[0] / 40.0)
    legend_posistion = (0.50, -0.14)

    # colors for highlight lines
    if xcolors is None:
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
    # --- SMALL GUARD 1: highlight models may be missing ---
    highlight_model1 = get_highlight_models(data_dict.get("model", []), model_name)
    if not highlight_model1:
        # No highlightable models → skip means/highlights later
        highlight_model1 = []
        have_highlights = False
    else:
        have_highlights = True

    # Only compute E3SM mean if we found highlight rows in the DF
    if have_highlights:
        if (highlight_model1[0] in set(data_dict["model"])) and (
            highlight_model1[-1] in set(data_dict["model"])
        ):
            irow_str = data_dict.index[data_dict["model"] == highlight_model1[0]][0]
            irow_end = (
                data_dict.index[data_dict["model"] == highlight_model1[-1]][0] + 1
            )
            data_dict.loc[mean2_name] = data_dict.iloc[irow_str:irow_end].mean(
                numeric_only=True, skipna=True
            )
            data_dict.at[mean2_name, "model"] = mean2_name
        else:
            have_highlights = False  # fallback if names weren’t found

    # CMIP/E3SM means only if we can split reliably
    if have_highlights:
        irow_sub = data_dict.index[data_dict["model"] == highlight_model1[0]][0]
        data_dict.loc[mean1_name] = data_dict.iloc[:irow_sub].mean(
            numeric_only=True, skipna=True
        )
        data_dict.at[mean1_name, "model"] = mean1_name
        data_dict.loc[mean2_name] = data_dict.iloc[irow_sub:].mean(
            numeric_only=True, skipna=True
        )
        data_dict.at[mean2_name, "model"] = mean2_name

    # --- SMALL GUARD 1: highlights models
    if not have_highlights:
        logger.warning(
            f"[ParCoord]: No highlightable models found for model_name={model_name}; "
            f"Skipping highlight and mean calculations."
        )

    model_list = data_dict["model"].astype(str).to_list()
    highlight_model2 = highlight_model1 + (
        [mean1_name, mean2_name] if have_highlights else []
    )

    # colors for highlight lines
    lncolors = []
    for i, model in enumerate(highlight_model2):
        if model == mean1_name:
            lncolors.append("#000000")
        elif model == mean2_name:
            lncolors.append("#5b5b5b")  # ("#999999")
        else:
            lncolors.append(xcolors[i % len(xcolors)])

    # --- SMALL GUARD 2: keep only existing, non-empty vars ---
    var_name1 = sorted(
        v for v in var_names if (v in data_dict.columns) and data_dict[v].notna().any()
    )
    if not var_name1:
        logger.warning(
            f"[ParCoord]: Nothing to plot for group={group}, region={region}, stat={stat}. "
            f"No valid variables found in metrics data (columns checked={len(var_names)})."
        )
        return

    # label information
    var_labels = []
    for v in var_name1:
        idx = var_names.index(v)
        if var_units is not None and idx < len(var_units):
            var_labels.append(var_names[idx] + "\n" + var_units[idx])
        else:
            var_labels.append(var_names[idx])

    # final plot data
    data_var = data_dict[var_name1].to_numpy()

    # --- SMALL GUARD 3: ensure at least 1 column for parallel-coords ---
    if data_var.ndim != 2 or data_var.shape[1] == 0:
        logger.warning(
            f"[ParCoord]: Not enough data to process parallel coordinate plots "
            f"(shape={data_var.shape}); returning without plot."
        )
        return

    xlabel = "Metric"
    ylabel = "{} ({})".format(stat_name, stat.upper())

    if "mean_climate" in [group, region]:
        title = f"Model Performance of Annual Climatology ({stat.upper()}, {region.upper()})"
    elif "variability_modes" in [group, region]:
        title = f"Model Performance of Modes Variability ({stat.upper()})"
    elif "enso" in [group, region]:
        title = f"Model Performance of ENSO ({stat.upper()})"
    else:
        title = f"Model Performance ({stat.upper()}, {region.upper()})"

    fig, ax = parallel_coordinate_plot(
        data_var,
        var_labels,
        model_list,
        model_names2=highlight_model1,
        group1_name=group1_name,
        group2_name=group2_name,
        models_to_highlight=highlight_model2,
        models_to_highlight_colors=lncolors,
        models_to_highlight_labels=highlight_model2,
        identify_all_models=identify_all_models,
        vertical_center=vertical_center,
        vertical_center_line=vertical_center_line,
        title="",
        figsize=figsize,
        colormap=color_map,
        show_boxplot=show_boxplot,
        show_violin=show_violin,
        violin_colors=violin_colors,
        legend_ncol=legend_ncol,
        legend_bbox_to_anchor=legend_posistion,
        legend_fontsize=fontsize * 0.85,
        xtick_labelsize=fontsize * 0.95,
        ytick_labelsize=fontsize * 0.95,
        logo_rect=logo_rect,
        logo_off=logo_off,
    )

    ax.set_xlabel(xlabel, fontsize=fontsize * 1.05)
    ax.set_ylabel(ylabel, fontsize=fontsize * 1.05)
    # ax.set_title(title, fontsize=fontsize * 1.05)

    # Add title
    fig.suptitle(f"{title}", fontsize=fontsize * 1.05, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # leave top 5 % free for title

    # Save figure as an image file
    outdir = os.path.join(out_path, region)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outfile = "{}_{}_parcoord_{}.{}".format(stat, region, group, fig_format)
    fig.savefig(os.path.join(outdir, outfile), facecolor="w", bbox_inches="tight")
    plt.close(fig)

    return
