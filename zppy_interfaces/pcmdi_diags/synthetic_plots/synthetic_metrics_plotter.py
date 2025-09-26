import os
import traceback
from collections import OrderedDict
from typing import List, Tuple

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
        logger.info("Generating synthetic metrics plots ...")
        at_least_one_success: bool = False
        for metric in metric_sets:
            logger.info(f"Processing metric: {metric}")
            self.parameter["test_path"] = self.base_test_input_path.replace(
                "%(group_type)", metric
            )
            self.parameter["diag_vars"] = self.metric_dict[metric]

            try:
                if metric == "mean_climate":
                    self._handle_mean_climate(metric)

                elif metric == "variability_modes":
                    self._handle_variability_modes(metric)

                elif metric == "enso_metric":
                    self._handle_enso_metric(metric)
                else:
                    raise ValueError(f"Invalid metric={metric}")
                at_least_one_success = True
            except Exception:
                traceback.print_exc()
                logger.error(f"Failed to handle metric={metric}")
        if not at_least_one_success:
            raise RuntimeError("No synthetic metrics plots could be generated.")

    def _handle_mean_climate(self, metric):
        logger.info(f"Handling mean climate for {metric}")
        self.parameter.update(
            {"cmip_path": self.cmip_clim_dir, "cmip_name": self.cmip_clim_set}
        )

        # Instantiate the collector
        collector = ClimMetricsReader(self.parameter)
        # Collect and merge metrics
        merge_lib = collector.collect()

        # merge_lib = collect_clim_metrics(self.parameter)
        for stat, vars_ in self.metric_dict[metric].items():
            logger.debug(
                f"Running mean climate plot driver for stat={stat} on metric_dict={vars_}"
            )
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
        logger.info(f"Handling variability_modes for {metric}")
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
        logger.info(f"Handling enso_metric for {metric}")
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
