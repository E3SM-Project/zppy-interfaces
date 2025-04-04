import glob
import math
import os
import traceback

import matplotlib as mpl
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset

from zppy_interfaces.global_time_series.utils import Parameters
from zppy_interfaces.multi_utils.logger import _setup_custom_logger

mpl.use("Agg")

logger = _setup_custom_logger(__name__)


# ---additional function to get moc time series
def getmoc(dir_in):
    files = sorted(glob.glob(dir_in + "mocTimeSeries*.nc"))
    nfiles = len(files)
    logger.info(f"{dir_in} {nfiles} moc files in total")
    var = np.array([])
    time = np.array([])
    for i in range(nfiles):
        # Open input file
        fin = Dataset(files[i], "r")
        time0 = fin["year"][:]
        var0 = fin["mocAtlantic26"][:]
        for iyear in range(int(time0[0]), int(time0[-1]) + 1):
            if i > 0 and iyear <= time[-1]:
                logger.info(
                    f"the amoc value for year {iyear} has been included in the moc time series from another moc file {files[i - 1]} {time[-1]} Skipping..."
                )
            else:
                imon = np.where(time0 == iyear)[0]
                if len(imon) == 12:
                    var = np.append(var, np.mean(var0[imon]))
                    time = np.append(time, iyear)
                else:
                    logger.error(f"error in input file : {files[i]}")

    return time, var


# -----------------------------------------------------------------------------
# Function to add horizontal line showing average value over a specified period
def add_line(year, var, year1, year2, ax, format="%4.2f", lw=1, color="b"):

    i1 = (np.abs(year - year1)).argmin()
    i2 = (np.abs(year - year2)).argmin()

    tmp = np.average(var[i1 : i2 + 1])
    ax.plot((year[i1], year[i2]), (tmp, tmp), lw=lw, color=color, label="average")
    ax.text(ax.get_xlim()[1] + 1, tmp, format % tmp, va="center", color=color)

    return


# -----------------------------------------------------------------------------
# Function to add line showing linear trend over a specified period
def add_trend(
    year,
    var,
    year1,
    year2,
    ax,
    format="%4.2f",
    lw=1,
    color="b",
    verbose=False,
    ohc=False,
    vol=False,
):

    i1 = (np.abs(year - year1)).argmin()
    i2 = (np.abs(year - year2)).argmin()
    x = year[i1 : i2 + 1]
    y = var[i1 : i2 + 1]

    fit = np.polyfit(x, y, 1)
    if verbose:
        logger.info(fit)
    fit_fn = np.poly1d(fit)
    ax.plot(x, fit_fn(x), lw=lw, ls="--", c=color, label="trend")
    if ohc:
        # Earth radius 6371229. from MPAS-O output files
        heat_uptake = fit[0] / (4.0 * math.pi * (6371229.0) ** 2 * 365.0 * 86400.0)
        ax.text(
            ax.get_xlim()[1] + 1,
            fit_fn(x[-1]),
            "%+4.2f W m$^{-2}$" % (heat_uptake),
            color=color,
        )
    if vol:
        # Earth radius 6371229. from MPAS-O output files
        # sea_lvl = fit[0] / ( 4.0*math.pi*(6371229.)**2*0.7)      #for oceanic portion of the Earth surface
        ax.text(
            ax.get_xlim()[1] + 1,
            fit_fn(x[-1]),
            "%+5.4f mm yr$^{-1}$" % (fit[0]),
            color=color,
        )

    return


# -----------------------------------------------------------------------------
# Function to get ylim
def get_ylim(standard_range, extreme_values):
    if len(extreme_values) > 0:
        has_extreme_values = True
        extreme_min = np.amin(extreme_values)
        extreme_max = np.amax(extreme_values)
    else:
        has_extreme_values = False
        extreme_min = None
        extreme_max = None
    if len(standard_range) == 2:
        has_standard_range = True
        standard_min = standard_range[0]
        standard_max = standard_range[1]
    else:
        has_standard_range = False
        standard_min = None
        standard_max = None
    if has_extreme_values and has_standard_range:
        # Use at least the standard range,
        # perhaps a wider window to include extremes
        if standard_min <= extreme_min:
            ylim_min = standard_min
        else:
            ylim_min = extreme_min
        if standard_max >= extreme_max:
            ylim_max = standard_max
        else:
            ylim_max = extreme_max
    elif has_extreme_values and not has_standard_range:
        ylim_min = extreme_min
        ylim_max = extreme_max
    elif has_standard_range and not has_extreme_values:
        ylim_min = standard_min
        ylim_max = standard_max
    else:
        raise ValueError("Not enough range information supplied")
    return [ylim_min, ylim_max]


# -----------------------------------------------------------------------------
# Plotting functions


# 1
def plot_net_toa_flux_restom(ax, xlim, exps, rgn):
    logger.info("Plot 1: plot_net_toa_flux_restom")
    param_dict = {
        "2nd_var": False,
        "axhline_y": 0,
        "check_exp_ocean": False,
        "check_exp_vol": False,
        "check_exp_year": True,
        "default_ylim": [-1.5, 1.5],
        "do_add_line": True,
        "do_add_trend": True,
        "format": "%4.2f",
        "glb_only": False,
        "lw": 1.0,
        "ohc": False,
        "set_axhline": True,
        "set_legend": True,
        "shorten_year": False,
        "title": "Net TOA flux (restom)",
        "use_getmoc": False,
        "var": lambda exp: np.array(exp["annual"]["RESTOM"][0]),
        "verbose": False,
        "vol": False,
        "ylabel": "W m-2",
    }
    plot(ax, xlim, exps, param_dict, rgn)


# 2
def plot_global_surface_air_temperature(ax, xlim, exps, rgn):
    logger.info("Plot 2: plot_global_surface_air_temperature")
    if rgn == "glb":
        region_title = "Global"
    elif rgn == "n":
        region_title = "Northern Hemisphere"
    elif rgn == "s":
        region_title = "Southern Hemisphere"
    else:
        raise RuntimeError(f"Invalid rgn={rgn}")
    param_dict = {
        "2nd_var": False,
        "axhline_y": None,
        "check_exp_ocean": False,
        "check_exp_vol": False,
        "check_exp_year": True,
        "default_ylim": [13, 15.5],
        "do_add_line": True,
        "do_add_trend": True,
        "format": "%4.2f",
        "glb_only": False,
        "lw": 1.0,
        "ohc": False,
        "set_axhline": False,
        "set_legend": True,
        "shorten_year": False,
        "title": f"{region_title} surface air temperature",
        "use_getmoc": False,
        "var": lambda exp: np.array(exp["annual"]["TREFHT"][0]) - 273.15,
        "verbose": False,
        "vol": False,
        "ylabel": "degC",
    }
    plot(ax, xlim, exps, param_dict, rgn)


# 3
def plot_toa_radiation(ax, xlim, exps, rgn):
    logger.info("Plot 3: plot_toa_radiation")
    param_dict = {
        "2nd_var": True,
        "axhline_y": None,
        "check_exp_ocean": False,
        "check_exp_vol": False,
        "check_exp_year": False,
        "default_ylim": [235, 245],
        "do_add_line": False,
        "do_add_trend": False,
        "format": None,
        "glb_only": False,
        "lw": 1.0,
        "ohc": False,
        "set_axhline": False,
        "set_legend": False,
        "shorten_year": False,
        "title": "TOA radiation: SW (solid), LW (dashed)",
        "use_getmoc": False,
        "var": lambda exp: np.array(exp["annual"]["FSNTOA"][0]),
        "verbose": None,
        "vol": None,
        "ylabel": "W m-2",
    }
    plot(ax, xlim, exps, param_dict, rgn)


# 4
def plot_net_atm_energy_imbalance(ax, xlim, exps, rgn):
    logger.info("Plot 4: plot_net_atm_energy_imbalance")
    param_dict = {
        "2nd_var": False,
        "axhline_y": None,
        "check_exp_ocean": False,
        "check_exp_vol": False,
        "check_exp_year": True,
        "default_ylim": [-0.3, 0.3],
        "do_add_line": True,
        "do_add_trend": False,
        "format": "%4.2f",
        "glb_only": False,
        "lw": 1.0,
        "ohc": False,
        "set_axhline": False,
        "set_legend": True,
        "shorten_year": False,
        "title": "Net atm energy imbalance (restom-ressurf)",
        "use_getmoc": False,
        "var": lambda exp: np.array(exp["annual"]["RESTOM"][0])
        - np.array(exp["annual"]["RESSURF"][0]),
        "verbose": False,
        "vol": False,
        "ylabel": "W m-2",
    }
    plot(ax, xlim, exps, param_dict, rgn)


# 5
def plot_change_ohc(ax, xlim, exps, rgn):
    logger.info("Plot 5: plot_change_ohc")
    param_dict = {
        "2nd_var": False,
        "axhline_y": 0,
        "check_exp_ocean": True,
        "check_exp_vol": False,
        "check_exp_year": False,
        "default_ylim": [-0.3e24, 0.9e24],
        "do_add_line": False,
        "do_add_trend": True,
        "format": "%4.2f",
        "glb_only": True,
        "lw": 1.5,
        "ohc": True,
        "set_axhline": True,
        "set_legend": True,
        "shorten_year": True,
        "title": "Change in ocean heat content",
        "use_getmoc": False,
        "var": lambda exp: np.array(exp["annual"]["ohc"]),
        "verbose": False,
        "vol": False,
        "ylabel": "J",
    }
    plot(ax, xlim, exps, param_dict, rgn)


# 6
def plot_max_moc(ax, xlim, exps, rgn):
    logger.info("Plot 6: plot_max_moc")
    param_dict = {
        "2nd_var": False,
        "axhline_y": 10,
        "check_exp_ocean": False,
        "check_exp_vol": False,
        "check_exp_year": False,
        "default_ylim": [4, 22],
        "do_add_line": False,
        "do_add_trend": True,
        "format": "%4.2f",
        "glb_only": True,
        "lw": 1.5,
        "ohc": False,
        "set_axhline": True,
        "set_legend": True,
        "shorten_year": False,
        "title": "Max MOC Atlantic streamfunction at 26.5N",
        "use_getmoc": True,
        "var": None,
        "verbose": True,
        "vol": None,
        "ylabel": "Sv",
    }
    plot(ax, xlim, exps, param_dict, rgn)


# 7
def plot_change_sea_level(ax, xlim, exps, rgn):
    logger.info("Plot 7: plot_change_sea_level")
    param_dict = {
        "2nd_var": False,
        "axhline_y": None,
        "check_exp_ocean": False,
        "check_exp_vol": True,
        "check_exp_year": True,
        "default_ylim": [4, 22],
        "do_add_line": False,
        "do_add_trend": True,
        "format": "%5.3f",
        "glb_only": True,
        "lw": 1.5,
        "ohc": False,
        "set_axhline": False,
        "set_legend": True,
        "shorten_year": True,
        "title": "Change in sea level",
        "use_getmoc": False,
        "var": lambda exp: (
            1e3
            * np.array(exp["annual"]["volume"])
            / (4.0 * math.pi * (6371229.0) ** 2 * 0.7)
        ),
        "verbose": True,
        "vol": True,
        "ylabel": "mm",
    }
    plot(ax, xlim, exps, param_dict, rgn)


# 8
def plot_net_atm_water_imbalance(ax, xlim, exps, rgn):
    logger.info("Plot 8: plot_net_atm_water_imbalance")
    param_dict = {
        "2nd_var": False,
        "axhline_y": None,
        "check_exp_ocean": False,
        "check_exp_vol": False,
        "check_exp_year": False,
        "default_ylim": [-1, 1],
        "do_add_line": True,
        "do_add_trend": False,
        "format": "%5.4f",
        "glb_only": False,
        "lw": 1.0,
        "ohc": False,
        "set_axhline": False,
        "set_legend": True,
        "shorten_year": False,
        "title": "Net atm water imbalance (evap-prec)",
        "use_getmoc": False,
        "var": lambda exp: (
            365
            * 86400
            * (
                np.array(exp["annual"]["QFLX"][0])
                - 1e3
                * (
                    np.array(exp["annual"]["PRECC"][0])
                    + np.array(exp["annual"]["PRECL"][0])
                )
            )
        ),
        "verbose": False,
        "vol": False,
        "ylabel": "mm yr-1",
    }
    plot(ax, xlim, exps, param_dict, rgn)


# Generic plot function
def plot_generic(ax, xlim, exps, var_name, rgn):
    logger.info(f"plot_generic for {var_name}, rgn={rgn}")
    param_dict = {
        "2nd_var": False,
        "axhline_y": 0,
        "check_exp_ocean": False,
        "check_exp_vol": False,
        "check_exp_year": True,
        "default_ylim": [],
        "do_add_line": True,
        "do_add_trend": True,
        "format": "%4.2f",
        "glb_only": False,
        "lw": 1.0,
        "ohc": False,
        "set_axhline": False,
        "set_legend": True,
        "shorten_year": False,
        "title": var_name,
        "use_getmoc": False,
        "var": lambda exp: np.array(exp["annual"][var_name][0]),
        "verbose": False,
        "vol": False,
        "ylabel": lambda exp: np.array(exp["annual"][var_name][1]),
    }
    plot(ax, xlim, exps, param_dict, rgn)


# FIXME: C901 'plot' is too complex (19)
def plot(ax, xlim, exps, param_dict, rgn):  # noqa: C901
    if param_dict["glb_only"] and (rgn != "glb"):
        return
    ax.set_xlim(xlim)
    extreme_values = []
    for exp in exps:
        # Relevant to "Plot 5: plot_change_ohc"
        if param_dict["check_exp_ocean"] and (exp["ocean"] == ""):
            continue
        # Relevant to "Plot 7: plot_change_sea_level"
        # This must be checked before plot 6,
        # otherwise, `param_dict["var"]` will be run,
        # but `exp["annual"]["volume"]` won't exist.
        if param_dict["check_exp_vol"] and (exp["vol"] == ""):
            continue
        # Relevant to "Plot 6: plot_max_moc"
        if param_dict["use_getmoc"]:
            if exp["moc"]:
                [year, var] = getmoc(exp["moc"])
            else:
                continue
        else:
            year = np.array(exp["annual"]["year"]) + exp["yoffset"]
            var = param_dict["var"](exp)
        extreme_values.append(np.amax(var))
        extreme_values.append(np.amin(var))
        if param_dict["shorten_year"]:
            year = year[: len(var)]
        try:
            ax.plot(
                year,
                var,
                lw=param_dict["lw"],
                marker=None,
                c=exp["color"],
                label=exp["name"],
            )
        except Exception:
            raise RuntimeError(f"{param_dict['title']} could not be plotted.")
        if param_dict["2nd_var"]:
            # Specifically for plot_toa_radiation
            # TODO: if more plots require a 2nd variable, we can change `var` to be a list,
            # but that will be a more significant refactoring.
            var = np.array(exp["annual"]["FLUT"][0])
            ax.plot(year, var, lw=1.0, marker=None, ls=":", c=exp["color"])
            continue
        if param_dict["check_exp_year"] and exp["yr"] is None:
            continue
        elif param_dict["do_add_line"] or param_dict["do_add_trend"]:
            for yrs in exp["yr"]:
                if param_dict["do_add_line"]:
                    add_line(
                        year,
                        var,
                        yrs[0],
                        yrs[1],
                        format=param_dict["format"],
                        ax=ax,
                        lw=2 * param_dict["lw"],
                        color=exp["color"],
                    )
                if param_dict["do_add_trend"]:
                    add_trend(
                        year,
                        var,
                        yrs[0],
                        yrs[1],
                        format=param_dict["format"],
                        ax=ax,
                        lw=2 * param_dict["lw"],
                        color=exp["color"],
                        ohc=param_dict["ohc"],
                        verbose=param_dict["verbose"],
                        vol=param_dict["vol"],
                    )
    ylim = get_ylim(param_dict["default_ylim"], extreme_values)
    ax.set_ylim(ylim)
    if param_dict["set_axhline"]:
        ax.axhline(y=param_dict["axhline_y"], lw=1, c="0.5")
    ax.set_title(param_dict["title"])
    ax.set_xlabel("Year")
    units = param_dict["ylabel"]
    c = callable(units)
    if c:
        units = units(exps[0])
    ax.set_ylabel(units)
    if param_dict["set_legend"]:
        ax.legend(loc="best")


PLOT_DICT = {
    "net_toa_flux_restom": plot_net_toa_flux_restom,
    "global_surface_air_temperature": plot_global_surface_air_temperature,
    "toa_radiation": plot_toa_radiation,
    "net_atm_energy_imbalance": plot_net_atm_energy_imbalance,
    "change_ohc": plot_change_ohc,  # only glb
    "max_moc": plot_max_moc,  # only glb
    "change_sea_level": plot_change_sea_level,  # only glb
    "net_atm_water_imbalance": plot_net_atm_water_imbalance,
}


# FIXME: C901 'make_plot_pdfs' is too complex (20)
def make_plot_pdfs(  # noqa: C901
    parameters: Parameters,
    rgn,
    component,
    xlim,
    exps,
    plot_list,
    valid_plots,
    invalid_plots,
):
    logger.info(f"make_plot_pdfs for rgn={rgn}, component={component}")
    num_plots = len(plot_list)
    if num_plots == 0:
        return

    # If make_viewer, then we want to do 1 plot per page.
    # However, the original plots are excluded from this restriction.
    # Note: if the user provides nrows=ncols=1, there will still be a single plot per page
    keep_user_dims = (not parameters.make_viewer) or (component == "original")
    if keep_user_dims:
        nrows = parameters.nrows
        ncols = parameters.ncols
    else:
        nrows = 1
        ncols = 1

    plots_per_page = nrows * ncols
    num_pages = math.ceil(num_plots / plots_per_page)

    counter = 0
    os.makedirs(parameters.results_dir, exist_ok=True)
    # https://stackoverflow.com/questions/58738992/save-multiple-figures-with-subplots-into-a-pdf-with-multiple-pages
    pdf = matplotlib.backends.backend_pdf.PdfPages(
        f"{parameters.results_dir}/{parameters.figstr}_{rgn}_{component}.pdf"
    )
    for page in range(num_pages):
        if plots_per_page == 1:
            logger.info("Using reduced figsize")
            fig = plt.figure(1, figsize=[13.5 / 2, 16.5 / 4])
        else:
            logger.info("Using standard figsize")
            fig = plt.figure(1, figsize=[13.5, 16.5])
        logger.info(f"Figure size={fig.get_size_inches() * fig.dpi}")
        fig.suptitle(f"{parameters.figstr}_{rgn}_{component}")
        for j in range(plots_per_page):
            logger.info(
                f"Plotting plot {j} on page {page}. This is plot {counter} in total."
            )
            # The final page doesn't need to be filled out with plots.
            if counter >= num_plots:
                break
            ax = plt.subplot(
                nrows,
                ncols,
                j + 1,
            )
            plot_name = plot_list[counter]
            if component == "original":
                try:
                    plot_function = PLOT_DICT[plot_name]
                except KeyError:
                    raise KeyError(f"Invalid plot name: {plot_name}")
                try:
                    plot_function(ax, xlim, exps, rgn)
                    valid_plots.append(plot_name)
                except Exception:
                    traceback.print_exc()
                    required_vars = []
                    if plot_name == "net_toa_flux_restom":
                        required_vars = ["RESTOM"]
                    elif plot_name == "net_atm_energy_imbalance":
                        required_vars = ["RESTOM", "RESSURF"]
                    elif plot_name == "global_surface_air_temperature":
                        required_vars = ["TREFHT"]
                    elif plot_name == "toa_radiation":
                        required_vars = ["FSNTOA", "FLUT"]
                    elif plot_name == "net_atm_water_imbalance":
                        required_vars = ["PRECC", "PRECL", "QFLX"]
                    logger.error(
                        f"Failed plot_function for {plot_name}. Check that {required_vars} are available."
                    )
                    invalid_plots.append(plot_name)
                counter += 1
            else:
                try:
                    plot_generic(ax, xlim, exps, plot_name, rgn)
                    valid_plots.append(plot_name)
                except Exception:
                    traceback.print_exc()
                    logger.error(
                        f"plot_generic failed. Invalid plot={plot_name}, rgn={rgn}"
                    )
                    invalid_plots.append(plot_name)
                counter += 1

        fig.tight_layout()
        pdf.savefig(1)
        # Always save individual PNGs for viewer mode
        if plots_per_page == 1:
            fig.savefig(
                f"{parameters.results_dir}/{parameters.figstr}_{rgn}_{component}_{plot_name}.png",
                dpi=150,
            )
        elif num_pages > 1:
            fig.savefig(
                f"{parameters.results_dir}/{parameters.figstr}_{rgn}_{component}_{page}.png",
                dpi=150,
            )
        else:
            fig.savefig(
                f"{parameters.results_dir}/{parameters.figstr}_{rgn}_{component}.png",
                dpi=150,
            )
        plt.close(fig)
    pdf.close()
