from typing import Any, Dict, List

import pytest

from zppy_interfaces.global_time_series.coupled_global import (
    Metric,
    Variable,
    construct_generic_variables,
    get_data_dir,
    get_exps,
    get_vars_original,
    get_ylim,
)
from zppy_interfaces.global_time_series.utils import (
    Parameters,
    get_region,
    param_get_list,
)

# Run tests with `pytest tests/global_time_series`


# Helper function
def get_var_names(vars: List[Variable]):
    return list(map(lambda v: v.variable_name, vars))


def test_param_get_list():
    assert param_get_list("None") == []

    assert param_get_list("a") == ["a"]
    assert param_get_list("a,b,c") == ["a", "b", "c"]

    assert param_get_list("") == [""]
    assert param_get_list("a,") == ["a", ""]
    assert param_get_list("a,b,c,") == ["a", "b", "c", ""]


def test_get_region():
    assert get_region("glb") == "glb"
    assert get_region("global") == "glb"
    assert get_region("GLB") == "glb"
    assert get_region("Global") == "glb"

    assert get_region("n") == "n"
    assert get_region("north") == "n"
    assert get_region("northern") == "n"
    assert get_region("N") == "n"
    assert get_region("North") == "n"
    assert get_region("Northern") == "n"

    assert get_region("s") == "s"
    assert get_region("south") == "s"
    assert get_region("southern") == "s"
    assert get_region("S") == "s"
    assert get_region("South") == "s"
    assert get_region("Southern") == "s"

    with pytest.raises(ValueError):
        get_region("not-a-region")


def test_Parameters_and_related_functions():
    # Consider the following parameters given by a user.
    args: Dict[str, str] = {
        "use_ocn": "True",
        "input": "/lcrc/group/e3sm2/ac.wlin/E3SMv3/v3.LR.historical_0051",
        "input_subdir": "archive/atm/hist",
        "moc_file": "mocTimeSeries_1985-1995.nc",
        "case_dir": "/lcrc/group/e3sm/ac.forsyth2/zppy_min_case_global_time_series_single_plots_output/test-616-20240930/v3.LR.historical_0051",
        "experiment_name": "v3.LR.historical_0051",
        "figstr": "v3.LR.historical_0051",
        "color": "Blue",
        "ts_num_years": "5",
        "plots_original": "None",
        "atmosphere_only": "False",
        "plots_atm": "TREFHT",
        "plots_ice": "None",
        "plots_lnd": "FSH,RH2M,LAISHA,LAISUN,QINTR,QOVER,QRUNOFF,QSOIL,QVEGE,QVEGT,SOILWATER_10CM,TSA,H2OSNO,TOTLITC,CWDC,SOIL1C,SOIL2C,SOIL3C,SOIL4C,WOOD_HARVESTC,TOTVEGC,NBP,GPP,AR,HR",
        "plots_ocn": "None",
        "nrows": "1",
        "ncols": "1",
        "results_dir": "results",
        "regions": "glb,n,s",
        "start_yr": "1985",
        "end_yr": "1989",
    }
    # Then:
    parameters: Parameters = Parameters(args)
    assert (
        parameters.case_dir
        == "/lcrc/group/e3sm/ac.forsyth2/zppy_min_case_global_time_series_single_plots_output/test-616-20240930/v3.LR.historical_0051"
    )
    assert parameters.experiment_name == "v3.LR.historical_0051"
    assert parameters.figstr == "v3.LR.historical_0051"
    assert parameters.year1 == 1985
    assert parameters.year2 == 1989
    assert parameters.color == "Blue"
    assert parameters.ts_num_years_str == "5"
    assert parameters.plots_original == []
    assert parameters.atmosphere_only == False  # noqa: E712
    assert parameters.plots_atm == ["TREFHT"]
    assert parameters.plots_ice == []
    assert parameters.plots_lnd == [
        "FSH",
        "RH2M",
        "LAISHA",
        "LAISUN",
        "QINTR",
        "QOVER",
        "QRUNOFF",
        "QSOIL",
        "QVEGE",
        "QVEGT",
        "SOILWATER_10CM",
        "TSA",
        "H2OSNO",
        "TOTLITC",
        "CWDC",
        "SOIL1C",
        "SOIL2C",
        "SOIL3C",
        "SOIL4C",
        "WOOD_HARVESTC",
        "TOTVEGC",
        "NBP",
        "GPP",
        "AR",
        "HR",
    ]
    assert parameters.plots_ocn == []
    assert parameters.nrows == 1
    assert parameters.ncols == 1
    assert parameters.regions == ["glb", "n", "s"]

    # test_get_data_dir
    assert (
        get_data_dir(parameters, "atm", True)
        == "/lcrc/group/e3sm/ac.forsyth2/zppy_min_case_global_time_series_single_plots_output/test-616-20240930/v3.LR.historical_0051/post/atm/glb/ts/monthly/5yr/"
    )
    assert get_data_dir(parameters, "atm", False) == ""
    assert (
        get_data_dir(parameters, "ice", True)
        == "/lcrc/group/e3sm/ac.forsyth2/zppy_min_case_global_time_series_single_plots_output/test-616-20240930/v3.LR.historical_0051/post/ice/glb/ts/monthly/5yr/"
    )
    assert get_data_dir(parameters, "ice", False) == ""
    assert (
        get_data_dir(parameters, "lnd", True)
        == "/lcrc/group/e3sm/ac.forsyth2/zppy_min_case_global_time_series_single_plots_output/test-616-20240930/v3.LR.historical_0051/post/lnd/glb/ts/monthly/5yr/"
    )
    assert get_data_dir(parameters, "lnd", False) == ""
    assert (
        get_data_dir(parameters, "ocn", True)
        == "/lcrc/group/e3sm/ac.forsyth2/zppy_min_case_global_time_series_single_plots_output/test-616-20240930/v3.LR.historical_0051/post/ocn/glb/ts/monthly/5yr/"
    )
    assert get_data_dir(parameters, "ocn", False) == ""

    # test_get_exps
    exps: List[Dict[str, Any]] = get_exps(parameters)
    assert len(exps) == 1
    expected = {
        "atmos": "/lcrc/group/e3sm/ac.forsyth2/zppy_min_case_global_time_series_single_plots_output/test-616-20240930/v3.LR.historical_0051/post/atm/glb/ts/monthly/5yr/",
        "ice": "",
        "land": "/lcrc/group/e3sm/ac.forsyth2/zppy_min_case_global_time_series_single_plots_output/test-616-20240930/v3.LR.historical_0051/post/lnd/glb/ts/monthly/5yr/",
        "ocean": "",
        "moc": "",
        "vol": "",
        "name": "v3.LR.historical_0051",
        "yoffset": 0.0,
        "yr": ([1985, 1989],),
        "color": "Blue",
    }
    assert exps[0] == expected
    # Change up parameters
    parameters.plots_original = "net_toa_flux_restom,global_surface_air_temperature,toa_radiation,net_atm_energy_imbalance,change_ohc,max_moc,change_sea_level,net_atm_water_imbalance".split(
        ","
    )
    parameters.plots_atm = []
    parameters.plots_lnd = []
    exps = get_exps(parameters)
    assert len(exps) == 1
    expected = {
        "atmos": "/lcrc/group/e3sm/ac.forsyth2/zppy_min_case_global_time_series_single_plots_output/test-616-20240930/v3.LR.historical_0051/post/atm/glb/ts/monthly/5yr/",
        "ice": "",
        "land": "",
        "ocean": "/lcrc/group/e3sm/ac.forsyth2/zppy_min_case_global_time_series_single_plots_output/test-616-20240930/v3.LR.historical_0051/post/ocn/glb/ts/monthly/5yr/",
        "moc": "/lcrc/group/e3sm/ac.forsyth2/zppy_min_case_global_time_series_single_plots_output/test-616-20240930/v3.LR.historical_0051/post/ocn/glb/ts/monthly/5yr/",
        "vol": "/lcrc/group/e3sm/ac.forsyth2/zppy_min_case_global_time_series_single_plots_output/test-616-20240930/v3.LR.historical_0051/post/ocn/glb/ts/monthly/5yr/",
        "name": "v3.LR.historical_0051",
        "yoffset": 0.0,
        "yr": ([1985, 1989],),
        "color": "Blue",
    }
    assert exps[0] == expected
    # Change up parameters
    parameters.atmosphere_only = True
    exps = get_exps(parameters)
    assert len(exps) == 1
    expected = {
        "atmos": "/lcrc/group/e3sm/ac.forsyth2/zppy_min_case_global_time_series_single_plots_output/test-616-20240930/v3.LR.historical_0051/post/atm/glb/ts/monthly/5yr/",
        "ice": "",
        "land": "",
        "ocean": "",
        "moc": "",
        "vol": "",
        "name": "v3.LR.historical_0051",
        "yoffset": 0.0,
        "yr": ([1985, 1989],),
        "color": "Blue",
    }
    assert exps[0] == expected


def test_Variable():
    v = Variable(
        "var_name",
        original_units="units1",
        final_units="units2",
        group="group_name",
        long_name="long name",
    )
    assert v.variable_name == "var_name"
    assert v.metric == Metric.AVERAGE
    assert v.scale_factor == 1.0
    assert v.original_units == "units1"
    assert v.final_units == "units2"
    assert v.group == "group_name"
    assert v.long_name == "long name"


def test_get_vars_original():
    assert get_var_names(get_vars_original(["net_toa_flux_restom"])) == ["RESTOM"]
    assert get_var_names(get_vars_original(["net_atm_energy_imbalance"])) == [
        "RESTOM",
        "RESSURF",
    ]
    assert get_var_names(get_vars_original(["global_surface_air_temperature"])) == [
        "TREFHT"
    ]
    assert get_var_names(get_vars_original(["toa_radiation"])) == ["FSNTOA", "FLUT"]
    assert get_var_names(get_vars_original(["net_atm_water_imbalance"])) == [
        "PRECC",
        "PRECL",
        "QFLX",
    ]
    assert get_var_names(
        get_vars_original(
            [
                "net_toa_flux_restom",
                "net_atm_energy_imbalance",
                "global_surface_air_temperature",
                "toa_radiation",
                "net_atm_water_imbalance",
            ]
        )
    ) == ["RESTOM", "RESSURF", "TREFHT", "FSNTOA", "FLUT", "PRECC", "PRECL", "QFLX"]
    assert get_var_names(get_vars_original(["invalid_plot"])) == []


def test_construct_generic_variables():
    vars: List[str] = ["a", "b", "c"]
    assert get_var_names(construct_generic_variables(vars)) == vars


def test_get_ylim():
    # Min is equal, max is equal
    assert get_ylim([-1, 1], [-1, 1]) == [-1, 1]
    # Min is lower, max is equal
    assert get_ylim([-1, 1], [-2, 1]) == [-2, 1]
    # Min is equal, max is higher
    assert get_ylim([-1, 1], [-1, 2]) == [-1, 2]
    # Min is lower, max is higher
    assert get_ylim([-1, 1], [-2, 2]) == [-2, 2]
    # Min is lower, max is higher, multiple extreme_values
    assert get_ylim([-1, 1], [-2, -0.5, 0.5, 2]) == [-2, 2]
    # No standard range
    assert get_ylim([], [-2, 2]) == [-2, 2]
    # No extreme range
    assert get_ylim([-1, 1], []) == [-1, 1]
