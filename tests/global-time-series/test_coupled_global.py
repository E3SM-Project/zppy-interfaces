import unittest
from typing import Any, Dict, List

from zppy_interfaces.global_time_series.coupled_global import (
    Metric,
    Variable,
    construct_generic_variables,
    get_data_dir,
    get_exps,
    get_vars_original,
    get_ylim,
)
from zppy_interfaces.global_time_series.global_time_series import (
    Parameters,
    get_region,
    param_get_list,
)


# Helper function
def get_var_names(vars: List[Variable]):
    return list(map(lambda v: v.variable_name, vars))


class TestGlobalTimeSeries(unittest.TestCase):

    # Useful classes and their helper functions ###############################
    def test_param_get_list(self):
        self.assertEqual(param_get_list("None"), [])

        self.assertEqual(param_get_list("a"), ["a"])
        self.assertEqual(param_get_list("a,b,c"), ["a", "b", "c"])

        self.assertEqual(param_get_list(""), [""])
        self.assertEqual(param_get_list("a,"), ["a", ""])
        self.assertEqual(param_get_list("a,b,c,"), ["a", "b", "c", ""])

    def test_get_region(self):
        self.assertEqual(get_region("glb"), "glb")
        self.assertEqual(get_region("global"), "glb")
        self.assertEqual(get_region("GLB"), "glb")
        self.assertEqual(get_region("Global"), "glb")

        self.assertEqual(get_region("n"), "n")
        self.assertEqual(get_region("north"), "n")
        self.assertEqual(get_region("northern"), "n")
        self.assertEqual(get_region("N"), "n")
        self.assertEqual(get_region("North"), "n")
        self.assertEqual(get_region("Northern"), "n")

        self.assertEqual(get_region("s"), "s")
        self.assertEqual(get_region("south"), "s")
        self.assertEqual(get_region("southern"), "s")
        self.assertEqual(get_region("S"), "s")
        self.assertEqual(get_region("South"), "s")
        self.assertEqual(get_region("Southern"), "s")

        self.assertRaises(ValueError, get_region, "not-a-region")

    def test_Parameters_and_related_functions(self):
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
        self.assertEqual(
            parameters.case_dir,
            "/lcrc/group/e3sm/ac.forsyth2/zppy_min_case_global_time_series_single_plots_output/test-616-20240930/v3.LR.historical_0051",
        )
        self.assertEqual(parameters.experiment_name, "v3.LR.historical_0051")
        self.assertEqual(parameters.figstr, "v3.LR.historical_0051")
        self.assertEqual(parameters.year1, 1985)
        self.assertEqual(parameters.year2, 1989)
        self.assertEqual(parameters.color, "Blue")
        self.assertEqual(parameters.ts_num_years_str, "5")
        self.assertEqual(parameters.plots_original, [])
        self.assertEqual(parameters.atmosphere_only, False)
        self.assertEqual(parameters.plots_atm, ["TREFHT"])
        self.assertEqual(parameters.plots_ice, [])
        self.assertEqual(
            parameters.plots_lnd,
            [
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
            ],
        )
        self.assertEqual(parameters.plots_ocn, [])
        self.assertEqual(parameters.nrows, 1)
        self.assertEqual(parameters.ncols, 1)
        self.assertEqual(parameters.regions, ["glb", "n", "s"])

        # test_get_data_dir
        self.assertEqual(
            get_data_dir(parameters, "atm", True),
            "/lcrc/group/e3sm/ac.forsyth2/zppy_min_case_global_time_series_single_plots_output/test-616-20240930/v3.LR.historical_0051/post/atm/glb/ts/monthly/5yr/",
        )
        self.assertEqual(get_data_dir(parameters, "atm", False), "")
        self.assertEqual(
            get_data_dir(parameters, "ice", True),
            "/lcrc/group/e3sm/ac.forsyth2/zppy_min_case_global_time_series_single_plots_output/test-616-20240930/v3.LR.historical_0051/post/ice/glb/ts/monthly/5yr/",
        )
        self.assertEqual(get_data_dir(parameters, "ice", False), "")
        self.assertEqual(
            get_data_dir(parameters, "lnd", True),
            "/lcrc/group/e3sm/ac.forsyth2/zppy_min_case_global_time_series_single_plots_output/test-616-20240930/v3.LR.historical_0051/post/lnd/glb/ts/monthly/5yr/",
        )
        self.assertEqual(get_data_dir(parameters, "lnd", False), "")
        self.assertEqual(
            get_data_dir(parameters, "ocn", True),
            "/lcrc/group/e3sm/ac.forsyth2/zppy_min_case_global_time_series_single_plots_output/test-616-20240930/v3.LR.historical_0051/post/ocn/glb/ts/monthly/5yr/",
        )
        self.assertEqual(get_data_dir(parameters, "ocn", False), "")

        # test_get_exps
        self.maxDiff = None
        exps: List[Dict[str, Any]] = get_exps(parameters)
        self.assertEqual(len(exps), 1)
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
        self.assertEqual(exps[0], expected)
        # Change up parameters
        parameters.plots_original = "net_toa_flux_restom,global_surface_air_temperature,toa_radiation,net_atm_energy_imbalance,change_ohc,max_moc,change_sea_level,net_atm_water_imbalance".split(
            ","
        )
        parameters.plots_atm = []
        parameters.plots_lnd = []
        exps = get_exps(parameters)
        self.assertEqual(len(exps), 1)
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
        self.assertEqual(exps[0], expected)
        # Change up parameters
        parameters.atmosphere_only = True
        exps = get_exps(parameters)
        self.assertEqual(len(exps), 1)
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
        self.assertEqual(exps[0], expected)

    def test_Variable(self):
        v = Variable(
            "var_name",
            original_units="units1",
            final_units="units2",
            group="group_name",
            long_name="long name",
        )
        self.assertEqual(v.variable_name, "var_name")
        self.assertEqual(v.metric, Metric.AVERAGE)
        self.assertEqual(v.scale_factor, 1.0)
        self.assertEqual(v.original_units, "units1")
        self.assertEqual(v.final_units, "units2")
        self.assertEqual(v.group, "group_name")
        self.assertEqual(v.long_name, "long name")

    def test_get_vars_original(self):
        self.assertEqual(
            get_var_names(get_vars_original(["net_toa_flux_restom"])), ["RESTOM"]
        )
        self.assertEqual(
            get_var_names(get_vars_original(["net_atm_energy_imbalance"])),
            ["RESTOM", "RESSURF"],
        )
        self.assertEqual(
            get_var_names(get_vars_original(["global_surface_air_temperature"])),
            ["TREFHT"],
        )
        self.assertEqual(
            get_var_names(get_vars_original(["toa_radiation"])), ["FSNTOA", "FLUT"]
        )
        self.assertEqual(
            get_var_names(get_vars_original(["net_atm_water_imbalance"])),
            ["PRECC", "PRECL", "QFLX"],
        )
        self.assertEqual(
            get_var_names(
                get_vars_original(
                    [
                        "net_toa_flux_restom",
                        "net_atm_energy_imbalance",
                        "global_surface_air_temperature",
                        "toa_radiation",
                        "net_atm_water_imbalance",
                    ]
                )
            ),
            ["RESTOM", "RESSURF", "TREFHT", "FSNTOA", "FLUT", "PRECC", "PRECL", "QFLX"],
        )
        self.assertEqual(get_var_names(get_vars_original(["invalid_plot"])), [])

    def test_construct_generic_variables(self):
        vars: List[str] = ["a", "b", "c"]
        self.assertEqual(get_var_names(construct_generic_variables(vars)), vars)

    def test_get_ylim(self):
        # Min is equal, max is equal
        self.assertEqual(get_ylim([-1, 1], [-1, 1]), [-1, 1])
        # Min is lower, max is equal
        self.assertEqual(get_ylim([-1, 1], [-2, 1]), [-2, 1])
        # Min is equal, max is higher
        self.assertEqual(get_ylim([-1, 1], [-1, 2]), [-1, 2])
        # Min is lower, max is higher
        self.assertEqual(get_ylim([-1, 1], [-2, 2]), [-2, 2])
        # Min is lower, max is higher, multiple extreme_values
        self.assertEqual(get_ylim([-1, 1], [-2, -0.5, 0.5, 2]), [-2, 2])
        # No standard range
        self.assertEqual(get_ylim([], [-2, 2]), [-2, 2])
        # No extreme range
        self.assertEqual(get_ylim([-1, 1], []), [-1, 1])


if __name__ == "__main__":
    unittest.main()
