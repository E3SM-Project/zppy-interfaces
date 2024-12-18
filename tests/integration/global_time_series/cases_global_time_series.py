import os
import shutil

from zppy_interfaces.global_time_series.__main__ import main
from zppy_interfaces.global_time_series.utils import Parameters

CASE_DIR = "/lcrc/group/e3sm/ac.forsyth2/zi-test-input-data"  # This is 44G.
WEB_DIR = "/lcrc/group/e3sm/public_html/diagnostic_output/ac.forsyth2/zi-test-webdir/"
RESULTS_DIR_PREFIX = "global_time_series_1985-1995_results"

parameters_custom: Parameters = Parameters(
    {
        "use_ocn": "False",
        "input": "/lcrc/group/e3sm2/ac.wlin/E3SMv3/v3.LR.historical_0051",
        "input_subdir": "archive/ocn/hist",
        "moc_file": "None",
        "case_dir": CASE_DIR,
        "experiment_name": "v3.LR.historical_0051",
        "figstr": "v3.LR.historical_0051",
        "color": "Blue",
        "ts_num_years": "5",
        "plots_original": "None",
        "plots_atm": "TREFHT,AODDUST",
        "plots_ice": "None",
        "plots_lnd": "FSH,RH2M,LAISHA,LAISUN,QINTR,QOVER,QRUNOFF,QSOIL,QVEGE,QVEGT,SOILWATER_10CM,TSA,H2OSNO,TOTLITC,CWDC,SOIL1C,SOIL2C,SOIL3C,SOIL4C,WOOD_HARVESTC,TOTVEGC,NBP,GPP,AR,HR",
        "plots_ocn": "None",
        "nrows": "4",
        "ncols": "2",
        "results_dir": f"{RESULTS_DIR_PREFIX}_custom",
        "regions": "glb,n,s",
        "start_yr": "1985",
        "end_yr": "1995",
    }
)


parameters_original_8_no_ocn: Parameters = Parameters(
    {
        "use_ocn": "False",
        "input": "/lcrc/group/e3sm2/ac.wlin/E3SMv3/v3.LR.historical_0051",
        "input_subdir": "archive/ocn/hist",
        "moc_file": "None",
        "case_dir": CASE_DIR,
        "experiment_name": "v3.LR.historical_0051",
        "figstr": "v3.LR.historical_0051",
        "color": "Blue",
        "ts_num_years": "5",
        "plots_original": "net_toa_flux_restom,global_surface_air_temperature,toa_radiation,net_atm_energy_imbalance,net_atm_water_imbalance",
        "plots_atm": "None",
        "plots_ice": "None",
        "plots_lnd": "None",
        "plots_ocn": "None",
        "nrows": "4",
        "ncols": "2",
        "results_dir": f"{RESULTS_DIR_PREFIX}_original_8_no_ocn",
        "regions": "glb,n,s",
        "start_yr": "1985",
        "end_yr": "1995",
    }
)

parameters_original_8: Parameters = Parameters(
    {
        "use_ocn": "True",
        "input": "/lcrc/group/e3sm2/ac.wlin/E3SMv3/v3.LR.historical_0051",
        "input_subdir": "archive/ocn/hist",
        "moc_file": "mocTimeSeries_1985-1995.nc",
        "case_dir": CASE_DIR,
        "experiment_name": "v3.LR.historical_0051",
        "figstr": "v3.LR.historical_0051",
        "color": "Blue",
        "ts_num_years": "5",
        "plots_original": "net_toa_flux_restom,global_surface_air_temperature,toa_radiation,net_atm_energy_imbalance,change_ohc,max_moc,change_sea_level,net_atm_water_imbalance",
        "plots_atm": "None",
        "plots_ice": "None",
        "plots_lnd": "None",
        "plots_ocn": "None",
        "nrows": "4",
        "ncols": "2",
        "results_dir": f"{RESULTS_DIR_PREFIX}_original_8",
        "regions": "glb,n,s",
        "start_yr": "1985",
        "end_yr": "1995",
    }
)

parameters_comprehensive_v3: Parameters = Parameters(
    {
        "use_ocn": "True",
        "input": "/lcrc/group/e3sm2/ac.wlin/E3SMv3/v3.LR.historical_0051",
        "input_subdir": "archive/ocn/hist",
        "moc_file": "mocTimeSeries_1985-1995.nc",
        "case_dir": CASE_DIR,
        "experiment_name": "v3.LR.historical_0051",
        "figstr": "v3.LR.historical_0051",
        "color": "Blue",
        "ts_num_years": "5",
        "plots_original": "net_toa_flux_restom,global_surface_air_temperature,toa_radiation,net_atm_energy_imbalance,change_ohc,max_moc,change_sea_level,net_atm_water_imbalance",
        "plots_atm": "None",
        "plots_ice": "None",
        "plots_lnd": "FSH,RH2M,LAISHA,LAISUN,QINTR,QOVER,QRUNOFF,QSOIL,QVEGE,QVEGT,SOILWATER_10CM,TSA,H2OSNO,TOTLITC,CWDC,SOIL1C,SOIL2C,SOIL3C,SOIL4C,WOOD_HARVESTC,TOTVEGC,NBP,GPP,AR,HR",
        "plots_ocn": "None",
        "nrows": "4",
        "ncols": "2",
        "results_dir": f"{RESULTS_DIR_PREFIX}_comprehensive_v3",
        "regions": "glb,n,s",
        "start_yr": "1985",
        "end_yr": "1995",
    }
)


def generate_results(parameters: Parameters):
    print(f"Generating results for {parameters.results_dir}")

    # CASE_DIR is large, so we don't want to copy it every time.
    # We also want to be able to reset it to the original state after running the test.
    # The only modification `global_time_series` makes to CASE_DIR is add an `ocn` subdirectory.
    # (It also adds a parameters.results_dir subdirectory to the current directory, which we then copy to WEB_DIR).
    # So, we just have to remove those
    for subdir in [
        f"{CASE_DIR}/ocn",
        parameters.results_dir,
        f"{WEB_DIR}/{parameters.results_dir}",
    ]:
        if os.path.exists(subdir):
            print(f"Removing {subdir}")
            shutil.rmtree(subdir)

    print("Running main")
    main(parameters)

    print(f"Copying {parameters.results_dir} to {WEB_DIR}/{parameters.results_dir}")
    shutil.copytree(parameters.results_dir, f"{WEB_DIR}/{parameters.results_dir}")


def run_all_cases():
    generate_results(parameters_custom)
    generate_results(parameters_original_8_no_ocn)
    generate_results(parameters_original_8)
    generate_results(parameters_comprehensive_v3)


if __name__ == "__main__":
    # TODO: Create actual pytest cases, including image comparison checks
    # (See https://github.com/E3SM-Project/zppy-interfaces/issues/5)
    run_all_cases()
