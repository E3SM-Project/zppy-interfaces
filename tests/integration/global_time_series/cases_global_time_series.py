import os
import shutil

from zppy_interfaces.global_time_series.__main__ import main
from zppy_interfaces.global_time_series.utils import Parameters

CASE_DIR = "/lcrc/group/e3sm/ac.forsyth2/zi-test-input-data"  # This is 44G.
WEB_DIR = "/lcrc/group/e3sm/public_html/diagnostic_output/ac.forsyth2/zi-test-webdir/"

"""
[global_time_series]
active = True
environment_commands = "source /gpfs/fs1/home/ac.forsyth2/miniforge3/etc/profile.d/conda.sh; conda activate zi_dev_weekly_20241122"
experiment_name = "v3.LR.historical_0051"
figstr = "v3.LR.historical_0051"
plots_original=""
plots_atm = "TREFHT,AODDUST"
plots_lnd = "FSH,RH2M,LAISHA,LAISUN,QINTR,QOVER,QRUNOFF,QSOIL,QVEGE,QVEGT,SOILWATER_10CM,TSA,H2OSNO,TOTLITC,CWDC,SOIL1C,SOIL2C,SOIL3C,SOIL4C,WOOD_HARVESTC,TOTVEGC,NBP,GPP,AR,HR"
ts_num_years = 5
walltime = "00:30:00"
years = "1985-1995",
"""
parameters_custom: Parameters = Parameters(
    {
        "use_ocn": "False",
        "input": "/lcrc/group/e3sm2/ac.wlin/E3SMv3/v3.LR.historical_0051",
        "input_subdir": "archive/ocn/hist",
        "moc_file": "mocTimeSeries_1985-1995.nc",
        "case_dir": CASE_DIR,
        "experiment_name": "v3.LR.historical_0051",
        "figstr": "v3.LR.historical_0051",
        "color": "Blue",
        "ts_num_years": "5",
        "plots_original": "None",
        "atmosphere_only": "False",
        "plots_atm": "TREFHT,AODDUST",
        "plots_ice": "None",
        "plots_lnd": "FSH,RH2M,LAISHA,LAISUN,QINTR,QOVER,QRUNOFF,QSOIL,QVEGE,QVEGT,SOILWATER_10CM,TSA,H2OSNO,TOTLITC,CWDC,SOIL1C,SOIL2C,SOIL3C,SOIL4C,WOOD_HARVESTC,TOTVEGC,NBP,GPP,AR,HR",
        "plots_ocn": "None",
        "nrows": "4",
        "ncols": "2",
        "results_dir": "global_time_series_1985-1995_results",
        "regions": "glb,n,s",
        "start_yr": "1985",
        "end_yr": "1995",
    }
)

"""
[global_time_series]
active = True
environment_commands = "#expand global_time_series_environment_commands#"
experiment_name = "#expand case_name#"
figstr = "#expand case_name#"
plots_original="net_toa_flux_restom,global_surface_air_temperature,toa_radiation,net_atm_energy_imbalance,net_atm_water_imbalance"
ts_num_years = 5
walltime = "00:30:00"
years = "1985-1995",
"""
parameters_original_8_no_ocn: Parameters = Parameters(
    {
        "use_ocn": "False",
        "input": "/lcrc/group/e3sm2/ac.wlin/E3SMv3/v3.LR.historical_0051",
        "input_subdir": "archive/ocn/hist",
        "moc_file": "mocTimeSeries_1985-1995.nc",
        "case_dir": CASE_DIR,
        "experiment_name": "v3.LR.historical_0051",
        "figstr": "v3.LR.historical_0051",
        "color": "Blue",
        "ts_num_years": "5",
        "plots_original": "net_toa_flux_restom,global_surface_air_temperature,toa_radiation,net_atm_energy_imbalance,net_atm_water_imbalance",
        "atmosphere_only": "True",
        "plots_atm": "None",
        "plots_ice": "None",
        "plots_lnd": "None",
        "plots_ocn": "None",
        "nrows": "4",
        "ncols": "2",
        "results_dir": "global_time_series_1985-1995_results",
        "regions": "glb,n,s",
        "start_yr": "1985",
        "end_yr": "1995",
    }
)

"""
[global_time_series]
active = True
climo_years = "1985-1989", "1990-1995",
environment_commands = "#expand global_time_series_environment_commands#"
experiment_name = "#expand case_name#"
figstr = "#expand case_name#"
moc_file=mocTimeSeries_1985-1995.nc
ts_num_years = 5
ts_years = "1985-1989", "1985-1995",
walltime = "00:30:00"
years = "1985-1995",
"""
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
        "atmosphere_only": "False",
        "plots_atm": "None",
        "plots_ice": "None",
        "plots_lnd": "None",
        "plots_ocn": "None",
        "nrows": "4",
        "ncols": "2",
        "results_dir": "global_time_series_1985-1995_results",
        "regions": "glb,n,s",
        "start_yr": "1985",
        "end_yr": "1995",
    }
)

"""
[global_time_series]
active = True
climo_years = "1985-1989", "1990-1995",
environment_commands = "source /gpfs/fs1/home/ac.forsyth2/miniforge3/etc/profile.d/conda.sh; conda activate zi_dev_weekly_20241122"
experiment_name = "v3.LR.historical_0051"
figstr = "v3.LR.historical_0051"
moc_file=mocTimeSeries_1985-1995.nc
plots_lnd = "FSH,RH2M,LAISHA,LAISUN,QINTR,QOVER,QRUNOFF,QSOIL,QVEGE,QVEGT,SOILWATER_10CM,TSA,H2OSNO,TOTLITC,CWDC,SOIL1C,SOIL2C,SOIL3C,SOIL4C,WOOD_HARVESTC,TOTVEGC,NBP,GPP,AR,HR"
ts_num_years = 5
ts_years = "1985-1989", "1985-1995",
walltime = "00:30:00"
years = "1985-1995",
"""
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
        "atmosphere_only": "False",
        "plots_atm": "None",
        "plots_ice": "None",
        "plots_lnd": "FSH,RH2M,LAISHA,LAISUN,QINTR,QOVER,QRUNOFF,QSOIL,QVEGE,QVEGT,SOILWATER_10CM,TSA,H2OSNO,TOTLITC,CWDC,SOIL1C,SOIL2C,SOIL3C,SOIL4C,WOOD_HARVESTC,TOTVEGC,NBP,GPP,AR,HR",
        "plots_ocn": "None",
        "nrows": "4",
        "ncols": "2",
        "results_dir": "global_time_series_1985-1995_results",
        "regions": "glb,n,s",
        "start_yr": "1985",
        "end_yr": "1995",
    }
)

if __name__ == "__main__":
    print("Generating parameters")
    parameters: Parameters = parameters_comprehensive_v3

    # TODO: Create actual pytest cases, including image comparison checks
    # (See https://github.com/E3SM-Project/zppy-interfaces/issues/5)

    # CASE_DIR is large, so we don't want to copy it every time.
    # We also want to be able to reset it to the original state after running the test.
    # The only modification `global_time_series` makes to CASE_DIR is add an `ocn` subdirectory.
    # (It also adds a parameters.results_dir subdirectory to the current directory).
    # So, we just have to remove those
    for subdir in [f"{CASE_DIR}/ocn", parameters.results_dir]:
        if os.path.exists(subdir):
            print(f"Removing {subdir}")
            shutil.rmtree(subdir)

    print("Running main")
    main(parameters)

    print(f"Copying {parameters.results_dir} to {WEB_DIR}")
    shutil.copytree(f"{parameters.results_dir}", WEB_DIR)
