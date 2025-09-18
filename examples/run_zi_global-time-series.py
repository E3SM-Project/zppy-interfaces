from zppy_interfaces.global_time_series.__main__ import main
import sys
import time

sys.argv.extend([
    "--use_ocn", "True",
    "--input", "/lcrc/group/e3sm2/ac.wlin/E3SMv3/v3.LR.historical_0051",
    "--input_subdir", "archive/ocn/hist",
    "--moc_file", "mocTimeSeries_1985-1995.nc",
    "--case_dir", "/lcrc/group/e3sm/ac.forsyth2/zppy_weekly_comprehensive_v3_output/test_issue-23-rebased-20250903/v3.LR.historical_0051",
    "--experiment_name", "v3.LR.historical_0051",
    "--figstr", "v3.LR.historical_0051",
    "--color", "Blue",
    "--ts_num_years", "5",
    "--plots_original","net_toa_flux_restom,global_surface_air_temperature,toa_radiation,net_atm_energy_imbalance,change_ohc,max_moc,change_sea_level,net_atm_water_imbalance",
    #"--plots_original","None",
    "--plots_atm", "TREFHT",
    "--plots_ice", "None",
    "--plots_lnd", "all",
    #"--plots_lnd", "FSH",
    "--plots_ocn", "None",
    "--nrows", "4",
    "--ncols", "2",
    "--results_dir", "/lcrc/group/e3sm/public_html/diagnostic_output/ac.zhang40/tests/zi",
    "--regions", "glb,n,s",
    "--make_viewer", "True",
    "--start_yr", "1985",
    "--end_yr", "1995"
])

start_time = time.time()
main()
end_time = time.time()

print(f"Execution time: {end_time - start_time:.2f} seconds")
