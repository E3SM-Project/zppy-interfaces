import os
import shutil

from zppy_interfaces.global_time_series.__main__ import main
from zppy_interfaces.global_time_series.utils import Parameters

CASE_DIR = "/lcrc/group/e3sm/ac.forsyth2/zi-test-input-data"  # This is 44G.
WEB_DIR = "/lcrc/group/e3sm/public_html/diagnostic_output/ac.forsyth2/zi-test-webdir/"
RESULTS_DIR_PREFIX = "global_time_series_1985-1995_results"

plots_lnd_metric_average = "FSH,RH2M,LAISHA,LAISUN,QINTR,QOVER,QRUNOFF,QSOIL,QVEGE,QVEGT,SOILWATER_10CM,TSA,H2OSNO,"
plots_lnd_metric_total = (
    "TOTLITC,CWDC,SOIL1C,SOIL2C,SOIL3C,SOIL4C,WOOD_HARVESTC,TOTVEGC,NBP,GPP,AR,HR"
)
plots_lnd_all = plots_lnd_metric_average + plots_lnd_metric_total

parameters_viewers: Parameters = Parameters(
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
        "plots_atm": "TREFHT",
        "plots_ice": "None",
        "plots_lnd": plots_lnd_all,
        "plots_ocn": "None",
        "nrows": "1",
        "ncols": "1",
        "results_dir": f"{RESULTS_DIR_PREFIX}_viewers",
        "regions": "glb,n,s",
        "make_viewer": "True",
        "start_yr": "1985",
        "end_yr": "1995",
    }
)

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
        "plots_atm": "TREFHT",
        "plots_ice": "None",
        "plots_lnd": plots_lnd_all,
        "plots_ocn": "None",
        "nrows": "4",
        "ncols": "2",
        "results_dir": f"{RESULTS_DIR_PREFIX}_custom",
        "regions": "glb,n,s",
        "make_viewer": "False",
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
        "make_viewer": "False",
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
        "make_viewer": "False",
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
        "plots_lnd": plots_lnd_all,
        "plots_ocn": "None",
        "nrows": "4",
        "ncols": "2",
        "results_dir": f"{RESULTS_DIR_PREFIX}_comprehensive_v3",
        "regions": "glb,n,s",
        "make_viewer": "False",
        "start_yr": "1985",
        "end_yr": "1995",
    }
)

# https://github.com/E3SM-Project/zppy/pull/694#discussion_r2023998012
# zi-global-time-series --use_ocn False --input /lcrc/group/e3sm2/ac.wlin/E3SMv3/v3.LR.historical_0051 --input_subdir archive/ocn/hist --moc_file mocTimeSeries_1985-2014.nc --case_dir /lcrc/group/e3sm/ac.forsyth2/E3SMv3_20250331_try2/v3.LR.historical_0051 --experiment_name v3.LR.historical_0051 --figstr v3.LR.historical_0051 --color Blue --ts_num_years 30 --plots_original None --plots_atm TREFHT --plots_ice None --plots_lnd FSH --plots_ocn None --nrows 4 --ncols 2 --results_dir ./zi --regions glb,n,s --make_viewer True --start_yr 1985 --end_yr 2014
#
# real	2m29.004s
# user	2m8.127s
# sys	0m8.645s
"""
Output, with extra lines removed:
Running main
2025-04-02 09:27:49,826 [INFO]: __main__.py(main:45) >> Update time series figures
2025-04-02 09:28:35,126 [INFO]: coupled_global.py(process_data:271) >> glb region globalAnnual was computed successfully for these variables: ['TREFHT', 'FSH']
2025-04-02 09:28:35,126 [ERROR]: coupled_global.py(process_data:274) >> glb region globalAnnual could not be computed for these variables: []
2025-04-02 09:28:35,126 [INFO]: coupled_global_plotting.py(make_plot_pdfs:554) >> make_plot_pdfs for rgn=glb, component=original
2025-04-02 09:28:35,126 [INFO]: coupled_global_plotting.py(make_plot_pdfs:554) >> make_plot_pdfs for rgn=glb, component=atm
2025-04-02 09:28:35,128 [INFO]: coupled_global_plotting.py(make_plot_pdfs:581) >> Using reduced figsize
2025-04-02 09:28:35,129 [INFO]: coupled_global_plotting.py(make_plot_pdfs:586) >> Figure size=[675.  412.5]
2025-04-02 09:28:35,129 [INFO]: coupled_global_plotting.py(make_plot_pdfs:589) >> Plotting plot 0 on page 0. This is plot 0 in total.
2025-04-02 09:28:35,140 [INFO]: coupled_global_plotting.py(plot_generic:413) >> plot_generic for TREFHT, rgn=glb
2025-04-02 09:28:36,572 [INFO]: coupled_global_plotting.py(make_plot_pdfs:554) >> make_plot_pdfs for rgn=glb, component=ice
2025-04-02 09:28:36,572 [INFO]: coupled_global_plotting.py(make_plot_pdfs:554) >> make_plot_pdfs for rgn=glb, component=lnd
2025-04-02 09:28:36,573 [INFO]: coupled_global_plotting.py(make_plot_pdfs:581) >> Using reduced figsize
2025-04-02 09:28:36,573 [INFO]: coupled_global_plotting.py(make_plot_pdfs:586) >> Figure size=[675.  412.5]
2025-04-02 09:28:36,573 [INFO]: coupled_global_plotting.py(make_plot_pdfs:589) >> Plotting plot 0 on page 0. This is plot 0 in total.
2025-04-02 09:28:36,580 [INFO]: coupled_global_plotting.py(plot_generic:413) >> plot_generic for FSH, rgn=glb
2025-04-02 09:28:38,787 [INFO]: coupled_global_plotting.py(make_plot_pdfs:554) >> make_plot_pdfs for rgn=glb, component=ocn
2025-04-02 09:28:38,787 [INFO]: coupled_global.py(run:312) >> These glb region plots generated successfully: ['TREFHT', 'FSH']
2025-04-02 09:28:38,787 [ERROR]: coupled_global.py(run:313) >> These glb region plots could not be generated successfully: []
2025-04-02 09:29:13,242 [INFO]: coupled_global.py(process_data:271) >> n region globalAnnual was computed successfully for these variables: ['TREFHT', 'FSH']
2025-04-02 09:29:13,243 [ERROR]: coupled_global.py(process_data:274) >> n region globalAnnual could not be computed for these variables: []
2025-04-02 09:29:13,243 [INFO]: coupled_global_plotting.py(make_plot_pdfs:554) >> make_plot_pdfs for rgn=n, component=original
2025-04-02 09:29:13,243 [INFO]: coupled_global_plotting.py(make_plot_pdfs:554) >> make_plot_pdfs for rgn=n, component=atm
2025-04-02 09:29:13,243 [INFO]: coupled_global_plotting.py(make_plot_pdfs:581) >> Using reduced figsize
2025-04-02 09:29:13,243 [INFO]: coupled_global_plotting.py(make_plot_pdfs:586) >> Figure size=[675.  412.5]
2025-04-02 09:29:13,244 [INFO]: coupled_global_plotting.py(make_plot_pdfs:589) >> Plotting plot 0 on page 0. This is plot 0 in total.
2025-04-02 09:29:13,250 [INFO]: coupled_global_plotting.py(plot_generic:413) >> plot_generic for TREFHT, rgn=n
2025-04-02 09:29:14,334 [INFO]: coupled_global_plotting.py(make_plot_pdfs:554) >> make_plot_pdfs for rgn=n, component=ice
2025-04-02 09:29:14,334 [INFO]: coupled_global_plotting.py(make_plot_pdfs:554) >> make_plot_pdfs for rgn=n, component=lnd
2025-04-02 09:29:14,335 [INFO]: coupled_global_plotting.py(make_plot_pdfs:581) >> Using reduced figsize
2025-04-02 09:29:14,335 [INFO]: coupled_global_plotting.py(make_plot_pdfs:586) >> Figure size=[675.  412.5]
2025-04-02 09:29:14,335 [INFO]: coupled_global_plotting.py(make_plot_pdfs:589) >> Plotting plot 0 on page 0. This is plot 0 in total.
2025-04-02 09:29:14,341 [INFO]: coupled_global_plotting.py(plot_generic:413) >> plot_generic for FSH, rgn=n
2025-04-02 09:29:16,778 [INFO]: coupled_global_plotting.py(make_plot_pdfs:554) >> make_plot_pdfs for rgn=n, component=ocn
2025-04-02 09:29:16,779 [INFO]: coupled_global.py(run:312) >> These n region plots generated successfully: ['TREFHT', 'FSH']
2025-04-02 09:29:16,779 [ERROR]: coupled_global.py(run:313) >> These n region plots could not be generated successfully: []
2025-04-02 09:29:50,685 [INFO]: coupled_global.py(process_data:271) >> s region globalAnnual was computed successfully for these variables: ['TREFHT', 'FSH']
2025-04-02 09:29:50,685 [ERROR]: coupled_global.py(process_data:274) >> s region globalAnnual could not be computed for these variables: []
2025-04-02 09:29:50,685 [INFO]: coupled_global_plotting.py(make_plot_pdfs:554) >> make_plot_pdfs for rgn=s, component=original
2025-04-02 09:29:50,685 [INFO]: coupled_global_plotting.py(make_plot_pdfs:554) >> make_plot_pdfs for rgn=s, component=atm
2025-04-02 09:29:50,685 [INFO]: coupled_global_plotting.py(make_plot_pdfs:581) >> Using reduced figsize
2025-04-02 09:29:50,686 [INFO]: coupled_global_plotting.py(make_plot_pdfs:586) >> Figure size=[675.  412.5]
2025-04-02 09:29:50,686 [INFO]: coupled_global_plotting.py(make_plot_pdfs:589) >> Plotting plot 0 on page 0. This is plot 0 in total.
2025-04-02 09:29:50,692 [INFO]: coupled_global_plotting.py(plot_generic:413) >> plot_generic for TREFHT, rgn=s
2025-04-02 09:29:51,784 [INFO]: coupled_global_plotting.py(make_plot_pdfs:554) >> make_plot_pdfs for rgn=s, component=ice
2025-04-02 09:29:51,784 [INFO]: coupled_global_plotting.py(make_plot_pdfs:554) >> make_plot_pdfs for rgn=s, component=lnd
2025-04-02 09:29:51,784 [INFO]: coupled_global_plotting.py(make_plot_pdfs:581) >> Using reduced figsize
2025-04-02 09:29:51,785 [INFO]: coupled_global_plotting.py(make_plot_pdfs:586) >> Figure size=[675.  412.5]
2025-04-02 09:29:51,785 [INFO]: coupled_global_plotting.py(make_plot_pdfs:589) >> Plotting plot 0 on page 0. This is plot 0 in total.
2025-04-02 09:29:51,791 [INFO]: coupled_global_plotting.py(plot_generic:413) >> plot_generic for FSH, rgn=s
2025-04-02 09:29:54,149 [INFO]: coupled_global_plotting.py(make_plot_pdfs:554) >> make_plot_pdfs for rgn=s, component=ocn
2025-04-02 09:29:54,149 [INFO]: coupled_global.py(run:312) >> These s region plots generated successfully: ['TREFHT', 'FSH']
2025-04-02 09:29:54,149 [ERROR]: coupled_global.py(run:313) >> These s region plots could not be generated successfully: []
2025-04-02 09:29:54,175 [INFO]: coupled_global_viewer.py(create_viewer:39) >> Creating viewer for atm
2025-04-02 09:29:54,175 [INFO]: coupled_global_viewer.py(create_viewer:45) >> Adding group All Variables
2025-04-02 09:29:54,271 [INFO]: coupled_global.py(coupled_global:352) >> Viewer URL for atm: table_atm/index.html
2025-04-02 09:29:54,271 [INFO]: coupled_global_viewer.py(create_viewer:39) >> Creating viewer for lnd
2025-04-02 09:29:54,271 [INFO]: coupled_global_viewer.py(create_viewer:45) >> Adding group Energy Flux
2025-04-02 09:29:54,290 [INFO]: coupled_global.py(coupled_global:352) >> Viewer URL for lnd: table_lnd/index.html
2025-04-02 09:29:54,291 [INFO]: coupled_global_viewer.py(create_viewer_index:91) >> Creating viewer index
2025-04-02 09:29:54,293 [INFO]: coupled_global.py(coupled_global:366) >> Viewer index URL: global_time_series_1985-2014_results_performance/index.html
Copying global_time_series_1985-2014_results_performance to /lcrc/group/e3sm/public_html/diagnostic_output/ac.forsyth2/zi-test-webdir//global_time_series_1985-2014_results_performance
"""
parameters_performance: Parameters = Parameters(
    {
        "use_ocn": "False",
        "input": "/lcrc/group/e3sm2/ac.wlin/E3SMv3/v3.LR.historical_0051",
        "input_subdir": "archive/ocn/hist",
        "moc_file": "mocTimeSeries_1985-2014.nc",
        "case_dir": "/lcrc/group/e3sm/ac.forsyth2/E3SMv3_20250331_try2/v3.LR.historical_0051",  # Note different case dir than the other Parameters objects in this file
        "experiment_name": "v3.LR.historical_0051",
        "figstr": "v3.LR.historical_0051",
        "color": "Blue",
        "ts_num_years": "30",
        "plots_original": "None",
        "plots_atm": "TREFHT",
        "plots_ice": "None",
        "plots_lnd": "FSH",
        "plots_ocn": "None",
        "nrows": "4",
        "ncols": "2",
        "results_dir": "global_time_series_1985-2014_results_performance",  # Note different results dir than the other Parameters objects in this file
        "regions": "glb,n,s",
        "make_viewer": "True",
        "start_yr": "1985",
        "end_yr": "2014",
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
    generate_results(parameters_viewers)
    generate_results(parameters_custom)
    generate_results(parameters_original_8_no_ocn)
    generate_results(parameters_original_8)
    generate_results(parameters_comprehensive_v3)


if __name__ == "__main__":
    # TODO: Create actual pytest cases, including image comparison checks
    # (See https://github.com/E3SM-Project/zppy-interfaces/issues/5)
    # run_all_cases()
    generate_results(parameters_performance)
