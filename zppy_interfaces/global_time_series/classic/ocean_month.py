# Compute time series of ocean heat content (ohc) using MPAS-O output

import glob
from datetime import datetime

import numpy as np
from mpas_tools.cime.constants import constants
from netCDF4 import Dataset, chartostring, date2num

from zppy_interfaces.multi_utils.logger import _setup_child_logger

logger = _setup_child_logger(__name__)


def ocean_month(
    path_in: str,
    subtask_name: str,
    case_dir: str,
    start_yr: int,
    end_yr: int,
    ts_num_years: int,
):
    path_out = f"{case_dir}/post/{subtask_name}/ocn/glb/ts/monthly/{ts_num_years}yr"

    # Ocean constants
    # specific heat [J/(kg*degC)]
    cp = constants["SHR_CONST_CPSW"]
    # [kg/m3]
    rho = constants["SHR_CONST_RHOSW"]
    # [J/(degC*m3)]
    fac = rho * cp

    # Time units, calendar
    tcalendar = "noleap"
    tunits = f"days since {start_yr:04d}-01-01 00:00:00"

    # Loop over year sets
    for y in range(start_yr, end_yr, ts_num_years):

        year1 = y
        year2 = y + ts_num_years - 1
        files = []
        for year in range(year1, year2 + 1):
            logger.info(f"year={year}")
            inFiles = (
                f"{path_in}/*mpaso.hist.am.timeSeriesStatsMonthly.{year:04d}-??-??.nc"
            )
            files.extend(sorted(glob.glob(inFiles)))
        out = f"{path_out}/mpaso.glb.{year1:04d}01-{year2:04d}12.nc"

        # Create output file
        fout = Dataset(out, "w", format="NETCDF4_CLASSIC")
        fout.createDimension("time", None)
        fout.createDimension("nbnd", 2)

        time = fout.createVariable("time", "f8", ("time",))
        time.long_name = "time"
        time.units = tunits
        time.calendar = tcalendar
        time.bounds = "time_bnds"

        time_bnds = fout.createVariable("time_bnds", "f8", ("time", "nbnd"))
        time_bnds.long_name = "time interval endpoints"

        ohc = fout.createVariable("ohc", "f8", ("time",))
        ohc.long_name = "total ocean heat content"
        ohc.units = "J"

        volume = fout.createVariable("volume", "f8", ("time",))
        volume.long_name = "sum of the volumeCell variable over the full domain, used to normalize global statistics"
        volume.units = "m^3"

        # OHC from monthly time series
        itime = 0
        for file in files:

            # Open current input file
            logger.info(f"mpaso file: {file}")
            f = Dataset(file, "r")

            # Time variables
            xtime_startMonthly = chartostring(f.variables["xtime_startMonthly"][:])
            xtime_endMonthly = chartostring(f.variables["xtime_endMonthly"][:])

            # Convert to datetime objects (assuming 0 UTC boundary)
            date_start = np.array(
                [datetime.strptime(x[0:10], "%Y-%m-%d") for x in xtime_startMonthly]
            )
            date_end = np.array(
                [datetime.strptime(x[0:10], "%Y-%m-%d") for x in xtime_endMonthly]
            )

            # Convert to netCDF4 time
            tstart = date2num(date_start, tunits, tcalendar)
            tend = date2num(date_end, tunits, tcalendar)
            t = 0.5 * (tstart + tend)

            # Variables needed to compute global OHC
            iregion = 6  # global average region
            sumLayerMaskValue = f.variables[
                "timeMonthly_avg_avgValueWithinOceanLayerRegion_sumLayerMaskValue"
            ][:, iregion, :]
            avgLayerArea = f.variables[
                "timeMonthly_avg_avgValueWithinOceanLayerRegion_avgLayerArea"
            ][:, iregion, :]
            avgLayerThickness = f.variables[
                "timeMonthly_avg_avgValueWithinOceanLayerRegion_avgLayerThickness"
            ][:, iregion, :]
            avgLayerTemperature = f.variables[
                "timeMonthly_avg_avgValueWithinOceanLayerRegion_avgLayerTemperature"
            ][:, iregion, :]

            # volumeCellGlobal
            volumeCell = f.variables["timeMonthly_avg_volumeCellGlobal"][:]

            # Close current input file
            f.close()

            # Compute OHC
            layerArea = sumLayerMaskValue * avgLayerArea
            layerVolume = layerArea * avgLayerThickness
            tmp = layerVolume * avgLayerTemperature
            ohc_tot = fac * np.sum(tmp, axis=1)

            # Diagnostics printout
            for i in range(len(date_start)):
                logger.info(
                    f"Start, End, OHC = {date_start[i]} ({tstart[i]}), {date_end[i]} ({tend[i]}), {ohc_tot[i]:.2e}"
                )

            # Write data
            time[itime:] = t
            time_bnds[itime:, 0] = tstart
            time_bnds[itime:, 1] = tend
            ohc[itime:] = ohc_tot
            volume[itime:] = volumeCell

            itime = itime + len(t)

        # Close output file
        fout.close()
