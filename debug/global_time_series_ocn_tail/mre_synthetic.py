"""Synthetic reproducer: change_ohc / change_sea_level fail with a tail year.

years = 1985-1995, ts_num_years = 5. atm has only the 5yr chunks 1985-1989
and 1990-1994; the ocean also has the final chunk 1995 (written by
create_ocean_ts since #66). See README.md.

Usage: python -I mre_synthetic.py
"""

import os
import tempfile
from types import SimpleNamespace

import cftime
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

from zppy_interfaces.global_time_series.coupled_global import (  # noqa: E402
    mix_pdf_original,
    plots_original,
)
from zppy_interfaces.global_time_series.coupled_global.utils import (  # noqa: E402
    get_vars_original,
)


def monthly_dataset(y1: int, y2: int) -> xr.Dataset:
    """Empty dataset with monthly noleap time and time_bnds."""
    edges = [
        cftime.DatetimeNoLeap(y, m, 1) for y in range(y1, y2 + 1) for m in range(1, 13)
    ] + [cftime.DatetimeNoLeap(y2 + 1, 1, 1)]
    time = [a + (b - a) / 2 for a, b in zip(edges[:-1], edges[1:])]
    ds = xr.Dataset(coords={"time": ("time", time, {"bounds": "time_bnds"})})
    ds["time_bnds"] = (("time", "nbnd"), np.array([edges[:-1], edges[1:]]).T)
    return ds


def write(ds: xr.Dataset, path: str) -> None:
    # time_bnds is decoded with time's units, so both need the same units
    units = {"units": "days since 1985-01-01", "calendar": "noleap"}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ds.to_netcdf(path, encoding={"time": units, "time_bnds": units})


def main() -> None:
    root = tempfile.mkdtemp()
    atm_dir = f"{root}/post/atm/glb/ts/monthly/5yr"
    ocn_dir = f"{root}/post/ocn/glb/ts/monthly/5yr"
    for y1, y2 in ((1985, 1989), (1990, 1994)):
        ds = monthly_dataset(y1, y2)
        ds["TREFHT"] = (("time", "rgn"), np.full((ds.sizes["time"], 3), 288.0))
        ds["TREFHT"].attrs["units"] = "K"
        write(ds, f"{atm_dir}/TREFHT_{y1}01_{y2}12.nc")
    for y1, y2 in ((1985, 1989), (1990, 1994), (1995, 1995)):
        ds = monthly_dataset(y1, y2)
        ds["ohc"] = ("time", np.full(ds.sizes["time"], 1e24), {"units": "J"})
        ds["volume"] = ("time", np.full(ds.sizes["time"], 1.3e18), {"units": "m^3"})
        write(ds, f"{ocn_dir}/mpaso.glb.{y1}01-{y2}12.nc")

    # Stand-ins with only the attributes process_data reads
    parameters = SimpleNamespace(
        case_dir=root,
        results_dir=f"{root}/post",
        experiment_name="mre",
        color="Blue",
        year1=1985,
        year2=1995,
        ts_num_years_str="5",
        plots_original=["global_surface_air_temperature", "change_ohc"],
    )
    requested_variables = SimpleNamespace(
        vars_original=get_vars_original(parameters.plots_original)
    )
    exps = plots_original.process_data(
        parameters, requested_variables  # type: ignore[arg-type]
    )
    annual = exps[0]["annual"]
    print(f"years: atm {len(annual['year'])}, ohc {len(annual['ohc'])}")

    # Raises RuntimeError when the lengths differ
    mix_pdf_original.plot_change_ohc(plt.subplots()[1], [1985, 1995], exps, "glb")
    print("change_ohc plotted OK")


if __name__ == "__main__":
    main()
