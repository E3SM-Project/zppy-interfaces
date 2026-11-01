"""MRE: global_time_series change_ohc / change_sea_level fail with a tail year.

Mirrors comprehensive_v3 (discussion #885, 2026-10-30 run):
  [ts] glb years = "1985:1995:5"  -> atm 5yr chunks 1985-1989, 1990-1994 only
  [global_time_series] years = "1985-1995", ts_num_years = 5
zppy-interfaces >= f01bb61 (#66) makes create_ocean_ts.ocean_month also write
the tail chunk mpaso.glb.199501-199512.nc into the *5yr/* directory, which the
ohc/vol reader globs in full. Years come from atm (10), ohc from ocn (11):
  ValueError: x and y must have same first dimension, but have shapes (10,) and (11,)

Usage: python -I mre_synthetic.py   (in a zppy-interfaces env; see README.md)
Exits with RuntimeError on the bug; prints "change_ohc and change_sea_level
plotted OK" with the fix.
"""

import os
import tempfile
from types import SimpleNamespace
from typing import Tuple

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

YEAR1, YEAR2, TS_NUM_YEARS = 1985, 1995, 5


def monthly_time(y1: int, y2: int, units: str) -> Tuple[np.ndarray, np.ndarray]:
    """Mid-month times and bounds (noleap) in days since `units` origin."""
    mdays = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    origin = int(units.split()[2][:4])
    month_starts = np.concatenate([[0], np.cumsum(mdays)[:-1]])
    starts = np.concatenate(
        [(y - origin) * 365 + month_starts for y in range(y1, y2 + 1)]
    ).astype("f8")
    ends = starts + np.tile(mdays, y2 - y1 + 1)
    return 0.5 * (starts + ends), np.stack([starts, ends], axis=1)


def write(path: str, y1: int, y2: int, data_vars: dict, units: str) -> None:
    t, tb = monthly_time(y1, y2, units)
    n = len(t)
    ds = xr.Dataset(
        {k: (dims, fn(n), attrs) for k, (dims, fn, attrs) in data_vars.items()},
        coords={"time": ("time", t, {"bounds": "time_bnds"})},
    )
    ds["time_bnds"] = (("time", "nbnd"), tb)
    ds["time"].attrs.update(units=units, calendar="noleap")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ds.to_netcdf(path)


def main() -> None:
    root = tempfile.mkdtemp(prefix="mre_gts_tail_ocn_")
    case_dir = f"{root}/case"
    results_dir = f"{root}/results"
    atm_dir = f"{case_dir}/post/atm/glb/ts/monthly/{TS_NUM_YEARS}yr"
    ocn_dir = f"{results_dir}/ocn/glb/ts/monthly/{TS_NUM_YEARS}yr"

    # atm: what zppy [ts] years = "1985:1995:5" produces (no 1yr/ tail chunk)
    for y1 in (1985, 1990):
        y2 = y1 + TS_NUM_YEARS - 1
        write(
            f"{atm_dir}/TREFHT_{y1:04d}01_{y2:04d}12.nc",
            y1,
            y2,
            {
                "TREFHT": (
                    ("time", "rgn"),
                    lambda n: np.full((n, 3), 288.0),
                    {"units": "K"},
                )
            },
            "days since 1985-01-01 00:00:00",
        )

    # ocn: what create_ocean_ts.ocean_month writes after zi #66, incl. the tail
    for y1, y2 in ((1985, 1989), (1990, 1994), (1995, 1995)):
        write(
            f"{ocn_dir}/mpaso.glb.{y1:04d}01-{y2:04d}12.nc",
            y1,
            y2,
            {
                "ohc": (("time",), lambda n: np.linspace(0, 1e23, n), {"units": "J"}),
                "volume": (("time",), lambda n: np.full(n, 1.3e18), {"units": "m^3"}),
            },
            "days since 1985-01-01 00:00:00",
        )

    parameters = SimpleNamespace(
        case_dir=case_dir,
        results_dir=results_dir,
        experiment_name="mre",
        color="Blue",
        year1=YEAR1,
        year2=YEAR2,
        ts_num_years_str=str(TS_NUM_YEARS),
        plots_original=[
            "global_surface_air_temperature",
            "change_ohc",
            "change_sea_level",
        ],
    )
    requested_variables = SimpleNamespace(
        vars_original=get_vars_original(parameters.plots_original)
    )

    # Stand-ins with only the attributes process_data reads
    exps = plots_original.process_data(
        parameters, requested_variables  # type: ignore[arg-type]
    )
    exp = exps[0]
    print(f"len(year) = {len(exp['annual']['year'])}  (from atm)")
    print(f"len(ohc)  = {len(exp['annual']['ohc'])}  (from ocn)")
    print(f"len(vol)  = {len(exp['annual']['volume'])}  (from ocn)")

    fig, ax = plt.subplots()
    # Raises RuntimeError when the lengths differ
    mix_pdf_original.plot_change_ohc(ax, [YEAR1, YEAR2], exps, "glb")
    mix_pdf_original.plot_change_sea_level(ax, [YEAR1, YEAR2], exps, "glb")
    print("change_ohc and change_sea_level plotted OK")


if __name__ == "__main__":
    main()
