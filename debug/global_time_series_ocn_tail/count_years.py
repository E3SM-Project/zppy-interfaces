"""Print the years global_time_series' original plots read for atm, ohc, volume.

Usage: python count_years.py <case_dir>   (called by check.sh)
Calls plots_original.process_data on a zppy reproducer's output, as the job
does: atm from <case_dir>/post/atm/glb/ts/monthly/5yr/ (plus a 1yr/ final
chunk, if any), ocean from <case_dir>/post/ocn/glb/ts/monthly/5yr/.
"""

import sys
from types import SimpleNamespace
from typing import List

from zppy_interfaces.global_time_series.coupled_global import plots_original
from zppy_interfaces.global_time_series.coupled_global.utils import (
    get_vars_original,
)


def span(years: List[int]) -> str:
    return f"{len(years)} years ({years[0]}-{years[-1]})"


def main() -> None:
    case_dir: str = sys.argv[1]
    # Stand-ins with only the attributes process_data reads
    parameters = SimpleNamespace(
        case_dir=case_dir,
        results_dir=f"{case_dir}/post",
        experiment_name="count_years",
        color="Blue",
        year1=1985,
        year2=1995,
        ts_num_years_str="5",
        plots_original=[
            "global_surface_air_temperature",
            "change_ohc",
            "change_sea_level",
        ],
    )
    requested_variables = SimpleNamespace(
        vars_original=get_vars_original(parameters.plots_original)
    )
    annual = plots_original.process_data(
        parameters, requested_variables  # type: ignore[arg-type]
    )[0]["annual"]

    print("x-axis (atm):", span(annual["year"]))
    for key in ("ohc", "volume"):
        print(f"{key:12s}:", span([t.year for t in annual[key].coords["time"].values]))


if __name__ == "__main__":
    main()
