"""Budget visualization functions using Bokeh and matplotlib."""

import os
from typing import Dict, List

import numpy as np

from .parser import Budget

# Seconds per year (365 days)
DT_SECONDS_PER_YEAR = 365.0 * 24.0 * 60.0 * 60.0

# Unit conversion factors
# Water: kg/m2s*1e6 -> mm (since 1 kg/m2 = 1 mm, multiply by dt/1e6)
WATER_CONVERSION = DT_SECONDS_PER_YEAR / 1e6
# Energy: W/m2 -> J/m2 *1e9 (since W = J/s, multiply by dt/1e9)
ENERGY_CONVERSION = DT_SECONDS_PER_YEAR / 1e9
# Carbon: kg-C/m2s*1e10 -> kg-C/m2 (multiply by dt/1e10)
CARBON_CONVERSION = DT_SECONDS_PER_YEAR / 1e10


def generate_html_plots(
    budgets: Dict[str, Budget], budget_names: List[str], output_dir: str
) -> None:
    """Generate interactive HTML plots using bokeh."""
    try:
        import itertools

        from bokeh.layouts import column
        from bokeh.models import ColumnDataSource, Legend
        from bokeh.palettes import Category10
        from bokeh.plotting import figure, output_file, save
    except ImportError as e:
        print(f"ERROR: bokeh package not available: {e}")
        return

    # Also try to import matplotlib for PNG fallback
    try:
        import matplotlib.pyplot as plt

        matplotlib_available = True
    except ImportError:
        matplotlib_available = False

    # Create budget plots
    plots = []
    for budget_name in budget_names:
        if budget_name not in budgets:
            continue

        if budgets[budget_name].data is None:
            continue

        b = budgets[budget_name]

        # List of colors for plots
        colors = itertools.cycle(Category10[10])

        # Determine unit conversion factor based on budget type
        if budget_name == "water":
            conversion_factor = WATER_CONVERSION
            converted_units = "mm"
        elif budget_name == "heat":
            conversion_factor = ENERGY_CONVERSION
            converted_units = "J/m2 *1e9"
        elif budget_name == "carbon":
            conversion_factor = CARBON_CONVERSION
            converted_units = "kg-C/m2"
        else:
            conversion_factor = 1.0
            converted_units = b.units

        # Create ColumnDataSource
        data = {}
        data["years"] = b.years

        for krow, vrow in b.irow.items():  # type: ignore
            for kcol, vcol in b.icol.items():  # type: ignore
                raw_data = b.data[:, vrow, vcol]  # type: ignore
                # Apply unit conversion and compute cumulative sum
                converted_data = raw_data * conversion_factor
                cumsum_data = np.cumsum(converted_data) - converted_data[0]
                data[krow + "_" + kcol] = cumsum_data
        source = ColumnDataSource(data=data)

        # Determine which row to plot
        if "*SUM*" in b.irow:  # type: ignore
            row_name = "*SUM*"
            plot_title = f"*SUM* annual cumulative {budget_name} budget"
        else:
            row_name = list(b.irow.keys())[0]  # type: ignore
            plot_title = f"{row_name} annual cumulative {budget_name} budget"

        # Create Bokeh plot
        p = figure(
            title=plot_title,
            height=400,
            width=1200,
            x_axis_label="year",
            y_axis_label=f"{budget_name} budget ({converted_units})",
        )
        p.add_layout(Legend(), "right")

        # Add lines to Bokeh plot
        for k, v in b.icol.items():  # type: ignore
            line_name = f"{row_name}_{k}"
            if line_name in data:
                p.line(
                    x="years",
                    y=line_name,
                    legend_label=k,
                    line_width=2,
                    color=next(colors),
                    source=source,
                )
        p.legend.click_policy = "hide"
        plots.append(p)

        # Create matplotlib PNG fallback
        if matplotlib_available:
            plt.figure(figsize=(12, 6))

            colors_mpl = plt.cm.tab10(np.linspace(0, 1, len(b.icol)))  # type: ignore
            for i, (k, v) in enumerate(b.icol.items()):  # type: ignore
                line_name = f"{row_name}_{k}"
                if line_name in data:
                    plt.plot(
                        data["years"],
                        data[line_name],
                        label=k,
                        linewidth=2,
                        color=colors_mpl[i],
                    )

            plt.title(plot_title)
            plt.xlabel("year")
            plt.ylabel(f"{budget_name} budget ({converted_units})")
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            png_file = os.path.join(output_dir, f"budget_{budget_name}.png")
            plt.savefig(png_file, dpi=150, bbox_inches="tight")
            plt.close()

    if plots:
        # Save Bokeh HTML
        html_file = os.path.join(output_dir, "budgets.html")

        c = column(children=plots, sizing_mode="stretch_width")
        output_file(html_file)
        save(c)

        print(f"Interactive HTML plots written to {html_file}")
        if matplotlib_available:
            print(f"PNG fallback plots also created in {output_dir}")
    else:
        print("No plots generated - no valid budget data found")
