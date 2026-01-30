"""Budget visualization functions using Bokeh and matplotlib."""

import os
from typing import Dict, List

import numpy as np

from .parser import Budget


def generate_ascii_output(
    budget_obj: Budget, budget_name: str, output_dir: str
) -> None:
    """Generate ASCII summary table for a budget."""
    if budget_obj.data is None:
        print(f"No data available for {budget_name} budget")
        return

    # Calculate average over the period
    avg = np.average(budget_obj.data[:, :, :], axis=0)

    # Generate output filename
    filename = os.path.join(output_dir, f"{budget_name}_budget_summary.txt")

    with open(filename, "w") as f:
        f.write(f"----- Average {budget_name} budget years {budget_obj.years[0]:04d} ")
        f.write(f"to {budget_obj.years[-1]:04d} ({budget_obj.units}) -----\n")
        f.write("\n")

        # Write header
        ncols = len(budget_obj.cols)  # type: ignore
        header_line = f"{'':10s}" + "".join([f"{col:>12s} " for col in budget_obj.cols])  # type: ignore
        f.write(header_line + "\n")

        # Write data rows
        for row in budget_obj.rows:  # type: ignore
            irow = budget_obj.irow[row]  # type: ignore
            data_line = f"{row:10s}" + "".join(
                [f"{avg[irow, i]:12.6f} " for i in range(ncols)]
            )
            f.write(data_line + "\n")

        f.write("-" * 60 + "\n")

    print(f"ASCII summary written to {filename}")


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

        # Create ColumnDataSource
        data = {}
        data["years"] = b.years

        for krow, vrow in b.irow.items():  # type: ignore
            for kcol, vcol in b.icol.items():  # type: ignore
                raw_data = b.data[:, vrow, vcol]  # type: ignore
                cumsum_data = np.cumsum(raw_data) - raw_data[0]
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
            y_axis_label=f"{budget_name} budget ({b.units})",
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
            plt.ylabel(f"{budget_name} budget ({b.units})")
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
