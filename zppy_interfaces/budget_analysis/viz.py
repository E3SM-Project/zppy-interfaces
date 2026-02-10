"""Visualization for whole-model budget analysis.

Generates an HTML report with Bokeh plots for each budget check result.
"""

import os
from typing import Dict, List

import numpy as np
import pandas as pd
from bokeh.layouts import column
from bokeh.models import Div
from bokeh.palettes import Category10
from bokeh.plotting import figure, output_file, save

from .checks import CheckResult
from .normalization import SECONDS_PER_YEAR

# Units per quantity for plot labels
QUANTITY_UNITS: Dict[str, Dict[str, str]] = {
    "water": {"flux": "mm/yr", "cumulative": "mm"},
    "heat": {"flux": "W/m2", "cumulative": "J/m2*1e9"},
}


def _cum_scale(quantity: str) -> float:
    """Scale factor for cumulative plots.

    Heat: W/m2 cumsum over years -> J/m2 (multiply by seconds_per_year).
    Water: mm/yr cumsum over years -> mm (no scaling needed).
    """
    if quantity == "heat":
        return SECONDS_PER_YEAR / 1e9
    return 1.0


def generate_budget_report(
    results: List[CheckResult],
    df: pd.DataFrame,
    output_dir: str,
    quantity: str = "water",
) -> str:
    """Generate an HTML report with budget check plots.

    Returns path to the HTML file.
    """
    units = QUANTITY_UNITS.get(quantity, {"flux": "", "cumulative": ""})
    flux_units = units["flux"]
    cum_units = units["cumulative"]
    scale = _cum_scale(quantity)
    plots: list = []

    for r in results:
        if r.name == f"cpl_{quantity}_component_fluxes":
            plots.append(
                Div(text=f"<h2>{quantity.title()} Budget Overview</h2>")
            )
            plots.append(
                _plot_cumulative_components(r, quantity, cum_units, scale)
            )

        elif r.name.endswith("_interface_match"):
            # name format: {component}_{quantity}_interface_match
            comp = r.name.replace(f"_{quantity}_interface_match", "")
            plots.append(Div(text=f"<h2>Interface Match: {comp}</h2>"))
            plots.append(
                _plot_comparison(
                    r, f"{comp} {quantity.title()} Flux (cpl vs {comp})",
                    flux_units,
                )
            )
            plots.append(
                _plot_residual(
                    r, f"Interface Residual (cpl - {comp})", flux_units
                )
            )
            plots.append(
                _plot_cumulative(
                    r, f"Interface Cumulative Residual (cpl - {comp})",
                    cum_units, scale,
                )
            )

        elif r.name == "lnd_closure":
            plots.append(Div(text="<h2>Land Water Closure</h2>"))
            plots.append(
                _plot_comparison(r, "Land ΔStorage vs ∫Flux dt", cum_units)
            )
            plots.append(_plot_residual(r, "Closure Residual", cum_units))
            plots.append(
                _plot_cumulative(r, "Closure Cumulative Residual", cum_units)
            )

        elif r.name.startswith("ocn_") and r.name.endswith("_closure"):
            label = "Water" if "water" in r.name else "Heat"
            change_label = "ΔMass" if label == "Water" else "ΔEnergy"
            plots.append(Div(text=f"<h2>Ocean {label} Closure</h2>"))
            plots.append(
                _plot_comparison(
                    r, f"Ocean {change_label} vs Net Flux", flux_units
                )
            )
            plots.append(
                _plot_residual(r, "Ocean Closure Residual", flux_units)
            )
            plots.append(
                _plot_cumulative(
                    r, "Ocean Closure Cumulative Residual",
                    cum_units, scale,
                )
            )

    if not plots:
        print(f"No {quantity} plots generated — no check results available")
        return ""

    html_path = os.path.join(output_dir, f"{quantity}_budget_report.html")
    output_file(html_path, title=f"E3SM {quantity.title()} Budget Analysis")
    save(column(plots, sizing_mode="stretch_width"))
    print(f"Report written to {html_path}")
    return html_path


def generate_landing_page(
    output_dir: str, report_paths: Dict[str, str]
) -> str:
    """Generate an index.html landing page linking to individual budget reports.

    Returns path to the landing page.
    """
    links = []
    for quantity, path in sorted(report_paths.items()):
        filename = os.path.basename(path)
        links.append(
            f'<li><a href="{filename}">{quantity.title()} Budget Report</a></li>'
        )
    html = f"""<!DOCTYPE html>
<html>
<head><title>E3SM Budget Analysis</title></head>
<body>
<h1>E3SM Budget Analysis</h1>
<ul>
{"".join(links)}
</ul>
</body>
</html>"""
    index_path = os.path.join(output_dir, "index.html")
    with open(index_path, "w") as f:
        f.write(html)
    print(f"Landing page written to {index_path}")
    return index_path


def _make_figure(title: str, y_label: str) -> figure:
    return figure(
        title=title,
        height=350,
        width=1200,
        x_axis_label="year",
        y_axis_label=y_label,
    )


def _plot_residual(r: CheckResult, title: str, units: str) -> figure:
    """Plot residual time series with a zero reference line."""
    p = _make_figure(title, f"residual ({units})")
    p.line(r.years, r.residual, line_width=2, color="red")
    p.line(
        r.years, np.zeros_like(r.years), line_width=1, color="gray",
        line_dash="dashed",
    )
    return p


def _plot_cumulative(
    r: CheckResult, title: str, units: str, scale: float = 1.0
) -> figure:
    """Plot cumulative residual, optionally scaled for unit conversion."""
    p = _make_figure(title, f"cumulative residual ({units})")
    p.line(
        r.years, r.cumulative_residual * scale, line_width=2, color="darkred"
    )
    return p


def _plot_comparison(r: CheckResult, title: str, units: str) -> figure:
    """Plot LHS and RHS on the same axes."""
    p = _make_figure(title, units)
    p.line(r.years, r.lhs, line_width=2, color="blue", legend_label=r.lhs_label)
    p.line(
        r.years, r.rhs, line_width=2, color="orange", legend_label=r.rhs_label
    )
    p.legend.click_policy = "hide"
    return p


def _plot_cumulative_components(
    r: CheckResult, quantity: str, units: str, scale: float = 1.0
) -> figure:
    """Cumulative net flux per component, with *SUM* residual highlighted."""
    p = _make_figure(
        f"Cumulative Net {quantity.title()} Flux per Component", units
    )
    if r.components is None:
        return p
    # Plot component lines, highlight *SUM* as thick dashed red
    other_names = sorted(k for k in r.components if k != "*SUM*")
    colors = Category10[max(3, len(other_names) + 1)]
    for i, name in enumerate(other_names):
        p.line(
            r.years,
            r.components[name] * scale,
            line_width=2,
            color=colors[i % len(colors)],
            legend_label=name,
        )
    if "*SUM*" in r.components:
        p.line(
            r.years,
            r.components["*SUM*"] * scale,
            line_width=3,
            color="red",
            line_dash="dashed",
            legend_label="*SUM* (residual)",
        )
    p.legend.click_policy = "hide"
    p.legend.location = "top_left"
    return p
