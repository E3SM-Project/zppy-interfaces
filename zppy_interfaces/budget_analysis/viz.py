"""Visualization for whole-model budget analysis.

Generates an HTML report with Bokeh plots for each budget check result.
"""

import os
from typing import List

import numpy as np
import pandas as pd
from bokeh.layouts import column
from bokeh.models import Div
from bokeh.palettes import Category10
from bokeh.plotting import figure, output_file, save

from .checks import CheckResult


def generate_budget_report(
    results: List[CheckResult],
    df: pd.DataFrame,
    output_dir: str,
) -> str:
    """Generate an HTML report with budget check plots.

    Returns path to the HTML file.
    """
    plots: list = []

    for r in results:
        if r.name == "cpl_component_fluxes":
            plots.append(Div(text="<h2>Conservation Overview</h2>"))
            plots.append(_plot_cumulative_components(r))

        elif r.name.endswith("_interface_match"):
            comp = r.name.replace("_interface_match", "")
            plots.append(Div(text=f"<h2>Interface Match: {comp}</h2>"))
            plots.append(
                _plot_comparison(r, f"{comp} Water Flux (cpl vs {comp})", "mm/yr")
            )
            plots.append(
                _plot_residual(r, f"Interface Residual (cpl - {comp})", "mm/yr")
            )

        elif r.name == "lnd_closure":
            plots.append(Div(text="<h2>Land Water Closure</h2>"))
            plots.append(_plot_comparison(r, "Land ΔStorage vs ∫Flux dt", "mm"))
            plots.append(_plot_residual(r, "Closure Residual", "mm"))
            plots.append(_plot_cumulative(r, "Closure Cumulative Residual", "mm"))

        elif r.name == "ocn_closure":
            plots.append(Div(text="<h2>Ocean Water Closure</h2>"))
            plots.append(_plot_comparison(r, "Ocean ΔMass vs Net Flux", "mm"))
            plots.append(_plot_residual(r, "Ocean Closure Residual", "mm"))
            plots.append(_plot_cumulative(r, "Ocean Closure Cumulative Residual", "mm"))

    if not plots:
        print("No plots generated — no check results available")
        return ""

    html_path = os.path.join(output_dir, "water_budget_report.html")
    output_file(html_path, title="E3SM Water Budget Analysis")
    save(column(plots, sizing_mode="stretch_width"))
    print(f"Report written to {html_path}")
    return html_path


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
        r.years, np.zeros_like(r.years), line_width=1, color="gray", line_dash="dashed"
    )
    return p


def _plot_cumulative(r: CheckResult, title: str, units: str) -> figure:
    """Plot cumulative residual."""
    p = _make_figure(title, f"cumulative residual ({units})")
    p.line(r.years, r.cumulative_residual, line_width=2, color="darkred")
    return p


def _plot_comparison(r: CheckResult, title: str, units: str) -> figure:
    """Plot LHS and RHS on the same axes."""
    p = _make_figure(title, units)
    p.line(r.years, r.lhs, line_width=2, color="blue", legend_label="LHS")
    p.line(r.years, r.rhs, line_width=2, color="orange", legend_label="RHS")
    p.legend.click_policy = "hide"
    return p


def _plot_cumulative_components(r: CheckResult) -> figure:
    """Cumulative net water flux per component, with *SUM* residual highlighted."""
    p = _make_figure("Cumulative Net Water Flux per Component", "mm")
    if r.components is None:
        return p
    # Plot component lines, highlight *SUM* as thick dashed red
    other_names = sorted(k for k in r.components if k != "*SUM*")
    colors = Category10[max(3, len(other_names) + 1)]
    for i, name in enumerate(other_names):
        p.line(
            r.years,
            r.components[name],
            line_width=2,
            color=colors[i % len(colors)],
            legend_label=name,
        )
    if "*SUM*" in r.components:
        p.line(
            r.years,
            r.components["*SUM*"],
            line_width=3,
            color="red",
            line_dash="dashed",
            legend_label="*SUM* (residual)",
        )
    p.legend.click_policy = "hide"
    p.legend.location = "top_left"
    return p
