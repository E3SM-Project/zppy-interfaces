import glob
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from jinja2 import Environment, FileSystemLoader


def collect_config(
    title: str = "E3SM-PMP Diagnostics",
    subtitle: str = "Model vs Obs.",
    version: str = "v3.8.2",
    case_id: str = "v3.LR.amip",
    run_dir: str = "model_vs_obs",
    out_dir: str = "./pcmdi_diags/viewer",
    diag_dir: str = "./pcmdi_diags",
    obs_dir: str = "./observations/Atm/time-series",
    pmp_dir: str = "./pcmdi_data",
    clim_period: str = "1985-2014",
    emov_period: str = "1985-2014",
    enso_period: str = "1985-2014",
    clim_regions: Optional[List[str]] = None,
    clim_vars: Optional[List[str]] = None,
    clim_reference: str = "obs",
    emov_vars: Optional[List[str]] = None,
    emov_reference: str = "obs",
    enso_vars: Optional[List[str]] = None,
    enso_reference: str = "obs",
) -> Dict[str, object]:
    """
    Collects and returns configuration information for the diagnostics viewer.
    Uses default settings when specific arguments are not provided.

    Returns:
        config (dict): A dictionary containing all configuration settings.
    """

    # Define default regions and variable lists
    if clim_regions is None:
        clim_regions = ["global", "ocean", "land", "NHEX", "SHEX", "TROPICS"]

    if clim_vars is None:
        # This list appears to dictate which variables show up under "Mean Climate Map"
        # That is, if plots for the variable don't exist, the links will be grayed out.
        clim_vars = [
            "pr",
            "prw",
            "psl",
            "rlds",
            "rldscs",
            "rltcre",
            "rstcre",
            "rsus",
            "rsuscs",
            "rlus",
            "rlut",
            "rlutcs",
            "rsds",
            "rsdscs",
            "rsdt",
            "rsut",
            "rsutcs",
            "rtmt",
            "sfcWind",
            "tas",
            "tauu",
            "tauv",
            "ts",
            "ta-200",
            "ta-850",
            "ua-200",
            "ua-850",
            "va-200",
            "va-850",
            "zg-500",
        ]

    if emov_vars is None:
        emov_vars = ["psl", "ts"]

    if enso_vars is None:
        enso_vars = [
            "psl",
            "pr",
            "prsn",
            "ts",
            "tas",
            "tauu",
            "tauv",
            "hfls",
            "hfss",
            "rlds",
            "rsds",
            "rlus",
            "rlut",
            "rsdt",
        ]

    # Derive additional paths
    template_dir = os.path.join(pmp_dir, "viewer")
    fig_dir = os.path.join(diag_dir, run_dir)

    # Consolidated configuration dictionary
    config: Dict[str, object] = {
        "title": title,
        "subtitle": subtitle,
        "version": version,
        "case_id": case_id,
        "clim_period": clim_period,
        "clim_vars": clim_vars,
        "clim_reference": clim_reference,
        "clim_region": clim_regions,
        "emov_period": emov_period,
        "emov_vars": emov_vars,
        "emov_reference": emov_reference,
        "enso_period": enso_period,
        "enso_vars": enso_vars,
        "enso_reference": enso_reference,
        "diag_dir": fig_dir,
        "obs_dir": obs_dir,
        "pmp_dir": pmp_dir,
        "template_dir": template_dir,
        "fig_dir": fig_dir,
        "out_dir": out_dir,
    }

    return config


def setup_jinja_env(template_dir):
    """
    Set up the Jinja2 environment
    """
    return Environment(loader=FileSystemLoader(template_dir))


def create_section(title, rows):
    """
    Prepare section data as a dictionary with title and rows.
    Each row is a tuple of (label, description).
    """
    return {"title": title, "rows": rows}


def add_section(sections: List[Dict[str, object]], title, rows):
    sections.append(
        {
            "title": title,
            "rows": [
                {"label": label, "description": description}
                for label, description in rows
            ],
        }
    )


def generate_methodology_html(config):
    """
    Generate the Methodology and Definitions HTML page for the E3SM-PMP Diagnostics Package.
    """
    # Data for content sections
    general_notes = [
        (
            "Source Code",
            "See <a href='https://github.com/PCMDI/pcmdi_metrics/' target='_blank'>GitHub</a>. The Program for Climate Model Diagnosis & Intercomparison (PCMDI) Metrics Package (PMP) emphasizes metrics of large- to global-scale annual cycle and both tropical and extra-tropical modes of variability. The package expects model data to be CF-compliant. Sample usage can be found at <a href='https://github.com/PCMDI/pcmdi_metrics/blob/main/doc/jupyter/Demo/README.md' target='_blank'>PMP Website</a>.",
        ),
        (
            "Reference",
            "The observations used were collected and processed by the <a href='https://github.com/E3SM-Project/e3sm_diags/' target='_blank'>E3SM Diagnostics</a>. More info: <a href='https://docs.e3sm.org/e3sm_diags/_build/html/main/index.html#/' target='_blank'>Documentation</a>.",
        ),
        (
            "Workflow",
            "The diagnostics were generated based on the <a href='https://docs.e3sm.org/zppy/_build/html/main/index.html' target='_blank'>E3SM-zppy</a> workflow.",
        ),
    ]
    pmp_metrics = [
        (
            "Mean Climate",
            "Mean climate summary statistics. See <a href='https://pcmdi.github.io/pcmdi_metrics/metrics_mean-clim.html' target='_blank'>PMP mean climate webpage</a>.",
        ),
        (
            "ENSO Metrics",
            "Summary metric for El Niño-Southern Oscillation (ENSO). See <a href='https://pcmdi.github.io/pcmdi_metrics/metrics_enso.html' target='_blank'>PMP ENSO webpage</a>.",
        ),
        (
            "EMoV Metrics",
            "Summary metric for extra-tropical modes of variability (EMoV). See <a href='https://pcmdi.github.io/pcmdi_metrics/metrics_mov.html' target='_blank'>PMP EMoV webpage</a>.",
        ),
        (
            "Method",
            "Summary metrics are featured with <a href='https://pcmdi.llnl.gov/research/metrics/v1.3.0/mean_clim/index.html' target='_blank'>Portrait Plot</a> and <a href='https://pcmdi.llnl.gov/research/metrics/v1.3.0/mean_clim/index.html' target='_blank'>Parallel Coordinate Plot</a>.",
        ),
        (
            "References",
            "See <a href='https://pcmdi.github.io/pcmdi_metrics/metrics.html' target='_blank'>PMP Metrics webpage</a>.",
        ),
    ]
    enso_metrics = [
        (
            "Region",
            "Diagnostics are based on monthly anomalies in the Nino3.4 region (5°S–5°N, 170°–120°W).",
        ),
        (
            "ENSO Perf",
            "ENSO performance metric collection, composed of 15 metrics to evaluate models on two aspects: a) Background climatology (double ITCZ, equator too dry, too cold cold tongue, shifted trade winds); b) Basic ENSO characteristics (amplitude, skewness, seasonality, SSTA pattern, lifecycle, duration, diversity).",
        ),
        (
            "ENSO Proc",
            "ENSO processes metric collection, composed of 11 metrics to evaluate models on four aspects: a) Background climatology (too cold cold tongue, shifted trade winds); b) Basic ENSO characteristics (amplitude, skewness, seasonality, SSTA pattern); c) Feedbacks (SSH-SST, SST-heat fluxes, SST-Taux, Taux-SSH); d) Ocean-driven SST change.",
        ),
        (
            "ENSO Telec",
            "ENSO teleconnection metric collection, composed of 7 metrics to evaluate models on two aspects: a) Basic ENSO characteristics (amplitude, seasonality, SSTA pattern); b) ENSO-related anomalies (precipitation and surface temperature) outside the equatorial Pacific during events.",
        ),
        (
            "Method",
            "Developed by the ENSO Metrics Working Group of the <a href='https://www.clivar.org/news/clivar-2020-enso-metrics-package' target='_blank'>International CLIVAR Pacific Panel</a>.",
        ),
        (
            "References",
            "For detailed algorithms and examples, see the <a href='https://pcmdi.github.io/pcmdi_metrics/metrics_enso.html' target='_blank'>ENSO metrics documentation</a>.",
        ),
    ]
    emov_metrics = [
        (
            "PDO",
            "Pacific Decadal Oscillation. The 1st EOF Mode of SST computed over the North Pacific basin (polewards of 20N).",
        ),
        (
            "NPGO",
            "North Pacific Gyre Oscillation. The 2nd EOF Mode of SST computed over the Northeast Pacific.",
        ),
        (
            "AMO",
            "Atlantic Multidecadal Oscillation. See the <a href='https://climatedataguide.ucar.edu/climate-data/atlantic-multi-decadal-oscillation-amo?qt-climatedatasetmaintabs=1' target='_blank'>Climate Data Guide</a>.",
        ),
        (
            "NAM",
            "Northern Annular Mode. The 1st EOF Mode of PSL computed over the NH region within [20-90N, 0-360].",
        ),
        (
            "PNA",
            "Pacific North American Pattern. The 1st EOFs of PSL computed over [20-85N, 120E-120W].",
        ),
        (
            "NPO",
            "North Pacific Oscillation. The 2nd EOFs of PSL computed over [20-85N, 120E-120W].",
        ),
        (
            "NAO",
            "North Atlantic Oscillation. The 1st EOF Mode of PSL computed over [20-80N, 40E-90W].",
        ),
        (
            "SAM",
            "Southern Annular Mode. The 1st EOF of PSL computed over [20-90S, 0-360].",
        ),
        (
            "PSA1",
            "Pacific South American Pattern 1. The 2nd EOF of PSL computed over [20-90S, 0-360].",
        ),
        (
            "PSA2",
            "Pacific South American Patterns 2. The 3rd EOF of PSL computed over [20-90S, 0-360].",
        ),
        (
            "Method",
            "The Common Basis Function approach (CBF) was employed to analyze variability, projecting model anomalies onto observed modes in addition to the traditional EOF approach.",
        ),
        (
            "References",
            "For detailed algorithms and examples, see the <a href='https://pcmdi.github.io/pcmdi_metrics/metrics_mov.html' target='_blank'>EMoV metrics documentation</a>.",
        ),
    ]
    clim_metrics = [
        (
            "Mean Bias",
            "Defined as climatological annual/seasonal mean differences between model and observations.",
        ),
        (
            "RMSE",
            "Root-Mean-Square Error (RMSE) is defined as the L2 error norm calculated against observations/reanalysis for large-scale seasonal and mean state climatologies.",
        ),
        (
            "Centered RMSE",
            "A variation of RMSE that removes the mean bias before computing the error.",
        ),
        (
            "Region",
            "Error metrics for mean climate were calculated over global, hemispheric, tropical, extra-tropical, and other selected domains.",
        ),
        (
            "References",
            "For detailed algorithms and examples, see the <a href='https://pcmdi.llnl.gov/metrics/mean_clim/' target='_blank'>PMP mean climate</a>.",
        ),
    ]

    # Prepare sections
    sections = [
        create_section("General Notes", general_notes),
        create_section("Summary Metrics", pmp_metrics),
        create_section("El Nino-Southern Oscillation (ENSO)", enso_metrics),
        create_section("Extra-Tropical Modes of Variability (EMoV)", emov_metrics),
        create_section("Mean Climate", clim_metrics),
    ]

    # Setup Jinja2 environment and load template
    env = setup_jinja_env(config["template_dir"])
    template = env.get_template("methodology_template.html")

    # Render the HTML with the sections data
    rendered_html = template.render(sections=sections)

    # Write the rendered HTML to the output file
    Path(os.path.join(config["out_dir"], "methodology.html")).write_text(rendered_html)
    print(f"HTML file written to: {config['out_dir']}")

    return


def generate_data_html(config):
    """
    Generate diagnostic output HTML pages for the E3SM-PMP Diagnostics Package.
    """

    clim_vars = ", ".join(config["clim_vars"])
    emov_vars = ", ".join(config["emov_vars"])
    enso_vars = ", ".join(config["enso_vars"])

    clim_reference = config["clim_reference"]
    emov_reference = config["emov_reference"]
    enso_reference = config["enso_reference"]

    sections: List[Dict[str, object]] = []

    # General Notes
    general_notes: List[Tuple[str, str]] = [
        (
            "Source Code",
            f"Diagnosis & Intercomparison (PCMDI) Metrics Package (PMP) <a href='https://github.com/PCMDI/pcmdi_metrics/' target='_blank'>Version {config['version']}</a>.",
        ),
        (
            "Reference",
            "The observations used were collected and processed by the <a href='https://github.com/E3SM-Project/e3sm_diags/' target='_blank'>E3SM Diagnostics</a>. More info: <a href='https://docs.e3sm.org/e3sm_diags/_build/html/main/index.html#/' target='_blank'>Documentation</a>.",
        ),
        ("Experiment", f"{config['case_id']}"),
        ("Output Path", f"{config['diag_dir']}/metrics_data"),
        ("Reference Path", f"{config['obs_dir']}"),
        ("PMP Path", f"{config['pmp_dir']}"),
    ]
    add_section(sections, "General Notes", general_notes)

    # Metrics for different periods
    clim_metrics = [
        (
            "Metrics",
            "Mean Bias, RMSE, Centered RMSE, etc., see full list in the <a href='https://pcmdi.github.io/pcmdi_metrics/metrics_mean-clim.html' target='_blank'>document</a>.",
        ),
        (
            "Region",
            "Global, North hemisphere, Southern Hemisphere, and Tropics. See <strong>'region/regions_specs.json'</strong> under <strong>PMP Path</strong>.",
        ),
        (
            "Variables",
            f"<strong>Corresponding to CMIP convention</strong>: {clim_vars}.",
        ),
        (
            "References",
            f"All used <strong>{clim_reference}</strong>, which is defined in 'reference/reference_alias.json' under <strong> PMP Path</strong>. Source data was linked from <strong> Reference Path</strong>.",
        ),
        (
            "Model Diagnostics",
            "JSON files in <strong>mean_climate</strong> directory under <strong>Output Path</strong>",
        ),
        (
            "CMIP Diagnostics",
            "Pre-generated PMP diagnostic datasets, see <a href='https://github.com/PCMDI/pcmdi_metrics_results_archive/tree/main/metrics_results/mean_climate' target='_blank'>CMIP mean climate</a>.",
        ),
    ]
    add_section(
        sections, f"Mean Climate Metrics Data ({config['clim_period']})", clim_metrics
    )

    emov_metrics = [
        (
            "Metrics",
            "PDO, NPGO, AMO, NAM, PNA, NPO, NAO, SAM, PSA1, PSA2. For NAM, NAO, SAM, PNA, and NPO the results are based on sea-level pressure (psl), while the results for AMO, PDO and NPGO are based on sea surface temperature(ts). See full list in the <a href='https://pcmdi.github.io/pcmdi_metrics/metrics_mov.html' target='_blank'>document</a>.",
        ),
        (
            "Region",
            "Specific regions defined in <strong>'region/regions_specs.json'</strong> under <strong> PMP Path</strong>.",
        ),
        (
            "Variables",
            f"<strong>Corresponding to CMIP convention</strong>: {emov_vars}.",
        ),
        (
            "References",
            f"Corresponding to <strong>Variables</strong>: <strong>{emov_reference}</strong>, which is defined in 'reference/reference_alias.json' under <strong> PMP Path</strong>. Source data was linked from <strong> Reference Path</strong>.",
        ),
        (
            "Model Diagnostics",
            "JSON files in <strong>variability_modes</strong> directory under <strong>Output Path</strong>",
        ),
        (
            "CMIP Diagnostics",
            "Pre-generated PMP diagnostic datasets, see <a href='https://github.com/PCMDI/pcmdi_metrics_results_archive/tree/main/metrics_results/variability_modes' target='_blank'>CMIP modes varibility</a>.",
        ),
    ]

    add_section(sections, f"EMoV Metrics Data ({config['emov_period']})", emov_metrics)

    enso_metrics = [
        (
            "Metrics",
            "Three groups including ENSO_perf, ENSO_proc, ENSO_tel. see full list in the <a href='https://pcmdi.github.io/pcmdi_metrics/metrics_enso.html' target='_blank'>document</a>.",
        ),
        ("Region", "Nino3.4 region (5°S–5°N, 170°–120°W)."),
        (
            "Variables",
            f"<strong>Corresponding to CMIP convention</strong>: {enso_vars}.",
        ),
        (
            "References",
            f"<strong>Corresponding to Variables</strong>: {enso_reference}, which are defined in 'reference/reference_alias.json' under <strong> PMP Path</strong>. Source data was linked from <strong> Reference Path</strong>.",
        ),
        (
            "Model Diagnostics",
            "JSON files in <strong>enso_metric/ENSO_perf</strong>, <strong>enso_metric/ENSO_proc</strong>, and <strong>enso_metric/ENSO_tel</strong>, saved at <strong> Output Path </strong>.",
        ),
        (
            "CMIP Diagnostics",
            "PMP diagnostic datasets, see <a href='https://github.com/PCMDI/pcmdi_metrics_results_archive/tree/main/metrics_results/enso_metric' target='_blank'>CMIP ENSO metric</a>.",
        ),
    ]

    add_section(sections, f"ENSO Metrics Data ({config['enso_period']})", enso_metrics)

    # Setup Jinja2 environment and load template
    env = setup_jinja_env(config["template_dir"])
    template = env.get_template("data_template.html")

    # Generate HTML content by rendering the template
    output_html = template.render(
        title="E3SM-PMP Diagnostics Package", sections=sections
    )

    # Write the generated HTML to the specified file
    Path(os.path.join(config["out_dir"], "diag_data.html")).write_text(output_html)
    print(f"HTML file written to: {config['out_dir']}")

    return


def to_relative_path(absolute_path, base_path=None):
    """
    Converts an absolute path to a relative path.

    Parameters:
    absolute_path (str): The absolute file path to convert.
    base_path (str): The base directory to make the path relative to.
                     If None, uses the current working directory.

    Returns:
    str: Relative path.
    """
    if base_path is None:
        base_path = os.getcwd()
    return os.path.join("..", os.path.relpath(absolute_path, start=base_path))


def safe_join(base, filename):
    if base.endswith("/"):
        return f"{base}{filename}"
    else:
        return f"{base}/{filename}"


def create_image_link(
    fig_dir, diag_dir, subdirs, filename_pattern, label, fallback_filename=None
):
    sub_path = Path(*subdirs)
    search_path = Path(fig_dir) / sub_path / filename_pattern
    matches = glob.glob(str(search_path))

    if matches:
        file_name = Path(matches[0]).name
    else:
        file_name = fallback_filename or filename_pattern

    full_path = Path(fig_dir) / sub_path / file_name
    href = Path(to_relative_path(fig_dir, diag_dir)) / sub_path / file_name
    href_str = str(href).replace("\\", "/")
    if full_path.is_file():
        return (
            f'<a href="{href_str}" target="_blank" style="margin-right: 4px;">'
            f'<span style="font-size: 11pt;">{label}</span></a>'
        )
    else:
        return f'<span style="font-size: 11pt; color:gray;">{label}</span>'


class SummaryTableBuilder:
    def __init__(self, diag_dir, fig_dir):
        self.diag_dir = diag_dir
        self.fig_dir = fig_dir
        self.regions = ["global", "ocean", "land", "NHEX", "SHEX", "TROPICS"]
        self.metrics = [
            ("Mean Bias", "mae_xy", "Portrait"),
            ("Pattern Corr.", "cor_xy", "Portrait"),
            ("RMSE", "rms_xy", "Portrait", "rms_xyt", "ParCoord"),
        ]

    def build_summary_table(self):
        clim_path = safe_join(str(self.fig_dir), "ERROR_metric/mean_climate")
        metric_table = []

        for i, region in enumerate(self.regions):
            row = []
            if i == 0:
                row.append(
                    {"content": "<b>Mean Climate</b>", "rowspan": len(self.regions)}
                )
            row.append({"content": region.upper()})

            for metric in self.metrics:
                if len(metric) == 3:
                    name, prefix, mode = metric
                    filename_pattern = f"{prefix}_{region}_portrait_mean_climate.png"
                    link = create_image_link(
                        fig_dir=clim_path,
                        diag_dir=self.diag_dir,
                        subdirs=[region],
                        filename_pattern=filename_pattern,
                        label=mode,
                    )
                    row.append({"colspan": 4, "content": f"{name}<br>{link}"})

                elif len(metric) == 5:
                    name, prefix1, mode1, prefix2, mode2 = metric

                    link1 = create_image_link(
                        fig_dir=clim_path,
                        diag_dir=self.diag_dir,
                        subdirs=[region],
                        filename_pattern=f"{prefix1}_{region}_portrait_mean_climate.png",
                        label=mode1,
                    )
                    link2 = create_image_link(
                        fig_dir=clim_path,
                        diag_dir=self.diag_dir,
                        subdirs=[region],
                        filename_pattern=f"{prefix2}_{region}_parcoord_mean_climate.png",
                        label=mode2,
                    )

                    row.append({"colspan": 4, "content": f"{name}<br>{link1} {link2}"})

            metric_table.append(row)

        return metric_table

    def build_enso_row(self):
        row: List[Dict[str, object]] = []
        row.append({"content": "<b>ENSO</b>"})
        row.append({"content": "TROPICS"})
        enso_path = safe_join(str(self.fig_dir), "ERROR_metric/enso_metric")

        link = create_image_link(
            fig_dir=enso_path,
            diag_dir=self.diag_dir,
            subdirs=[],
            filename_pattern="enso_metric_skill_portrait.png",
            label="Portrait",
        )

        row.append({"colspan": 12, "content": f"Performance Skill<br>{link}"})
        return row

    def build_emov_row(self):
        row: List[Dict[str, object]] = []
        row.append({"content": "<b>EMoVs</b>"})
        row.append({"content": "Extra-TROPICS"})
        emov_path = safe_join(str(self.fig_dir), "ERROR_metric/variability_modes")

        modes_metrics = [
            (
                "PC_Std_Dev",
                "stdv_pc_ratio_to_obs_variability_modes_portrait_mon.png",
                "Portrait",
                "stdv_pc_ratio_to_obs_variability_modes_parcoord_mon.png",
                "ParCoord",
            ),
            (
                "Centered RMSE",
                "rmsc_variability_modes_portrait_mon.png",
                "Portrait",
                "rmsc_variability_modes_parcoord_mon.png",
                "ParCoord",
            ),
            (
                "RMSE",
                "rms_variability_modes_portrait_mon.png",
                "Portrait",
                "rms_variability_modes_parcoord_mon.png",
                "ParCoord",
            ),
        ]

        for name, f1, mode1, f2, mode2 in modes_metrics:
            link1 = create_image_link(emov_path, self.diag_dir, [], f1, mode1)
            link2 = create_image_link(emov_path, self.diag_dir, [], f2, mode2)
            row.append({"colspan": 4, "content": f"{name}<br>{link1} {link2}"})
        return row


def generate_summary_table(diag_dir, fig_dir):
    """
    Wrapper function to generate climatology table using SummaryTableBuilder.
    Returns the table as summary_metric_table.
    """
    builder = SummaryTableBuilder(diag_dir, fig_dir)
    summary_metric_table = builder.build_summary_table()
    summary_metric_table.append(builder.build_enso_row())
    summary_metric_table.append(builder.build_emov_row())
    return summary_metric_table


class CMVARGroupBuilder:
    def __init__(self):
        self.regions = {
            "Global": {"All": "01", "El/La": "02"},
            "Africa": {"All": "03", "El/La": "08"},
            "CONUS": {"All": "04", "El/La": "09"},
            "SA": {"All": "05", "El/La": "10"},
            "SCS": {"All": "06", "El/La": "11"},
            "AUS": {"All": "07", "El/La": "12"},
        }

    def create_metric_group(self, *metrics):
        return {
            name: f"divedown{str(i + 1).zfill(2)}" for i, name in enumerate(metrics)
        }

    def create_region_maps(self):
        region_metrics = {}
        for reg, types in self.regions.items():
            for typ, val in types.items():
                region_metrics[f"{reg}({typ})"] = f"divedown{str(val).zfill(2)}"
        return region_metrics

    def construct(self):
        region_metrics = self.create_region_maps()

        return {
            "Perf": {
                "ENSO_perf": {
                    "BiasPrLat": self.create_metric_group("Skill(Lat)", "Pattern(SRF)"),
                    "BiasPrLon": self.create_metric_group("Skill(Lon)", "Pattern(SRF)"),
                    "BiasSstLon": self.create_metric_group(
                        "Skill(Lon)", "Pattern(SRF)"
                    ),
                    "BiasTauxLon": self.create_metric_group(
                        "Skill(Lon)", "Pattern(SRF)"
                    ),
                    "EnsoAmpl": self.create_metric_group(
                        "Skill(All)", "Skill(Lon)", "Pattern(SRF)"
                    ),
                    "EnsoDuration": self.create_metric_group(
                        "Skill(All)", "Skill(AC)", "Skill(El/La)"
                    ),
                    "EnsoSstDiversity": self.create_metric_group(
                        "Skill(All)", "Skill(El/La)"
                    ),
                    "EnsoSstSkew": self.create_metric_group(
                        "Skill(All)", "Skill(Lon)", "Pattern(SRF)"
                    ),
                    "EnsoSstLonRmse": self.create_metric_group(
                        "Skill(Lon)", "Pattern(SRF)", "Skill(El/La)", "Pattern(El/La)"
                    ),
                    "EnsoSstTsRmse": self.create_metric_group(
                        "Skill(All)", "Pattern(Hov)", "Sill(El/La)", "Pattern(El/La)"
                    ),
                    "EnsoSeasonality": self.create_metric_group(
                        "Skill(All)",
                        "Skill(AC)",
                        "Pattern(Hov)",
                        "Skill(El/La)",
                        "Pattern(El/La)",
                    ),
                    "SeasonalPrLat": self.create_metric_group(
                        "Skill(All)", "Pattern(SRF)", "Pattern(Hov)"
                    ),
                    "SeasonalPrLon": self.create_metric_group(
                        "Skill(All)", "Pattern(SRF)", "Pattern(Hov)"
                    ),
                    "SeasonalSstLon": self.create_metric_group(
                        "Skill(All)", "Pattern(SRF)", "Pattern(Hov)"
                    ),
                    "SeasonalTauxLon": self.create_metric_group(
                        "Skill(All)", "Pattern(SRF)", "Pattern(Hov)"
                    ),
                }
            },
            "Proc": {
                "ENSO_proc": {
                    "BiasSstLon": self.create_metric_group(
                        "Skill(Lon)", "Pattern(SRF)"
                    ),
                    "BiasTauxLon": self.create_metric_group(
                        "Skill(Lon)", "Pattern(SRF)"
                    ),
                    "EnsoAmpl": self.create_metric_group(
                        "Skill(All)", "Skill(Lon)", "Pattern(SRF)"
                    ),
                    "EnsoFbSstTaux": self.create_metric_group(
                        "Skill(CPL)", "Skill(NLIN)", "SKill(FDBK)", "Pattern(Hov)"
                    ),
                    "EnsoSeasonality": self.create_metric_group(
                        "Skill(All)",
                        "Skill(AC)",
                        "Pattern(Hov)",
                        "Skill(Lon)",
                        "Pattern(SRF)",
                    ),
                    "EnsoSstLonRmse": self.create_metric_group(
                        "Skill(Lon)", "Pattern(SRF)", "Skill(El/La)", "Pattern(El/La)"
                    ),
                    "EnsoSstSkew": self.create_metric_group(
                        "Skill(All)", "Skill(Lon)", "Pattern(SRF)"
                    ),
                }
            },
            "Telec": {
                "ENSO_tel": {
                    "EnsoAmpl": self.create_metric_group(
                        "Skill(All)", "Skill(Lon)", "Pattern(SRF)"
                    ),
                    "EnsoSeasonality": self.create_metric_group(
                        "Skill(All)",
                        "Skill(AC)",
                        "Pattern(Hov)",
                        "Skill(El/La)",
                        "Pattern(El/La)",
                    ),
                    "EnsoSstLonRmse": self.create_metric_group(
                        "Skill(Lon)", "Pattern(SRF)", "Skill(El/La)", "Pattern(El/La)"
                    ),
                    "EnsoPrMapDjf": region_metrics,
                    "EnsoPrMapJja": region_metrics,
                    "EnsoSstMapDjf": region_metrics,
                    "EnsoSstMapJja": region_metrics,
                }
            },
        }

    @staticmethod
    def reshape_1d_to_2d(lst, num_cols, fill_value=""):
        num_rows = -(-len(lst) // num_cols)
        padded = lst + [fill_value] * (num_rows * num_cols - len(lst))
        return [padded[i * num_cols : (i + 1) * num_cols] for i in range(num_rows)], (
            num_rows,
            num_cols,
        )

    @staticmethod
    def build_cmvar_cell(fig_dir, diag_dir, group, variable, keys_dict):
        content = f"{variable}<br>"
        keys = list(keys_dict.items())
        for i, (label, code) in enumerate(keys):
            link = create_image_link(
                fig_dir=fig_dir,
                diag_dir=diag_dir,
                subdirs=[group],
                filename_pattern=f"{group}_{variable}*_{code}.png",
                label=label,
                fallback_filename=f"{group}_{variable}_{code}.png",
            )
            if len(keys) > 2 and (i - 1) % 2 == 0:
                content += f" {link}<br>"
            else:
                content += f" {link}"
        return content

    def generate_mcpl_row(self, mode, diag_dir, fig_dir):
        groups = {
            "MOV_eoftest": {
                "EOF Spectr": {
                    "EG Spec(Yearly)": {"EG_Spec": "yearly"},
                    "EG Spec(Monthly)": {"EG_Spec": "monthly"},
                }
            },
            "MOV_compose": {
                "EOF Compos": {
                    "CBF(Yearly)": {"cbf": "yearly"},
                    "CBF(Monthly)": {"cbf": "monthly"},
                    "EOF(Yearly)": {"cbf": "yearly"},
                    "EOF(Monthly)": {"cbf": "monthly"},
                }
            },
            "MOV_pattern": {
                "EOF Pattern": {
                    "EOF1(Yearly)": {"eof1": "yearly"},
                    "EOF1(Monthly)": {"eof1": "monthly"},
                    "EOF2(Yearly)": {"eof2": "yearly"},
                    "EOF2(Monthly)": {"eof2": "monthly"},
                    "EOF3(Yearly)": {"eof3": "yearly"},
                    "EOF3(Monthly)": {"eof3": "monthly"},
                }
            },
            "MOV_telecon": {
                "EOF Telec": {
                    "CBF(Yearly)": {"cbf": "yearly"},
                    "CBF(Monthly)": {"cbf": "monthly"},
                    "EOF1(Yearly)": {"eof1": "yearly"},
                    "EOF1(Monthly)": {"eof1": "monthly"},
                    "EOF2(Yearly)": {"eof2": "yearly"},
                    "EOF2(Monthly)": {"eof2": "monthly"},
                    "EOF3(Yearly)": {"eof3": "yearly"},
                    "EOF3(Monthly)": {"eof3": "monthly"},
                }
            },
        }

        row = [{"content": f"<b>{mode} (SST)</b>", "rowspan": 1}]
        for src_grp, cat_dict in groups.items():
            category_label = next(iter(cat_dict))
            display_items = cat_dict[category_label]
            content = f"{category_label}<br>"

            for idx, (label, fileinfo) in enumerate(display_items.items()):
                key_type, season = next(iter(fileinfo.items()))
                pattern = f"{src_grp}_{mode}_*{key_type}.png"
                fallback = f"{src_grp}_{mode}_{key_type}.png"
                link = create_image_link(
                    fig_dir=fig_dir,
                    diag_dir=diag_dir,
                    subdirs=[src_grp, season],
                    filename_pattern=pattern,
                    fallback_filename=fallback,
                    label=label,
                )

                if len(display_items) > 2 and (idx % 2 == 1):
                    content += f"{link}<br>"
                else:
                    content += f"{link} "

            row.append({"colspan": 4, "content": content.strip()})
        return row


def generate_cmvar_table(diag_dir, fig_dir):
    """
    Generate the Coupled Modes Variability (CMVAR) diagnostic table using CMVARGroupBuilder.
    """
    builder = CMVARGroupBuilder()
    cmvar_groups = builder.construct()
    cmvar_table = []
    enso_path = safe_join(str(fig_dir), "ENSO_metric")
    emov_path = safe_join(str(fig_dir), "MOV_metric")
    for mode_label, group_data in cmvar_groups.items():
        # Each group_data should have only one key (like 'ENSO_perf', etc.)
        group_name = next(iter(group_data))
        var_dict = group_data[group_name]

        variables, (nrows, ncols) = builder.reshape_1d_to_2d(list(var_dict.keys()), 4)

        for i, row_vars in enumerate(variables):
            row = []
            if i == 0:
                row.append({"content": f"<b>ENSO {mode_label}</b>", "rowspan": nrows})

            for var in row_vars:
                if var == "":
                    row.append({"colspan": 4, "content": "--"})
                else:
                    content = builder.build_cmvar_cell(
                        fig_dir=enso_path,
                        diag_dir=diag_dir,
                        group=group_name,
                        variable=var,
                        keys_dict=var_dict[var],
                    )
                    row.append({"colspan": 4, "content": content})

            cmvar_table.append(row)

    # Add PDO, NPGO, AMO rows
    for mode in ["PDO", "NPGO", "AMO"]:
        mcpl_row = builder.generate_mcpl_row(mode, diag_dir, emov_path)
        cmvar_table.append(mcpl_row)

    return cmvar_table


class EMOVGroupBuilder:
    def __init__(self, diag_dir, fig_dir):
        self.diag_dir = diag_dir
        self.fig_dir = fig_dir
        self.modes = {
            "NAM": "EOF1",
            "PNA": "EOF1",
            "NPO": "EOF2",
            "NAO": "EOF1",
            "SAM": "EOF1",
            "PSA1": "EOF2",
            "PSA2": "EOF3",
        }
        self.seasons = ["DJF", "MAM", "JJA", "SON", "yearly", "monthly"]
        self.rowspecs = [
            ("Composite (CBF)", "MOV_compose_{}_{}_cbf.png"),
            ("Composite (EOF)", "MOV_compose_{}_{}_{}.png"),
            ("North Test", "MOV_eoftest_{}_{}_EG_Spec.png"),
            ("EOF1 Pattern", "MOV_pattern_{}_{}_eof1.png"),
            ("EOF1 Telecon.", "MOV_telecon_{}_{}_eof1.png"),
            ("EOF2 Pattern", "MOV_pattern_{}_{}_eof2.png"),
            ("EOF2 Telecon.", "MOV_telecon_{}_{}_eof2.png"),
            ("EOF3 Pattern", "MOV_pattern_{}_{}_eof3.png"),
            ("EOF3 Telecon.", "MOV_telecon_{}_{}_eof3.png"),
        ]

    def build(self):
        emov_path = safe_join(str(self.fig_dir), "MOV_metric")
        table = []

        for mode, eof in self.modes.items():
            for i, (label, pattern) in enumerate(self.rowspecs):
                row = []
                if i == 0:
                    row.append(
                        {
                            "content": f"<b>{mode} (PSL)</b>",
                            "rowspan": len(self.rowspecs),
                        }
                    )

                row.append(
                    {
                        "colspan": 4,
                        "content": f'<span style="font-size: 11pt;">{label}</span>',
                    }
                )

                for season in self.seasons:
                    try:
                        filename = pattern.format(mode, season, eof.lower())
                    except IndexError:
                        filename = pattern.format(mode, season)

                    subdir_parts = ["_".join(filename.split("_")[:2]), season]

                    link = create_image_link(
                        fig_dir=emov_path,
                        diag_dir=self.diag_dir,
                        subdirs=subdir_parts,
                        filename_pattern=filename,
                        fallback_filename=filename,
                        label=season.upper(),
                    )

                    row.append({"colspan": 4, "content": link})

                table.append(row)

        return table


def generate_emovs_table(diag_dir, fig_dir):
    builder = EMOVGroupBuilder(diag_dir, fig_dir)
    emovs_table = builder.build()
    return emovs_table


class MeanClimateTableBuilder:
    def __init__(self, diag_dir, fig_dir, variables=None, regions=None):
        self.diag_dir = Path(diag_dir)
        self.fig_dir = Path(fig_dir)
        self.variables = variables or ["pr", "psl", "tas", "ts", "rlds", "rlut"]
        self.regions = regions or ["global", "land", "ocean", "TROPICS", "NHEX", "SHEX"]
        self.seasons = ["DJF", "MAM", "JJA", "SON", "AC"]  # AC = Annual Cycle

    def build_table(self):
        """
        Constructs a list of table row dictionaries for use in an HTML diagnostic viewer.
        """
        clim_path = safe_join(str(self.fig_dir), "CLIM_patttern")
        table = []

        for var in self.variables:
            for i, region in enumerate(self.regions):
                row = []
                if i == 0:
                    row.append(
                        {"content": f"<b>{var}</b>", "rowspan": len(self.regions)}
                    )

                row.append(
                    {
                        "colspan": 4,
                        "content": f'<span style="font-size: 11pt;">{region}</span>',
                    }
                )

                for season in self.seasons:
                    label = "Yearly" if season == "AC" else season
                    filename = f"{var.strip()}_{region}_{season}.png"
                    subdir = safe_join(str(region), str(season))
                    link = create_image_link(
                        fig_dir=clim_path,
                        diag_dir=self.diag_dir,
                        subdirs=[subdir],
                        filename_pattern=filename,
                        label=label,
                        fallback_filename=filename,
                    )
                    row.append({"colspan": 4, "content": link})

                table.append(row)

        return table


def generate_clim_table(diag_dir, fig_dir, variables=None, regions=None):
    """
    Wrapper to generate the climatology diagnostics table.

    Args:
        diag_dir (str): Path to diagnostics directory.
        fig_dir (str): Path to figures directory.
        variable_str (str): Comma-separated variables string.
        region_str (str): Comma-separated regions string.

    Returns:
        list: HTML-ready row data for climatology table.
    """
    builder = MeanClimateTableBuilder(diag_dir, fig_dir, variables, regions)
    clim_table = builder.build_table()
    return clim_table


def generate_viewer_html(config):
    """
    Generate overview HTML page for the E3SM-PMP Diagnostics
    """
    env = setup_jinja_env(config["template_dir"])
    template = env.get_template("index_template.html")

    # Add 'mean climate' row
    metric_table = generate_summary_table(config["diag_dir"], config["fig_dir"])

    # Add 'Coupled Modes Variability' table
    cmvars_table = generate_cmvar_table(config["diag_dir"], config["fig_dir"])

    # Add 'Extratropical Modes Variability' tabel
    emovs_table = generate_emovs_table(config["diag_dir"], config["fig_dir"])

    # Add 'Mean Climate Map' tabel
    clim_table = generate_clim_table(
        config["diag_dir"],
        config["fig_dir"],
        config["clim_vars"],
        config["clim_region"],
    )

    # Render final HTML
    output_html = template.render(
        title=config["title"],
        subtitle=config["subtitle"],
        version=config["version"],
        clim_period=config["clim_period"],
        emov_period=config["emov_period"],
        enso_period=config["enso_period"],
        created=datetime.now().strftime("%Y-%m-%d"),
        metric_table=metric_table,
        cmvars_table=cmvars_table,
        emovs_table=emovs_table,
        clim_table=clim_table,
    )

    # Write the generated HTML to the specified file
    Path(os.path.join(config["out_dir"], "index.html")).write_text(output_html)
    print(f"HTML file written to: {config['out_dir']}")
    return
