import glob
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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
    clim_viewer: bool = True,
    clim_period: str = "1985-2014",
    clim_vars: Optional[List[str]] = None,
    clim_regions: Optional[List[str]] = None,
    clim_reference: str = "obs",
    mova_viewer: bool = True,
    mova_modes: Optional[List[str]] = None,
    mova_seasons: Optional[List[str]] = None,
    mova_period: str = "1985-2014",
    mova_vars: Optional[List[str]] = None,
    mova_reference: str = "obs",
    movc_viewer: bool = True,
    movc_modes: Optional[List[str]] = None,
    movc_seasons: Optional[List[str]] = None,
    movc_period: str = "1985-2014",
    movc_vars: Optional[List[str]] = None,
    movc_reference: str = "obs",
    enso_viewer: bool = True,
    enso_period: str = "1985-2014",
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

    if mova_vars is None:
        mova_vars = ["psl"]

    if mova_modes is None:
        mova_modes = ["NAM", "PNA", "NPO", "NAO", "SAM", "PSA1", "PSA2"]

    if mova_seasons is None:
        mova_seasons = ["DJF", "MAM", "JJA", "SON", "yearly", "monthly"]

    if movc_vars is None:
        movc_vars = ["ts"]

    if movc_modes is None:
        movc_modes = ["PDO", "NPGO", "AMO"]

    if movc_seasons is None:
        movc_seasons = ["yearly", "monthly"]

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
        "clim_viewer": clim_viewer,
        "clim_period": clim_period,
        "clim_vars": clim_vars,
        "clim_reference": clim_reference,
        "clim_regions": clim_regions,
        "mova_viewer": mova_viewer,
        "mova_period": mova_period,
        "mova_vars": mova_vars,
        "mova_modes": mova_modes,
        "mova_seasons": mova_seasons,
        "mova_reference": mova_reference,
        "movc_viewer": movc_viewer,
        "movc_period": movc_period,
        "movc_vars": movc_vars,
        "movc_modes": movc_modes,
        "movc_seasons": movc_seasons,
        "movc_reference": movc_reference,
        "enso_viewer": enso_viewer,
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
    Expects:
      - config["template_dir"], config["out_dir"]
      - booleans: "clim_viewer", "mova_viewer", "movc_viewer", "enso_viewer"
    """

    # --- Safe getters with defaults
    def cfg(key, default=None):
        return config.get(key, default)

    # Data for content sections
    general_notes = [
        (
            "Source Code",
            (
                "See <a href='https://github.com/PCMDI/pcmdi_metrics/' target='_blank' rel='noopener'>GitHub</a>. "
                "The Program for Climate Model Diagnosis &amp; Intercomparison (PCMDI) Metrics Package (PMP) emphasizes metrics "
                "of large- to global-scale annual cycle and both tropical and extra-tropical modes of variability. "
                "The package expects model data to be CF-compliant. Sample usage: "
                "<a href='https://github.com/PCMDI/pcmdi_metrics/blob/main/doc/jupyter/Demo/README.md' target='_blank' rel='noopener'>PMP Website</a>."
            ),
        ),
        (
            "Reference",
            (
                "The observations used were collected and processed by the "
                "<a href='https://github.com/E3SM-Project/e3sm_diags/' target='_blank' rel='noopener'>E3SM Diagnostics</a>. "
                "More info: <a href='https://docs.e3sm.org/e3sm_diags/_build/html/main/index.html#/' target='_blank' rel='noopener'>Documentation</a>."
            ),
        ),
        (
            "Workflow",
            (
                "The diagnostics were generated using the "
                "<a href='https://docs.e3sm.org/zppy/_build/html/main/index.html' target='_blank' rel='noopener'>E3SM-zppy</a> workflow."
            ),
        ),
    ]

    # -------------------- PMP summary metrics --------------------
    pmp_metrics = []
    if cfg("clim_viewer", False):
        pmp_metrics.append(
            (
                "Mean Climate",
                "Mean climate summary statistics. See <a href='https://pcmdi.github.io/pcmdi_metrics/metrics_mean-clim.html' target='_blank' rel='noopener'>PMP mean climate webpage</a>.",
            )
        )
    if cfg("mova_viewer", False) or cfg("movc_viewer", False):
        pmp_metrics.append(
            (
                "EMoV Metrics",
                "Summary metric for extra-tropical modes of variability (EMoV). See <a href='https://pcmdi.github.io/pcmdi_metrics/metrics_mov.html' target='_blank' rel='noopener'>PMP EMoV webpage</a>.",
            )
        )
    if cfg("enso_viewer", False):
        pmp_metrics.append(
            (
                "ENSO Metrics",
                "Summary metric for El Niño–Southern Oscillation (ENSO). See <a href='https://pcmdi.github.io/pcmdi_metrics/metrics_enso.html' target='_blank' rel='noopener'>PMP ENSO webpage</a>.",
            )
        )
    pmp_metrics.extend(
        [
            (
                "Method",
                "Summary metrics are featured with <a href='https://pcmdi.llnl.gov/research/metrics/v1.3.0/mean_clim/index.html' target='_blank' rel='noopener'>Portrait Plot</a> and "
                "<a href='https://pcmdi.llnl.gov/research/metrics/v1.3.0/mean_clim/index.html' target='_blank' rel='noopener'>Parallel Coordinate Plot</a>.",
            ),
            (
                "References",
                "See <a href='https://pcmdi.github.io/pcmdi_metrics/metrics.html' target='_blank' rel='noopener'>PMP Metrics webpage</a>.",
            ),
        ]
    )

    # -------------------- ENSO metrics --------------------
    enso_metrics = []
    if cfg("enso_viewer", False):
        enso_metrics.extend(
            [
                (
                    "Region",
                    "Diagnostics are based on monthly anomalies in the Niño&nbsp;3.4 region (5&deg;S–5&deg;N, 170&deg;–120&deg;W).",
                ),
                (
                    "ENSO Perf",
                    "Performance collection (15 metrics) covering: (a) background climatology (double ITCZ, equator too dry, cold tongue bias, shifted trade winds); "
                    "(b) basic ENSO characteristics (amplitude, skewness, seasonality, SSTA pattern, lifecycle, duration, diversity).",
                ),
                (
                    "ENSO Proc",
                    "Processes collection (11 metrics) covering: (a) background climatology (cold tongue bias, shifted trade winds); "
                    "(b) basic characteristics (amplitude, skewness, seasonality, SSTA pattern); "
                    "(c) feedbacks (SSH–SST, SST–heat fluxes, SST–&tau;<sub>x</sub>, &tau;<sub>x</sub>–SSH); "
                    "(d) ocean-driven SST change.",
                ),
                (
                    "ENSO Telec",
                    "Teleconnections collection (7 metrics) covering: (a) basic characteristics (amplitude, seasonality, SSTA pattern); "
                    "(b) ENSO-related anomalies (precipitation and surface temperature) outside the equatorial Pacific during events.",
                ),
                (
                    "Method",
                    "Developed by the ENSO Metrics Working Group of the <a href='https://www.clivar.org/news/clivar-2020-enso-metrics-package' target='_blank' rel='noopener'>International CLIVAR Pacific Panel</a>.",
                ),
                (
                    "References",
                    "See the <a href='https://pcmdi.github.io/pcmdi_metrics/metrics_enso.html' target='_blank' rel='noopener'>ENSO metrics documentation</a> for algorithms and examples.",
                ),
            ]
        )

    # -------------------- EMoV (Atmos/Coupled modes) --------------------
    emov_metrics = []
    if cfg("mova_viewer", False):
        emov_metrics.extend(
            [
                (
                    "NAM",
                    "Northern Annular Mode. 1st EOF mode of PSL over 20–90&deg;N, 0–360&deg;.",
                ),
                (
                    "PNA",
                    "Pacific–North American pattern. 1st EOF mode of PSL over 20–85&deg;N, 120&deg;E–120&deg;W.",
                ),
                (
                    "NPO",
                    "North Pacific Oscillation. 2nd EOF mode of PSL over 20–85&deg;N, 120&deg;E–120&deg;W.",
                ),
                (
                    "NAO",
                    "North Atlantic Oscillation. 1st EOF mode of PSL over 20–80&deg;N, 40&deg;E–90&deg;W.",
                ),
                (
                    "SAM",
                    "Southern Annular Mode. 1st EOF mode of PSL over 20–90&deg;S, 0–360&deg;.",
                ),
                (
                    "PSA1",
                    "Pacific–South American pattern 1. 2nd EOF mode of PSL over 20–90&deg;S, 0–360&deg;.",
                ),
                (
                    "PSA2",
                    "Pacific–South American pattern 2. 3rd EOF mode of PSL over 20–90&deg;S, 0–360&deg;.",
                ),
            ]
        )
    if cfg("movc_viewer", False):
        emov_metrics.extend(
            [
                (
                    "PDO",
                    "Pacific Decadal Oscillation. 1st EOF mode of SST over the North Pacific (poleward of 20&deg;N).",
                ),
                (
                    "NPGO",
                    "North Pacific Gyre Oscillation. 2nd EOF mode of SST over the Northeast Pacific.",
                ),
                (
                    "AMO",
                    "Atlantic Multidecadal Oscillation, typically derived from detrended North Atlantic SST averages (method varies by index).",
                ),
            ]
        )
    emov_metrics.extend(
        [
            (
                "Method",
                "The Common Basis Function (CBF) approach is employed in addition to traditional EOFs, projecting model anomalies onto observed modes.",
            ),
            (
                "References",
                "See the <a href='https://pcmdi.github.io/pcmdi_metrics/metrics_mov.html' target='_blank' rel='noopener'>EMoV metrics documentation</a>.",
            ),
        ]
    )

    # -------------------- Mean climate metrics --------------------
    clim_metrics = [
        (
            "Mean Bias",
            "Climatological annual/seasonal mean differences between model and observations.",
        ),
        (
            "RMSE",
            "Root-Mean-Square Error (L2 norm) against observations/reanalyses for seasonal and mean-state climatologies.",
        ),
        ("Centered RMSE", "RMSE of anomalies after removing the mean bias."),
        (
            "Region",
            "Metrics computed over global, hemispheric, tropical, extra-tropical, and other selected domains.",
        ),
        (
            "References",
            "See <a href='https://pcmdi.llnl.gov/metrics/mean_clim/' target='_blank' rel='noopener'>PMP mean climate</a>.",
        ),
    ]

    # Build sections (skip empty bodies gracefully if your create_section handles it)
    sections = [
        create_section("General Notes", general_notes),
        create_section("Summary Metrics", pmp_metrics),
    ]

    # ENSO
    if cfg("enso_viewer", False):
        sections.append(
            create_section("El Niño–Southern Oscillation (ENSO)", enso_metrics)
        )

    # MOVs (MOVA/MOVC)
    movs_on = cfg("movs_viewer", None)
    if movs_on is None:
        movs_on = cfg("mova_viewer", False) or cfg("movc_viewer", False)

    if movs_on:
        sections.append(
            create_section("Extra-Tropical Modes of Variability (EMoV)", emov_metrics)
        )

    # Mean climate
    if cfg("clim_viewer", False):
        sections.append(create_section("Mean Climate", clim_metrics))

    # Setup Jinja2 environment and load template
    env = setup_jinja_env(cfg("template_dir"))
    template = env.get_template("methodology_template.html")

    # Render and write
    rendered_html = template.render(sections=sections)
    out_path = os.path.join(cfg("out_dir"), "methodology.html")
    Path(out_path).write_text(rendered_html)
    print(f"HTML file written to: {cfg('out_dir')}")

    return out_path


def generate_data_html(config):
    """
    Generate diagnostic output HTML pages for the E3SM-PMP Diagnostics Package.
    """

    # Safe getters with defaults
    def cfg(key, default=None):
        return config.get(key, default)

    # Join lists safely
    def join_list(key):
        vals = cfg(key, []) or []
        return ", ".join(vals)

    clim_vars = join_list("clim_vars")
    mova_vars = join_list("mova_vars")
    movc_vars = join_list("movc_vars")
    enso_vars = join_list("enso_vars")

    clim_reference = cfg("clim_reference", "")
    mova_reference = cfg("mova_reference", "")
    movc_reference = cfg("movc_reference", "")
    enso_reference = cfg("enso_reference", "")

    sections: List[Dict[str, object]] = []

    # ---------------- General Notes ----------------
    general_notes: List[Tuple[str, str]] = [
        (
            "Source Code",
            (
                "Diagnosis &amp; Intercomparison (PCMDI) Metrics Package (PMP) "
                f"<a href='https://github.com/PCMDI/pcmdi_metrics/' target='_blank' rel='noopener'>Version {cfg('version','')}</a>."
            ),
        ),
        (
            "Reference",
            "The observations used were collected and processed by the "
            "<a href='https://github.com/E3SM-Project/e3sm_diags/' target='_blank' rel='noopener'>E3SM Diagnostics</a>. "
            "More info: <a href='https://docs.e3sm.org/e3sm_diags/_build/html/main/index.html#/' target='_blank' rel='noopener'>Documentation</a>.",
        ),
        ("Experiment", f"{cfg('case_id','')}"),
        ("Output Path", f"{cfg('diag_dir','')}/metrics_data"),
        ("Reference Path", f"{cfg('obs_dir','')}"),
        ("PMP Path", f"{cfg('pmp_dir','')}"),
    ]
    add_section(sections, "General Notes", general_notes)

    # ---------------- Mean Climate ----------------
    if cfg("clim_viewer", False):
        clim_metrics = [
            (
                "Metrics",
                "Mean Bias, RMSE, Centered RMSE, etc.; see full list in the "
                "<a href='https://pcmdi.github.io/pcmdi_metrics/metrics_mean-clim.html' target='_blank' rel='noopener'>document</a>.",
            ),
            (
                "Region",
                "Global, Northern Hemisphere, Southern Hemisphere, and Tropics. "
                "See <strong>region/regions_specs.json</strong> under <strong>PMP Path</strong>.",
            ),
            (
                "Variables",
                f"<strong>CMIP conventions</strong>: {clim_vars}.",
            ),
            (
                "References",
                f"All use <strong>{clim_reference}</strong>, defined in <strong>reference/reference_alias.json</strong> under <strong>PMP Path</strong>. "
                "Source data linked from <strong>Reference Path</strong>.",
            ),
            (
                "Model Diagnostics",
                "JSON files in <strong>mean_climate</strong> under <strong>Output Path</strong>.",
            ),
            (
                "CMIP Diagnostics",
                "Pre-generated datasets: "
                "<a href='https://github.com/PCMDI/pcmdi_metrics_results_archive/tree/main/metrics_results/mean_climate' target='_blank' rel='noopener'>CMIP mean climate</a>.",
            ),
        ]
        add_section(
            sections,
            f"Mean Climate Metrics Data ({cfg('clim_period','')})",
            clim_metrics,
        )

    # ---------------- EMoV (Atmos/Coupled modes) ----------------
    metric_string = variable_string = reference_string = ""
    mova_view = cfg("mova_viewer", False)
    movc_view = cfg("movc_viewer", False)

    if mova_view and movc_view:
        metric_string = (
            "The coupled modes considered include the Pacific Decadal Oscillation (PDO), "
            "North Pacific Gyre Oscillation (NPGO), and Atlantic Multidecadal Oscillation (AMO), "
            "while the atmospheric modes include the Northern Annular Mode (NAM), Pacific–North American pattern (PNA), "
            "North Pacific Oscillation (NPO), North Atlantic Oscillation (NAO), Southern Annular Mode (SAM), "
            "and the Pacific–South American patterns (PSA1 and PSA2). "
            "Metrics were derived from empirical orthogonal function (EOF) analysis, using sea-level pressure (PSL) for atmospheric modes "
            "and sea surface temperature (TS) for coupled modes."
        )
        variable_string = f"{mova_vars} (ATM) and {movc_vars} (CPL)"
        reference_string = f"{mova_reference} (ATM) and {movc_reference} (CPL)"
    elif mova_view:
        metric_string = (
            "Atmospheric modes include NAM, PNA, NPO, NAO, SAM, and PSA1/PSA2. "
            "Metrics were derived from EOF analysis based on sea-level pressure (PSL)."
        )
        variable_string = f"{mova_vars} (ATM)"
        reference_string = f"{mova_reference} (ATM)"
    elif movc_view:
        metric_string = (
            "The coupled modes considered include PDO, NPGO, and AMO. "
            "Metrics were derived from EOF analysis based on sea surface temperature (TS)."
        )
        variable_string = f"{movc_vars} (CPL)"
        reference_string = f"{movc_reference} (CPL)"

    if mova_view or movc_view:
        emov_metrics = [
            (
                "Metrics",
                f"{metric_string} See full descriptions in the "
                "<a href='https://pcmdi.github.io/pcmdi_metrics/metrics_mov.html' target='_blank' rel='noopener'>document</a>.",
            ),
            (
                "Region",
                "Regions defined in <strong>region/regions_specs.json</strong> under <strong>PMP Path</strong>.",
            ),
            (
                "Variables",
                f"<strong>CMIP conventions</strong>: <strong>{variable_string}</strong>.",
            ),
            (
                "References",
                f"Following <strong>Variables</strong>: <strong>{reference_string}</strong>, defined in "
                "<strong>reference/reference_alias.json</strong> under <strong>PMP Path</strong>. "
                "Source data linked from <strong>Reference Path</strong>.",
            ),
            (
                "Model Diagnostics",
                "JSON files in <strong>variability_modes</strong> under <strong>Output Path</strong>.",
            ),
            (
                "CMIP Diagnostics",
                "Pre-generated datasets: "
                "<a href='https://github.com/PCMDI/pcmdi_metrics_results_archive/tree/main/metrics_results/variability_modes' target='_blank' rel='noopener'>CMIP modes variability</a>.",
            ),
        ]
        add_section(
            sections,
            f"EMoV Metrics Data (coupled modes: {cfg('movc_period','')}; atmospheric modes: {cfg('mova_period','')})",
            emov_metrics,
        )

    # ---------------- ENSO ----------------
    if cfg("enso_viewer", False):
        enso_metrics = [
            (
                "Metrics",
                "Three groups (ENSO_perf, ENSO_proc, ENSO_tel); see full list in the "
                "<a href='https://pcmdi.github.io/pcmdi_metrics/metrics_enso.html' target='_blank' rel='noopener'>document</a>.",
            ),
            ("Region", "Niño&nbsp;3.4 region (5&deg;S–5&deg;N, 170&deg;–120&deg;W)."),
            ("Variables", f"<strong>CMIP conventions</strong>: {enso_vars}."),
            (
                "References",
                f"<strong>Following Variables</strong>: {enso_reference}, defined in "
                "<strong>reference/reference_alias.json</strong> under <strong>PMP Path</strong>. "
                "Source data linked from <strong>Reference Path</strong>.",
            ),
            (
                "Model Diagnostics",
                "JSON files in <strong>enso_metric/ENSO_perf</strong>, "
                "<strong>enso_metric/ENSO_proc</strong>, and <strong>enso_metric/ENSO_tel</strong> under <strong>Output Path</strong>.",
            ),
            (
                "CMIP Diagnostics",
                "Pre-generated datasets: "
                "<a href='https://github.com/PCMDI/pcmdi_metrics_results_archive/tree/main/metrics_results/enso_metric' target='_blank' rel='noopener'>CMIP ENSO metrics</a>.",
            ),
        ]
        add_section(
            sections, f"ENSO Metrics Data ({cfg('enso_period','')})", enso_metrics
        )

    # ---------------- Render ----------------
    env = setup_jinja_env(cfg("template_dir"))
    template = env.get_template("data_template.html")

    output_html = template.render(
        title="E3SM-PMP Diagnostics Package", sections=sections
    )

    out_path = os.path.join(cfg("out_dir"), "diag_data.html")
    Path(out_path).write_text(output_html)
    print(f"HTML file written to: {cfg('out_dir')}")

    return out_path


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

    def build_summary_table(self, regions=None, metrics=None):
        clim_path = safe_join(str(self.fig_dir), "ERROR_metric/mean_climate")
        metric_table = []

        if regions is None:
            regions = self.regions

        if metrics is None:
            metrics = self.metrics

        for i, region in enumerate(regions):
            row = []
            if i == 0:
                row.append(
                    {"content": "<b>Mean Climate</b>", "rowspan": len(self.regions)}
                )
            row.append({"content": region.upper()})

            for metric in metrics:
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


def generate_summary_table(
    diag_dir: str,
    fig_dir: str,
    clim_show: Union[bool, str] = True,
    clim_regions: Optional[Union[List[str], str]] = None,
    clim_metrics: Optional[List[Tuple[str, ...]]] = None,
    mova_show: Union[bool, str] = True,
    movc_show: Union[bool, str] = True,
    enso_show: Union[bool, str] = True,
) -> List[list]:
    """
    Build the summary metrics table (Mean Climate, ENSO, EMoV).
    Returns a list of rows (each row is a list of cell dicts).
    """
    # Coerce possible string flags
    for name in ("clim_show", "mova_show", "movc_show", "enso_show"):
        val = locals()[name]
        if not isinstance(val, bool):
            locals()[name] = str(val).strip().lower() in {
                "1",
                "true",
                "t",
                "yes",
                "y",
                "on",
            }
    clim_show, mova_show, movc_show, enso_show = (
        clim_show,
        mova_show,
        movc_show,
        enso_show,
    )

    builder = SummaryTableBuilder(diag_dir, fig_dir)
    table: List[list] = []

    if clim_show:
        table.extend(
            builder.build_summary_table(regions=clim_regions, metrics=clim_metrics)
        )

    if enso_show:
        table.append(builder.build_enso_row())

    if mova_show or movc_show:
        table.append(builder.build_emov_row())

    return table


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


def generate_cmvar_table(
    diag_dir: str,
    fig_dir: str,
    enso_show: Union[bool, str] = True,
    movc_show: Union[bool, str] = True,
    movc_modes: Optional[Union[List[str], str]] = None,
) -> List[list]:
    """
    Build the Coupled Modes Variability (CMVAR) table (ENSO + coupled ocean modes).

    Returns:
        list: HTML-ready rows (list of lists of cell dicts).
    """
    # Coerce flags if they might come from text config
    if not isinstance(enso_show, bool):
        enso_show = str(enso_show).strip().lower() in {
            "1",
            "true",
            "t",
            "yes",
            "y",
            "on",
        }
    if not isinstance(movc_show, bool):
        movc_show = str(movc_show).strip().lower() in {
            "1",
            "true",
            "t",
            "yes",
            "y",
            "on",
        }

    # Accept comma-separated modes
    if movc_modes is None:
        movc_modes = ["PDO", "NPGO", "AMO"]
    elif isinstance(movc_modes, str):
        movc_modes = [s.strip() for s in movc_modes.split(",") if s.strip()]

    cmvar_table: List[list] = []

    builder = CMVARGroupBuilder()

    # --- ENSO block ---
    if enso_show:
        # Prefer existing dir; fallback keeps old behavior
        enso_path_candidates = (
            safe_join(str(fig_dir), "enso_metric"),
            safe_join(str(fig_dir), "ENSO_metric"),
        )
        enso_path = next(
            (p for p in enso_path_candidates if os.path.isdir(p)),
            enso_path_candidates[-1],
        )

        cmvar_groups = builder.construct() or {}
        for mode_label, group_data in cmvar_groups.items():
            if not isinstance(group_data, dict) or not group_data:
                continue
            group_name = next(iter(group_data))
            var_dict = group_data[group_name] or {}

            variables, (nrows, _ncols) = builder.reshape_1d_to_2d(
                list(var_dict.keys()), 4
            )

            for i, row_vars in enumerate(variables):
                row = []
                if i == 0:
                    row.append(
                        {"content": f"<b>ENSO {mode_label}</b>", "rowspan": nrows}
                    )

                for var in row_vars:
                    if not var:
                        row.append({"colspan": 4, "content": "--"})
                        continue
                    content = builder.build_cmvar_cell(
                        fig_dir=enso_path,
                        diag_dir=diag_dir,
                        group=group_name,
                        variable=var,
                        keys_dict=var_dict.get(var, {}),
                    )
                    row.append({"colspan": 4, "content": content})

                cmvar_table.append(row)

    # --- Coupled modes (PDO/NPGO/AMO) block ---
    if movc_show:
        emov_path = safe_join(str(fig_dir), "MOV_metric")
        for mode in movc_modes:
            mcpl_row = builder.generate_mcpl_row(mode, diag_dir, emov_path)
            if mcpl_row:
                cmvar_table.append(mcpl_row)

    return cmvar_table


class EMOVGroupBuilder:
    def __init__(
        self,
        diag_dir,
        fig_dir,
        modes_names=None,
        modes_seasons=None,
    ):
        self.diag_dir = diag_dir
        self.fig_dir = fig_dir
        self.modes = self.map_modes(modes_names)
        self.seasons = self.map_seasons(modes_seasons)

        # (UI label, filename template)
        self.rowspecs = [
            ("Composite (CBF)", "MOV_compose_{}_{}_cbf.png"),
            ("Composite (EOF)", "MOV_compose_{}_{}_{}.png"),  # needs 3 args
            ("North Test", "MOV_eoftest_{}_{}_EG_Spec.png"),
            ("EOF1 Pattern", "MOV_pattern_{}_{}_eof1.png"),
            ("EOF1 Telecon.", "MOV_telecon_{}_{}_eof1.png"),
            ("EOF2 Pattern", "MOV_pattern_{}_{}_eof2.png"),
            ("EOF2 Telecon.", "MOV_telecon_{}_{}_eof2.png"),
            ("EOF3 Pattern", "MOV_pattern_{}_{}_eof3.png"),
            ("EOF3 Telecon.", "MOV_telecon_{}_{}_eof3.png"),
        ]

    def map_modes(self, names):
        default_modes = {
            "NAM": "EOF1",
            "PNA": "EOF1",
            "NAO": "EOF1",
            "SAM": "EOF1",
            "NPO": "EOF2",
            "PSA1": "EOF2",
            "PSA2": "EOF3",
        }
        if names is None:
            return default_modes
        return {k: default_modes.get(k, "EOF1") for k in names}

    def map_seasons(self, names):
        default_seasons = ["DJF", "MAM", "JJA", "SON", "yearly", "monthly"]
        if names is None:
            return default_seasons

        seasons = []
        for sea in names:
            s = str(sea).strip().lower()
            if s in {"ann", "year", "yearly"}:
                seasons.append("yearly")
            elif s in {"mon", "month", "monthly"}:
                seasons.append("monthly")
            else:
                # preserve canonical case for DJF/MAM/JJA/SON if provided
                seasons.append(sea)
        return seasons

    def _format_filename(self, pattern, mode, season, eof_tag):
        n = pattern.count("{}")
        if n == 3:
            return pattern.format(mode, season, eof_tag)
        elif n == 2:
            return pattern.format(mode, season)
        else:
            # Unexpected template; return as-is to avoid crashing
            return pattern

    def build(self):
        emov_path = safe_join(str(self.fig_dir), "MOV_metric")
        table = []

        for mode, eof in self.modes.items():
            eof_tag = str(eof).lower()  # e.g., "eof1"
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
                    filename = self._format_filename(pattern, mode, season, eof_tag)

                    # subdir convention: e.g., "MOV_compose/<SEASON>"
                    prefix = "_".join(filename.split("_")[:2])  # "MOV_compose"
                    subdir_parts = [prefix, season]

                    link = create_image_link(
                        fig_dir=emov_path,
                        diag_dir=self.diag_dir,
                        subdirs=subdir_parts,
                        filename_pattern=filename,
                        fallback_filename=filename,
                        label=str(season).upper(),
                    )

                    row.append({"colspan": 4, "content": link})

                table.append(row)

        return table


def generate_emovs_table(
    diag_dir: str,
    fig_dir: str,
    show: Union[bool, str] = True,
    modes: Optional[Union[List[str], str]] = None,
) -> List[list]:
    """
    Build the Extratropical Modes of Variability (EMoV) table.

    Args:
        diag_dir: Path to diagnostics directory (for relative links).
        fig_dir: Path to figures directory.
        show: Whether to build the table. If a string (e.g., "false"), it will be coerced.
        modes: List of modes or comma-separated string (e.g., "PDO,NPGO,AMO").

    Returns:
        list: HTML-ready rows (list of lists of cell dicts).
    """
    # Coerce show if it might come in as a string
    if not isinstance(show, bool):
        show = str(show).strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    if not show:
        return []

    # Accept comma-separated string for modes
    if modes is None:
        modes = ["PDO", "NPGO", "AMO"]
    elif isinstance(modes, str):
        modes = [s.strip() for s in modes.split(",") if s.strip()]

    builder = EMOVGroupBuilder(
        diag_dir=diag_dir,
        fig_dir=fig_dir,
        modes_names=modes,
    )
    return builder.build()


class MeanClimateTableBuilder:
    def __init__(self, diag_dir, fig_dir, variables=None, regions=None, seasons=None):
        self.diag_dir = Path(diag_dir)
        self.fig_dir = Path(fig_dir)
        self.variables = self.map_vars(variables)
        self.regions = self.map_regions(regions)
        self.seasons = self.map_seasons(seasons)

    def map_vars(self, names):
        default_vars = ["pr", "psl", "tas", "ts", "rlds", "rlut"]
        if names is None:
            variables = default_vars
        else:
            variables = names
        return variables

    def map_regions(self, names):
        default_regions = ["global", "land", "ocean", "TROPICS", "NHEX", "SHEX"]
        if names is None:
            seasons = default_regions
        else:
            seasons = names
        return seasons

    def map_seasons(self, names):
        default_seasons = ["DJF", "MAM", "JJA", "SON", "AC"]  # AC = Annual Cycle
        if names is None:
            return default_seasons
        seasons = []
        for sea in names:
            s = str(sea).strip().lower()
            if s in {"ann", "year", "yearly"}:
                seasons.append("AC")
            else:
                # preserve canonical case for DJF/MAM/JJA/SON if provided
                seasons.append(sea)
        return seasons

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


def generate_clim_table(
    diag_dir: str,
    fig_dir: str,
    show: Union[bool, str] = True,
    variables: Optional[Union[List[str], str]] = None,
    regions: Optional[Union[List[str], str]] = None,
) -> List[list]:
    """
    Build the climatology diagnostics table.

    Args:
        diag_dir: Path to the diagnostics directory (used to compute relative links).
        fig_dir: Path to the figures directory.
        show: Whether to build the table. If a string (e.g., "true"/"false"), it will be coerced.
        variables: List of variable names (or a comma-separated string). If None, builder defaults are used.
        regions: List of region names (or a comma-separated string). If None, builder defaults are used.

    Returns:
        A list of table rows (each row is a list of cell dicts) ready for HTML rendering.
    """
    # Coerce 'show' if it might come from text config
    # In case some unexpected guess values are passed
    if not isinstance(show, bool):
        show = str(show).strip().lower() in {"1", "true", "t", "yes", "y", "on"}

    # In case comma-separated strings is passed
    if isinstance(variables, str):
        variables = [s.strip() for s in variables.split(",") if s.strip()]
    if isinstance(regions, str):
        regions = [s.strip() for s in regions.split(",") if s.strip()]

    if not show:
        return []

    builder = MeanClimateTableBuilder(
        diag_dir,
        fig_dir,
        variables=variables,
        regions=regions,
    )
    return builder.build_table()


def generate_viewer_html(config):
    """
    Generate overview HTML page for the E3SM-PMP Diagnostics
    """
    env = setup_jinja_env(config["template_dir"])
    template = env.get_template("index_template.html")

    # Add 'summary metrics' row
    metric_table = generate_summary_table(
        config["diag_dir"],
        config["fig_dir"],
        clim_show=config["clim_viewer"],
        clim_regions=config["clim_regions"],
        mova_show=config["mova_viewer"],
        movc_show=config["movc_viewer"],
        enso_show=config["enso_viewer"],
    )

    # Add 'Coupled Modes Variability' table
    cmvars_table = generate_cmvar_table(
        config["diag_dir"],
        config["fig_dir"],
        enso_show=config["enso_viewer"],
        movc_show=config["movc_viewer"],
        movc_modes=config["movc_modes"],
    )

    # Add 'Extratropical Modes Variability' tabel
    emovs_table = generate_emovs_table(
        config["diag_dir"],
        config["fig_dir"],
        show=config["mova_viewer"],
        modes=config["mova_modes"],
    )

    # Add 'Mean Climate Map' tabel
    clim_table = generate_clim_table(
        config["diag_dir"],
        config["fig_dir"],
        show=config["clim_viewer"],
        variables=config["clim_vars"],
        regions=config["clim_regions"],
    )

    # Render final HTML
    output_html = template.render(
        title=config["title"],
        subtitle=config["subtitle"],
        version=config["version"],
        clim_period=config["clim_period"],
        mova_period=config["mova_period"],
        movc_period=config["movc_period"],
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
