import json

import pytest

from zppy_interfaces.pcmdi_diags import pcmdi_synthetic_plots
from zppy_interfaces.pcmdi_diags.pcmdi_synthetic_plots import (
    SyntheticPlotsParameters,
)


def _base_args(tmp_path):
    return {
        "figure_format": "png",
        "www": str(tmp_path / "www"),
        "results_dir": "model_vs_obs",
        "case": "E3SM_r1",
        "model_name": "CMIP6.historical.E3SM.r1i1p1f1",
        "model_tableID": "Amon",
        "web_dir": str(tmp_path / "web"),
        "pcmdi_webtitle": "Diagnostics",
        "pcmdi_version": "v1",
        "run_type": "model_vs_obs",
        "pcmdi_external_prefix": str(tmp_path / "external"),
        "pcmdi_viewer_template": "viewer-template",
        "clim_viewer": False,
        "mova_viewer": False,
        "movc_viewer": False,
        "enso_viewer": True,
        "enso_years": "1985-2014",
        "cmip_enso_dir": str(tmp_path / "cmip-enso"),
        "cmip_enso_set": "CMIP6.historical.dataset",
    }


def test_SyntheticPlotsParameters_normalizes_optional_lists(tmp_path):
    args = _base_args(tmp_path)
    args.update(
        {
            "clim_vars": " pr, tas, ,",
            "clim_regions": [" global ", "", "ocean"],
            "enso_vars": " ts, tauu ",
        }
    )

    parameters = SyntheticPlotsParameters(args)

    assert parameters.clim_vars == ["pr", "tas"]
    assert parameters.clim_regions == ["global", "ocean"]
    assert parameters.enso_vars == ["ts", "tauu"]


def test_SyntheticPlotsParameters_defaults_enabled_modes(tmp_path):
    args = _base_args(tmp_path)
    args.update(
        {
            "mova_viewer": True,
            "mova_years": "1985-2014",
            "cmip_movs_dir": str(tmp_path / "cmip-movs"),
            "cmip_movs_set": "CMIP6.historical.dataset",
        }
    )

    parameters = SyntheticPlotsParameters(args)

    assert parameters.mova_modes == [
        "NAM",
        "PNA",
        "NPO",
        "NAO",
        "SAM",
        "PSA1",
        "PSA2",
    ]


def test_SyntheticPlotsParameters_rejects_missing_required_value(tmp_path):
    args = _base_args(tmp_path)
    args["web_dir"] = " "

    with pytest.raises(ValueError, match="--web_dir"):
        SyntheticPlotsParameters(args)


def test_SyntheticPlotsParameters_validates_enabled_viewer_inputs(tmp_path):
    args = _base_args(tmp_path)
    args["cmip_enso_set"] = None

    with pytest.raises(ValueError, match="--cmip_enso_set"):
        SyntheticPlotsParameters(args)


def test_SyntheticPlotsParameters_requires_enabled_viewer(tmp_path):
    args = _base_args(tmp_path)
    args["enso_viewer"] = False

    with pytest.raises(ValueError, match="At least one diagnostics viewer"):
        SyntheticPlotsParameters(args)


def test_main_passes_configured_variable_lists_to_viewer(
    tmp_path, monkeypatch
):
    args = _base_args(tmp_path)
    args.update(
        {
            "mova_vars": "psl, zg",
            "movc_vars": "ts, tos",
            "enso_vars": "ts, tauu",
        }
    )
    monkeypatch.chdir(tmp_path)
    (tmp_path / "synthetic_metrics_list.json").write_text(json.dumps({}))
    monkeypatch.setattr(pcmdi_synthetic_plots, "_get_args", lambda: args)

    class FakePlotter:
        def __init__(self, **kwargs):
            pass

        def generate(self):
            pass

    captured = {}

    def fake_collect_config(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(pcmdi_synthetic_plots, "SyntheticMetricsPlotter", FakePlotter)
    monkeypatch.setattr(pcmdi_synthetic_plots, "collect_config", fake_collect_config)
    monkeypatch.setattr(pcmdi_synthetic_plots, "generate_methodology_html", lambda _: None)
    monkeypatch.setattr(pcmdi_synthetic_plots, "generate_data_html", lambda _: None)
    monkeypatch.setattr(pcmdi_synthetic_plots, "generate_viewer_html", lambda _: None)

    pcmdi_synthetic_plots.main()

    assert captured["mova_vars"] == ["psl", "zg"]
    assert captured["movc_vars"] == ["ts", "tos"]
    assert captured["enso_vars"] == ["ts", "tauu"]
