from zppy_interfaces.pcmdi_diags.synthetic_plots.utils import get_highlight_models


def test_get_highlight_models():
    all_models = [
        "CESM2-FV2",
        "CESM2-WACCM",
        "CESM2-WACCM-FV2",
        "GFDL-AM4",
        "GFDL-CM4",
        "GFDL-ESM4",
        "E3SM-1-0",
        "E3SM-2-0",
    ]
    model_name = ["CESM2-FV2", "E3SM"]
    actual = get_highlight_models(all_models, model_name)
    expected = [
        "E3SM-1-0",
        "E3SM-2-0",
        "CESM2-FV2",
    ]
    assert actual == expected
