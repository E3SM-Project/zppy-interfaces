import pandas as pd

from zppy_interfaces.pcmdi_diags.synthetic_plots import synthetic_metrics_plotter
from zppy_interfaces.pcmdi_diags.synthetic_plots.synthetic_metrics_plotter import (
    drop_vars,
    mean_climate_plot_driver,
)


def _mean_climate_frame(**values):
    data = {
        "model": ["CMIP", "E3SM"],
        "run": ["r1", "r1"],
        "model_run": ["CMIP_r1", "E3SM_r1"],
    }
    data.update(values)
    return pd.DataFrame(data)


def test_drop_vars_removes_requested_variables_missing_from_dataframe():
    data_dict, var_names, var_units = drop_vars(
        _mean_climate_frame(pr=[1.0, 2.0]),
        ["pr", "prw"],
        ["mm/day", "kg/m2"],
    )

    assert "prw" not in data_dict.columns
    assert var_names == ["pr"]
    assert var_units == ["mm/day"]


def test_mean_climate_portrait_skips_region_missing_variables(monkeypatch, tmp_path):
    captured = {}

    def fake_portrait_metric_plot(
        region,
        stat,
        group,
        data_dict,
        stat_name,
        model_name,
        var_list,
        model_list,
        out_path,
        fig_format,
    ):
        captured["region"] = region
        captured["var_list"] = var_list
        captured["data_dict"] = data_dict

    monkeypatch.setattr(
        synthetic_metrics_plotter,
        "portrait_metric_plot",
        fake_portrait_metric_plot,
    )

    metric_dict = {
        "type": ["portrait"],
        "region": ["ocean"],
        "season": ["djf", "mam", "jja", "son"],
        "name": "Mean Bias",
    }
    df_dict = {
        "djf": {"ocean": _mean_climate_frame(pr=[1.0, 2.0], prw=[3.0, 4.0])},
        "mam": {"ocean": _mean_climate_frame(pr=[1.0, 2.0])},
        "jja": {"ocean": _mean_climate_frame(pr=[1.0, 2.0], prw=[3.0, 4.0])},
        "son": {"ocean": _mean_climate_frame(pr=[1.0, 2.0], prw=[3.0, 4.0])},
    }

    mean_climate_plot_driver(
        metric="mean_climate",
        stat="mae_xy",
        regions=["ocean"],
        model_name=["E3SM"],
        metric_dict=metric_dict,
        df_dict=df_dict,
        var_list=["pr", "prw"],
        var_unit_list=["mm/day", "kg/m2"],
        save_data=False,
        out_path=str(tmp_path),
        fig_format="png",
    )

    assert captured["region"] == "ocean"
    assert captured["var_list"] == ["pr"]
    assert all(values.shape == (1, 2) for values in captured["data_dict"].values())
