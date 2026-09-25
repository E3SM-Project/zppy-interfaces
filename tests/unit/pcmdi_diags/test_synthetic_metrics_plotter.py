import json
from types import SimpleNamespace

import pandas as pd
import pytest

from zppy_interfaces.pcmdi_diags.synthetic_plots import synthetic_metrics_plotter
from zppy_interfaces.pcmdi_diags.synthetic_plots.enso_metrics_reader import (
    EnsoMetricsReader,
)
from zppy_interfaces.pcmdi_diags.synthetic_plots.synthetic_metrics_plotter import (
    SyntheticMetricsPlotter,
    drop_vars,
    enso_plot_driver,
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


def test_handle_mean_climate_uses_all_data_when_filters_are_unspecified(
    monkeypatch, tmp_path
):
    merge_lib = SimpleNamespace(
        var_list=["pr", "tas"],
        var_unit_list=["mm/day", "K"],
        regions=["global", "ocean"],
        df_dict={"mae_xy": {}},
    )

    class FakeClimMetricsReader:
        def __init__(self, parameter):
            pass

        def collect(self):
            return merge_lib

    captured = {}

    def fake_plot_driver(*args):
        captured["regions"] = args[2]
        captured["var_list"] = args[6]
        captured["var_unit_list"] = args[7]

    monkeypatch.setattr(
        synthetic_metrics_plotter, "ClimMetricsReader", FakeClimMetricsReader
    )
    monkeypatch.setattr(
        synthetic_metrics_plotter, "mean_climate_plot_driver", fake_plot_driver
    )
    plotter = SyntheticMetricsPlotter(
        case_name="E3SM_r1",
        test_name="CMIP6.historical.E3SM.r1i1p1f1",
        table_id="Amon",
        figure_format="png",
        metric_dict={
            "mean_climate": {"mae_xy": {}},
            "variability_modes": {},
            "enso_metric": {},
        },
        save_data=False,
        base_test_input_path="/base/%(group_type)/put_model_here",
        results_dir=str(tmp_path),
        clim_viewer=True,
        clim_vars=None,
        clim_regions=None,
        mova_viewer=False,
        movc_viewer=False,
        enso_viewer=False,
    )

    plotter._handle_mean_climate("mean_climate")

    assert captured["regions"] == ["global", "ocean"]
    assert captured["var_list"] == ["pr", "tas"]
    assert captured["var_unit_list"] == ["mm/day", "K"]


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


def test_enso_plot_driver_builds_portrait_plot_paths(monkeypatch, tmp_path):
    captured = {}

    def fake_enso_portrait_plot(
        metrics_collections,
        list_project,
        list_obs,
        dict_json_path,
        figure_name,
        reduced_set,
    ):
        captured["metrics_collections"] = metrics_collections
        captured["list_project"] = list_project
        captured["list_obs"] = list_obs
        captured["dict_json_path"] = dict_json_path
        captured["figure_name"] = figure_name
        captured["reduced_set"] = reduced_set
        return None, {}

    monkeypatch.setattr(
        synthetic_metrics_plotter, "enso_portrait_plot", fake_enso_portrait_plot
    )

    parameter = {
        "diag_vars": {
            "cor_xy": {
                "type": ["portrait"],
                "collection": ["ENSO_perf"],
            }
        },
        "cmip_name": "CMIP6.historical",
        "model_name": ["E3SM_r1"],
        "out_dir": str(tmp_path),
    }

    plotted = enso_plot_driver(
        "enso_metric", "cor_xy", "path/to/dict_json.json", parameter, "png"
    )

    assert captured["metrics_collections"] == ["ENSO_perf"]
    assert captured["list_project"] == ["CMIP6", "E3SM_r1"]
    assert captured["list_obs"] == []
    assert captured["dict_json_path"] == "path/to/dict_json.json"
    assert captured["reduced_set"] is True
    assert plotted is True

    expected_outdir = tmp_path / "enso_metric"
    assert expected_outdir.is_dir()
    assert captured["figure_name"] == str(
        expected_outdir / "enso_metric_cor_xy_portrait.png"
    )


def test_handle_enso_metric_dispatches_to_reader_and_plot_driver(monkeypatch, tmp_path):
    captured_reader_args = {}
    captured_plot_args = {}

    class FakeEnsoMetricsReader:
        def __init__(self, parameter, stat, metric_dict, mips, collections):
            captured_reader_args["parameter"] = parameter
            captured_reader_args["stat"] = stat
            captured_reader_args["metric_dict"] = metric_dict
            captured_reader_args["mips"] = mips
            captured_reader_args["collections"] = collections

        def run(self):
            return {"fake": "path"}

    def fake_enso_plot_driver(metric, stat, dict_json_path, parameter, fig_format):
        captured_plot_args["metric"] = metric
        captured_plot_args["stat"] = stat
        captured_plot_args["dict_json_path"] = dict_json_path
        captured_plot_args["fig_format"] = fig_format
        return True

    monkeypatch.setattr(
        synthetic_metrics_plotter, "EnsoMetricsReader", FakeEnsoMetricsReader
    )
    monkeypatch.setattr(
        synthetic_metrics_plotter, "enso_plot_driver", fake_enso_plot_driver
    )

    plotter = SyntheticMetricsPlotter(
        case_name="E3SM_r1",
        test_name="CMIP6.historical.E3SM.r1i1p1f1",
        table_id="Amon",
        figure_format="png",
        metric_dict={
            "mean_climate": {},
            "variability_modes": {},
            "enso_metric": {"cor_xy": {"collection": ["ENSO_perf"]}},
        },
        save_data=False,
        base_test_input_path="/base/%(group_type)/put_model_here",
        results_dir=str(tmp_path),
        clim_viewer=False,
        mova_viewer=False,
        movc_viewer=False,
        enso_viewer=True,
        cmip_enso_dir="/cmip",
        cmip_enso_set="CMIP6.historical",
    )

    plotter.generate()

    assert captured_reader_args["stat"] == "cor_xy"
    assert captured_reader_args["mips"] == ["CMIP6", "E3SM_r1"]
    assert captured_reader_args["collections"] == ["ENSO_perf"]
    assert captured_plot_args["metric"] == "enso_metric"
    assert captured_plot_args["stat"] == "cor_xy"
    assert captured_plot_args["dict_json_path"] == {"fake": "path"}
    assert captured_plot_args["fig_format"] == "png"


def test_enso_metrics_reader_dispatches_uppercase_cmip(monkeypatch, tmp_path):
    cmip_file = tmp_path / "cmip.json"
    cmip_file.touch()
    reader = EnsoMetricsReader(
        parameter={},
        stat="cor_xy",
        metric_dict={},
        mips=["CMIP6"],
        collections=["ENSO_perf"],
    )

    monkeypatch.setattr(
        reader,
        "_get_cmip_json_path",
        lambda mip, collection: str(cmip_file),
    )

    def fail_test_lookup(mip, collection):
        raise AssertionError("CMIP6 must not use the test-model lookup")

    monkeypatch.setattr(reader, "_get_test_json_path", fail_test_lookup)

    assert reader.run() == {"CMIP6": {"ENSO_perf": str(cmip_file)}}


def test_enso_metrics_reader_rejects_nonexistent_collected_path(monkeypatch, tmp_path):
    reader = EnsoMetricsReader(
        parameter={},
        stat="cor_xy",
        metric_dict={},
        mips=["CMIP6"],
        collections=["ENSO_perf"],
    )
    missing_file = tmp_path / "missing.json"
    monkeypatch.setattr(
        reader,
        "_get_cmip_json_path",
        lambda mip, collection: str(missing_file),
    )

    with pytest.raises(FileNotFoundError, match="CMIP6.*ENSO_perf"):
        reader.run()


@pytest.mark.parametrize(
    ("mips", "collections", "message"),
    [
        ([], ["ENSO_perf"], "mips is empty"),
        (["CMIP6"], [], "metrics_collections is empty"),
    ],
)
def test_enso_metrics_reader_rejects_empty_inputs(mips, collections, message):
    reader = EnsoMetricsReader({}, "cor_xy", {}, mips, collections)

    with pytest.raises(ValueError, match=message):
        reader.run()


def test_enso_metrics_reader_maps_each_test_model_to_its_own_file(tmp_path):
    collection = "ENSO_perf"
    model_names = ["ModelA", "ModelB"]
    expected_paths = {}

    for model_name in model_names:
        model_dir = tmp_path / model_name / collection
        model_dir.mkdir(parents=True)
        json_path = model_dir / f"metrics.{model_name}.results.v20250923.json"
        json_path.write_text(json.dumps({"RESULTS": {"model": {"old": {}}}}))
        expected_paths[model_name] = str(json_path)

    reader = EnsoMetricsReader(
        parameter={
            "model_name": model_names,
            "test_path": str(tmp_path / "put_model_here"),
        },
        stat="cor_xy",
        metric_dict={},
        mips=model_names,
        collections=[collection],
    )

    assert reader.run() == {
        model_name: {collection: expected_paths[model_name]}
        for model_name in model_names
    }
    for model_name, json_path in expected_paths.items():
        with open(json_path) as json_file:
            data = json.load(json_file)
        assert list(data["RESULTS"]["model"]) == [model_name]


def test_enso_metrics_reader_rejects_missing_model_block(tmp_path):
    model_dir = tmp_path / "ModelA" / "ENSO_perf"
    model_dir.mkdir(parents=True)
    (model_dir / "metrics.ModelA.results.v20250923.json").write_text(
        json.dumps({"RESULTS": {}})
    )
    reader = EnsoMetricsReader(
        parameter={
            "model_name": ["ModelA"],
            "test_path": str(tmp_path / "put_model_here"),
        },
        stat="cor_xy",
        metric_dict={},
        mips=["ModelA"],
        collections=["ENSO_perf"],
    )

    with pytest.raises(KeyError, match="RESULTS.model"):
        reader.run()


def test_generate_fails_when_all_enso_stats_fail(monkeypatch, tmp_path):
    class FailingEnsoMetricsReader:
        def __init__(self, *args, **kwargs):
            pass

        def run(self):
            raise FileNotFoundError("missing ENSO metrics")

    monkeypatch.setattr(
        synthetic_metrics_plotter, "EnsoMetricsReader", FailingEnsoMetricsReader
    )
    plotter = SyntheticMetricsPlotter(
        case_name="E3SM_r1",
        test_name="CMIP6.historical.E3SM.r1i1p1f1",
        table_id="Amon",
        figure_format="png",
        metric_dict={
            "mean_climate": {},
            "variability_modes": {},
            "enso_metric": {"cor_xy": {"collection": ["ENSO_perf"]}},
        },
        save_data=False,
        base_test_input_path="/base/%(group_type)/put_model_here",
        results_dir=str(tmp_path),
        clim_viewer=False,
        mova_viewer=False,
        movc_viewer=False,
        enso_viewer=True,
        cmip_enso_dir="/cmip",
        cmip_enso_set="CMIP6.historical.dataset",
    )

    with pytest.raises(RuntimeError, match="No synthetic metrics plots"):
        plotter.generate()
