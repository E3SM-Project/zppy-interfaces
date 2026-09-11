import json
from collections import OrderedDict

import pytest

from zppy_interfaces.pcmdi_diags import pcmdi_setup
from zppy_interfaces.pcmdi_diags.pcmdi_setup import (
    CoreParameters,
    DataCatalogueBuilder,
    LandSeaMaskGenerator,
    set_up,
)


def _core_args(tmp_path, run_type="model_vs_obs"):
    return {
        "num_workers": "2",
        "multiprocessing": "true",
        "subsection": "mean_climate",
        "climo_ts_dir_primary": str(tmp_path / "test"),
        "climo_ts_dir_ref": str(tmp_path / "ref"),
        "model_name": "CMIP6.historical.E3SM.r1i1p1f1",
        "model_tableID": "Amon",
        "figure_format": "png",
        "run_type": run_type,
        "obs_sets": " default, alternate " if run_type == "model_vs_obs" else None,
        "model_name_ref": (
            "CMIP6.historical.Reference.r1i1p1f1"
            if run_type == "model_vs_model"
            else None
        ),
        "vars": " pr, tas, ,",
        "tableID_ref": "Amon" if run_type == "model_vs_model" else None,
        "generate_sftlf": "false",
        "case_id": "v20250923",
        "results_dir": "model_vs_obs",
    }


def test_CoreParameters_validates_and_normalizes_values(tmp_path):
    parameters = CoreParameters(_core_args(tmp_path))

    assert parameters.num_workers == 2
    assert parameters.multiprocessing is True
    assert parameters.variables == ["pr", "tas"]
    assert parameters.obs_sets == "default,alternate"


@pytest.mark.parametrize("num_workers", ["0", "-1", "not-an-integer"])
def test_CoreParameters_rejects_invalid_num_workers(tmp_path, num_workers):
    args = _core_args(tmp_path)
    args["num_workers"] = num_workers

    with pytest.raises(ValueError, match="num_workers"):
        CoreParameters(args)


def test_CoreParameters_rejects_invalid_multiprocessing_value(tmp_path):
    args = _core_args(tmp_path)
    args["multiprocessing"] = "sometimes"

    with pytest.raises(ValueError, match="multiprocessing"):
        CoreParameters(args)


@pytest.mark.parametrize(
    ("run_type", "missing_arg"),
    [
        ("model_vs_obs", "obs_sets"),
        ("model_vs_model", "model_name_ref"),
        ("model_vs_model", "tableID_ref"),
    ],
)
def test_CoreParameters_validates_run_specific_arguments(
    tmp_path, run_type, missing_arg
):
    args = _core_args(tmp_path, run_type=run_type)
    args[missing_arg] = None

    with pytest.raises(ValueError, match=missing_arg):
        CoreParameters(args)


def test_DataCatalogueBuilder():
    dcb = DataCatalogueBuilder("", [], "", [], [], "", "")

    assert dcb._get_base_varname("ta-200") == "ta"
    assert dcb._get_base_varname("ta_200") == "ta"
    assert dcb._get_base_varname("pr") == "pr"


def test_DataCatalogueBuilder_builds_logical_variable_from_alias(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    test_dir = tmp_path / "test"
    ref_dir = tmp_path / "ref"
    output_dir = tmp_path / "catalogues"
    test_dir.mkdir()
    ref_dir.mkdir()
    output_dir.mkdir()

    (test_dir / "cmip.exp.model.r1.Amon.ts.198501-201412.nc").touch()
    (ref_dir / "obs.exp.dataset.r1.Amon.sst.198501-201412.nc").touch()

    builder = DataCatalogueBuilder(
        "test",
        ["exp"],
        "ref",
        ["exp"],
        ["sst"],
        "enso",
        "catalogues",
        variable_aliases={"sst": "ts"},
    )
    test_info, ref_info = builder.build_catalogues()

    assert list(test_info) == ["sst"]
    assert list(ref_info) == ["sst"]
    assert test_info["sst"]["model"]["var_name"] == "sst"
    assert test_info["sst"]["model"]["var_in_file"] == "ts"
    assert ref_info["sst"]["dataset"]["var_in_file"] == "sst"
    assert (output_dir / "test_enso_catalogue.json").exists()
    assert (output_dir / "ref_enso_catalogue.json").exists()
    assert builder.test_catalogue_path == "catalogues/test_enso_catalogue.json"
    assert builder.ref_catalogue_path == "catalogues/ref_enso_catalogue.json"

    # Preserve the established path when both the requested name and alias exist.
    (test_dir / "cmip.exp.model.r1.Amon.sst.198501-201412.nc").touch()
    builder = DataCatalogueBuilder(
        "test",
        ["exp"],
        "ref",
        ["exp"],
        ["sst"],
        "enso",
        "catalogues",
        variable_aliases={"sst": "ts"},
    )
    test_info, _ = builder.build_catalogues()

    assert test_info["sst"]["model"]["var_in_file"] == "sst"


def test_DataCatalogueBuilder_saves_absolute_source_under_output_dir(tmp_path):
    source_dir = tmp_path / "inputs" / "test"
    output_dir = tmp_path / "catalogues"
    builder = DataCatalogueBuilder(
        str(source_dir), [], "ref", [], [], "enso", str(output_dir)
    )

    catalogue_path = builder._save_catalogue(str(source_dir), OrderedDict())

    assert catalogue_path == str(output_dir / "test_enso_catalogue.json")
    assert (output_dir / "test_enso_catalogue.json").is_file()


def test_LandSeaMaskGenerator_uses_group_basename_for_absolute_path(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    catalogue_dir = tmp_path / "pcmdi_diags"
    catalogue_dir.mkdir()
    catalogue = catalogue_dir / "test_mean_climate_catalogue.json"
    catalogue.write_text(json.dumps({}))
    generator = LandSeaMaskGenerator(
        str(tmp_path / "inputs" / "test"),
        str(tmp_path / "inputs" / "ref"),
        "mean_climate",
    )

    generator._process_group(str(tmp_path / "inputs" / "test"))


def test_set_up_derives_missing_reference_when_test_variable_exists(
    tmp_path, monkeypatch
):
    test_dir = tmp_path / "test"
    ref_dir = tmp_path / "ref"
    test_dir.mkdir()
    ref_dir.mkdir()
    (test_dir / "case.rstcre.198501-201412.nc").touch()
    parameters = CoreParameters(_core_args(tmp_path, run_type="model_vs_model"))
    parameters.variables = ["rstcre"]
    derived = []

    monkeypatch.setattr(
        pcmdi_setup,
        "derive_missing_variable",
        lambda variable, path, model_id: derived.append((variable, path, model_id)),
    )

    class FakeBuilder:
        def __init__(self, *args, **kwargs):
            self.test_catalogue_path = "pcmdi_diags/test_mean_climate_catalogue.json"

        def build_catalogues(self):
            return {}, {"rstcre": {"set": "default"}}

    class FakeMaskGenerator:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, flag):
            pass

    monkeypatch.setattr(pcmdi_setup, "DataCatalogueBuilder", FakeBuilder)
    monkeypatch.setattr(pcmdi_setup, "LandSeaMaskGenerator", FakeMaskGenerator)

    set_up(parameters)

    assert derived == [
        (
            "rstcre",
            str(ref_dir),
            "CMIP6.historical.Reference.r1i1p1f1.Amon",
        )
    ]


def test_LandSeaMaskGenerator():
    lsmg = LandSeaMaskGenerator("", "", "", "")
    assert lsmg._parse_flag("True")
    assert lsmg._parse_flag("Y")
    assert lsmg._parse_flag("Yes")
    assert lsmg._parse_flag("true")
    assert lsmg._parse_flag("y")
    assert lsmg._parse_flag("yes")
    assert not lsmg._parse_flag("False")
