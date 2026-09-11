from zppy_interfaces.pcmdi_diags.pcmdi_setup import (
    DataCatalogueBuilder,
    LandSeaMaskGenerator,
)


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


def test_LandSeaMaskGenerator():
    lsmg = LandSeaMaskGenerator("", "", "", "")
    assert lsmg._parse_flag("True")
    assert lsmg._parse_flag("Y")
    assert lsmg._parse_flag("Yes")
    assert lsmg._parse_flag("true")
    assert lsmg._parse_flag("y")
    assert lsmg._parse_flag("yes")
    assert not lsmg._parse_flag("False")
