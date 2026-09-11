import shlex

import pytest

from zppy_interfaces.pcmdi_diags.pcmdi_variability_modes import (
    VariabilityMetricsCollector,
    VariabilityModesParameters,
    generate_varmode_cmds,
)


def _collector(tmp_path):
    return VariabilityMetricsCollector(
        ["NPGO"],
        "png",
        "CMIP6",
        "historical",
        "E3SM",
        "r1i1p1f1",
        "v20250923",
        str(tmp_path / "%(output_type)" / "%(metric_type)"),
        str(tmp_path / "output" / "%(group_type)"),
    )


def test_VariabilityModesParameters_normalizes_modes_and_variable():
    parameters = VariabilityModesParameters(
        {"var_modes": " npgo, PDO, npgo, ,", "vars": " psl "}
    )

    assert parameters.var_modes == ["NPGO", "PDO"]
    assert parameters.vars == "psl"


@pytest.mark.parametrize("variables", ["", " , ", "psl,ts"])
def test_VariabilityModesParameters_requires_one_variable(variables):
    with pytest.raises(ValueError, match="vars|exactly one variable"):
        VariabilityModesParameters({"var_modes": "NPO", "vars": variables})


def test_VariabilityMetricsCollector():
    vmc = VariabilityMetricsCollector(
        ["mode"],
        "png",
        "mip",
        "exp",
        "model",
        "relm",
        "v20250923",
        "dir_%(metric_type)",
        "",
    )
    assert vmc.input_dir == "dir_variability_modes"
    assert vmc.model_name == "mip.exp.model_relm"
    assert vmc.seasons == ["DJF", "MAM", "JJA", "SON", "yearly", "monthly"]
    assert vmc.fig_sets["MOV_eoftest"] == ["diagnostic_results", "EG_Spec*"]
    assert vmc.fig_sets["MOV_compose"] == ["graphics", "*compare_obs"]
    assert vmc.fig_sets["MOV_telecon"] == ["graphics", "*teleconnection"]
    assert vmc.fig_sets["MOV_pattern"] == ["graphics", "*"]
    with pytest.raises(ValueError, match="Could not classify"):
        vmc._classify_output_name("graphics", "mode", "DJF", "invalid.txt")
    assert (
        vmc._classify_output_name("graphics", "mode", "DJF", "North_test.txt")
        == "graphics_mode_DJF_EG_Spec.png"
    )
    assert (
        vmc._classify_output_name("graphics", "mode", "DJF", "_cbf_.txt")
        == "graphics_mode_DJF_cbf.png"
    )
    assert (
        vmc._classify_output_name("graphics", "mode", "DJF", "EOF1.txt")
        == "graphics_mode_DJF_eof1.png"
    )
    assert (
        vmc._classify_output_name("graphics", "mode", "DJF", "EOF2.txt")
        == "graphics_mode_DJF_eof2.png"
    )
    assert (
        vmc._classify_output_name("graphics", "mode", "DJF", "EOF3.txt")
        == "graphics_mode_DJF_eof3.png"
    )


def test_generate_varmode_cmds():
    actual = generate_varmode_cmds(
        ["mode1", "mode2"],
        "varOBS",
        "reftyrs",
        "reftyre",
        "refname",
        "refpath",
        "v20250923",
    )
    expected = [
        "variability_modes_driver.py -p parameterfile.py --variability_mode MODE1 --eofn_mod 1 --eofn_obs 1 --varOBS varOBS --osyear reftyrs --oeyear reftyre --reference_data_name refname --reference_data_path refpath --case_id v20250923",
        "variability_modes_driver.py -p parameterfile.py --variability_mode MODE2 --eofn_mod 1 --eofn_obs 1 --varOBS varOBS --osyear reftyrs --oeyear reftyre --reference_data_name refname --reference_data_path refpath --case_id v20250923",
    ]
    assert actual == expected


def test_generate_varmode_cmds_quotes_every_dynamic_argument():
    command = generate_varmode_cmds(
        ["npgo"],
        "psl; false",
        "1985",
        "2014",
        "obs $(false)",
        "/path with spaces/$(false)",
        "case; false",
    )[0]

    parsed = shlex.split(command)
    assert parsed[parsed.index("--variability_mode") + 1] == "NPGO"
    assert parsed[parsed.index("--eofn_mod") + 1] == "2"
    assert parsed[parsed.index("--varOBS") + 1] == "psl; false"
    assert parsed[parsed.index("--reference_data_name") + 1] == "obs $(false)"
    assert (
        parsed[parsed.index("--reference_data_path") + 1]
        == "/path with spaces/$(false)"
    )
    assert parsed[parsed.index("--case_id") + 1] == "case; false"


def test_generate_varmode_cmds_rejects_empty_modes():
    with pytest.raises(ValueError, match="var_modes"):
        generate_varmode_cmds([], "psl", 1985, 2014, "obs", "/ref", "case")


def test_collect_metrics_uses_portable_paths_and_safe_move(tmp_path):
    collector = _collector(tmp_path)
    source_dir = (
        tmp_path / "metrics_results" / "variability_modes" / "NPGO" / "Reference"
    )
    source_dir.mkdir(parents=True)
    source = source_dir / "metrics.json"
    source.write_text("{}")

    collector._collect_metrics()

    destination = (
        tmp_path
        / "output"
        / "metrics_data"
        / "variability_modes"
        / "NPGO"
        / "Reference"
        / "var_mode_NPGO.EOF2.CMIP6.historical.E3SM_r1i1p1f1.vs.Reference.v20250923.json"
    )
    assert destination.read_text() == "{}"
    assert not source.exists()


def test_collectors_fail_when_outputs_are_entirely_missing(tmp_path):
    collector = _collector(tmp_path)

    with pytest.raises(FileNotFoundError, match="metrics JSON"):
        collector._collect_metrics()
    with pytest.raises(FileNotFoundError, match="diagnostic NetCDF"):
        collector._collect_diags()
    with pytest.raises(FileNotFoundError, match="figures"):
        collector._collect_figures()
