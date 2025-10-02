from zppy_interfaces.pcmdi_diags.pcmdi_variability_modes import (
    VariabilityMetricsCollector,
    generate_varmode_cmds,
)


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
    assert (
        vmc._classify_output_name("graphics", "mode", "DJF", "invalid.txt")
        == "graphics_mode_DJF_unknown.png"
    )
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
        "variability_modes_driver.py -p parameterfile.py --variability_mode mode1 --eofn_mod 1 --eofn_obs 1 --varOBS varOBS --osyear reftyrs --oeyear reftyre --reference_data_name refname --reference_data_path refpath --case_id v20250923",
        "variability_modes_driver.py -p parameterfile.py --variability_mode mode2 --eofn_mod 1 --eofn_obs 1 --varOBS varOBS --osyear reftyrs --oeyear reftyre --reference_data_name refname --reference_data_path refpath --case_id v20250923",
    ]
    assert actual == expected
