import json

import pytest

from zppy_interfaces.pcmdi_diags.pcmdi_enso import (
    EnsoDiagnosticsCollector,
    ENSOParameters,
    build_enso_obsvar_catalog,
    build_enso_obsvar_landmask,
    check_enso_input,
    check_output_dirs,
    check_vars,
    generate_enso_cmds,
    normalize_enso_model_catalogue,
)


def test_ENSOParameters_requires_enso_groups():
    with pytest.raises(ValueError):
        ENSOParameters({})


def test_ENSOParameters_parses_enso_groups():
    params = ENSOParameters({"enso_groups": "ENSO_perf,ENSO_proc"})
    assert params.enso_groups == "ENSO_perf,ENSO_proc"


def test_EnsoDiagnosticsCollector_init_paths_and_model_name():
    collector = EnsoDiagnosticsCollector(
        fig_format="png",
        refname="obsname",
        model_name_parts=["CMIP6", "historical", "E3SM", "r1i1p1f1"],
        case_id="v20250923",
        input_dir="dir_%(metric_type)/%(output_type)",
        output_dir="out_%(group_type)",
    )
    assert collector.mip == "CMIP6"
    assert collector.exp == "historical"
    assert collector.model == "E3SM"
    assert collector.relm == "r1i1p1f1"
    assert collector.model_name == "CMIP6.historical.E3SM_r1i1p1f1"
    assert collector.input_dir == "dir_enso_metric/%(output_type)"
    assert collector.diag_metric == "enso_metric"
    assert collector.fig_sets == {"ENSO_metric": ["graphics", "*"]}


def test_EnsoDiagnosticsCollector_invalid_model_name_parts_raises():
    with pytest.raises(ValueError):
        EnsoDiagnosticsCollector(
            fig_format="png",
            refname="obsname",
            model_name_parts=["CMIP6", "historical"],
            case_id="v20250923",
            input_dir="dir_%(metric_type)",
            output_dir="out",
        )


def test_collect_figures_uses_tail_after_last_model_marker(tmp_path):
    collector = EnsoDiagnosticsCollector(
        fig_format="png",
        refname="obsname",
        model_name_parts=["CMIP6", "historical", "E3SM", "r1i1p1f1"],
        case_id="v20250923",
        input_dir=str(tmp_path / "input" / "%(metric_type)" / "%(output_type)"),
        output_dir=str(tmp_path / "output" / "%(group_type)"),
    )
    group = "ENSO_perf"
    source_dir = tmp_path / "input" / "enso_metric" / "graphics" / group
    source_dir.mkdir(parents=True)
    source = source_dir / "prefix_E3SM_r1i1p1f1_middle_E3SM_r1i1p1f1_suffix.png"
    source.touch()

    assert collector.collect_figures([group]) is True

    destination = tmp_path / "output" / "ENSO_metric" / group / "ENSO_perf_suffix.png"
    assert destination.is_file()


def test_generate_enso_cmds(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "parameterfile.py").touch()

    actual = generate_enso_cmds("ENSO_perf, ENSO_proc", "v20250923")

    assert actual == [
        "enso_driver.py -p parameterfile.py --metricsCollection ENSO_perf --case_id v20250923",
        "enso_driver.py -p parameterfile.py --metricsCollection ENSO_proc --case_id v20250923",
    ]


def test_generate_enso_cmds_missing_param_file_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError):
        generate_enso_cmds("ENSO_perf", "v20250923")


def test_check_vars_all_variables_found(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ts_dir = tmp_path / "ts"
    ts_dir.mkdir()
    (ts_dir / "case.ts.198501_201412.nc").touch()
    (ts_dir / "ts_files.txt").touch()
    (ts_dir / "pr_198501_201412.nc").touch()
    (ts_dir / "pr_files.txt").touch()

    stdout = "list_variables = ['ts', 'pr']\n"

    assert check_vars(stdout) is True


@pytest.mark.parametrize("variable", ["ssh", "thf", "taux"])
def test_check_vars_missing_optional_variables_still_passes(
    tmp_path, monkeypatch, variable
):
    monkeypatch.chdir(tmp_path)
    ts_dir = tmp_path / "ts"
    ts_dir.mkdir()
    (ts_dir / "case.ts.198501_201412.nc").touch()
    (ts_dir / "ts_files.txt").touch()
    stdout = f"list_variables = ['ts', '{variable}']\n"

    assert check_vars(stdout) is True


@pytest.mark.parametrize("variable", ["tas", "tauy"])
def test_check_vars_missing_required_variable_fails(tmp_path, monkeypatch, variable):
    monkeypatch.chdir(tmp_path)
    ts_dir = tmp_path / "ts"
    ts_dir.mkdir()
    (ts_dir / "case.ts.198501_201412.nc").touch()
    (ts_dir / "ts_files.txt").touch()
    stdout = f"list_variables = ['ts', '{variable}']\n"

    assert check_vars(stdout) is False


def test_check_vars_no_variable_list_found_fails():
    assert check_vars("no variable list in this output") is False


def test_check_enso_input_uses_configured_directory(tmp_path):
    input_dir = tmp_path / "arbitrary-input"
    input_dir.mkdir()
    source_nc = input_dir / "case.ts.198501-201412.nc"
    source_txt = input_dir / "ts_files.txt"
    source_nc.touch()
    source_txt.touch()

    check_enso_input(str(input_dir))

    assert (input_dir / "case.sst.198501-201412.nc").is_symlink()
    assert (input_dir / "sst_files.txt").is_symlink()


def test_check_enso_input_links_all_matching_files_and_repairs_broken_link(tmp_path):
    input_dir = tmp_path / "arbitrary-input"
    input_dir.mkdir()
    first_source = input_dir / "case.ts.198501-199912.nc"
    second_source = input_dir / "case.ts.200001-201412.nc"
    first_source.touch()
    second_source.touch()
    broken_link = input_dir / "case.sst.198501-199912.nc"
    broken_link.symlink_to(input_dir / "missing.ts.nc")

    check_enso_input(str(input_dir))

    first_link = input_dir / "case.sst.198501-199912.nc"
    second_link = input_dir / "case.sst.200001-201412.nc"
    assert first_link.resolve() == first_source
    assert second_link.resolve() == second_source


def test_check_vars_uses_configured_directory(tmp_path):
    input_dir = tmp_path / "arbitrary-input"
    input_dir.mkdir()
    (input_dir / "case.ts.198501-201412.nc").touch()
    (input_dir / "ts_files.txt").touch()

    assert check_vars("list_variables = ['ts']\n", str(input_dir)) is True


def test_check_output_dirs_all_present_and_nonempty(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for subdir in ["graphics_out", "diag_out", "metrics_out"]:
        d = tmp_path / subdir
        d.mkdir()
        (d / "placeholder.txt").touch()

    stdout = (
        "output directory for graphics: graphics_out\n"
        "output directory for diagnostic_results: diag_out\n"
        "output directory for metrics_results: metrics_out\n"
    )

    assert check_output_dirs(stdout) is True


def test_check_output_dirs_empty_directory_fails(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "graphics_out").mkdir()
    (tmp_path / "graphics_out" / "placeholder.txt").touch()
    (tmp_path / "diag_out").mkdir()
    (tmp_path / "diag_out" / "placeholder.txt").touch()
    (tmp_path / "metrics_out").mkdir()  # left empty -- should fail the check

    stdout = (
        "output directory for graphics: graphics_out\n"
        "output directory for diagnostic_results: diag_out\n"
        "output directory for metrics_results: metrics_out\n"
    )

    assert check_output_dirs(stdout) is False


def test_check_output_dirs_missing_lines_fail_validation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert check_output_dirs("") is False


def test_build_enso_obsvar_catalog(tmp_path):
    obs_dic = {
        "ts": {"set": "primary", "primary": "run1", "run1": {"file": "ts.nc"}},
        "pr": {"set": "alt", "alt": "run2", "run2": {"file": "pr.nc"}},
    }
    output_file = tmp_path / "obs_catalogue.json"

    build_enso_obsvar_catalog(obs_dic, ["ts", "pr-200"], str(output_file))

    with open(output_file) as f:
        result = json.load(f)

    assert result == {
        "run1": {"ts": {"file": "ts.nc"}},
        "run2": {"pr": {"file": "pr.nc"}},
    }


def test_build_enso_obsvar_catalog_missing_key_raises():
    with pytest.raises(KeyError):
        build_enso_obsvar_catalog({}, ["sst"])


def test_build_enso_obsvar_catalog_resolves_source_alias(tmp_path):
    obs_dic = {
        "ts": {
            "set": "primary",
            "primary": "run1",
            "run1": {
                "var_name": "ts",
                "var_in_file": "ts",
                "file_path": "/data/run1.ts.198501-201412.nc",
            },
        }
    }
    output_file = tmp_path / "obs_catalogue.json"

    build_enso_obsvar_catalog(obs_dic, ["sst"], str(output_file))

    with open(output_file) as f:
        result = json.load(f)

    assert list(result["run1"]) == ["sst"]
    assert result["run1"]["sst"]["var_name"] == "sst"
    assert result["run1"]["sst"]["var_in_file"] == "ts"
    assert result["run1"]["sst"]["file_path"].endswith("run1.ts.198501-201412.nc")


def test_build_enso_obsvar_landmask(tmp_path):
    obs_dic = {
        "ts": {"set": "primary", "primary": "run1"},
    }
    output_file = tmp_path / "obs_landmask.json"

    build_enso_obsvar_landmask(obs_dic, ["ts"], str(output_file), mask_dir="fixed")

    with open(output_file) as f:
        result = json.load(f)

    assert result == {"run1": "fixed/sftlf.run1.nc"}


def test_build_enso_obsvar_landmask_resolves_source_alias(tmp_path):
    obs_dic = {"ts": {"set": "primary", "primary": "run1"}}
    output_file = tmp_path / "obs_landmask.json"

    build_enso_obsvar_landmask(obs_dic, ["sst"], str(output_file), mask_dir="fixed")

    with open(output_file) as f:
        result = json.load(f)

    assert result == {"run1": "fixed/sftlf.run1.nc"}


def test_normalize_enso_model_catalogue_adds_logical_variable_from_source(tmp_path):
    catalogue_file = tmp_path / "ts_enso_catalogue.json"
    catalogue = {
        "ts": {
            "set": "primary",
            "primary": "run1",
            "run1": {
                "var_name": "ts",
                "var_in_file": "ts",
                "file_path": "/data/run1.ts.198501_201412.nc",
                "template": "run1.ts.%(time).nc",
            },
        }
    }
    catalogue_file.write_text(json.dumps(catalogue))

    # "sst" is the logical ENSO variable name; the catalogue only has the
    # CMIP source variable "ts" (see ALT_OBS_MAP).
    normalize_enso_model_catalogue(["sst"], catalogue_file=str(catalogue_file))

    with open(catalogue_file) as f:
        result = json.load(f)

    assert "sst" in result
    sst_entry = result["sst"]["run1"]
    assert sst_entry["var_name"] == "sst"
    assert sst_entry["var_in_file"] == "ts"
    assert sst_entry["file_path"] == "/data/run1.sst.198501_201412.nc"
    assert sst_entry["template"] == "run1.sst.%(time).nc"


def test_normalize_enso_model_catalogue_rewrites_only_variable_component(tmp_path):
    catalogue_file = tmp_path / "ts_enso_catalogue.json"
    source_path = "/data/archive.ts/run1.ts.prefix.ts.198501_201412.nc"
    source_template = "run1.ts.prefix.ts.%(time).nc"
    catalogue = {
        "ts": {
            "set": "primary",
            "primary": "run1",
            "metadata": {"aliases": ["ts"]},
            "run1": {
                "var_name": "ts",
                "var_in_file": "ts",
                "file_path": source_path,
                "template": source_template,
            },
        }
    }
    catalogue_file.write_text(json.dumps(catalogue))

    normalize_enso_model_catalogue(["sst"], catalogue_file=str(catalogue_file))

    with open(catalogue_file) as f:
        result = json.load(f)

    assert result["sst"]["run1"]["file_path"] == (
        "/data/archive.ts/run1.ts.prefix.sst.198501_201412.nc"
    )
    assert result["sst"]["run1"]["template"] == ("run1.ts.prefix.sst.%(time).nc")
    assert result["ts"]["run1"]["file_path"] == source_path
    assert result["ts"]["run1"]["template"] == source_template


def test_normalize_enso_model_catalogue_skips_when_file_missing(tmp_path):
    missing_file = tmp_path / "does_not_exist.json"

    # Should not raise, and should not create the file.
    normalize_enso_model_catalogue(["sst"], catalogue_file=str(missing_file))

    assert not missing_file.exists()
