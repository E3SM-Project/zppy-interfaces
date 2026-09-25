from typing import List

import pytest

from zppy_interfaces.pcmdi_diags.pcmdi_mean_climate import (
    MeanClimateMetricsCollector,
    MeanClimateParameters,
    _move_output,
    generate_mean_clim_cmds,
)


def test_MeanClimateParameters_normalizes_regions():
    parameters = MeanClimateParameters({"regions": " global, ocean, ,land "})

    assert parameters.regions == ["global", "ocean", "land"]


@pytest.mark.parametrize("regions", [None, "", " , "])
def test_MeanClimateParameters_requires_nonempty_regions(regions):
    with pytest.raises(ValueError, match="regions"):
        MeanClimateParameters({"regions": regions})


def test_generate_mean_clim_cmds():
    # Sample of mean_climate default vars in zppy/defaults/default.ini
    variables: List[str] = ["tauu", "tauv", "ta-200"]
    # Example: /lcrc/group/e3sm/ac.forsyth2/zppy_pr719_output/unique_id_48/v3.LR.amip_0101/post/scripts/tmp.pcmdi_diags_mean_climate_model_vs_obs_2005-2014.915900.07Jq/pcmdi_diags/climo_ref_mean_climate_catalogue.json
    obs_dic = {
        "tauu": {"set": "default"},
        "tauv": {"set": "default"},
        "ta": {"set": "default"},  # Not in the example json file above
    }
    # Example: Appears after "AC" in .nc files in /lcrc/group/e3sm/ac.forsyth2/zppy_pr719_output/unique_id_48/v3.LR.amip_0101/post/scripts/tmp.pcmdi_diags_mean_climate_model_vs_obs_2005-2014.915900.07Jq/climo
    case_id: str = "v20250923"
    actual = generate_mean_clim_cmds(
        variables=variables,
        obs_dic=obs_dic,
        case_id=case_id,
    )
    expected: List[str] = [
        "mean_climate_driver.py -p parameterfile.py --vars tauu -r default --case_id v20250923",
        "mean_climate_driver.py -p parameterfile.py --vars tauv -r default --case_id v20250923",
        "mean_climate_driver.py -p parameterfile.py --vars ta-200 -r default --case_id v20250923",
    ]
    assert actual == expected


def test_generate_mean_clim_cmds_fails_when_no_variables_can_be_resolved():
    with pytest.raises(ValueError, match="No mean-climate commands"):
        generate_mean_clim_cmds(
            variables=["missing"],
            obs_dic={"pr": {"set": "default"}},
            case_id="v20250923",
        )


def test_MeanClimateMetricsCollector_rejects_malformed_metrics_filename(tmp_path):
    metrics_dir = tmp_path / "metrics_results" / "mean_climate"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "malformed.json").touch()
    collector = MeanClimateMetricsCollector(
        regions=["global"],
        variables=["pr"],
        fig_format="png",
        model_info=("CMIP6", "historical", "E3SM", "r1i1p1f1"),
        case_id="v20250923",
        input_template=str(tmp_path / "%(output_type)" / "%(metric_type)"),
        output_dir=str(tmp_path / "output" / "%(group_type)"),
    )

    with pytest.raises(ValueError, match="Unexpected metrics filename format"):
        collector._collect_metrics()


def test_move_output_replaces_file_and_rejects_directory(tmp_path):
    source = tmp_path / "source.txt"
    destination = tmp_path / "destination.txt"
    source.write_text("new")
    destination.write_text("old")

    _move_output(str(source), str(destination))

    assert destination.read_text() == "new"
    assert not source.exists()

    next_source = tmp_path / "next.txt"
    directory_destination = tmp_path / "directory"
    next_source.touch()
    directory_destination.mkdir()
    with pytest.raises(IsADirectoryError):
        _move_output(str(next_source), str(directory_destination))
    assert next_source.exists()
