from typing import List

from zppy_interfaces.pcmdi_diags.pcmdi_mean_cimate import generate_mean_clim_cmds


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
