from zppy_interfaces.pcmdi_diags.pcmdi_setup import (
    DataCatalogueBuilder,
    LandSeaMaskGenerator,
)


def test_DataCatalogueBuilder():
    dcb = DataCatalogueBuilder("", [], "", [], [], "", "")

    assert dcb._get_base_varname("ta-200") == "ta"
    assert dcb._get_base_varname("ta_200") == "ta"
    assert dcb._get_base_varname("pr") == "pr"


def test_LandSeaMaskGenerator():
    lsmg = LandSeaMaskGenerator("", "", "", "")
    assert lsmg._parse_flag("True")
    assert lsmg._parse_flag("Y")
    assert lsmg._parse_flag("Yes")
    assert lsmg._parse_flag("true")
    assert lsmg._parse_flag("y")
    assert lsmg._parse_flag("yes")
    assert not lsmg._parse_flag("False")
