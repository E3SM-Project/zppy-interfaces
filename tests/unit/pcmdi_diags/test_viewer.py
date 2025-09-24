from zppy_interfaces.pcmdi_diags.viewer import safe_join


def test_safe_join():
    assert safe_join("a", "b") == "a/b"
    assert safe_join("a/", "b") == "a/b"
