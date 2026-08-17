from zppy_interfaces.pcmdi_diags.viewer import (
    CMVARGroupBuilder,
    generate_cmvar_table,
    generate_data_html,
    generate_emovs_table,
    safe_join,
)


def test_safe_join():
    assert safe_join("a", "b") == "a/b"
    assert safe_join("a/", "b") == "a/b"


def test_coupled_mov_compose_eof_uses_mode_eof(tmp_path):
    fig_dir = tmp_path / "figures"
    diag_dir = tmp_path / "viewer"
    yearly_dir = fig_dir / "MOV_compose" / "yearly"
    yearly_dir.mkdir(parents=True)
    diag_dir.mkdir()

    (yearly_dir / "MOV_compose_AMO_yearly_eof1.png").touch()
    (yearly_dir / "MOV_compose_PDO_yearly_eof1.png").touch()
    (yearly_dir / "MOV_compose_NPGO_yearly_eof2.png").touch()

    builder = CMVARGroupBuilder()

    amo_row = builder.generate_mcpl_row("AMO", str(diag_dir), str(fig_dir))
    amo_compose_cell = amo_row[2]["content"]
    assert "AMO (SST)" in amo_row[0]["content"]
    assert "MOV_compose_AMO_yearly_eof1.png" in amo_compose_cell

    pdo_row = builder.generate_mcpl_row("PDO", str(diag_dir), str(fig_dir))
    pdo_compose_cell = pdo_row[2]["content"]
    assert "PDO (SST)" in pdo_row[0]["content"]
    assert "MOV_compose_PDO_yearly_eof1.png" in pdo_compose_cell
    assert "MOV_compose_PDO_yearly_cbf.png" not in pdo_compose_cell

    npgo_row = builder.generate_mcpl_row("NPGO", str(diag_dir), str(fig_dir))
    npgo_compose_cell = npgo_row[2]["content"]
    assert "NPGO (SST)" in npgo_row[0]["content"]
    assert "MOV_compose_NPGO_yearly_eof2.png" in npgo_compose_cell
    assert "MOV_compose_NPGO_yearly_eof1.png" not in npgo_compose_cell


def test_coupled_modes_are_normalized_from_config(tmp_path):
    fig_dir = tmp_path / "figures"
    diag_dir = tmp_path / "viewer"
    yearly_dir = fig_dir / "MOV_metric" / "MOV_compose" / "yearly"
    yearly_dir.mkdir(parents=True)
    diag_dir.mkdir()

    (yearly_dir / "MOV_compose_NPGO_yearly_eof2.png").touch()

    table = generate_cmvar_table(
        str(diag_dir),
        str(fig_dir),
        enso_show=False,
        movc_show=True,
        movc_modes=" npgo ",
    )

    assert len(table) == 1
    assert "NPGO (SST)" in table[0][0]["content"]
    assert "MOV_compose_NPGO_yearly_eof2.png" in table[0][2]["content"]
    assert "MOV_compose_NPGO_yearly_eof1.png" not in table[0][2]["content"]


def test_emovs_table_defaults_to_atmospheric_modes(tmp_path):
    table = generate_emovs_table(str(tmp_path / "viewer"), str(tmp_path / "figures"))

    first_cells = [row[0]["content"] for row in table if row and "rowspan" in row[0]]
    assert first_cells == [
        "<b>NAM (PSL)</b>",
        "<b>PNA (PSL)</b>",
        "<b>NPO (PSL)</b>",
        "<b>NAO (PSL)</b>",
        "<b>SAM (PSL)</b>",
        "<b>PSA1 (PSL)</b>",
        "<b>PSA2 (PSL)</b>",
    ]


def test_emovs_modes_are_normalized_from_config(tmp_path):
    fig_dir = tmp_path / "figures"
    compose_dir = fig_dir / "MOV_metric" / "MOV_compose" / "DJF"
    compose_dir.mkdir(parents=True)
    (compose_dir / "MOV_compose_NPO_DJF_eof2.png").touch()
    (compose_dir / "MOV_compose_PSA2_DJF_eof3.png").touch()

    table = generate_emovs_table(
        str(tmp_path / "viewer"),
        str(fig_dir),
        modes=" npo, psa2 ",
    )

    first_cells = [row[0]["content"] for row in table if row and "rowspan" in row[0]]
    assert first_cells == ["<b>NPO (PSL)</b>", "<b>PSA2 (PSL)</b>"]
    assert "MOV_compose_NPO_DJF_eof2.png" in table[1][1]["content"]
    assert "MOV_compose_PSA2_DJF_eof3.png" in table[10][1]["content"]


def test_generate_data_html_creates_out_dir_and_keeps_string_lists(tmp_path):
    template_dir = tmp_path / "templates"
    out_dir = tmp_path / "viewer"
    template_dir.mkdir(parents=True)
    (template_dir / "data_template.html").write_text(
        "{% for section in sections %}"
        "{% for row in section.rows %}{{ row.description }}\n{% endfor %}"
        "{% endfor %}"
    )

    out_path = generate_data_html(
        {
            "template_dir": str(template_dir),
            "out_dir": str(out_dir),
            "clim_viewer": True,
            "clim_vars": "pr,tas",
        }
    )

    html = (out_dir / "diag_data.html").read_text()
    assert out_path == str(out_dir / "diag_data.html")
    assert out_dir.is_dir()
    assert "pr,tas" in html
    assert "p, r" not in html
