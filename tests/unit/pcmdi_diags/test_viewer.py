from zppy_interfaces.pcmdi_diags.viewer import (
    CMVARGroupBuilder,
    SummaryTableBuilder,
    generate_cmvar_table,
    generate_data_html,
    generate_emovs_table,
    generate_summary_table,
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


def test_summary_table_builder_enso_row_links_to_existing_figure(tmp_path):
    fig_dir = tmp_path / "figures"
    diag_dir = tmp_path / "viewer"
    enso_dir = fig_dir / "ERROR_metric" / "enso_metric"
    enso_dir.mkdir(parents=True)
    diag_dir.mkdir()
    (enso_dir / "enso_metric_skill_portrait.png").touch()

    builder = SummaryTableBuilder(str(diag_dir), str(fig_dir))
    row = builder.build_enso_row()

    assert row[0]["content"] == "<b>ENSO</b>"
    assert row[1]["content"] == "TROPICS"
    assert "enso_metric_skill_portrait.png" in row[2]["content"]
    assert "color:gray" not in row[2]["content"]


def test_summary_table_builder_enso_row_greys_out_missing_figure(tmp_path):
    fig_dir = tmp_path / "figures"
    diag_dir = tmp_path / "viewer"
    fig_dir.mkdir()
    diag_dir.mkdir()

    builder = SummaryTableBuilder(str(diag_dir), str(fig_dir))
    row = builder.build_enso_row()

    assert "color:gray" in row[2]["content"]


def test_generate_summary_table_includes_enso_row_when_enabled(tmp_path):
    table = generate_summary_table(
        str(tmp_path / "viewer"),
        str(tmp_path / "figures"),
        clim_show=False,
        mova_show=False,
        movc_show=False,
        enso_show=True,
    )

    assert len(table) == 1
    assert table[0][0]["content"] == "<b>ENSO</b>"


def test_generate_summary_table_omits_enso_row_when_disabled(tmp_path):
    table = generate_summary_table(
        str(tmp_path / "viewer"),
        str(tmp_path / "figures"),
        clim_show=False,
        mova_show=False,
        movc_show=False,
        enso_show=False,
    )

    assert table == []


def test_generate_cmvar_table_enso_block_builds_perf_proc_telec_groups(tmp_path):
    fig_dir = tmp_path / "figures"
    diag_dir = tmp_path / "viewer"
    fig_dir.mkdir()
    diag_dir.mkdir()

    table = generate_cmvar_table(
        str(diag_dir),
        str(fig_dir),
        enso_show=True,
        movc_show=False,
    )

    section_labels = [row[0]["content"] for row in table if "rowspan" in row[0]]
    assert "<b>ENSO Perf</b>" in section_labels
    assert "<b>ENSO Proc</b>" in section_labels
    assert "<b>ENSO Telec</b>" in section_labels

    # Spot check that a known ENSO_perf variable shows up in a cell.
    assert any(
        "BiasPrLat" in cell["content"]
        for row in table
        for cell in row
        if "content" in cell
    )


def test_generate_cmvar_table_enso_disabled_skips_enso_rows(tmp_path):
    fig_dir = tmp_path / "figures"
    diag_dir = tmp_path / "viewer"
    fig_dir.mkdir()
    diag_dir.mkdir()

    table = generate_cmvar_table(
        str(diag_dir),
        str(fig_dir),
        enso_show=False,
        movc_show=False,
    )

    assert table == []


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


def test_generate_data_html_includes_enso_section_when_enabled(tmp_path):
    template_dir = tmp_path / "templates"
    out_dir = tmp_path / "viewer"
    template_dir.mkdir(parents=True)
    (template_dir / "data_template.html").write_text(
        "{% for section in sections %}" "{{ section.title }}\n" "{% endfor %}"
    )

    generate_data_html(
        {
            "template_dir": str(template_dir),
            "out_dir": str(out_dir),
            "enso_viewer": True,
            "enso_vars": "ts,tauu",
            "enso_period": "1985-2014",
        }
    )

    html = (out_dir / "diag_data.html").read_text()
    assert "ENSO Metrics Data (1985-2014)" in html


def test_generate_data_html_omits_enso_section_when_disabled(tmp_path):
    template_dir = tmp_path / "templates"
    out_dir = tmp_path / "viewer"
    template_dir.mkdir(parents=True)
    (template_dir / "data_template.html").write_text(
        "{% for section in sections %}" "{{ section.title }}\n" "{% endfor %}"
    )

    generate_data_html(
        {
            "template_dir": str(template_dir),
            "out_dir": str(out_dir),
            "enso_viewer": False,
        }
    )

    html = (out_dir / "diag_data.html").read_text()
    assert "ENSO" not in html
