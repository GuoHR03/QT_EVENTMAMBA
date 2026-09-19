from tools.validate_source_text import validate_source_text


def test_validate_source_text_accepts_utf8(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "valid.py").write_text(
        'MESSAGE = "中文"\n',
        encoding="utf-8",
    )

    assert validate_source_text(tmp_path) == []


def test_validate_source_text_reports_invalid_utf8_and_replacement_character(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "invalid.py").write_bytes(b"\xff\xfe")
    (tmp_path / "README.md").write_text("broken: \ufffd\n", encoding="utf-8")

    errors = validate_source_text(tmp_path)

    assert any("invalid UTF-8" in error for error in errors)
    assert any("U+FFFD" in error for error in errors)
