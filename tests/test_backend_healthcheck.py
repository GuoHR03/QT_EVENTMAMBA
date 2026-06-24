from backend.backend_healthcheck import read_backend_log_tail


def test_read_backend_log_tail_decodes_and_trims(tmp_path):
    log_path = tmp_path / "backend.log"
    log_path.write_bytes(("a" * 20 + "就绪").encode("utf-8"))

    assert read_backend_log_tail(str(log_path), max_chars=4) == "aa就绪"


def test_read_backend_log_tail_returns_empty_for_missing_file(tmp_path):
    assert read_backend_log_tail(str(tmp_path / "missing.log")) == ""
