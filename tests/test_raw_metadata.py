import json
import struct

from backend.raw_metadata import raw_duration_from_sidecar, raw_info_sidecar_path, raw_tmp_index_path


def test_raw_duration_from_sidecar_reads_metavision_info_json(tmp_path):
    raw_path = tmp_path / "recording.raw"
    sidecar_path = raw_info_sidecar_path(raw_path)
    sidecar_path.write_text(json.dumps({"duration": 654321}), encoding="utf-8")

    assert raw_duration_from_sidecar(raw_path) == 654321


def test_raw_duration_from_sidecar_reads_tmp_index_when_info_json_missing(tmp_path):
    raw_path = tmp_path / "recording.raw"
    raw_path.write_bytes(b"\0" * 500)
    tmp_index_path = raw_tmp_index_path(raw_path)
    tmp_index_path.write_bytes(
        b"% bookmark_period_us 2000\n% index_version 2.0\n% end\n"
        + struct.pack("<QqI", 2000, 100, 10)
        + struct.pack("<QqI", 4000, 450, 20)
        + struct.pack("<QqI", 6000, 9999, 30)
    )

    assert raw_duration_from_sidecar(raw_path) == 4000


def test_raw_duration_from_sidecar_ignores_tmp_index_sentinel_bookmarks(tmp_path):
    raw_path = tmp_path / "recording.raw"
    raw_path.write_bytes(b"\0" * 500)
    raw_tmp_index_path(raw_path).write_bytes(
        b"% bookmark_period_us 2000\n% index_version 2.0\n% end\n"
        + struct.pack("<QqI", (1 << 64) - 1, 289, 0)
        + struct.pack("<QqI", 2000, 100, 10)
        + struct.pack("<QqI", 4000, 450, 20)
    )

    assert raw_duration_from_sidecar(raw_path) == 4000


def test_raw_duration_from_sidecar_prefers_tmp_index_over_info_json(tmp_path):
    raw_path = tmp_path / "recording.raw"
    raw_path.write_bytes(b"\0" * 500)
    raw_info_sidecar_path(raw_path).write_text(json.dumps({"duration": 1234}), encoding="utf-8")
    raw_tmp_index_path(raw_path).write_bytes(
        b"% bookmark_period_us 2000\n% end\n"
        + struct.pack("<QqI", 4000, 450, 20)
    )

    assert raw_duration_from_sidecar(raw_path) == 4000


def test_raw_duration_from_sidecar_returns_zero_when_missing(tmp_path):
    assert raw_duration_from_sidecar(tmp_path / "missing.raw") == 0
