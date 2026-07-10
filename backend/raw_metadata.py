import json
import logging
import struct
from pathlib import Path

LOGGER = logging.getLogger(__name__)
TMP_INDEX_RECORD_DTYPE = "<QqI"
TMP_INDEX_RECORD_SIZE = struct.calcsize(TMP_INDEX_RECORD_DTYPE)
TMP_INDEX_INVALID_TIMESTAMP = (1 << 64) - 1


def raw_info_sidecar_path(input_path):
    raw_path = Path(input_path)
    return raw_path.with_name(f"{raw_path.stem}_info.json")


def raw_tmp_index_path(input_path):
    raw_path = Path(input_path)
    return raw_path.with_name(f"{raw_path.name}.tmp_index")


def raw_duration_from_sidecar(input_path):
    tmp_index_duration = raw_duration_from_tmp_index(input_path)
    if tmp_index_duration > 0:
        return tmp_index_duration
    return raw_duration_from_info_json(input_path)


def raw_duration_from_info_json(input_path):
    info_path = raw_info_sidecar_path(input_path)
    if not info_path.exists():
        return 0
    try:
        with info_path.open("r", encoding="utf-8") as handle:
            info = json.load(handle)
    except Exception:
        return 0
    try:
        return max(0, int(info.get("duration", 0)))
    except (TypeError, ValueError):
        return 0


def raw_duration_from_tmp_index(input_path):
    tmp_index_path = raw_tmp_index_path(input_path)
    if not tmp_index_path.exists():
        return 0

    try:
        payload = tmp_index_path.read_bytes()
    except OSError:
        return 0

    payload_start = _tmp_index_payload_start(payload)
    if payload_start is None:
        return 0

    raw_size = _raw_file_size(input_path)
    max_timestamp_us = 0
    record_count = (len(payload) - payload_start) // TMP_INDEX_RECORD_SIZE
    for index in range(record_count):
        offset = payload_start + index * TMP_INDEX_RECORD_SIZE
        timestamp_us, byte_offset, _event_count = struct.unpack_from(TMP_INDEX_RECORD_DTYPE, payload, offset)
        if _valid_tmp_index_bookmark(timestamp_us, byte_offset, raw_size):
            max_timestamp_us = max(max_timestamp_us, int(timestamp_us))
    return max_timestamp_us


def compute_raw_duration(input_path):
    duration = raw_duration_from_sidecar(input_path)
    if duration > 0:
        return duration

    try:
        from metavision_core.event_io.raw_info import get_raw_info

        info = get_raw_info(input_path)
    except Exception as exc:
        LOGGER.warning("Failed to compute RAW duration for %s: %s", input_path, exc)
        return 0

    try:
        return max(0, int(info.get("duration", 0)))
    except (AttributeError, TypeError, ValueError):
        return 0


def _tmp_index_payload_start(payload):
    for marker in (b"% end\n", b"% end\r\n"):
        header_end = payload.find(marker)
        if header_end >= 0:
            return header_end + len(marker)
    return None


def _raw_file_size(input_path):
    try:
        return Path(input_path).stat().st_size
    except OSError:
        return None


def _valid_tmp_index_bookmark(timestamp_us, byte_offset, raw_size):
    if timestamp_us <= 0 or timestamp_us == TMP_INDEX_INVALID_TIMESTAMP or byte_offset < 0:
        return False
    if raw_size is not None and byte_offset > raw_size:
        return False
    return True
