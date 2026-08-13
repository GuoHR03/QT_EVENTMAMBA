import json
import pickle
import re
from pathlib import Path

import numpy as np
import pytest

from backend import zmq_protocol as wire


_PICKLE_TRIPWIRE = {"executed": False}


def _activate_pickle_tripwire():
    _PICKLE_TRIPWIRE["executed"] = True


class _MaliciousPickle:
    def __reduce__(self):
        return (_activate_pickle_tripwire, ())


class _ReceivingSocket:
    def __init__(self, frames):
        self._frames = frames

    def recv_multipart(self):
        return self._frames


def _json_frame(msg_type="PING", **updates):
    message = {
        "protocol": wire.PROTOCOL_NAME,
        "version": wire.PROTOCOL_VERSION,
        "msg_type": msg_type,
    }
    message.update(updates)
    return json.dumps(message, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _event_array():
    return np.arange(
        wire.EVENT_SHAPE[0] * wire.EVENT_SHAPE[1],
        dtype=wire.EVENT_DTYPE,
    ).reshape(wire.EVENT_SHAPE)


def _event_frames(body=None, **metadata_updates):
    events = _event_array()
    metadata = {
        "timestamp": 123456789,
        "cropped": False,
        "dtype": wire.EVENT_DTYPE_NAME,
        "shape": list(wire.EVENT_SHAPE),
        "nbytes": wire.EVENT_BODY_BYTES,
    }
    metadata.update(metadata_updates)
    if body is None:
        body = events.tobytes(order="C")
    return [_json_frame("EVENTS", **metadata), body]


def _round_trip_request(message):
    decoded = wire.decode_message(wire.encode_message(message))
    assert wire.validate_request_message(decoded) is decoded
    return decoded


def _round_trip_response(message):
    decoded = wire.decode_message(wire.encode_response(message))
    assert wire.validate_response_message(decoded) is decoded
    return decoded


@pytest.mark.parametrize(
    "message",
    [
        {"msg_type": "PING", "instance_nonce": "nonce-123"},
        {
            "msg_type": "CONFIG",
            "width": 1280,
            "height": 720,
            "prediction_mode": "ellipse",
        },
    ],
)
def test_json_request_round_trip(message):
    assert _round_trip_request(message) == message


def test_events_request_round_trip_uses_fixed_binary_layout():
    events = _event_array()
    message = {
        "msg_type": "EVENTS",
        "data": events,
        "timestamp": 987654321,
        "cropped": True,
    }

    frames = wire.encode_message(message)
    decoded = wire.decode_message(frames)

    assert len(frames) == 2
    assert len(frames[1]) == wire.EVENT_BODY_BYTES
    assert wire.validate_request_message(decoded) is decoded
    assert decoded["timestamp"] == message["timestamp"]
    assert decoded["cropped"] is True
    assert decoded["data"].dtype == wire.EVENT_DTYPE
    assert decoded["data"].shape == wire.EVENT_SHAPE
    assert decoded["data"].flags.c_contiguous
    assert decoded["data"].flags.owndata
    np.testing.assert_array_equal(decoded["data"], events)


@pytest.mark.parametrize(
    "message",
    [
        {
            "msg_type": "READY",
            "status": "READY",
            "instance_nonce": "nonce-123",
            "pid": 4321,
        },
        {
            "msg_type": "STATUS",
            "message": "configured",
            "width": 1280,
            "height": 720,
            "mode": "center",
        },
        {
            "msg_type": "PREDICTION",
            "values": [0.25, 0.75],
            "cropped": False,
            "mode": "center",
        },
        {
            "msg_type": "PREDICTION",
            "values": [0.25, 0.75, 0.5, 0.1, 0.2],
            "cropped": True,
            "mode": "ellipse",
        },
        {
            "msg_type": "ERROR",
            "message": "invalid request",
            "code": "invalid_request",
        },
    ],
)
def test_response_round_trip(message):
    assert _round_trip_response(message) == message


@pytest.mark.parametrize(
    "frames",
    [
        [b"\xff"],
        [b"{"],
        [b"[]"],
        [b"null"],
    ],
)
def test_decode_rejects_invalid_utf8_json_and_non_object_headers(frames):
    with pytest.raises(wire.ZmqProtocolError):
        wire.decode_message(frames)


def test_decode_rejects_duplicate_json_keys():
    duplicate_header = (
        b'{"protocol":"eventmamba","version":1,"msg_type":"PING",'
        b'"msg_type":"CONFIG"}'
    )

    with pytest.raises(wire.ZmqProtocolError, match="duplicate"):
        wire.decode_message([duplicate_header])


@pytest.mark.parametrize(
    "updates",
    [
        {"protocol": "different-protocol"},
        {"version": 2},
        {"version": True},
        {"msg_type": "EXECUTE"},
        {"msg_type": 1},
    ],
)
def test_decode_rejects_unknown_protocol_version_and_message_type(updates):
    with pytest.raises(wire.ZmqProtocolError):
        wire.decode_message([_json_frame(**updates)])


def test_encode_rejects_unknown_message_type():
    with pytest.raises(wire.ZmqProtocolError, match="unknown message type"):
        wire.encode_message({"msg_type": "EXECUTE", "command": "whoami"})


@pytest.mark.parametrize(
    "frames",
    [
        [],
        [_json_frame(), b"", b""],
        [_json_frame(), b"unexpected"],
    ],
)
def test_decode_rejects_invalid_frame_counts(frames):
    with pytest.raises(wire.ZmqProtocolError):
        wire.decode_message(frames)


def test_decode_rejects_events_without_a_body_frame():
    with pytest.raises(wire.ZmqProtocolError, match="two frames"):
        wire.decode_message(_event_frames()[:1])


@pytest.mark.parametrize(
    "dtype_name",
    [">f4", "float32", "<f8", 4, None],
)
def test_decode_rejects_invalid_event_dtype_metadata(dtype_name):
    with pytest.raises(wire.ZmqProtocolError, match="dtype"):
        wire.decode_message(_event_frames(dtype=dtype_name))


@pytest.mark.parametrize(
    "shape",
    [
        [True, 3],
        [1024.0, 3],
        [2**62, 3],
        [1024],
        [1024, 3, 1],
        "1024x3",
        None,
    ],
)
def test_decode_rejects_invalid_event_shape_metadata(shape):
    with pytest.raises(wire.ZmqProtocolError, match="shape"):
        wire.decode_message(_event_frames(shape=shape))


@pytest.mark.parametrize(
    "nbytes",
    [
        True,
        float(wire.EVENT_BODY_BYTES),
        wire.EVENT_BODY_BYTES - 1,
        wire.EVENT_BODY_BYTES + 1,
        2**62,
        str(wire.EVENT_BODY_BYTES),
        None,
    ],
)
def test_decode_rejects_invalid_event_nbytes_metadata(nbytes):
    with pytest.raises(wire.ZmqProtocolError, match="byte count"):
        wire.decode_message(_event_frames(nbytes=nbytes))


@pytest.mark.parametrize("size_delta", [-1, 1])
def test_decode_rejects_wrong_event_body_length(size_delta):
    body = b"\x00" * (wire.EVENT_BODY_BYTES + size_delta)

    with pytest.raises(wire.ZmqProtocolError, match="length"):
        wire.decode_message(_event_frames(body=body))


@pytest.mark.parametrize("non_finite", [np.nan, np.inf, -np.inf])
def test_encode_and_decode_reject_non_finite_event_values(non_finite):
    events = _event_array()
    events[0, 0] = non_finite
    message = {
        "msg_type": "EVENTS",
        "data": events,
        "timestamp": 1,
        "cropped": False,
    }

    with pytest.raises(wire.ZmqProtocolError, match="finite"):
        wire.encode_message(message)
    with pytest.raises(wire.ZmqProtocolError, match="finite"):
        wire.decode_message(_event_frames(body=events.tobytes(order="C")))


@pytest.mark.parametrize(
    "events",
    [
        np.zeros((1024, 3), dtype=np.float64),
        np.zeros((1023, 3), dtype=wire.EVENT_DTYPE),
        np.zeros((3, 1024), dtype=wire.EVENT_DTYPE).T,
    ],
)
def test_encode_rejects_invalid_event_array_layout(events):
    message = {
        "msg_type": "EVENTS",
        "data": events,
        "timestamp": 1,
        "cropped": False,
    }

    with pytest.raises(wire.ZmqProtocolError):
        wire.encode_message(message)


def test_oversized_header_is_rejected_before_json_parsing(monkeypatch):
    oversized_header = b"{" + (b" " * wire.MAX_HEADER_BYTES)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("json.loads must not inspect an oversized header")

    monkeypatch.setattr(wire.json, "loads", fail_if_called)

    with pytest.raises(wire.ZmqProtocolError, match="header exceeds"):
        wire.decode_message([oversized_header])


@pytest.mark.parametrize("receiver", [wire.receive_request, wire.receive_response])
def test_malicious_pickle_is_rejected_without_executing_reduce_target(receiver):
    _PICKLE_TRIPWIRE["executed"] = False
    payload = pickle.dumps(_MaliciousPickle(), protocol=pickle.HIGHEST_PROTOCOL)
    assert _PICKLE_TRIPWIRE["executed"] is False

    with pytest.raises(wire.ZmqProtocolError):
        receiver(_ReceivingSocket([payload]))

    assert _PICKLE_TRIPWIRE["executed"] is False


def test_production_zmq_code_contains_no_pickle_or_pyobj_api():
    project_root = Path(__file__).resolve().parents[1]
    source_files = list((project_root / "backend").rglob("*.py"))
    source_files.extend(
        path
        for path in (project_root / "linux_backend.py", project_root / "windows_backend.py")
        if path.exists()
    )
    source_files.extend((project_root / "tools").glob("*.py"))

    import_pattern = re.compile(
        r"(?m)^\s*(?:import\s+[^#\n]*\bpickle\b|from\s+pickle\s+import\b)"
    )
    banned_tokens = (
        "send_pyobj",
        "recv_pyobj",
        "pickle.loads",
        "pickle.dumps",
        "zmq.CONFLATE",
    )
    violations = []

    for path in sorted(source_files):
        source = path.read_text(encoding="utf-8-sig")
        if import_pattern.search(source):
            violations.append("%s: pickle import" % path.relative_to(project_root))
        for token in banned_tokens:
            if token in source:
                violations.append("%s: %s" % (path.relative_to(project_root), token))

    assert not violations, "unsafe ZMQ serialization API remains:\n%s" % "\n".join(violations)


def test_linux_backend_does_not_bind_inference_to_all_interfaces():
    project_root = Path(__file__).resolve().parents[1]
    source = (project_root / "linux_backend.py").read_text(encoding="utf-8-sig")

    assert "0.0.0.0" not in source
    assert '"127.0.0.1"' in source
