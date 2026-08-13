import json
import math

import numpy as np
import zmq


PROTOCOL_NAME = "eventmamba"
PROTOCOL_VERSION = 1

MAX_HEADER_BYTES = 16 * 1024
MAX_MESSAGE_BYTES = 64 * 1024
MAX_TEXT_CHARS = 4096
MAX_NONCE_CHARS = 256
MAX_SENSOR_DIMENSION = 16384

EVENT_SHAPE = (1024, 3)
EVENT_DTYPE_NAME = "<f4"
EVENT_DTYPE = np.dtype(EVENT_DTYPE_NAME)
EVENT_BODY_BYTES = int(np.prod(EVENT_SHAPE)) * EVENT_DTYPE.itemsize

REQUEST_MESSAGE_TYPES = frozenset(("PING", "CONFIG", "EVENTS"))
RESPONSE_MESSAGE_TYPES = frozenset(("READY", "STATUS", "PREDICTION", "ERROR"))
DIAGNOSTIC_MESSAGE_TYPES = frozenset(("ACK", "STOP", "STOPPED"))
MESSAGE_TYPES = REQUEST_MESSAGE_TYPES | RESPONSE_MESSAGE_TYPES | DIAGNOSTIC_MESSAGE_TYPES


class ZmqProtocolError(ValueError):
    """Raised when a ZMQ wire message does not match the safe protocol."""


def configure_socket_limits(socket):
    """Bound the size of every frame before Python receives its contents."""
    try:
        socket.setsockopt(zmq.MAXMSGSIZE, MAX_MESSAGE_BYTES)
    except (AttributeError, zmq.ZMQError):
        # Older libzmq builds may not expose MAXMSGSIZE. The decoder still
        # enforces the same bound after receipt.
        pass


def encode_message(message):
    """Encode one internal message as strict JSON plus an optional event frame."""
    if not isinstance(message, dict):
        raise ZmqProtocolError("message must be an object")

    header = dict(message)
    for reserved_key in ("protocol", "version", "dtype", "shape", "nbytes"):
        if reserved_key in header:
            raise ZmqProtocolError("message contains reserved wire fields")

    msg_type = header.get("msg_type")
    if type(msg_type) is not str or msg_type not in MESSAGE_TYPES:
        raise ZmqProtocolError("unknown message type")

    body = None
    if msg_type == "EVENTS":
        if "data" not in header:
            raise ZmqProtocolError("EVENTS message is missing data")
        events = np.asarray(header.pop("data"))
        if events.shape != EVENT_SHAPE:
            raise ZmqProtocolError("EVENTS data has an invalid shape")
        if events.dtype != EVENT_DTYPE:
            raise ZmqProtocolError("EVENTS data must use little-endian float32")
        if not events.flags.c_contiguous:
            raise ZmqProtocolError("EVENTS data must be C-contiguous")
        if not np.isfinite(events).all():
            raise ZmqProtocolError("EVENTS data must contain only finite values")
        body = events.tobytes(order="C")
        header.update(
            {
                "dtype": EVENT_DTYPE_NAME,
                "shape": list(EVENT_SHAPE),
                "nbytes": EVENT_BODY_BYTES,
            }
        )
    elif "data" in header:
        raise ZmqProtocolError("only EVENTS messages may contain binary data")

    header["protocol"] = PROTOCOL_NAME
    header["version"] = PROTOCOL_VERSION
    try:
        header_bytes = json.dumps(
            header,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
            default=_json_default,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ZmqProtocolError("message is not JSON serializable") from exc

    if not header_bytes or len(header_bytes) > MAX_HEADER_BYTES:
        raise ZmqProtocolError("JSON header exceeds the protocol limit")

    frames = [header_bytes]
    if body is not None:
        frames.append(body)
    if sum(len(frame) for frame in frames) > MAX_MESSAGE_BYTES:
        raise ZmqProtocolError("message exceeds the protocol limit")
    return frames


def decode_message(frames):
    """Decode frames without executing payload-controlled Python code."""
    if not isinstance(frames, (list, tuple)):
        raise ZmqProtocolError("wire message must be a frame sequence")
    if len(frames) not in (1, 2):
        raise ZmqProtocolError("wire message has an invalid frame count")

    checked_frames = []
    total_bytes = 0
    for frame in frames:
        if not isinstance(frame, (bytes, bytearray, memoryview)):
            raise ZmqProtocolError("wire frame must contain bytes")
        frame_size = len(frame)
        total_bytes += frame_size
        if total_bytes > MAX_MESSAGE_BYTES:
            raise ZmqProtocolError("message exceeds the protocol limit")
        checked_frames.append(frame)

    header_frame = checked_frames[0]
    if not header_frame or len(header_frame) > MAX_HEADER_BYTES:
        raise ZmqProtocolError("JSON header exceeds the protocol limit")
    try:
        header_text = bytes(header_frame).decode("utf-8", errors="strict")
        message = json.loads(
            header_text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ZmqProtocolError, RecursionError) as exc:
        if isinstance(exc, ZmqProtocolError):
            raise
        raise ZmqProtocolError("invalid JSON header") from exc

    if not isinstance(message, dict):
        raise ZmqProtocolError("JSON header must be an object")
    if message.get("protocol") != PROTOCOL_NAME:
        raise ZmqProtocolError("unknown protocol")
    if type(message.get("version")) is not int or message["version"] != PROTOCOL_VERSION:
        raise ZmqProtocolError("unsupported protocol version")

    msg_type = message.get("msg_type")
    if type(msg_type) is not str or msg_type not in MESSAGE_TYPES:
        raise ZmqProtocolError("unknown message type")

    if msg_type == "EVENTS":
        if len(checked_frames) != 2:
            raise ZmqProtocolError("EVENTS message must contain two frames")
        _validate_event_metadata(message)
        body = checked_frames[1]
        if len(body) != EVENT_BODY_BYTES:
            raise ZmqProtocolError("EVENTS body has an invalid length")
        events = np.frombuffer(body, dtype=EVENT_DTYPE, count=int(np.prod(EVENT_SHAPE)))
        events = events.reshape(EVENT_SHAPE).copy(order="C")
        if not np.isfinite(events).all():
            raise ZmqProtocolError("EVENTS data must contain only finite values")
        message["data"] = events
        for key in ("dtype", "shape", "nbytes"):
            message.pop(key, None)
    else:
        if len(checked_frames) != 1:
            raise ZmqProtocolError("JSON-only message contains an unexpected body")
        if any(key in message for key in ("data", "dtype", "shape", "nbytes")):
            raise ZmqProtocolError("JSON-only message contains binary metadata")

    message.pop("protocol", None)
    message.pop("version", None)
    return message


def send_message(socket, message):
    socket.send_multipart(encode_message(message))


def receive_message(socket):
    return decode_message(socket.recv_multipart())


def send_request(socket, message):
    validate_request_message(message)
    send_message(socket, message)


def receive_request(socket):
    message = receive_message(socket)
    validate_request_message(message)
    return message


def send_response(socket, message):
    socket.send_multipart(encode_response(message))


def encode_response(message):
    validate_response_message(message)
    return encode_message(message)


def receive_response(socket):
    message = receive_message(socket)
    validate_response_message(message)
    return message


def validate_request_message(message):
    if not isinstance(message, dict):
        raise ZmqProtocolError("request must be an object")
    msg_type = message.get("msg_type")
    if type(msg_type) is not str or msg_type not in REQUEST_MESSAGE_TYPES:
        raise ZmqProtocolError("unknown request type")

    if msg_type == "PING":
        _require_keys(message, ("msg_type",), ("instance_nonce",))
        if "instance_nonce" in message:
            _validate_nonce(message["instance_nonce"], allow_none=False)
        return message

    if msg_type == "CONFIG":
        _require_keys(
            message,
            ("msg_type", "width", "height", "prediction_mode"),
        )
        _validate_dimension(message["width"], "width")
        _validate_dimension(message["height"], "height")
        _validate_mode(message["prediction_mode"])
        return message

    _require_keys(message, ("msg_type", "data", "timestamp", "cropped"))
    events = message["data"]
    if not isinstance(events, np.ndarray):
        raise ZmqProtocolError("EVENTS data must be a NumPy array")
    if events.shape != EVENT_SHAPE:
        raise ZmqProtocolError("EVENTS data has an invalid shape")
    if events.dtype != EVENT_DTYPE:
        raise ZmqProtocolError("EVENTS data must use little-endian float32")
    if not events.flags.c_contiguous:
        raise ZmqProtocolError("EVENTS data must be C-contiguous")
    if not np.isfinite(events).all():
        raise ZmqProtocolError("EVENTS data must contain only finite values")
    timestamp = message["timestamp"]
    if type(timestamp) is not int or timestamp < 0 or timestamp > (2**63 - 1):
        raise ZmqProtocolError("timestamp must be a non-negative int64")
    if type(message["cropped"]) is not bool:
        raise ZmqProtocolError("cropped must be a boolean")
    return message


def validate_response_message(message):
    if not isinstance(message, dict):
        raise ZmqProtocolError("response must be an object")
    msg_type = message.get("msg_type")
    if type(msg_type) is not str or msg_type not in RESPONSE_MESSAGE_TYPES:
        raise ZmqProtocolError("unknown response type")

    if msg_type == "READY":
        _require_keys(
            message,
            ("msg_type", "status", "instance_nonce", "pid"),
        )
        if message["status"] != "READY":
            raise ZmqProtocolError("READY response has an invalid status")
        _validate_nonce(message["instance_nonce"], allow_none=True)
        pid = message["pid"]
        if type(pid) is not int or pid <= 0 or pid > (2**63 - 1):
            raise ZmqProtocolError("READY response has an invalid pid")
        return message

    if msg_type in ("STATUS", "ERROR"):
        optional = ("width", "height", "mode") if msg_type == "STATUS" else ("code",)
        _require_keys(message, ("msg_type", "message"), optional)
        _validate_text(message["message"], "message")
        if msg_type == "STATUS":
            if "width" in message:
                _validate_dimension(message["width"], "width")
            if "height" in message:
                _validate_dimension(message["height"], "height")
            if "mode" in message:
                _validate_mode(message["mode"])
        elif "code" in message:
            _validate_text(message["code"], "code", max_chars=128)
        return message

    _require_keys(message, ("msg_type", "values", "cropped", "mode"))
    _validate_mode(message["mode"])
    if type(message["cropped"]) is not bool:
        raise ZmqProtocolError("cropped must be a boolean")
    values = message["values"]
    expected_length = 2 if message["mode"] == "center" else 5
    if not isinstance(values, (list, tuple)) or len(values) != expected_length:
        raise ZmqProtocolError("prediction has an invalid value count")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise ZmqProtocolError("prediction values must be numeric")
        if not math.isfinite(float(value)):
            raise ZmqProtocolError("prediction values must be finite")
    return message


def _validate_event_metadata(message):
    dtype_name = message.get("dtype")
    if type(dtype_name) is not str or dtype_name != EVENT_DTYPE_NAME:
        raise ZmqProtocolError("EVENTS metadata has an invalid dtype")
    shape = message.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != len(EVENT_SHAPE)
        or any(type(value) is not int for value in shape)
        or tuple(shape) != EVENT_SHAPE
    ):
        raise ZmqProtocolError("EVENTS metadata has an invalid shape")
    nbytes = message.get("nbytes")
    if type(nbytes) is not int or nbytes != EVENT_BODY_BYTES:
        raise ZmqProtocolError("EVENTS metadata has an invalid byte count")


def _require_keys(message, required, optional=()):
    required = frozenset(required)
    allowed = required | frozenset(optional)
    actual = frozenset(message)
    if not required.issubset(actual) or not actual.issubset(allowed):
        raise ZmqProtocolError("message fields do not match the protocol schema")


def _validate_dimension(value, label):
    if type(value) is not int or not 1 <= value <= MAX_SENSOR_DIMENSION:
        raise ZmqProtocolError("%s is outside the supported range" % label)


def _validate_mode(value):
    if value not in ("center", "ellipse"):
        raise ZmqProtocolError("unsupported prediction mode")


def _validate_nonce(value, allow_none):
    if allow_none and value is None:
        return
    if not isinstance(value, str) or not value or len(value) > MAX_NONCE_CHARS:
        raise ZmqProtocolError("instance nonce is invalid")


def _validate_text(value, label, max_chars=MAX_TEXT_CHARS):
    if not isinstance(value, str) or len(value) > max_chars:
        raise ZmqProtocolError("%s is invalid" % label)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError("unsupported JSON value")


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ZmqProtocolError("JSON object contains duplicate keys")
        result[key] = value
    return result


def _reject_json_constant(value):
    raise ZmqProtocolError("JSON constants NaN and Infinity are not allowed")
