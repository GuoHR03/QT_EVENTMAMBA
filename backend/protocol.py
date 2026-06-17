PREDICTION_RESPONSE = "PREDICTION"
STATUS_RESPONSE = "STATUS"
ERROR_RESPONSE = "ERROR"


def make_status_response(message, **extra):
    payload = {"msg_type": STATUS_RESPONSE, "message": str(message)}
    payload.update(extra)
    return payload


def make_error_response(message, **extra):
    payload = {"msg_type": ERROR_RESPONSE, "message": str(message)}
    payload.update(extra)
    return payload


def make_prediction_response(values, cropped=True, mode=None):
    return {
        "msg_type": PREDICTION_RESPONSE,
        "values": list(values),
        "cropped": bool(cropped),
        "mode": mode,
    }


def is_prediction_response(payload):
    return isinstance(payload, dict) and payload.get("msg_type") == PREDICTION_RESPONSE


def response_message(payload):
    if isinstance(payload, dict):
        return payload.get("message", str(payload))
    return str(payload)
