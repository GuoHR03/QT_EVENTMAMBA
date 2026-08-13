PREDICTION_RESPONSE = "PREDICTION"
STATUS_RESPONSE = "STATUS"
ERROR_RESPONSE = "ERROR"
LOCAL_ROI_CONTEXT = "_eventmamba_effective_roi"


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
