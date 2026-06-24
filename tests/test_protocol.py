from backend.protocol import (
    ERROR_RESPONSE,
    PREDICTION_RESPONSE,
    make_error_response,
    make_prediction_response,
    make_status_response,
    response_message,
    is_prediction_response,
)


def test_prediction_response_shape():
    payload = make_prediction_response((0.25, 0.75), cropped=True, mode="center")

    assert payload == {
        "msg_type": PREDICTION_RESPONSE,
        "values": [0.25, 0.75],
        "cropped": True,
        "mode": "center",
    }
    assert is_prediction_response(payload)


def test_status_and_error_message_helpers():
    status = make_status_response("ready", port=5555)
    error = make_error_response("failed", code="load_error")

    assert response_message(status) == "ready"
    assert response_message(error) == "failed"
    assert error["msg_type"] == ERROR_RESPONSE
    assert error["code"] == "load_error"
