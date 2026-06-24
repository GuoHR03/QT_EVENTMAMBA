import numpy as np
import pytest

from backend.inference_request import (
    apply_config_request,
    parse_event_data,
    process_inference_request,
    unpack_events_request,
)
from backend.protocol import PREDICTION_RESPONSE, STATUS_RESPONSE


class FakeInnerPredictor:
    def __init__(self):
        self.last_event_data = None

    def predict(self, event_data):
        self.last_event_data = event_data
        return [1.0, 2.0]


class FakePredictor:
    def __init__(self):
        self.width = 640
        self.height = 480
        self.current_mode = "center"
        self.load_message = "loaded"
        self.predictor = FakeInnerPredictor()

    def set_mode(self, mode):
        self.current_mode = mode


def test_parse_event_data_accepts_n_by_3():
    data = parse_event_data([[1, 2, 3], [4, 5, 6]])

    assert data.dtype == np.float32
    assert data.shape == (2, 3)
    assert data.flags["C_CONTIGUOUS"]


def test_parse_event_data_accepts_3_by_n():
    data = parse_event_data(np.array([[1, 4], [2, 5], [3, 6]], dtype=np.float32))

    assert data.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_parse_event_data_rejects_bad_shape():
    with pytest.raises(ValueError, match="输入形状"):
        parse_event_data(np.zeros((2, 2), dtype=np.float32))


def test_unpack_events_request_extracts_cropped_flag():
    event_data, is_cropped = unpack_events_request({"data": [[1, 2, 3]], "cropped": False})

    assert event_data.shape == (1, 3)
    assert is_cropped is False


def test_apply_config_request_updates_predictor_state():
    predictor = FakePredictor()

    mode = apply_config_request(
        predictor,
        {"msg_type": "CONFIG", "width": 320, "height": 240, "prediction_mode": "ellipse"},
    )

    assert mode == "ellipse"
    assert predictor.width == 320
    assert predictor.height == 240
    assert predictor.current_mode == "ellipse"


def test_process_inference_request_returns_status_for_config():
    predictor = FakePredictor()

    response = process_inference_request(
        predictor,
        {"msg_type": "CONFIG", "width": 320, "height": 240, "prediction_mode": "center"},
    )

    assert response["msg_type"] == STATUS_RESPONSE
    assert response["width"] == 320
    assert response["height"] == 240
    assert response["mode"] == "center"
    assert "loaded" in response["message"]


def test_process_inference_request_returns_prediction_for_events():
    predictor = FakePredictor()

    response = process_inference_request(predictor, {"data": [[1, 2, 3]], "cropped": False})

    assert response["msg_type"] == PREDICTION_RESPONSE
    assert response["values"] == [1.0, 2.0]
    assert response["cropped"] is False
    assert response["mode"] == "center"
    assert predictor.predictor.last_event_data.tolist() == [[1.0, 2.0, 3.0]]
