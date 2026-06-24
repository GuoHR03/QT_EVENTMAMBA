import numpy as np

from backend.protocol import make_prediction_response, make_status_response
from backend.settings import DEFAULT_SENSOR_HEIGHT, DEFAULT_SENSOR_WIDTH


def parse_event_data(data):
    if data is None:
        raise ValueError("没有收到推理数据")

    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"输入维度应为二维，实际为 {data.shape}")
    if data.shape[1] == 3:
        event_data = data
    elif data.shape[0] == 3:
        event_data = data.T
    else:
        raise ValueError(f"输入形状应为 (N, 3)，实际为 {data.shape}")

    return np.ascontiguousarray(event_data, dtype=np.float32)


def unpack_events_request(data):
    is_cropped = True
    if isinstance(data, dict):
        is_cropped = bool(data.get("cropped", True))
        data = data.get("data")
    return parse_event_data(data), is_cropped


def apply_config_request(predictor, data):
    predictor.width = data.get("width", DEFAULT_SENSOR_WIDTH)
    predictor.height = data.get("height", DEFAULT_SENSOR_HEIGHT)
    prediction_mode = data.get("prediction_mode", predictor.current_mode)
    predictor.set_mode(prediction_mode)
    return prediction_mode


def make_config_response(predictor, prediction_mode):
    if predictor.load_message:
        message = (
            f"{predictor.load_message}\n"
            f"相机参数初始化成功\n"
            f"相机参数: {predictor.width}x{predictor.height}\n"
            f"预测模式: {prediction_mode}"
        )
    else:
        message = f"相机参数初始化成功\n预测模式: {prediction_mode}"
    return make_status_response(
        message,
        width=predictor.width,
        height=predictor.height,
        mode=prediction_mode,
    )


def process_inference_request(predictor, data):
    if isinstance(data, dict) and data.get("msg_type") == "CONFIG":
        prediction_mode = apply_config_request(predictor, data)
        return make_config_response(predictor, prediction_mode)

    event_data, is_cropped = unpack_events_request(data)
    result = predictor.predictor.predict(event_data)
    return make_prediction_response(
        result,
        cropped=is_cropped,
        mode=predictor.current_mode,
    )
