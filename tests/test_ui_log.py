from app.ui_log import PredictionLogThrottle, log_level_for_message


def test_log_level_detects_errors_and_warnings():
    assert log_level_for_message("加载模型失败：missing weights") == "error"
    assert log_level_for_message("Warning: replay timeout") == "warning"


def test_log_level_detects_success_and_runtime_info():
    assert log_level_for_message("WSL 推理服务已就绪") == "success"
    assert log_level_for_message("[NoiseFilter] Disabled") == "info"
    assert log_level_for_message("[ROI] updated") == "info"


def test_log_level_defaults_for_prediction_output():
    assert log_level_for_message("center=(12.5, 18.0)") == "default"


def test_prediction_log_throttle_limits_only_successful_predictions():
    timestamps = iter((10.0, 10.2, 11.0))
    throttle = PredictionLogThrottle(interval_s=1.0, clock=lambda: next(timestamps))

    assert throttle.should_log({"msg_type": "PREDICTION", "values": [0.1, 0.2]})
    assert not throttle.should_log({"msg_type": "PREDICTION", "values": [0.2, 0.3]})
    assert throttle.should_log({"msg_type": "ERROR", "message": "backend failed"})
    assert throttle.should_log({"msg_type": "PREDICTION", "values": [0.3, 0.4]})


def test_prediction_log_throttle_reset_allows_immediate_sample():
    throttle = PredictionLogThrottle(interval_s=60.0, clock=lambda: 1.0)
    prediction = {"msg_type": "PREDICTION", "values": [0.1, 0.2]}

    assert throttle.should_log(prediction)
    assert not throttle.should_log(prediction)
    throttle.reset()
    assert throttle.should_log(prediction)
