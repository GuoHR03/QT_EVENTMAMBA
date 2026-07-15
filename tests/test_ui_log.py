from app.ui_log import log_level_for_message


def test_log_level_detects_errors_and_warnings():
    assert log_level_for_message("加载模型失败：missing weights") == "error"
    assert log_level_for_message("Warning: replay timeout") == "warning"


def test_log_level_detects_success_and_runtime_info():
    assert log_level_for_message("WSL 推理服务已就绪") == "success"
    assert log_level_for_message("[NoiseFilter] Disabled") == "info"
    assert log_level_for_message("[ROI] updated") == "info"


def test_log_level_defaults_for_prediction_output():
    assert log_level_for_message("center=(12.5, 18.0)") == "default"
