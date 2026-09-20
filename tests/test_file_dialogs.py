from app import file_dialogs


def test_windows_model_dialog_primary_filter_includes_randla(monkeypatch):
    captured = {}

    def fake_get_open_file_name(parent, title, initial_dir, file_filter):
        captured.update(
            title=title,
            initial_dir=initial_dir,
            file_filter=file_filter,
        )
        return "", ""

    monkeypatch.setattr(
        file_dialogs.QFileDialog,
        "getOpenFileName",
        fake_get_open_file_name,
    )

    assert file_dialogs.choose_weights_file(None, runtime_kind="windows") == ""
    assert captured["title"] == "选择 Windows ONNX 模型"
    primary_filter = captured["file_filter"].split(";;", 1)[0]
    assert "*_native_fps.onnx" in primary_filter
    assert "*_native.onnx" in primary_filter
