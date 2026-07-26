from backend.windows_process import (
    build_windows_backend_command,
    build_windows_backend_executable_command,
)


def test_build_windows_backend_command_uses_native_assets():
    command = build_windows_backend_command(
        r"E:\repo\.venv-onnx-win\Scripts\python.exe",
        r"E:\repo\windows_backend.py",
        r"E:\repo\artifacts\center.onnx",
        r"E:\repo\artifacts\ellipse.onnx",
        r"E:\repo\artifacts\ellipse_matrix.npy",
        r"E:\repo\native\selective_scan.dll",
        "ellipse",
        6000,
    )

    assert command == [
        r"E:\repo\.venv-onnx-win\Scripts\python.exe",
        r"E:\repo\windows_backend.py",
        "--center-model",
        r"E:\repo\artifacts\center.onnx",
        "--ellipse-model",
        r"E:\repo\artifacts\ellipse.onnx",
        "--ellipse-matrix",
        r"E:\repo\artifacts\ellipse_matrix.npy",
        "--custom-op-library",
        r"E:\repo\native\selective_scan.dll",
        "--initial-mode",
        "ellipse",
        "--port",
        "6000",
    ]


def test_build_windows_backend_executable_command_omits_python_script():
    command = build_windows_backend_executable_command(
        r"E:\repo\backend_runtime\UI_Event_Backend.exe",
        r"E:\repo\artifacts\center.onnx",
        r"E:\repo\artifacts\ellipse.onnx",
        r"E:\repo\artifacts\ellipse_matrix.npy",
        r"E:\repo\native\selective_scan.dll",
        "center",
        5555,
    )

    assert command == [
        r"E:\repo\backend_runtime\UI_Event_Backend.exe",
        "--center-model",
        r"E:\repo\artifacts\center.onnx",
        "--ellipse-model",
        r"E:\repo\artifacts\ellipse.onnx",
        "--ellipse-matrix",
        r"E:\repo\artifacts\ellipse_matrix.npy",
        "--custom-op-library",
        r"E:\repo\native\selective_scan.dll",
        "--initial-mode",
        "center",
        "--port",
        "5555",
    ]
