import os
from pathlib import Path

import pytest

from backend.inference_service import InferenceService
from backend.settings import (
    DEFAULT_CENTER_ONNX_MODEL,
    DEFAULT_ELLIPSE_MATRIX,
    DEFAULT_ELLIPSE_ONNX_MODEL,
    DEFAULT_SELECTIVE_SCAN_DLL,
    DEFAULT_WINDOWS_BACKEND_EXECUTABLE,
    DEFAULT_WINDOWS_PYTHON,
)


def _touch(root, relative_path):
    path = root / Path(relative_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def _windows_service(root, frozen):
    service = InferenceService.__new__(InferenceService)
    service.windows_python = DEFAULT_WINDOWS_PYTHON
    service.windows_backend_executable = DEFAULT_WINDOWS_BACKEND_EXECUTABLE
    service.center_onnx_model = DEFAULT_CENTER_ONNX_MODEL
    service.ellipse_onnx_model = DEFAULT_ELLIPSE_ONNX_MODEL
    service.ellipse_matrix = DEFAULT_ELLIPSE_MATRIX
    service.selective_scan_dll = DEFAULT_SELECTIVE_SCAN_DLL
    service.active_model_path = None
    service._is_frozen_ui = lambda: frozen

    _touch(root, DEFAULT_CENTER_ONNX_MODEL)
    _touch(root, DEFAULT_ELLIPSE_ONNX_MODEL)
    _touch(root, DEFAULT_ELLIPSE_MATRIX)
    _touch(root, DEFAULT_SELECTIVE_SCAN_DLL)
    return service


def test_source_ui_keeps_python_and_windows_backend_script(tmp_path):
    service = _windows_service(tmp_path, frozen=False)
    python_executable = _touch(tmp_path, DEFAULT_WINDOWS_PYTHON)
    backend_script = _touch(tmp_path, "windows_backend.py")

    command = service._build_windows_command(
        str(tmp_path),
        str(tmp_path / "checkpoint" / "center.pth"),
        "center",
        5555,
    )

    assert command[:2] == [
        os.path.abspath(python_executable),
        os.path.abspath(backend_script),
    ]
    assert service.active_model_path == os.path.abspath(
        tmp_path / DEFAULT_CENTER_ONNX_MODEL
    )


def test_frozen_ui_uses_installed_backend_executable_and_relative_assets(tmp_path):
    service = _windows_service(tmp_path, frozen=True)
    backend_executable = _touch(tmp_path, DEFAULT_WINDOWS_BACKEND_EXECUTABLE)
    selected_model = _touch(tmp_path, "models/selected.onnx")

    command = service._build_windows_command(
        str(tmp_path),
        "models/selected.onnx",
        "ellipse",
        6000,
    )

    assert command[0] == os.path.abspath(backend_executable)
    assert "windows_backend.py" not in command
    assert DEFAULT_WINDOWS_PYTHON not in command
    assert command[command.index("--ellipse-model") + 1] == os.path.abspath(
        selected_model
    )
    assert service.active_model_path == os.path.abspath(selected_model)


def test_frozen_ui_reports_missing_backend_executable_instead_of_python(tmp_path):
    service = _windows_service(tmp_path, frozen=True)
    _touch(tmp_path, DEFAULT_WINDOWS_PYTHON)
    _touch(tmp_path, "windows_backend.py")

    with pytest.raises(FileNotFoundError) as exc_info:
        service._build_windows_command(
            str(tmp_path),
            str(tmp_path / "checkpoint" / "center.pth"),
            "center",
            5555,
        )

    message = str(exc_info.value)
    assert "Windows backend executable" in message
    assert "Windows Python" not in message
