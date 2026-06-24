import os

import pytest

from backend.inference_runtime import (
    decode_backend_log,
    default_backend_log_path,
    read_float_env,
    runtime_root_dir,
    to_wsl_path,
)


def test_to_wsl_path_converts_windows_drive_paths():
    assert to_wsl_path(r"E:\Code\Qt\UI_Event-main\model.pth") == "/mnt/e/Code/Qt/UI_Event-main/model.pth"


def test_to_wsl_path_converts_matching_wsl_unc_path():
    path = r"\\wsl$\EventMamba_mini\home\user\model.pth"

    assert to_wsl_path(path, "EventMamba_mini") == "/home/user/model.pth"


def test_to_wsl_path_converts_other_wsl_distro_path():
    path = r"\\wsl$\Ubuntu\home\user\model.pth"

    assert to_wsl_path(path, "EventMamba_mini") == "/mnt/wsl/Ubuntu/home/user/model.pth"


def test_to_wsl_path_leaves_linux_and_empty_paths_unchanged():
    assert to_wsl_path("/home/user/model.pth") == "/home/user/model.pth"
    assert to_wsl_path("") == ""
    assert to_wsl_path(None) is None


def test_runtime_root_dir_uses_frozen_meipass_when_available():
    assert runtime_root_dir(__file__, frozen=True, meipass=r"C:\Temp\bundle") == r"C:\Temp\bundle"


def test_runtime_root_dir_uses_module_parent_for_source_tree():
    root = runtime_root_dir(os.path.join("repo", "backend", "inference_service.py"), frozen=False)

    assert root.endswith("repo")


def test_default_backend_log_path_prefers_localappdata():
    log_path = default_backend_log_path(
        runtime_root=r"C:\Project",
        environ={"LOCALAPPDATA": r"C:\Users\me\AppData\Local", "TEMP": r"C:\Temp"},
    )

    assert log_path == os.path.join(r"C:\Users\me\AppData\Local", "UI_Event", "eventmamba_backend.log")


def test_decode_backend_log_handles_utf8_utf16_and_gbk():
    assert decode_backend_log("ready".encode("utf-8")) == "ready"
    assert decode_backend_log("ready".encode("utf-16")) == "ready"
    assert decode_backend_log("就绪".encode("gbk")) == "就绪"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2.5", 2.5),
        ("bad", 180.0),
        (None, 180.0),
    ],
)
def test_read_float_env(value, expected):
    environ = {}
    if value is not None:
        environ["EVENTMAMBA_BACKEND_READY_TIMEOUT_S"] = value

    assert read_float_env("EVENTMAMBA_BACKEND_READY_TIMEOUT_S", 180.0, environ=environ) == expected
