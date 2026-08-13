from types import SimpleNamespace

import pytest

from backend import wsl_process
from backend.wsl_process import build_backend_command


def test_build_backend_command_uses_center_weight_argument():
    cmd = build_backend_command(
        "EventMamba_mini",
        "/env/bin/python",
        "/repo/linux_backend.py",
        "/weights/center.pth",
        "center",
        5555,
    )

    assert cmd == [
        "wsl",
        "-d",
        "EventMamba_mini",
        "/env/bin/python",
        "/repo/linux_backend.py",
        "--center-weights",
        "/weights/center.pth",
        "--port",
        "5555",
    ]


def test_build_backend_command_uses_ellipse_weight_argument():
    cmd = build_backend_command(
        "EventMamba_mini",
        "/env/bin/python",
        "/repo/linux_backend.py",
        "/weights/ellipse.pth",
        "ellipse",
        6000,
    )

    assert "--ellipse-weights" in cmd
    assert "--center-weights" not in cmd
    assert cmd[-1] == "6000"


def test_build_backend_command_can_bind_healthcheck_nonce():
    cmd = build_backend_command(
        "EventMamba_mini",
        "/env/bin/python",
        "/repo/linux_backend.py",
        "/weights/center.pth",
        "center",
        5555,
        instance_nonce="nonce-123",
    )

    assert cmd[-2:] == ["--instance-nonce", "nonce-123"]


def test_wsl_cleanup_targets_only_verified_pid(monkeypatch):
    calls = []

    def fake_run(_distro, command, timeout_s, capture_output=False):
        calls.append(command)
        if command[0] == "cat":
            return SimpleNamespace(
                returncode=0,
                stdout=b"python\0linux_backend.py\0--instance-nonce\0nonce-123\0",
            )
        if command[:2] == ["kill", "-0"]:
            return SimpleNamespace(returncode=1, stdout=b"")
        return SimpleNamespace(returncode=0, stdout=b"")

    monkeypatch.setattr(wsl_process, "_run_wsl", fake_run)

    assert wsl_process.kill_stale_backend_processes(
        "EventMamba_mini",
        pid=321,
        instance_nonce="nonce-123",
    )
    assert calls == [
        ["cat", "/proc/321/cmdline"],
        ["kill", "-TERM", "321"],
        ["kill", "-0", "321"],
    ]
    assert all("pkill" not in command for command in calls)


def test_wsl_cleanup_refuses_pid_with_different_nonce(monkeypatch):
    monkeypatch.setattr(
        wsl_process,
        "_run_wsl",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=b"python\0linux_backend.py\0--instance-nonce\0someone-else\0",
        ),
    )

    with pytest.raises(RuntimeError, match="Refusing to stop"):
        wsl_process.kill_stale_backend_processes(
            "EventMamba_mini",
            pid=321,
            instance_nonce="owned",
        )
