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
