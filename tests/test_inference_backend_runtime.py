import pytest

from backend.inference_backend_runtime import (
    InferenceRuntimeSettings,
    build_wsl_launch,
    finite_positive_timeout,
)
from backend.settings import (
    ENV_BACKEND_READY_TIMEOUT_S,
    ENV_INFERENCE_RUNTIME,
    ENV_WSL_DISTRO,
)


def test_runtime_settings_validate_and_normalize_environment():
    settings = InferenceRuntimeSettings.from_environment(
        {
            ENV_INFERENCE_RUNTIME: " WSL ",
            ENV_WSL_DISTRO: "Ubuntu-Test",
            ENV_BACKEND_READY_TIMEOUT_S: "12.5",
        }
    )

    assert settings.kind == "wsl"
    assert settings.display_name == "WSL"
    assert settings.wsl_distro == "Ubuntu-Test"
    assert settings.ready_timeout_s == 12.5


def test_runtime_settings_reject_unknown_backend_kind():
    with pytest.raises(ValueError, match="Unsupported inference runtime"):
        InferenceRuntimeSettings.from_environment(
            {ENV_INFERENCE_RUNTIME: "remote"}
        )


@pytest.mark.parametrize("value", (0, -1, "bad", float("inf")))
def test_runtime_timeout_requires_a_finite_positive_value(value):
    assert finite_positive_timeout(value, 30) == 30.0


def test_wsl_launch_owns_platform_specific_command_conversion():
    converted = []

    def convert(path):
        converted.append(str(path))
        return "/converted/" + str(path).replace("\\", "/")

    launch = build_wsl_launch(
        r"E:\project",
        r"E:\models\center.pth",
        "center",
        6000,
        "nonce",
        distro="Ubuntu-Test",
        linux_python="/opt/venv/bin/python",
        path_converter=convert,
    )

    assert len(converted) == 2
    assert launch.command[0:3] == [
        "wsl",
        "-d",
        "Ubuntu-Test",
    ]
    assert launch.command[-4:] == ["--port", "6000", "--instance-nonce", "nonce"]
    assert launch.active_model_path == r"E:\models\center.pth"
