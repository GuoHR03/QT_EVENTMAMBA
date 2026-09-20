"""Emit a bounded GitHub Actions annotation for the ONNX rewrite failure."""

from tests.test_onnx_hierarchical_fps_rewrite import _source_model
from tools.onnx_insert_hierarchical_fps import rewrite_model


def _escape_workflow_command(value: str) -> str:
    return (
        value.replace("%", "%25")
        .replace("\r", "%0D")
        .replace("\n", "%0A")
    )


try:
    rewrite_model(_source_model())
except Exception as exc:
    detail = f"{type(exc).__name__}: {exc}"[:1000]
    print(
        "::error title=ONNX rewrite diagnostic::"
        + _escape_workflow_command(detail)
    )
    raise SystemExit(1)

print("ONNX rewrite diagnostic passed")
