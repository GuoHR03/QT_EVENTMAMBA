"""Emit a bounded GitHub Actions annotation for the ONNX rewrite failure."""

import importlib.util
from pathlib import Path
import traceback

import onnx
from onnx import TensorProto, helper

from tools.onnx_insert_hierarchical_fps import (
    CUSTOM_DOMAIN,
    CUSTOM_OPSET_VERSION,
    FPS_OUTPUT_SIZES,
    rewrite_model,
)


def _source_model() -> onnx.ModelProto:
    events = helper.make_tensor_value_info(
        "events", TensorProto.FLOAT, [1, 3, 1024]
    )
    fps_inputs = [
        helper.make_tensor_value_info(name, TensorProto.INT64, [1, count])
        for name, count in FPS_OUTPUT_SIZES
    ]
    nodes = []
    outputs = []
    for index, (name, count) in enumerate(FPS_OUTPUT_SIZES):
        output_name = f"cast_fps{index}"
        nodes.append(
            helper.make_node(
                "Cast",
                [name],
                [output_name],
                name=f"/group{index}/Cast",
                to=TensorProto.INT64,
            )
        )
        outputs.append(
            helper.make_tensor_value_info(
                output_name, TensorProto.INT64, [1, count]
            )
        )
    graph = helper.make_graph(
        nodes,
        "eventmamba_fixture",
        [events, *fps_inputs],
        outputs,
    )
    model = helper.make_model(
        graph,
        producer_name="eventmamba-ci-diagnostic",
        opset_imports=(
            helper.make_opsetid("", 17),
            helper.make_opsetid(CUSTOM_DOMAIN, CUSTOM_OPSET_VERSION),
        ),
    )
    model.ir_version = 8
    return model


def _escape_workflow_command(value: str) -> str:
    return (
        value.replace("%", "%25")
        .replace("\r", "%0D")
        .replace("\n", "%0A")
    )


try:
    rewrite_model(_source_model())
    test_path = (
        Path(__file__).resolve().parents[1]
        / "tests"
        / "test_onnx_hierarchical_fps_rewrite.py"
    )
    spec = importlib.util.spec_from_file_location(
        "ui_event_onnx_rewrite_test", test_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load diagnostic test: {test_path}")
    test_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_module)
    test_module.test_rewrite_inserts_exact_custom_op_contract_and_preserves_existing_nodes()
except Exception as exc:
    frame = traceback.extract_tb(exc.__traceback__)[-1]
    detail = (
        f"{type(exc).__name__} at {Path(frame.filename).name}:{frame.lineno}: "
        f"{frame.line or ''}; {exc}"
    )[:1000]
    print(
        "::error title=ONNX rewrite diagnostic::"
        + _escape_workflow_command(detail)
    )
    raise SystemExit(1)

print("ONNX rewrite diagnostic passed")
