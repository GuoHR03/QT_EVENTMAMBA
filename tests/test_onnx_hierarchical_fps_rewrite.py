from pathlib import Path

import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper

from tools.onnx_insert_hierarchical_fps import (
    CUSTOM_DOMAIN,
    CUSTOM_OPSET_VERSION,
    CUSTOM_OP_TYPE,
    FPS_OUTPUT_SIZES,
    NODE_NAME,
    REWRITE_METADATA_KEY,
    REWRITE_VERSION,
    rewrite_file,
    rewrite_model,
)
from tools.validate_windows_inference_artifacts import validate_artifacts


def _source_model(*, domain_opset=CUSTOM_OPSET_VERSION, bad_consumer=None):
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
        op_type = bad_consumer if index == 0 and bad_consumer else "Cast"
        output_name = f"cast_fps{index}"
        if op_type == "Cast":
            node = helper.make_node(
                op_type,
                [name],
                [output_name],
                name=f"/group{index}/Cast",
                to=TensorProto.INT64,
            )
        else:
            node = helper.make_node(
                op_type,
                [name],
                [output_name],
                name=f"/group{index}/{op_type}",
            )
        nodes.append(node)
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
        producer_name="eventmamba-test",
        opset_imports=(
            helper.make_opsetid("", 17),
            helper.make_opsetid(CUSTOM_DOMAIN, domain_opset),
        ),
    )
    model.ir_version = 8
    return model


def _metadata(model):
    return {item.key: item.value for item in model.metadata_props}


def test_rewrite_inserts_exact_custom_op_contract_and_preserves_existing_nodes():
    source = _source_model()
    existing_nodes = [node.SerializeToString() for node in source.graph.node]

    rewritten = rewrite_model(source)

    assert [value.name for value in source.graph.input] == [
        "events",
        "fps0",
        "fps1",
        "fps2",
    ]
    assert [value.name for value in rewritten.graph.input] == [
        "events",
        "fps_starts",
    ]
    node = rewritten.graph.node[0]
    assert node.name == NODE_NAME
    assert node.domain == CUSTOM_DOMAIN
    assert node.op_type == CUSTOM_OP_TYPE
    assert list(node.input) == ["events", "fps_starts"]
    assert list(node.output) == ["fps0", "fps1", "fps2"]
    assert not node.attribute
    assert [item.SerializeToString() for item in rewritten.graph.node[1:]] == existing_nodes
    assert {value.name for value in rewritten.graph.value_info} == {
        "fps0",
        "fps1",
        "fps2",
    }
    assert _metadata(rewritten)[REWRITE_METADATA_KEY] == str(REWRITE_VERSION)
    onnx.checker.check_model(rewritten)


def test_rewrite_is_idempotent_for_an_already_rewritten_model():
    first = rewrite_model(_source_model())

    second = rewrite_model(first)

    assert second.SerializeToString(deterministic=True) == first.SerializeToString(
        deterministic=True
    )


def test_rewrite_rejects_an_unexpected_fps_consumer():
    with pytest.raises(RuntimeError, match="Unexpected consumer for 'fps0'"):
        rewrite_model(_source_model(bad_consumer="Identity"))


def test_rewrite_rejects_an_unsupported_custom_domain_version():
    with pytest.raises(RuntimeError, match="Unsupported 'com.eventmamba' opset 2"):
        rewrite_model(_source_model(domain_opset=2))


def test_rewrite_file_is_repeatable_without_overwriting(tmp_path):
    source_path = tmp_path / "source.onnx"
    output_path = tmp_path / "native_fps.onnx"
    onnx.save_model(_source_model(), str(source_path))

    first = rewrite_file(source_path, output_path)
    first_mtime = output_path.stat().st_mtime_ns
    second = rewrite_file(source_path, output_path)

    assert first.status == "rewritten"
    assert second.status == "up_to_date"
    assert output_path.stat().st_mtime_ns == first_mtime


def test_rewrite_file_refuses_a_different_existing_output(tmp_path):
    source_path = tmp_path / "source.onnx"
    output_path = tmp_path / "native_fps.onnx"
    onnx.save_model(_source_model(), str(source_path))
    rewrite_file(source_path, output_path)
    existing = onnx.load(str(output_path))
    existing.producer_version = "different-but-still-valid"
    onnx.save_model(existing, str(output_path))

    with pytest.raises(FileExistsError, match="Output already exists and differs"):
        rewrite_file(source_path, output_path)


@pytest.mark.parametrize(
    "source_name,output_name",
    (
        (
            "eventmamba_center_selective_scan_cuda.onnx",
            "eventmamba_center_native_fps.onnx",
        ),
        (
            "eventmamba_ellipse_selective_scan_cuda.onnx",
            "eventmamba_ellipse_native_fps.onnx",
        ),
    ),
)
def test_checked_in_native_fps_models_match_their_sources(source_name, output_name):
    artifacts = Path(__file__).resolve().parents[1] / "artifacts"
    source = onnx.load(str(artifacts / source_name))
    expected = rewrite_model(source)
    actual = onnx.load(str(artifacts / output_name))

    assert actual.SerializeToString(deterministic=True) == expected.SerializeToString(
        deterministic=True
    )
    onnx.checker.check_model(actual)


def test_release_artifact_validator_accepts_matching_native_bundle():
    root = Path(__file__).resolve().parents[1]
    artifacts = root / "artifacts"

    result = validate_artifacts(
        artifacts / "eventmamba_center_native_fps.onnx",
        artifacts / "eventmamba_ellipse_native_fps.onnx",
        artifacts / "eventmamba_ellipse_matrix_A.npy",
        root / "native/selective_scan_ort/bin/eventmamba_selective_scan.dll",
    )

    assert result["status"] == "verified"
    assert result["center"]["inputs"] == ["events", "fps_starts"]
    assert result["ellipse"]["native_fps_nodes"] == 1
    assert result["custom_op_library"]["bytes"] > 0


def test_release_artifact_validator_rejects_legacy_model():
    root = Path(__file__).resolve().parents[1]
    artifacts = root / "artifacts"

    with pytest.raises(RuntimeError, match="Rewritten graph inputs"):
        validate_artifacts(
            artifacts / "eventmamba_center_selective_scan_cuda.onnx",
            artifacts / "eventmamba_ellipse_native_fps.onnx",
            artifacts / "eventmamba_ellipse_matrix_A.npy",
            root / "native/selective_scan_ort/bin/eventmamba_selective_scan.dll",
        )
