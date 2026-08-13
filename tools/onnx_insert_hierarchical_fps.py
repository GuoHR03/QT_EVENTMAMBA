"""Insert the native hierarchical FPS custom op into an EventMamba ONNX model.

The exported EventMamba models currently accept three precomputed FPS index
tensors.  This rewrite replaces those graph inputs with a single native custom
operator while deliberately leaving every existing graph node untouched::

    events [B, 3, 1024], fps_starts [B, 3]
        -> com.eventmamba::HierarchicalFarthestPointSampling
        -> fps0 [B, 512], fps1 [B, 256], fps2 [B, 128]

The command refuses to overwrite its input or a different existing output.
Repeating the same command is safe: an identical, valid output is reported as
``up_to_date`` without being rewritten.
"""

import argparse
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import onnx
from onnx import TensorProto, helper


CUSTOM_DOMAIN = "com.eventmamba"
CUSTOM_OP_TYPE = "HierarchicalFarthestPointSampling"
CUSTOM_OPSET_VERSION = 1
REWRITE_VERSION = 1
REWRITE_METADATA_KEY = "eventmamba.hierarchical_fps_rewrite_version"
NODE_NAME = "/HierarchicalFarthestPointSampling_v1"

EVENTS_INPUT = "events"
STARTS_INPUT = "fps_starts"
FPS_OUTPUT_SIZES: Tuple[Tuple[str, int], ...] = (
    ("fps0", 512),
    ("fps1", 256),
    ("fps2", 128),
)


@dataclass(frozen=True)
class RewriteResult:
    status: str
    input: str
    output: str
    rewrite_version: int
    custom_domain: str
    custom_op: str
    custom_opset: int
    graph_inputs: Tuple[str, ...]
    graph_outputs: Tuple[str, ...]


def _fail(message: str) -> None:
    raise RuntimeError(message)


def _input_map(model: onnx.ModelProto) -> Dict[str, onnx.ValueInfoProto]:
    result: Dict[str, onnx.ValueInfoProto] = {}
    for value in model.graph.input:
        if value.name in result:
            _fail(f"Duplicate graph input: {value.name}")
        result[value.name] = value
    return result


def _metadata_map(model: onnx.ModelProto) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for item in model.metadata_props:
        if item.key in result:
            _fail(f"Duplicate model metadata key: {item.key}")
        result[item.key] = item.value
    return result


def _domain_opset(model: onnx.ModelProto, domain: str) -> int:
    versions = [item.version for item in model.opset_import if item.domain == domain]
    if len(versions) != 1:
        _fail(
            f"Expected exactly one opset import for {domain!r}, found {versions}"
        )
    version = versions[0]
    if version != CUSTOM_OPSET_VERSION:
        _fail(
            f"Unsupported {domain!r} opset {version}; expected "
            f"{CUSTOM_OPSET_VERSION}"
        )
    return version


def _dimension_key(dimension: onnx.TensorShapeProto.Dimension) -> Tuple[str, object]:
    kind = dimension.WhichOneof("value")
    if kind == "dim_value":
        if dimension.dim_value <= 0:
            _fail(f"Dimension must be positive, got {dimension.dim_value}")
        return kind, dimension.dim_value
    if kind == "dim_param" and dimension.dim_param:
        return kind, dimension.dim_param
    _fail("Dynamic dimensions must have a non-empty symbolic name")
    raise AssertionError("unreachable")


def _tensor_signature(
    value: onnx.ValueInfoProto,
) -> Tuple[int, Tuple[Tuple[str, object], ...]]:
    value_type = value.type
    if not value_type.HasField("tensor_type"):
        _fail(f"{value.name!r} must be a tensor")
    tensor_type = value_type.tensor_type
    if not tensor_type.HasField("shape"):
        _fail(f"{value.name!r} must have a declared shape")
    return tensor_type.elem_type, tuple(
        _dimension_key(dimension) for dimension in tensor_type.shape.dim
    )


def _validate_tensor(
    value: onnx.ValueInfoProto,
    *,
    element_type: int,
    dimensions: Sequence[Tuple[str, object]],
) -> None:
    actual_type, actual_dimensions = _tensor_signature(value)
    expected_dimensions = tuple(dimensions)
    if actual_type != element_type or actual_dimensions != expected_dimensions:
        _fail(
            f"Unexpected signature for {value.name!r}: "
            f"type={actual_type}, dimensions={actual_dimensions}; expected "
            f"type={element_type}, dimensions={expected_dimensions}"
        )


def _batch_dimension(events: onnx.ValueInfoProto) -> Tuple[str, object]:
    element_type, dimensions = _tensor_signature(events)
    if element_type != TensorProto.FLOAT:
        _fail(
            f"{EVENTS_INPUT!r} must be float32, got tensor element type "
            f"{element_type}"
        )
    if len(dimensions) != 3 or dimensions[1:] != (
        ("dim_value", 3),
        ("dim_value", 1024),
    ):
        _fail(
            f"{EVENTS_INPUT!r} must have shape [B, 3, 1024], got "
            f"{dimensions}"
        )
    return dimensions[0]


def _shape_argument(dimension: Tuple[str, object]) -> object:
    return dimension[1]


def _consumers(model: onnx.ModelProto) -> Mapping[str, List[Tuple[onnx.NodeProto, int]]]:
    result: Dict[str, List[Tuple[onnx.NodeProto, int]]] = {}
    for node in model.graph.node:
        for index, value in enumerate(node.input):
            if value:
                result.setdefault(value, []).append((node, index))
    return result


def _validate_fps_consumers(model: onnx.ModelProto) -> None:
    consumers = _consumers(model)
    for value_name, _ in FPS_OUTPUT_SIZES:
        value_consumers = consumers.get(value_name, [])
        if len(value_consumers) != 1:
            _fail(
                f"Expected exactly one consumer for {value_name!r}, found "
                f"{[(node.name, index) for node, index in value_consumers]}"
            )
        node, input_index = value_consumers[0]
        cast_to = [attribute.i for attribute in node.attribute if attribute.name == "to"]
        if (
            node.op_type != "Cast"
            or node.domain not in ("", "ai.onnx")
            or input_index != 0
            or len(node.input) != 1
            or cast_to != [TensorProto.INT64]
        ):
            _fail(
                f"Unexpected consumer for {value_name!r}: "
                f"name={node.name!r}, op={node.domain!r}::{node.op_type}, "
                f"input_index={input_index}, cast_to={cast_to}"
            )


def _validate_no_name_collisions(model: onnx.ModelProto, names: Iterable[str]) -> None:
    names = set(names)
    initializer_names = {item.name for item in model.graph.initializer}
    sparse_initializer_names = {item.values.name for item in model.graph.sparse_initializer}
    output_names = {value for node in model.graph.node for value in node.output if value}
    graph_output_names = {value.name for value in model.graph.output}
    collisions = {
        "initializers": sorted(names & initializer_names),
        "sparse_initializers": sorted(names & sparse_initializer_names),
        "node_outputs": sorted(names & output_names),
        "graph_outputs": sorted(names & graph_output_names),
    }
    collisions = {kind: values for kind, values in collisions.items() if values}
    if collisions:
        _fail(f"Tensor name collision(s): {collisions}")


def _validate_source(model: onnx.ModelProto) -> Tuple[Tuple[str, object], Dict[str, onnx.ValueInfoProto]]:
    custom_nodes = [
        node
        for node in model.graph.node
        if node.domain == CUSTOM_DOMAIN and node.op_type == CUSTOM_OP_TYPE
    ]
    if custom_nodes:
        _fail("Model already contains the hierarchical FPS custom op")

    inputs = _input_map(model)
    expected_names = {EVENTS_INPUT, *(name for name, _ in FPS_OUTPUT_SIZES)}
    if set(inputs) != expected_names:
        _fail(
            f"Expected graph inputs {sorted(expected_names)}, got "
            f"{list(inputs)}"
        )

    _domain_opset(model, CUSTOM_DOMAIN)
    batch_dimension = _batch_dimension(inputs[EVENTS_INPUT])
    for name, sample_count in FPS_OUTPUT_SIZES:
        _validate_tensor(
            inputs[name],
            element_type=TensorProto.INT64,
            dimensions=(batch_dimension, ("dim_value", sample_count)),
        )

    _validate_fps_consumers(model)
    _validate_no_name_collisions(
        model,
        (STARTS_INPUT, *(name for name, _ in FPS_OUTPUT_SIZES)),
    )
    value_info_names = {value.name for value in model.graph.value_info}
    reserved_value_info = value_info_names & {
        STARTS_INPUT,
        *(name for name, _ in FPS_OUTPUT_SIZES),
    }
    if reserved_value_info:
        _fail(
            "Unexpected pre-existing value_info for rewrite tensors: "
            f"{sorted(reserved_value_info)}"
        )
    metadata = _metadata_map(model)
    if REWRITE_METADATA_KEY in metadata:
        _fail(
            f"Source already declares {REWRITE_METADATA_KEY}="
            f"{metadata[REWRITE_METADATA_KEY]!r}"
        )
    return batch_dimension, inputs


def _validate_rewritten(model: onnx.ModelProto) -> None:
    _domain_opset(model, CUSTOM_DOMAIN)
    inputs = _input_map(model)
    if list(inputs) != [EVENTS_INPUT, STARTS_INPUT]:
        _fail(
            f"Rewritten graph inputs must be [{EVENTS_INPUT!r}, "
            f"{STARTS_INPUT!r}], got {list(inputs)}"
        )
    batch_dimension = _batch_dimension(inputs[EVENTS_INPUT])
    _validate_tensor(
        inputs[STARTS_INPUT],
        element_type=TensorProto.INT64,
        dimensions=(batch_dimension, ("dim_value", 3)),
    )

    nodes = [
        node
        for node in model.graph.node
        if node.domain == CUSTOM_DOMAIN and node.op_type == CUSTOM_OP_TYPE
    ]
    if len(nodes) != 1:
        _fail(f"Expected exactly one hierarchical FPS node, found {len(nodes)}")
    node = nodes[0]
    if (
        node.name != NODE_NAME
        or list(node.input) != [EVENTS_INPUT, STARTS_INPUT]
        or list(node.output) != [name for name, _ in FPS_OUTPUT_SIZES]
        or node.attribute
    ):
        _fail(
            "Hierarchical FPS node contract mismatch: "
            f"name={node.name!r}, inputs={list(node.input)}, "
            f"outputs={list(node.output)}, attributes="
            f"{[attribute.name for attribute in node.attribute]}"
        )

    value_info = {value.name: value for value in model.graph.value_info}
    for name, sample_count in FPS_OUTPUT_SIZES:
        if name not in value_info:
            _fail(f"Missing value_info for {name!r}")
        _validate_tensor(
            value_info[name],
            element_type=TensorProto.INT64,
            dimensions=(batch_dimension, ("dim_value", sample_count)),
        )
    _validate_fps_consumers(model)

    metadata = _metadata_map(model)
    actual_version = metadata.get(REWRITE_METADATA_KEY)
    if actual_version != str(REWRITE_VERSION):
        _fail(
            f"Missing or unsupported rewrite version {actual_version!r}; "
            f"expected {REWRITE_VERSION}"
        )
    onnx.checker.check_model(model)


def rewrite_model(model: onnx.ModelProto) -> onnx.ModelProto:
    """Return a rewritten copy of *model* and validate the full contract."""

    rewritten = onnx.ModelProto()
    rewritten.CopyFrom(model)

    existing_nodes = [
        node
        for node in rewritten.graph.node
        if node.domain == CUSTOM_DOMAIN and node.op_type == CUSTOM_OP_TYPE
    ]
    if existing_nodes:
        _validate_rewritten(rewritten)
        return rewritten

    batch_dimension, inputs = _validate_source(rewritten)
    old_nodes = [node.SerializeToString() for node in rewritten.graph.node]

    events_input = onnx.ValueInfoProto()
    events_input.CopyFrom(inputs[EVENTS_INPUT])
    starts_input = helper.make_tensor_value_info(
        STARTS_INPUT,
        TensorProto.INT64,
        [_shape_argument(batch_dimension), 3],
    )
    del rewritten.graph.input[:]
    rewritten.graph.input.extend((events_input, starts_input))

    for name, _ in FPS_OUTPUT_SIZES:
        output_value_info = onnx.ValueInfoProto()
        output_value_info.CopyFrom(inputs[name])
        rewritten.graph.value_info.append(output_value_info)

    fps_node = helper.make_node(
        CUSTOM_OP_TYPE,
        [EVENTS_INPUT, STARTS_INPUT],
        [name for name, _ in FPS_OUTPUT_SIZES],
        domain=CUSTOM_DOMAIN,
        name=NODE_NAME,
    )
    rewritten.graph.node.insert(0, fps_node)

    metadata = _metadata_map(rewritten)
    metadata[REWRITE_METADATA_KEY] = str(REWRITE_VERSION)
    helper.set_model_props(rewritten, metadata)

    if [node.SerializeToString() for node in rewritten.graph.node[1:]] != old_nodes:
        _fail("Existing graph nodes changed during FPS rewrite")
    _validate_rewritten(rewritten)
    return rewritten


def validate_rewritten_model(model: onnx.ModelProto) -> None:
    """Validate a release model against the native FPS graph contract."""
    _validate_rewritten(model)


def _deterministic_bytes(model: onnx.ModelProto) -> bytes:
    return model.SerializeToString(deterministic=True)


def _save_checked(model: onnx.ModelProto, output: Path) -> None:
    output.parent.mkdir(parents=False, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.stem}.", suffix=output.suffix, dir=str(output.parent)
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        onnx.save_model(model, str(temporary_path))
        saved = onnx.load(str(temporary_path), load_external_data=True)
        _validate_rewritten(saved)
        os.replace(str(temporary_path), str(output))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def rewrite_file(input_path: Path, output_path: Path, *, overwrite: bool = False) -> RewriteResult:
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input ONNX model does not exist: {input_path}")
    if input_path == output_path:
        raise ValueError("Input and output paths must be different")
    if not output_path.parent.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {output_path.parent}")

    source = onnx.load(str(input_path), load_external_data=True)
    rewritten = rewrite_model(source)
    status = "rewritten"
    if output_path.exists():
        existing = onnx.load(str(output_path), load_external_data=True)
        _validate_rewritten(existing)
        if _deterministic_bytes(existing) == _deterministic_bytes(rewritten):
            status = "up_to_date"
        elif not overwrite:
            raise FileExistsError(
                f"Output already exists and differs: {output_path}. "
                "Pass --overwrite to replace that versioned output explicitly."
            )
        else:
            _save_checked(rewritten, output_path)
    else:
        _save_checked(rewritten, output_path)

    return RewriteResult(
        status=status,
        input=str(input_path),
        output=str(output_path),
        rewrite_version=REWRITE_VERSION,
        custom_domain=CUSTOM_DOMAIN,
        custom_op=CUSTOM_OP_TYPE,
        custom_opset=CUSTOM_OPSET_VERSION,
        graph_inputs=tuple(value.name for value in rewritten.graph.input),
        graph_outputs=tuple(value.name for value in rewritten.graph.output),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Insert EventMamba's native hierarchical FPS ONNX custom op."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace a different existing output (never the input).",
    )
    args = parser.parse_args()
    result = rewrite_file(args.input, args.output, overwrite=args.overwrite)
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
