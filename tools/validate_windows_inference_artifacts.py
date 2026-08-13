"""Fail closed when Windows inference release assets do not match."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.onnx_insert_hierarchical_fps import validate_rewritten_model


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _tensor_dimensions(value):
    dimensions = []
    for dimension in value.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            dimensions.append(dimension.dim_value)
        elif dimension.dim_param:
            dimensions.append(dimension.dim_param)
        else:
            dimensions.append(None)
    return dimensions


def _validate_model(path, expected_width):
    model = onnx.load(str(path), load_external_data=True)
    validate_rewritten_model(model)
    custom_nodes = [
        node
        for node in model.graph.node
        if node.domain == "com.eventmamba"
        and node.op_type == "HierarchicalFarthestPointSampling"
    ]
    scan_nodes = [
        node
        for node in model.graph.node
        if node.domain == "com.eventmamba" and node.op_type == "SelectiveScanCore"
    ]
    if len(custom_nodes) != 1:
        raise RuntimeError(f"{path.name}: expected one native FPS node")
    if len(scan_nodes) != 6:
        raise RuntimeError(
            f"{path.name}: expected six SelectiveScanCore nodes, got {len(scan_nodes)}"
        )
    if len(model.graph.output) != 1:
        raise RuntimeError(f"{path.name}: expected one model output")
    output = model.graph.output[0]
    dimensions = _tensor_dimensions(output)
    if (
        output.type.tensor_type.elem_type != TensorProto.FLOAT
        or len(dimensions) != 2
        or dimensions[1] != expected_width
    ):
        raise RuntimeError(
            f"{path.name}: unexpected output signature "
            f"type={output.type.tensor_type.elem_type}, shape={dimensions}"
        )
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "nodes": len(model.graph.node),
        "native_fps_nodes": len(custom_nodes),
        "selective_scan_core_nodes": len(scan_nodes),
        "inputs": [value.name for value in model.graph.input],
        "output": {"name": output.name, "shape": dimensions},
    }


def validate_artifacts(center, ellipse, matrix, custom_op_library):
    paths = {
        name: Path(value).resolve()
        for name, value in {
            "center": center,
            "ellipse": ellipse,
            "matrix": matrix,
            "custom_op_library": custom_op_library,
        }.items()
    }
    for name, path in paths.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"{name} artifact is missing or empty: {path}")

    matrix = np.load(str(paths["matrix"]), allow_pickle=False)
    if matrix.shape != (2, 512) or matrix.dtype != np.float32:
        raise RuntimeError(
            "Ellipse matrix must have shape (2, 512) and dtype float32, "
            f"got shape={matrix.shape}, dtype={matrix.dtype}"
        )
    if not np.isfinite(matrix).all():
        raise RuntimeError("Ellipse matrix contains NaN or infinity")

    return {
        "status": "verified",
        "center": _validate_model(paths["center"], 2),
        "ellipse": _validate_model(paths["ellipse"], 1024),
        "matrix": {
            "path": str(paths["matrix"]),
            "bytes": paths["matrix"].stat().st_size,
            "sha256": _sha256(paths["matrix"]),
            "shape": list(matrix.shape),
            "dtype": str(matrix.dtype),
        },
        "custom_op_library": {
            "path": str(paths["custom_op_library"]),
            "bytes": paths["custom_op_library"].stat().st_size,
            "sha256": _sha256(paths["custom_op_library"]),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--center",
        default="artifacts/eventmamba_center_native_fps.onnx",
    )
    parser.add_argument(
        "--ellipse",
        default="artifacts/eventmamba_ellipse_native_fps.onnx",
    )
    parser.add_argument(
        "--matrix",
        default="artifacts/eventmamba_ellipse_matrix_A.npy",
    )
    parser.add_argument(
        "--custom-op-library",
        default="native/selective_scan_ort/bin/eventmamba_selective_scan.dll",
    )
    args = parser.parse_args()
    result = validate_artifacts(
        args.center,
        args.ellipse,
        args.matrix,
        args.custom_op_library,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
