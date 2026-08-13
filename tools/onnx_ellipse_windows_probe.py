"""Validate the ellipse ONNX/CUDA model against its PyTorch export reference."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.ellipse_decoder import decode_ellipse_vsa
from backend.windows_onnx_runtime import prepare_windows_cuda_runtime


LEGACY_INPUT_NAMES = {"events", "fps0", "fps1", "fps2"}
NATIVE_FPS_INPUT_NAMES = {"events", "fps_starts"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--custom-op-library", required=True)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    prepare_windows_cuda_runtime()
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.register_custom_ops_library(str(Path(args.custom_op_library).resolve()))
    session = ort.InferenceSession(
        str(Path(args.model).resolve()),
        sess_options=options,
        providers=["CUDAExecutionProvider"],
    )
    if session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"Provider fell back to {session.get_providers()[0]}")

    reference = np.load(args.reference)
    input_names = {item.name for item in session.get_inputs()}
    if input_names == LEGACY_INPUT_NAMES:
        fps_mode = "legacy-python"
        inputs = {
            name: np.ascontiguousarray(reference[name])
            for name in ("events", "fps0", "fps1", "fps2")
        }
    elif input_names == NATIVE_FPS_INPUT_NAMES:
        fps_mode = "native-custom-op"
        starts = np.asarray(
            [
                int(reference["fps0"][0, 0]),
                int(reference["fps1"][0, 0]),
                int(reference["fps2"][0, 0]),
            ],
            dtype=np.int64,
        )
        inputs = {
            "events": np.ascontiguousarray(reference["events"]),
            "fps_starts": starts[None],
        }
    else:
        raise RuntimeError(f"Unsupported model inputs: {sorted(input_names)}")
    session.run(None, inputs)
    times = []
    actual_raw = None
    for _ in range(args.repeats):
        started = time.perf_counter()
        actual_raw = session.run(None, inputs)[0]
        times.append(time.perf_counter() - started)

    expected_raw = reference["raw_output"]
    matrix_a = np.load(args.matrix)
    actual_decoded = decode_ellipse_vsa(actual_raw, matrix_a)
    expected_decoded = reference["decoded_output"]
    raw_error = float(np.max(np.abs(actual_raw - expected_raw)))
    decoded_error = float(np.max(np.abs(actual_decoded - expected_decoded)))
    raw_allclose = bool(
        np.allclose(actual_raw, expected_raw, rtol=1e-3, atol=1e-4)
    )
    decoded_allclose = bool(
        np.allclose(actual_decoded, expected_decoded, rtol=1e-3, atol=1e-4)
    )
    result = {
        "status": "verified" if raw_allclose and decoded_allclose else "mismatch",
        "provider": session.get_providers()[0],
        "fps_mode": fps_mode,
        "raw_output_shape": list(actual_raw.shape),
        "raw_max_abs_error": raw_error,
        "decoded_output": actual_decoded.tolist(),
        "decoded_max_abs_error": decoded_error,
        "raw_allclose": raw_allclose,
        "decoded_allclose": decoded_allclose,
        "repeats": len(times),
        "mean_ms": 1000 * sum(times) / len(times),
        "min_ms": 1000 * min(times),
        "max_ms": 1000 * max(times),
    }
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["status"] == "verified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
