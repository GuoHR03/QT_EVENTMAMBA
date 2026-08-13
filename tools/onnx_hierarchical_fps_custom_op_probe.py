"""Validate and benchmark the native hierarchical FPS ONNX Runtime op."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.windows_onnx_runtime import prepare_windows_cuda_runtime


def _fps_indices(points, count, start):
    points = np.asarray(points, dtype=np.float32)
    centroids = np.zeros(count, dtype=np.int64)
    distance = np.full(len(points), np.float32(1e10), dtype=np.float32)
    farthest = int(start)
    for index in range(count):
        centroids[index] = farthest
        delta = points - points[farthest]
        distance = np.minimum(distance, np.sum(delta * delta, axis=1))
        farthest = int(np.argmax(distance))
    return centroids


def _reference(events, starts):
    batch_size = events.shape[0]
    outputs = [
        np.empty((batch_size, 512), dtype=np.int64),
        np.empty((batch_size, 256), dtype=np.int64),
        np.empty((batch_size, 128), dtype=np.int64),
    ]
    for batch in range(batch_size):
        xyz0 = events[batch].T
        outputs[0][batch] = _fps_indices(xyz0, 512, starts[batch, 0])
        xyz1 = xyz0[np.sort(outputs[0][batch])]
        outputs[1][batch] = _fps_indices(xyz1, 256, starts[batch, 1])
        xyz2 = xyz1[np.sort(outputs[1][batch])]
        outputs[2][batch] = _fps_indices(xyz2, 128, starts[batch, 2])
    return outputs


def _make_probe_model():
    node = helper.make_node(
        "HierarchicalFarthestPointSampling",
        ["events", "fps_starts"],
        ["fps0", "fps1", "fps2"],
        name="HierarchicalFarthestPointSamplingProbe",
        domain="com.eventmamba",
    )
    graph = helper.make_graph(
        [node],
        "eventmamba_hierarchical_fps_probe",
        [
            helper.make_tensor_value_info(
                "events", TensorProto.FLOAT, ["batch", 3, 1024]
            ),
            helper.make_tensor_value_info(
                "fps_starts", TensorProto.INT64, ["batch", 3]
            ),
        ],
        [
            helper.make_tensor_value_info(
                "fps0", TensorProto.INT64, ["batch", 512]
            ),
            helper.make_tensor_value_info(
                "fps1", TensorProto.INT64, ["batch", 256]
            ),
            helper.make_tensor_value_info(
                "fps2", TensorProto.INT64, ["batch", 128]
            ),
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="eventmamba-native-fps-probe",
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("com.eventmamba", 1),
        ],
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def _run_case(session, events, starts):
    inputs = {
        "events": np.ascontiguousarray(events, dtype=np.float32),
        "fps_starts": np.ascontiguousarray(starts, dtype=np.int64),
    }
    actual = session.run(None, inputs)
    expected = _reference(inputs["events"], inputs["fps_starts"])
    matches = [bool(np.array_equal(left, right)) for left, right in zip(actual, expected)]
    if not all(matches):
        mismatch = next(index for index, match in enumerate(matches) if not match)
        locations = np.argwhere(actual[mismatch] != expected[mismatch])
        first = locations[0].tolist() if locations.size else None
        raise AssertionError(f"FPS stage {mismatch} differs at {first}")
    return actual


def _expect_rejected(session, events, starts, message_fragment):
    try:
        session.run(
            None,
            {
                "events": np.ascontiguousarray(events, dtype=np.float32),
                "fps_starts": np.ascontiguousarray(starts, dtype=np.int64),
            },
        )
    except Exception as exc:
        if message_fragment not in str(exc):
            raise AssertionError(
                f"Expected rejection containing {message_fragment!r}, got {exc}"
            ) from exc
        return
    raise AssertionError(f"Expected native FPS rejection: {message_fragment}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--custom-op-library",
        default="native/selective_scan_ort/bin/eventmamba_selective_scan.dll",
    )
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")

    library = Path(args.custom_op_library).resolve()
    if not library.is_file():
        raise FileNotFoundError(f"Custom-op library not found: {library}")

    prepare_windows_cuda_runtime()
    options = ort.SessionOptions()
    options.log_severity_level = 4
    options.register_custom_ops_library(str(library))
    session = ort.InferenceSession(
        _make_probe_model().SerializeToString(),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    rng = np.random.default_rng(11)
    random_events = rng.standard_normal((2, 3, 1024), dtype=np.float32)
    random_starts = np.asarray([[967, 320, 175], [3, 7, 11]], dtype=np.int64)
    random_outputs = _run_case(session, random_events, random_starts)

    tie_events = np.zeros((1, 3, 1024), dtype=np.float32)
    tie_starts = np.asarray([[19, 23, 29]], dtype=np.int64)
    _run_case(session, tie_events, tie_starts)

    invalid_starts = random_starts[:1].copy()
    invalid_starts[0, 0] = 1024
    _expect_rejected(
        session,
        random_events[:1],
        invalid_starts,
        "fps_starts values are out of range",
    )
    non_finite_events = random_events[:1].copy()
    non_finite_events[0, 0, 0] = np.nan
    _expect_rejected(
        session,
        non_finite_events,
        random_starts[:1],
        "events values must be finite",
    )

    benchmark_inputs = {
        "events": np.ascontiguousarray(random_events[:1]),
        "fps_starts": np.ascontiguousarray(random_starts[:1]),
    }
    for _ in range(args.warmups):
        session.run(None, benchmark_inputs)
    times = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        session.run(None, benchmark_inputs)
        times.append((time.perf_counter() - started) * 1000.0)

    result = {
        "status": "verified",
        "provider": session.get_providers()[0],
        "custom_op_library": str(library),
        "random_batch": int(random_events.shape[0]),
        "output_shapes": [list(value.shape) for value in random_outputs],
        "tie_break_verified": True,
        "invalid_start_rejected": True,
        "non_finite_input_rejected": True,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "mean_ms": float(np.mean(times)),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
    }
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
