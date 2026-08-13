"""Run the exported EventMamba model with native Windows ONNX Runtime."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.onnx_cuda_runtime import preload_cuda_dlls


LEGACY_INPUT_NAMES = {"events", "fps0", "fps1", "fps2"}
NATIVE_FPS_INPUT_NAMES = {"events", "fps_starts"}


def _fps_indices(points, count, start):
    """Exact NumPy oracle for the production farthest-point sampler."""
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


def _legacy_fps_inputs(events, starts):
    xyz0 = events[0].T
    fps0 = _fps_indices(xyz0, 512, starts[0])
    xyz1 = xyz0[np.sort(fps0)]
    fps1 = _fps_indices(xyz1, 256, starts[1])
    xyz2 = xyz1[np.sort(fps1)]
    fps2 = _fps_indices(xyz2, 128, starts[2])
    return {
        "fps0": fps0[None],
        "fps1": fps1[None],
        "fps2": fps2[None],
    }


def _next_starts(rng):
    # Keep the three scalar draws identical to the legacy predictor.
    return np.asarray(
        [
            int(rng.integers(0, 1024)),
            int(rng.integers(0, 512)),
            int(rng.integers(0, 256)),
        ],
        dtype=np.int64,
    )


def _percentile_ms(samples, percentile):
    return float(np.percentile(samples, percentile) * 1000.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sample")
    parser.add_argument("--save-result")
    parser.add_argument("--full-output", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--custom-op-library")
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")

    model_path = Path(args.model).resolve()
    onnx.checker.check_model(onnx.load(str(model_path)))
    available = ort.get_available_providers()
    if args.provider not in available:
        raise RuntimeError(f"Requested {args.provider}; available: {available}")
    if args.provider == "CUDAExecutionProvider":
        preload_cuda_dlls()

    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3
    session_options.enable_profiling = args.profile
    dll_directory_handles = []
    if args.custom_op_library:
        custom_op_library = Path(args.custom_op_library).resolve()
        cuda_12_bin = Path(
            os.environ.get(
                "CUDA_PATH_V12_2",
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2",
            )
        ) / "bin"
        if os.name == "nt" and cuda_12_bin.is_dir():
            dll_directory_handles.append(os.add_dll_directory(str(cuda_12_bin)))
        session_options.register_custom_ops_library(str(custom_op_library))
    session = ort.InferenceSession(
        str(model_path), sess_options=session_options, providers=[args.provider]
    )
    if session.get_providers()[0] != args.provider:
        raise RuntimeError(f"Provider fell back to {session.get_providers()[0]}")

    input_names = {item.name for item in session.get_inputs()}
    if input_names == LEGACY_INPUT_NAMES:
        fps_mode = "legacy-python"
    elif input_names == NATIVE_FPS_INPUT_NAMES:
        fps_mode = "native-custom-op"
    else:
        raise RuntimeError(f"Unsupported model inputs: {sorted(input_names)}")

    rng = np.random.default_rng(args.seed)
    if args.sample:
        sample = np.load(args.sample)
        sample_events = np.asarray(sample["events"], dtype=np.float32)
        if sample_events.shape == (1024, 3):
            events = np.ascontiguousarray(sample_events.T[None])
        elif sample_events.shape == (1, 3, 1024):
            events = np.ascontiguousarray(sample_events)
        else:
            raise ValueError(
                "Sample events must have shape (1024, 3) or (1, 3, 1024), "
                f"got {sample_events.shape}"
            )
    else:
        events = rng.standard_normal((1, 3, 1024), dtype=np.float32)

    def make_inputs(starts):
        if fps_mode == "native-custom-op":
            return {"events": events, "fps_starts": starts[None]}
        return {"events": events, **_legacy_fps_inputs(events, starts)}

    fixed_starts = _next_starts(rng)
    fixed_inputs = make_inputs(fixed_starts)
    for _ in range(max(args.warmups, 0)):
        session.run(None, fixed_inputs)

    session_times = []
    output = None
    for _ in range(args.repeats):
        started = time.perf_counter()
        output = session.run(None, fixed_inputs)[0]
        session_times.append(time.perf_counter() - started)

    prep_times = []
    end_to_end_times = []
    last_starts = fixed_starts
    for _ in range(args.repeats):
        request_started = time.perf_counter()
        prep_started = request_started
        last_starts = _next_starts(rng)
        request_inputs = make_inputs(last_starts)
        prepared = time.perf_counter()
        output = session.run(None, request_inputs)[0]
        finished = time.perf_counter()
        prep_times.append(prepared - prep_started)
        end_to_end_times.append(finished - request_started)

    if args.save_result:
        reference_fps = _legacy_fps_inputs(events, last_starts)
        np.savez(
            args.save_result,
            output=output,
            events=events,
            fps_starts=last_starts[None],
            **reference_fps,
        )

    result = {
        "status": "verified",
        "platform": "Windows",
        "onnxruntime_version": ort.__version__,
        "provider": session.get_providers()[0],
        "fps_mode": fps_mode,
        "model_bytes": model_path.stat().st_size,
        "output_shape": list(output.shape),
        "output": (
            output.tolist()
            if args.full_output
            else np.asarray(output).reshape(-1)[:8].tolist()
        ),
        "sample": str(Path(args.sample).resolve()) if args.sample else "synthetic",
        "custom_op_library": (
            str(Path(args.custom_op_library).resolve())
            if args.custom_op_library
            else None
        ),
        "warmups": max(args.warmups, 0),
        "repeats": len(session_times),
        "input_prep_ms": {
            "mean": 1000 * sum(prep_times) / len(prep_times),
            "p50": _percentile_ms(prep_times, 50),
            "p95": _percentile_ms(prep_times, 95),
        },
        "session_ms": {
            "mean": 1000 * sum(session_times) / len(session_times),
            "p50": _percentile_ms(session_times, 50),
            "p95": _percentile_ms(session_times, 95),
        },
        "end_to_end_ms": {
            "mean": 1000 * sum(end_to_end_times) / len(end_to_end_times),
            "p50": _percentile_ms(end_to_end_times, 50),
            "p95": _percentile_ms(end_to_end_times, 95),
        },
    }
    if args.profile:
        profile_path = session.end_profiling()
        with open(profile_path, "r", encoding="utf-8") as handle:
            profile_events = json.load(handle)
        providers = {}
        for event in profile_events:
            if event.get("cat") != "Node":
                continue
            provider = event.get("args", {}).get("provider", "unassigned")
            entry = providers.setdefault(provider, {"events": 0, "duration_us": 0})
            entry["events"] += 1
            entry["duration_us"] += int(event.get("dur", 0))
        result["profile_path"] = profile_path
        result["profile_by_provider"] = providers
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
