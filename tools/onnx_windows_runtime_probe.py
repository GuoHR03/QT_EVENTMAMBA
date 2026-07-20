"""Run the exported EventMamba model with native Windows ONNX Runtime."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

from onnx_cuda_runtime import preload_cuda_dlls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sample")
    parser.add_argument("--save-result")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--custom-op-library")
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    onnx.checker.check_model(onnx.load(str(model_path)))
    available = ort.get_available_providers()
    if args.provider not in available:
        raise RuntimeError(f"Requested {args.provider}; available: {available}")
    if args.provider == "CUDAExecutionProvider":
        preload_cuda_dlls()

    session_options = ort.SessionOptions()
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

    rng = np.random.default_rng(args.seed)
    if args.sample:
        sample = np.load(args.sample)
        events = np.asarray(sample["events"], dtype=np.float32).T[None]
    else:
        events = rng.standard_normal((1, 3, 1024), dtype=np.float32)

    def fps(points, count):
        centroids = np.zeros(count, dtype=np.int64)
        distance = np.full(len(points), 1e10, dtype=np.float32)
        farthest = int(rng.integers(0, len(points)))
        for index in range(count):
            centroids[index] = farthest
            delta = points - points[farthest]
            distance = np.minimum(distance, np.sum(delta * delta, axis=1))
            farthest = int(np.argmax(distance))
        return centroids

    xyz0 = events[0].T
    fps0 = fps(xyz0, 512)
    xyz1 = xyz0[np.sort(fps0)]
    fps1 = fps(xyz1, 256)
    xyz2 = xyz1[np.sort(fps1)]
    fps2 = fps(xyz2, 128)
    inputs = {
        "events": events,
        "fps0": fps0[None],
        "fps1": fps1[None],
        "fps2": fps2[None],
    }
    session.run(None, inputs)
    times = []
    output = None
    for _ in range(args.repeats):
        started = time.perf_counter()
        output = session.run(None, inputs)[0]
        times.append(time.perf_counter() - started)

    if args.save_result:
        np.savez(args.save_result, output=output, **inputs)

    result = {
        "status": "verified",
        "platform": "Windows",
        "onnxruntime_version": ort.__version__,
        "provider": session.get_providers()[0],
        "model_bytes": model_path.stat().st_size,
        "output": output.tolist(),
        "sample": str(Path(args.sample).resolve()) if args.sample else "synthetic",
        "custom_op_library": (
            str(Path(args.custom_op_library).resolve())
            if args.custom_op_library
            else None
        ),
        "repeats": len(times),
        "mean_seconds": sum(times) / len(times),
        "min_seconds": min(times),
        "max_seconds": max(times),
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
