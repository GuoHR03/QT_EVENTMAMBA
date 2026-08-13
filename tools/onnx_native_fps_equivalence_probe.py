"""Compare legacy precomputed-FPS models with native-FPS model rewrites."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.windows_onnx_runtime import prepare_windows_cuda_runtime
from tools.onnx_windows_runtime_probe import _legacy_fps_inputs, _next_starts


def _load_session(model_path, custom_op_library):
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.register_custom_ops_library(str(custom_op_library))
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CUDAExecutionProvider"],
    )
    if session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"Provider fell back to {session.get_providers()[0]}")
    return session


def _load_events(path):
    sample = np.load(path)
    events = np.asarray(sample["events"], dtype=np.float32)
    if events.shape == (1024, 3):
        return np.ascontiguousarray(events.T[None])
    if events.shape == (1, 3, 1024):
        return np.ascontiguousarray(events)
    raise ValueError(f"Unexpected sample event shape: {events.shape}")


def _compare_pair(name, legacy_path, native_path, library, event_cases, seed):
    legacy = _load_session(legacy_path, library)
    native = _load_session(native_path, library)
    rng = np.random.default_rng(seed)
    cases = []
    verified = True
    for case_index, events in enumerate(event_cases):
        starts = _next_starts(rng)
        fps_inputs = _legacy_fps_inputs(events, starts)
        legacy_output = legacy.run(
            None,
            {"events": events, **fps_inputs},
        )[0]
        native_inputs = {
            "events": events,
            "fps_starts": starts[None],
        }
        native_output = native.run(None, native_inputs)[0]
        repeated_output = native.run(None, native_inputs)[0]
        allclose = bool(
            np.allclose(native_output, legacy_output, rtol=1e-3, atol=1e-4)
        )
        repeat_equal = bool(np.array_equal(native_output, repeated_output))
        repeat_allclose = bool(
            np.allclose(native_output, repeated_output, rtol=1e-3, atol=1e-4)
        )
        verified = verified and allclose and repeat_allclose
        cases.append(
            {
                "case": case_index,
                "starts": starts.tolist(),
                "shape": list(native_output.shape),
                "max_abs_error": float(
                    np.max(np.abs(native_output - legacy_output))
                ),
                "allclose": allclose,
                "repeat_bitwise_equal": repeat_equal,
                "repeat_max_abs_error": float(
                    np.max(np.abs(native_output - repeated_output))
                ),
                "repeat_allclose": repeat_allclose,
            }
        )
    return {
        "mode": name,
        "status": "verified" if verified else "mismatch",
        "legacy_model": str(legacy_path),
        "native_model": str(native_path),
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--legacy-center",
        default="artifacts/eventmamba_center_selective_scan_cuda.onnx",
    )
    parser.add_argument(
        "--native-center",
        default="artifacts/eventmamba_center_native_fps.onnx",
    )
    parser.add_argument(
        "--legacy-ellipse",
        default="artifacts/eventmamba_ellipse_selective_scan_cuda.onnx",
    )
    parser.add_argument(
        "--native-ellipse",
        default="artifacts/eventmamba_ellipse_native_fps.onnx",
    )
    parser.add_argument(
        "--custom-op-library",
        default="native/selective_scan_ort/bin/eventmamba_selective_scan.dll",
    )
    parser.add_argument("--sample", default="artifacts/real_raw_sample.npz")
    parser.add_argument("--synthetic-cases", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.synthetic_cases < 0:
        parser.error("--synthetic-cases must be non-negative")

    paths = {
        key: Path(value).resolve()
        for key, value in {
            "legacy_center": args.legacy_center,
            "native_center": args.native_center,
            "legacy_ellipse": args.legacy_ellipse,
            "native_ellipse": args.native_ellipse,
            "library": args.custom_op_library,
            "sample": args.sample,
        }.items()
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} not found: {path}")

    prepare_windows_cuda_runtime()
    event_cases = [_load_events(paths["sample"])]
    synthetic_rng = np.random.default_rng(args.seed + 1000)
    for _ in range(args.synthetic_cases):
        event_cases.append(
            synthetic_rng.standard_normal((1, 3, 1024), dtype=np.float32)
        )

    results = [
        _compare_pair(
            "center",
            paths["legacy_center"],
            paths["native_center"],
            paths["library"],
            event_cases,
            args.seed,
        ),
        _compare_pair(
            "ellipse",
            paths["legacy_ellipse"],
            paths["native_ellipse"],
            paths["library"],
            event_cases,
            args.seed,
        ),
    ]
    verified = all(item["status"] == "verified" for item in results)
    payload = {
        "status": "verified" if verified else "mismatch",
        "provider": "CUDAExecutionProvider",
        "sample": str(paths["sample"]),
        "synthetic_cases": args.synthetic_cases,
        "results": results,
    }
    print(json.dumps(payload, indent=2), flush=True)
    return 0 if verified else 2


if __name__ == "__main__":
    raise SystemExit(main())
