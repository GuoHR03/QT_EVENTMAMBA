"""Validate the Windows CUDA SelectiveScan custom operator."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

from onnx_cuda_runtime import preload_cuda_dlls


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUDA_12_BIN = Path(
    os.environ.get(
        "CUDA_PATH_V12_2",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2",
    )
) / "bin"
_DLL_DIRECTORY_HANDLES = []


def stable_softplus(value):
    return np.logaddexp(np.float32(0.0), value).astype(np.float32)


def silu(value):
    return value / (np.float32(1.0) + np.exp(-value))


def reference_scan(inputs):
    u = inputs["u"]
    delta = inputs["delta"]
    a_term = inputs["A"]
    b_term = inputs["B"]
    c_term = inputs["C"]
    d_skip = inputs["D"]
    z = inputs["z"]
    delta_bias = inputs["delta_bias"]
    batch, channels, length = u.shape
    state_size = a_term.shape[1]
    state = np.zeros((batch, channels, state_size), dtype=np.float32)
    output = np.empty_like(u)
    for position in range(length):
        dt = stable_softplus(delta[:, :, position] + delta_bias[None, :])
        state = (
            np.exp(dt[:, :, None] * a_term[None, :, :]) * state
            + dt[:, :, None]
            * b_term[:, None, :, position]
            * u[:, :, position, None]
        )
        value = np.sum(
            state * c_term[:, None, :, position], axis=2, dtype=np.float32
        )
        value += u[:, :, position] * d_skip[None, :]
        output[:, :, position] = value * silu(z[:, :, position])
    return output


def make_inputs(rng, batch, channels, length, state_size):
    normal = lambda shape, scale=1.0: (
        rng.standard_normal(shape, dtype=np.float32) * np.float32(scale)
    )
    return {
        "u": normal((batch, channels, length), 0.35),
        "delta": normal((batch, channels, length), 0.4),
        "A": -np.exp(normal((channels, state_size), 0.2)).astype(np.float32),
        "B": normal((batch, state_size, length), 0.25),
        "C": normal((batch, state_size, length), 0.25),
        "D": normal((channels,), 0.2),
        "z": normal((batch, channels, length), 0.4),
        "delta_bias": normal((channels,), 0.2),
    }


def create_model(path, inputs):
    value_infos = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, value.shape)
        for name, value in inputs.items()
    ]
    output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, inputs["u"].shape
    )
    node = helper.make_node(
        "SelectiveScan",
        list(inputs),
        ["output"],
        domain="com.eventmamba",
        name="SelectiveScanCudaPoc",
    )
    graph = helper.make_graph([node], "selective_scan_cuda_poc", value_infos, [output])
    model = helper.make_model(
        graph,
        producer_name="eventmamba-selective-scan-poc",
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("com.eventmamba", 1),
        ],
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)


def run_case(dll, rng, case, repeats):
    batch, channels, length, state_size = case
    inputs = make_inputs(rng, batch, channels, length, state_size)
    expected = reference_scan(inputs)
    model_path = (
        PROJECT_ROOT
        / "artifacts"
        / f"selective_scan_custom_{channels}x{length}x{state_size}.onnx"
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    create_model(model_path, inputs)

    options = ort.SessionOptions()
    options.register_custom_ops_library(str(dll))
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CUDAExecutionProvider"],
    )
    for _ in range(3):
        actual = session.run(None, inputs)[0]
    durations = []
    for _ in range(repeats):
        started = time.perf_counter()
        actual = session.run(None, inputs)[0]
        durations.append(time.perf_counter() - started)

    max_error = float(np.max(np.abs(actual - expected)))
    allclose = bool(np.allclose(actual, expected, rtol=2e-4, atol=2e-5))
    return {
        "shape": [batch, channels, length, state_size],
        "allclose": allclose,
        "max_abs_error": max_error,
        "mean_ms": 1000.0 * sum(durations) / len(durations),
        "min_ms": 1000.0 * min(durations),
        "max_ms": 1000.0 * max(durations),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dll",
        default="native/selective_scan_ort/build/vs174_mismatch/eventmamba_selective_scan.dll",
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    dll = (PROJECT_ROOT / args.dll).resolve()
    if not dll.is_file():
        raise FileNotFoundError(f"Custom operator DLL not found: {dll}")
    if CUDA_12_BIN.is_dir() and os.name == "nt":
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(CUDA_12_BIN)))
    cuda_runtime_dirs = preload_cuda_dlls()
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(f"CUDA provider unavailable: {ort.get_available_providers()}")

    rng = np.random.default_rng(args.seed)
    cases = [
        (1, 128, 512, 16),
        (1, 256, 256, 16),
        (1, 512, 128, 16),
    ]
    results = [run_case(dll, rng, case, args.repeats) for case in cases]
    payload = {
        "status": "verified" if all(item["allclose"] for item in results) else "mismatch",
        "provider": "CUDAExecutionProvider",
        "onnxruntime_version": ort.__version__,
        "dll": str(dll),
        "cuda_12_bin": str(CUDA_12_BIN),
        "cuda_runtime_dirs": cuda_runtime_dirs,
        "cases": results,
        "estimated_six_scan_ms": 2.0 * sum(item["mean_ms"] for item in results),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "verified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
