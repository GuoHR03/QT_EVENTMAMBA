"""Minimal EventMamba -> ONNX export and parity probe.

Run this inside the same Linux/WSL environment used by inference so the
custom mamba_ssm operators are available.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def _model_for(mode):
    if mode == "center":
        from backend.models.eventmamba_v1 import EventMamba as CenterEventMamba

        return CenterEventMamba(num_classes=2)
    from backend.models.eventmamba_v3 import EventMamba as EllipseEventMamba

    return EllipseEventMamba(num_classes=1024)


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def _load_model(mode, weights_path, device):
    model = _model_for(mode).to(device)
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(_extract_state_dict(checkpoint))
    model.eval()
    return model


def _verify_with_onnxruntime(onnx_path, dummy_input, expected):
    import onnxruntime as ort

    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    actual = session.run(None, {"events": dummy_input.cpu().numpy()})[0]
    expected = expected.cpu().numpy()
    return {
        "max_abs_error": float(np.max(np.abs(actual - expected))),
        "mean_abs_error": float(np.mean(np.abs(actual - expected))),
        "allclose_rtol_1e-3_atol_1e-4": bool(
            np.allclose(actual, expected, rtol=1e-3, atol=1e-4)
        ),
    }


def run(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    print(f"[1/4] loading {args.mode} model on {device}", flush=True)
    load_started = time.perf_counter()
    model = _load_model(args.mode, args.weights, device)
    print(f"[1/4] model loaded in {time.perf_counter() - load_started:.2f}s", flush=True)
    dummy_input = torch.randn(1, 3, 1024, device=device, dtype=torch.float32)

    print("[2/4] running PyTorch reference forward", flush=True)
    forward_started = time.perf_counter()
    with torch.inference_mode():
        expected = model(dummy_input).detach().cpu()
    print(
        f"[2/4] forward completed in {time.perf_counter() - forward_started:.2f}s; "
        f"output={list(expected.shape)}",
        flush=True,
    )

    if args.forward_only:
        print(
            json.dumps(
                {
                    "status": "forward_passed",
                    "mode": args.mode,
                    "device": str(device),
                    "input_shape": list(dummy_input.shape),
                    "output_shape": list(expected.shape),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[3/4] exporting ONNX to {output_path}", flush=True)
    started = time.perf_counter()
    torch.onnx.export(
        model,
        (dummy_input,),
        str(output_path),
        input_names=["events"],
        output_names=["prediction"],
        opset_version=args.opset,
        do_constant_folding=True,
    )
    export_seconds = time.perf_counter() - started
    print(f"[3/4] export completed in {export_seconds:.2f}s", flush=True)

    result = {
        "status": "exported",
        "mode": args.mode,
        "device": str(device),
        "torch_version": torch.__version__,
        "input_shape": list(dummy_input.shape),
        "output_shape": list(expected.shape),
        "opset": args.opset,
        "onnx_path": str(output_path),
        "onnx_bytes": output_path.stat().st_size,
        "export_seconds": export_seconds,
    }
    try:
        print("[4/4] validating with ONNX tooling", flush=True)
        import onnx

        model_proto = onnx.load(str(output_path))
        onnx.checker.check_model(model_proto)
        result["onnx_checker"] = "passed"
    except ImportError:
        result["onnx_checker"] = "skipped: onnx is not installed"

    try:
        result["onnxruntime"] = _verify_with_onnxruntime(
            output_path, dummy_input, expected
        )
    except ImportError:
        result["onnxruntime"] = "skipped: onnxruntime is not installed"

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("center", "ellipse"), default="center")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", default="artifacts/eventmamba_poc.onnx")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--forward-only", action="store_true")
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        result = {
            "status": "failed",
            "stage": "load, forward, or ONNX export",
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
