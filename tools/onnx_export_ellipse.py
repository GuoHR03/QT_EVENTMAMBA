"""Export the ellipse checkpoint to an ONNX model with selective-scan Loops."""

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

from backend.ellipse_decoder import decode_ellipse_vsa
from backend.models import vsa
from tools.onnx_exportable_eventmamba import (
    EllipseCheckpointModel,
    PrecomputedFpsEllipseModel,
    precompute_fps_indices,
)


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--sample")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output",
        default="artifacts/eventmamba_ellipse_selective_scan_loop.onnx",
    )
    parser.add_argument(
        "--matrix-output",
        default="artifacts/eventmamba_ellipse_matrix_A.npy",
    )
    parser.add_argument(
        "--reference-output",
        default="artifacts/eventmamba_ellipse_export_reference.npz",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    source = EllipseCheckpointModel()
    source.load_state_dict(extract_state_dict(checkpoint), strict=True)
    model = PrecomputedFpsEllipseModel(source).to(device).eval()
    matrix_a = torch.load(
        args.matrix,
        map_location=device,
        weights_only=False,
    ).float()

    if args.sample:
        events_np = np.asarray(np.load(args.sample)["events"], dtype=np.float32)
        events = torch.from_numpy(events_np.T[None]).to(device)
    else:
        torch.manual_seed(args.seed)
        events = torch.randn(1, 3, 1024, device=device)
    fps_indices = precompute_fps_indices(events, args.seed)

    with torch.inference_mode():
        started = time.perf_counter()
        raw_output = model(events, *fps_indices)
        torch.cuda.synchronize() if device.type == "cuda" else None
        reference_seconds = time.perf_counter() - started
        real_part, imag_part = torch.chunk(raw_output, chunks=2, dim=-1)
        decoded_torch = vsa.Decode_VSA(
            torch.complex(real_part, imag_part),
            matrix_a,
            isELL=True,
        )
    decoded_numpy = decode_ellipse_vsa(
        raw_output.detach().cpu().numpy(),
        matrix_a.detach().cpu().numpy(),
    )
    decoder_error = float(
        np.max(np.abs(decoded_numpy - decoded_torch.detach().cpu().numpy()))
    )
    if not np.allclose(
        decoded_numpy,
        decoded_torch.detach().cpu().numpy(),
        rtol=1e-4,
        atol=1e-5,
    ):
        raise RuntimeError(f"NumPy VSA decoder mismatch: {decoder_error}")

    output_path = (PROJECT_ROOT / args.output).resolve()
    matrix_output = (PROJECT_ROOT / args.matrix_output).resolve()
    reference_output = (PROJECT_ROOT / args.reference_output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(matrix_output, matrix_a.detach().cpu().numpy())
    np.savez(
        reference_output,
        events=events.detach().cpu().numpy(),
        fps0=fps_indices[0].detach().cpu().numpy(),
        fps1=fps_indices[1].detach().cpu().numpy(),
        fps2=fps_indices[2].detach().cpu().numpy(),
        raw_output=raw_output.detach().cpu().numpy(),
        decoded_output=decoded_torch.detach().cpu().numpy(),
    )

    scripted = torch.jit.script(model)
    export_started = time.perf_counter()
    torch.onnx.export(
        scripted,
        (events, *fps_indices),
        str(output_path),
        input_names=["events", "fps0", "fps1", "fps2"],
        output_names=["prediction"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print(
        json.dumps(
            {
                "status": "exported",
                "onnx_path": str(output_path),
                "onnx_bytes": output_path.stat().st_size,
                "matrix_path": str(matrix_output),
                "reference_path": str(reference_output),
                "raw_output_shape": list(raw_output.shape),
                "decoded_output": decoded_torch.detach().cpu().numpy().tolist(),
                "numpy_decoder_max_abs_error": decoder_error,
                "reference_seconds": reference_seconds,
                "export_seconds": time.perf_counter() - export_started,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
