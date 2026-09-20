"""Export the RandLA random-sample ellipse checkpoint to ONNX."""

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
from tools.onnx_exportable_randla_eventmamba import (
    PrecomputedRandomSampleEllipseModel,
    RandLAEllipseCheckpointModel,
    precompute_random_sample_indices,
)
from tools.onnx_fix_topk_k import normalize_topk_inputs
from tools.onnx_replace_selective_scan_loops import replace_loops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--sample")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--loop-output",
        default="artifacts/experimental/eventmamba_ellipse_randla_loop.onnx",
    )
    parser.add_argument(
        "--output",
        default="artifacts/sources/eventmamba_ellipse_randla_selective_scan_cuda.onnx",
    )
    parser.add_argument(
        "--matrix-output",
        default="artifacts/eventmamba_ellipse_randla_matrix_A.npy",
    )
    parser.add_argument(
        "--reference-output",
        default="artifacts/experimental/eventmamba_ellipse_randla_reference.npz",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    source = RandLAEllipseCheckpointModel()
    source.load_state_dict(checkpoint, strict=True)
    model = PrecomputedRandomSampleEllipseModel(source).to(device).eval()
    matrix_a = torch.load(
        args.matrix,
        map_location=device,
        weights_only=False,
    ).float()

    if args.sample:
        sample = np.load(args.sample)
        events_np = np.asarray(sample["events"], dtype=np.float32)
        if events_np.shape == (1024, 3):
            events_np = events_np.T[None]
        events = torch.from_numpy(events_np).to(device)
    else:
        rng = np.random.default_rng(args.seed)
        events = torch.from_numpy(
            rng.standard_normal((1, 3, 1024), dtype=np.float32)
        ).to(device)
    sample_indices = tuple(
        value.to(device)
        for value in precompute_random_sample_indices(events.shape[0], args.seed)
    )

    with torch.inference_mode():
        started_at = time.perf_counter()
        raw_output = model(events, *sample_indices)
        if device.type == "cuda":
            torch.cuda.synchronize()
        reference_ms = (time.perf_counter() - started_at) * 1000.0
        real_part, imag_part = torch.chunk(raw_output, chunks=2, dim=-1)
        decoded_torch = vsa.Decode_VSA(
            torch.complex(real_part, imag_part),
            matrix_a,
            isELL=True,
        )
    raw_numpy = raw_output.detach().cpu().numpy()
    matrix_numpy = matrix_a.detach().cpu().numpy()
    decoded_numpy = decode_ellipse_vsa(raw_numpy, matrix_numpy)
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

    loop_path = (PROJECT_ROOT / args.loop_output).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()
    matrix_path = (PROJECT_ROOT / args.matrix_output).resolve()
    reference_path = (PROJECT_ROOT / args.reference_output).resolve()
    for path in (loop_path, output_path, matrix_path, reference_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    np.save(matrix_path, matrix_numpy)
    np.savez(
        reference_path,
        events=events.detach().cpu().numpy(),
        sample0=sample_indices[0].detach().cpu().numpy(),
        sample1=sample_indices[1].detach().cpu().numpy(),
        sample2=sample_indices[2].detach().cpu().numpy(),
        raw_output=raw_numpy,
        decoded_output=decoded_torch.detach().cpu().numpy(),
    )

    scripted = torch.jit.script(model)
    export_started_at = time.perf_counter()
    torch.onnx.export(
        scripted,
        (events, *sample_indices),
        str(loop_path),
        input_names=["events", "sample0", "sample1", "sample2"],
        output_names=["prediction"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    normalized_topk = normalize_topk_inputs(
        str(loop_path),
        str(loop_path),
        profile="randla-sample",
    )
    replacements = replace_loops(str(loop_path), str(output_path))
    print(
        json.dumps(
            {
                "status": "exported",
                "loop_model": str(loop_path),
                "output": str(output_path),
                "matrix": str(matrix_path),
                "reference": str(reference_path),
                "raw_output_shape": list(raw_numpy.shape),
                "decoded_output": decoded_numpy.tolist(),
                "decoder_max_abs_error": decoder_error,
                "reference_ms": reference_ms,
                "export_ms": (time.perf_counter() - export_started_at) * 1000.0,
                "selective_scan_replacements": len(replacements),
                "normalized_topk_inputs": normalized_topk,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
