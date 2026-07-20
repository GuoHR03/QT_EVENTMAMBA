"""Compare the full EventMamba model with exportable reference Mamba blocks."""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.models.eventmamba_v1 import EventMamba
from tools.onnx_feasibility import _extract_state_dict
from tools.onnx_mamba_component_probe import ReferenceBiMambaV2


def _replace_mamba_blocks(model):
    for name in ("mamba1", "mamba2", "mamba3"):
        setattr(model, name, ReferenceBiMambaV2(getattr(model, name)))
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.weights, map_location=device)
    fast_model = EventMamba(num_classes=2).to(device).eval()
    fast_model.load_state_dict(_extract_state_dict(checkpoint))
    reference_model = _replace_mamba_blocks(copy.deepcopy(fast_model)).eval()
    sample = torch.randn(1, 3, 1024, device=device)

    torch.manual_seed(args.seed)
    started = time.perf_counter()
    with torch.inference_mode():
        fast_output = fast_model(sample)
    fast_seconds = time.perf_counter() - started

    torch.manual_seed(args.seed)
    started = time.perf_counter()
    with torch.inference_mode():
        reference_output = reference_model(sample)
    reference_seconds = time.perf_counter() - started

    max_abs_error = torch.max(torch.abs(fast_output - reference_output)).item()
    allclose = torch.allclose(
        fast_output, reference_output, rtol=1e-3, atol=1e-4
    )
    print(
        json.dumps(
            {
                "status": "verified" if allclose else "mismatch",
                "input_shape": list(sample.shape),
                "output_shape": list(fast_output.shape),
                "fast_seconds": fast_seconds,
                "reference_seconds": reference_seconds,
                "max_abs_error": max_abs_error,
                "allclose": bool(allclose),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0 if allclose else 2


if __name__ == "__main__":
    raise SystemExit(main())
