"""Compare a saved Windows ONNX result with the PyTorch reference on WSL."""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.models.eventmamba_v1 import EventMamba
from tools.onnx_feasibility import _extract_state_dict
from tools.onnx_precomputed_fps_probe import PrecomputedFpsCenterModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    saved = np.load(args.result)
    device = torch.device(args.device)
    model = EventMamba(num_classes=2).to(device).eval()
    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(_extract_state_dict(checkpoint))
    reference = PrecomputedFpsCenterModel(copy.deepcopy(model)).to(device).eval()
    inputs = [
        torch.from_numpy(saved[name]).to(device)
        for name in ("events", "fps0", "fps1", "fps2")
    ]
    with torch.inference_mode():
        expected = reference(*inputs).cpu().numpy()
    actual = saved["output"]
    error = float(np.max(np.abs(actual - expected)))
    allclose = bool(np.allclose(actual, expected, rtol=1e-3, atol=1e-4))
    print(json.dumps({
        "status": "verified" if allclose else "mismatch",
        "windows_onnx_output": actual.tolist(),
        "wsl_pytorch_reference": expected.tolist(),
        "max_abs_error": error,
        "allclose": allclose,
    }, indent=2))
    return 0 if allclose else 2


if __name__ == "__main__":
    raise SystemExit(main())
