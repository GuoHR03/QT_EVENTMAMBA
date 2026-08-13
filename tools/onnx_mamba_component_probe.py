"""ONNX probe for an exportable reference implementation of BiMamba v2."""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.models.mamba_layer import MambaBlock
from tools.onnx_selective_scan_loop_probe import ScriptedSelectiveScan


class ReferenceScanBranch(nn.Module):
    def __init__(self, conv, x_proj, dt_proj, a_log, d_skip):
        super().__init__()
        self.conv = conv
        self.x_proj = x_proj
        self.dt_proj = dt_proj
        self.a_log = a_log
        self.d_skip = d_skip
        self.scan = torch.jit.script(ScriptedSelectiveScan())

    def forward(self, xz):
        _, _, sequence_length = xz.shape
        x, z = xz.chunk(2, dim=1)
        x = F.silu(self.conv(x)[..., :sequence_length])
        x_dbl = x.transpose(1, 2).reshape(-1, x.shape[1])
        x_dbl = F.linear(x_dbl, self.x_proj.weight)
        dt_rank = self.dt_proj.weight.shape[1]
        d_state = self.a_log.shape[1]
        dt, b_term, c_term = torch.split(
            x_dbl, [dt_rank, d_state, d_state], dim=-1
        )
        batch = x.shape[0]
        dt = F.linear(dt, self.dt_proj.weight).transpose(0, 1)
        dt = dt.reshape(x.shape[1], batch, sequence_length).permute(1, 0, 2)
        b_term = b_term.reshape(batch, sequence_length, d_state).transpose(1, 2)
        c_term = c_term.reshape(batch, sequence_length, d_state).transpose(1, 2)
        a_term = -torch.exp(self.a_log.float())
        return self.scan(
            x, dt, a_term, b_term, c_term, self.d_skip.float(), z,
            self.dt_proj.bias.float(),
        )


class ReferenceBiMambaV2(nn.Module):
    """Pure PyTorch equivalent of the project's fused BiMamba v2 path."""

    def __init__(self, block, use_onnx_loop=False):
        super().__init__()
        mixer = block.mixer
        self.norm = block.norm
        self.in_proj = mixer.in_proj
        self.forward_branch = ReferenceScanBranch(
            mixer.conv1d, mixer.x_proj, mixer.dt_proj, mixer.A_log, mixer.D
        )
        self.backward_branch = ReferenceScanBranch(
            mixer.conv1d_b,
            mixer.x_proj_b,
            mixer.dt_proj_b,
            mixer.A_b_log,
            mixer.D_b,
        )
        self.out_proj = mixer.out_proj
        self.if_divide_out = mixer.if_divide_out

    def forward(self, hidden_states):
        residual = hidden_states
        normalized = self.norm(residual.to(self.norm.weight.dtype))
        xz = self.in_proj(normalized).transpose(1, 2)
        forward = self.forward_branch(xz)
        backward = self.backward_branch(xz.flip(-1)).flip(-1)
        combined = (forward + backward).transpose(1, 2)
        if self.if_divide_out:
            combined = combined / 2
        output = self.out_proj(combined)
        return output, residual


def _max_error(actual, expected):
    return float(np.max(np.abs(actual - expected)))


def main():
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fast_block = MambaBlock(dim=64, layer_idx=0, bimamba_type="v2").to(device).eval()
    reference_block = ReferenceBiMambaV2(fast_block).to(device).eval()
    sample = torch.randn(1, 16, 64, device=device)

    print(f"[1/3] comparing fused and reference paths on {device}", flush=True)
    with torch.inference_mode():
        fast_output, fast_residual = fast_block(sample)
        reference_output, reference_residual = reference_block(sample)
    reference_error = torch.max(torch.abs(fast_output - reference_output)).item()
    print(f"[1/3] fused/reference max error={reference_error:.9g}", flush=True)

    output = PROJECT_ROOT / "artifacts" / "mamba_block_reference.onnx"
    output.parent.mkdir(parents=True, exist_ok=True)
    print("[2/3] exporting reference BiMamba block", flush=True)
    scripted_reference = torch.jit.script(reference_block)
    torch.onnx.export(
        scripted_reference,
        (sample,),
        str(output),
        input_names=["hidden_states"],
        output_names=["output", "residual"],
        opset_version=17,
        do_constant_folding=True,
    )

    print("[3/3] validating with ONNX Runtime", flush=True)
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(str(output)))
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    actual, actual_residual = session.run(
        None, {"hidden_states": sample.detach().cpu().numpy()}
    )
    expected = fast_output.detach().cpu().numpy()
    expected_residual = fast_residual.detach().cpu().numpy()
    max_abs_error = _max_error(actual, expected)
    residual_max_abs_error = _max_error(actual_residual, expected_residual)
    result = {
        "status": "verified" if np.allclose(actual, expected, rtol=1e-3, atol=1e-4) else "mismatch",
        "component": "ReferenceBiMambaV2",
        "onnx_path": str(output),
        "onnx_bytes": output.stat().st_size,
        "fused_reference_max_abs_error": reference_error,
        "onnx_fast_max_abs_error": max_abs_error,
        "residual_max_abs_error": residual_max_abs_error,
        "allclose": bool(
            np.allclose(actual, expected, rtol=1e-3, atol=1e-4)
            and np.allclose(
                actual_residual, expected_residual, rtol=1e-3, atol=1e-4
            )
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if result["allclose"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
