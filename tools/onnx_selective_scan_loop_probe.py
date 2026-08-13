"""Verify exporting selective scan as an ONNX Loop instead of unrolling it."""

import json
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ScriptedSelectiveScan(nn.Module):
    def forward(self, u, delta, a_term, b_term, c_term, d_skip, z, delta_bias):
        input_dtype = u.dtype
        u_float = u.float()
        delta_float = F.softplus(delta.float() + delta_bias[:, None].float())
        b_float = b_term.float()
        c_float = c_term.float()
        state = torch.zeros(
            (u.shape[0], u.shape[1], a_term.shape[1]),
            dtype=a_term.dtype,
            device=u.device,
        )
        delta_a = torch.exp(torch.einsum("bdl,dn->bdln", delta_float, a_term))
        delta_b_u = torch.einsum(
            "bdl,bnl,bdl->bdln", delta_float, b_float, u_float
        )
        outputs = torch.jit.annotate(List[torch.Tensor], [])
        for index in range(u.shape[2]):
            state = delta_a[:, :, index] * state + delta_b_u[:, :, index]
            value = torch.einsum("bdn,bn->bd", state, c_float[:, :, index])
            outputs.append(value)
        result = torch.stack(outputs, dim=2)
        result = result + u_float * d_skip[:, None]
        result = result * F.silu(z)
        return result.to(dtype=input_dtype)


def main():
    from mamba_ssm.ops.selective_scan_interface import selective_scan_ref

    torch.manual_seed(7)
    batch, channels, length, state_size = 1, 8, 16, 4
    u = torch.randn(batch, channels, length)
    delta = torch.randn(batch, channels, length)
    a_term = -torch.exp(torch.randn(channels, state_size))
    b_term = torch.randn(batch, state_size, length)
    c_term = torch.randn(batch, state_size, length)
    d_skip = torch.randn(channels)
    z = torch.randn(batch, channels, length)
    delta_bias = torch.randn(channels)
    inputs = (u, delta, a_term, b_term, c_term, d_skip, z, delta_bias)

    module = ScriptedSelectiveScan().eval()
    scripted = torch.jit.script(module)
    expected = selective_scan_ref(
        u,
        delta,
        a_term,
        b_term,
        c_term,
        d_skip,
        z=z,
        delta_bias=delta_bias,
        delta_softplus=True,
    )
    scripted_output = scripted(*inputs)
    scripted_error = torch.max(torch.abs(expected - scripted_output)).item()

    output = PROJECT_ROOT / "artifacts" / "selective_scan_loop.onnx"
    output.parent.mkdir(parents=True, exist_ok=True)
    names = ["u", "delta", "A", "B", "C", "D", "z", "delta_bias"]
    torch.onnx.export(
        scripted,
        inputs,
        str(output),
        input_names=names,
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
    )

    import onnx
    import onnxruntime as ort

    graph = onnx.load(str(output))
    onnx.checker.check_model(graph)
    operation_types = [node.op_type for node in graph.graph.node]
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    actual = session.run(
        None, {name: value.numpy() for name, value in zip(names, inputs)}
    )[0]
    expected_np = expected.numpy()
    ort_error = float(np.max(np.abs(actual - expected_np)))
    allclose = bool(np.allclose(actual, expected_np, rtol=1e-3, atol=1e-4))
    result = {
        "status": "verified" if allclose and "Loop" in operation_types else "failed",
        "scripted_reference_max_abs_error": scripted_error,
        "onnxruntime_max_abs_error": ort_error,
        "onnx_top_level_nodes": len(graph.graph.node),
        "contains_loop": "Loop" in operation_types,
        "operation_types": operation_types,
        "onnx_bytes": output.stat().st_size,
        "allclose": allclose,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if result["status"] == "verified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
