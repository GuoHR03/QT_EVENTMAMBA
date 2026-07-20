"""Validate moving farthest-point sampling outside the ONNX model."""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.models.eventmamba_v1 import EventMamba
from backend.models.modules import furthest_point_sample, index_points, square_distance
from tools.onnx_feasibility import _extract_state_dict
from tools.onnx_mamba_component_probe import ReferenceBiMambaV2


def exportable_index_points(points, indices):
    channels = points.shape[-1]
    if indices.dim() == 2:
        gather_indices = indices.unsqueeze(-1).expand(-1, -1, channels)
        return torch.gather(points, 1, gather_indices)
    expanded_points = points.unsqueeze(1).expand(
        -1, indices.shape[1], -1, -1
    )
    gather_indices = indices.unsqueeze(-1).expand(-1, -1, -1, channels)
    return torch.gather(expanded_points, 2, gather_indices)


def exportable_square_distance(source, target):
    distance = -2 * torch.matmul(source, target.permute(0, 2, 1))
    distance = distance + torch.sum(source ** 2, -1).unsqueeze(-1)
    distance = distance + torch.sum(target ** 2, -1).unsqueeze(1)
    return distance


class IndexedLocalGrouper(nn.Module):
    """LocalGrouper with FPS indices supplied as an explicit model input."""

    def __init__(self, source):
        super().__init__()
        self.groups = source.groups
        self.kneighbors = source.kneighbors
        self.use_xyz = source.use_xyz
        self.normalize = source.normalize
        if source.normalize is not None:
            self.affine_alpha = source.affine_alpha
            self.affine_beta = source.affine_beta

    def forward(self, xyz, points, fps_idx):
        batch, _, _ = xyz.shape
        fps_idx, _ = torch.sort(fps_idx.long(), dim=1)
        new_xyz = exportable_index_points(xyz, fps_idx)
        new_points = exportable_index_points(points, fps_idx)
        distances = exportable_square_distance(new_xyz, xyz)
        neighbor_idx = distances.argsort()[:, :, : self.kneighbors]
        neighbor_idx = neighbor_idx.sort(dim=-1)[0]
        grouped_xyz = exportable_index_points(xyz, neighbor_idx)
        grouped_points = exportable_index_points(points, neighbor_idx)
        if self.use_xyz:
            grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
        if self.normalize is not None:
            if self.normalize == "center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            else:
                mean = (
                    torch.cat([new_points, new_xyz], dim=-1)
                    if self.use_xyz
                    else new_points
                ).unsqueeze(-2)
            std = torch.std(
                (grouped_points - mean).reshape(batch, -1),
                dim=-1,
                keepdim=True,
            ).unsqueeze(-1).unsqueeze(-1)
            grouped_points = (grouped_points - mean) / (std + 1e-5)
            grouped_points = self.affine_alpha * grouped_points + self.affine_beta
        repeated_anchor = new_points.reshape(batch, self.groups, 1, -1).repeat(
            1, 1, self.kneighbors, 1
        )
        return new_xyz, torch.cat([grouped_points, repeated_anchor], dim=-1)


class PrecomputedFpsCenterModel(nn.Module):
    def __init__(self, source_model, use_onnx_loop=False):
        super().__init__()
        self.group0 = IndexedLocalGrouper(source_model.group)
        self.group1 = IndexedLocalGrouper(source_model.group_1)
        self.group2 = IndexedLocalGrouper(source_model.group_2)
        for name in (
            "embed_dim", "conv1", "conv1_1", "conv2", "conv2_1",
            "conv3", "conv3_1", "attention_1", "attention_2",
            "attention_3", "attention_4", "classifier",
        ):
            setattr(self, name, getattr(source_model, name))
        self.mamba1 = ReferenceBiMambaV2(
            source_model.mamba1, use_onnx_loop=use_onnx_loop
        )
        self.mamba2 = ReferenceBiMambaV2(
            source_model.mamba2, use_onnx_loop=use_onnx_loop
        )
        self.mamba3 = ReferenceBiMambaV2(
            source_model.mamba3, use_onnx_loop=use_onnx_loop
        )

    def forward(self, x, fps0, fps1, fps2):
        xyz = x.permute(0, 2, 1)
        xyz, features = self.group0(xyz, x.permute(0, 2, 1), fps0)
        batch, groups, channels, neighbors = features.permute(0, 1, 3, 2).shape
        features = features.permute(0, 1, 3, 2).reshape(-1, channels, neighbors)
        features = self.conv1(self.embed_dim(features))
        features = features.permute(0, 2, 1)
        weights = self.attention_1(features)
        features = torch.bmm(weights.unsqueeze(1), features).squeeze(1)
        features = features.reshape(batch, groups, -1)
        features, _ = self.mamba1(features)
        features = self.conv1_1(features.permute(0, 2, 1)).permute(0, 2, 1)

        xyz, features = self.group1(xyz, features, fps1)
        batch, groups, channels, neighbors = features.permute(0, 1, 3, 2).shape
        features = features.permute(0, 1, 3, 2).reshape(-1, channels, neighbors)
        features = self.conv2(features).permute(0, 2, 1)
        weights = self.attention_2(features)
        features = torch.bmm(weights.unsqueeze(1), features).squeeze(1)
        features = features.reshape(batch, groups, -1)
        features, _ = self.mamba2(features)
        features = self.conv2_1(features.permute(0, 2, 1)).permute(0, 2, 1)

        xyz, features = self.group2(xyz, features, fps2)
        batch, groups, channels, neighbors = features.permute(0, 1, 3, 2).shape
        features = features.permute(0, 1, 3, 2).reshape(-1, channels, neighbors)
        features = self.conv3(features).permute(0, 2, 1)
        weights = self.attention_3(features)
        features = torch.bmm(weights.unsqueeze(1), features).squeeze(1)
        features = features.reshape(batch, groups, -1)
        features, _ = self.mamba3(features)
        features = self.conv3_1(features.permute(0, 2, 1)).permute(0, 2, 1)
        weights = self.attention_4(features)
        features = torch.bmm(weights.unsqueeze(1), features).squeeze(1)
        return self.classifier(features)


def precompute_fps_indices(events, seed):
    torch.manual_seed(seed)
    xyz0 = events.permute(0, 2, 1)
    fps0 = furthest_point_sample(xyz0, 512).long()
    xyz1 = index_points(xyz0, torch.sort(fps0, dim=1)[0])
    fps1 = furthest_point_sample(xyz1, 256).long()
    xyz2 = index_points(xyz1, torch.sort(fps1, dim=1)[0])
    fps2 = furthest_point_sample(xyz2, 128).long()
    return fps0, fps1, fps2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--export", action="store_true")
    parser.add_argument(
        "--ort-provider",
        default="CPUExecutionProvider",
        choices=["CPUExecutionProvider", "CUDAExecutionProvider"],
    )
    parser.add_argument("--ort-repeats", type=int, default=1)
    parser.add_argument(
        "--output", default="artifacts/eventmamba_center_precomputed_fps.onnx"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    model = EventMamba(num_classes=2).to(device).eval()
    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(_extract_state_dict(checkpoint))
    wrapper = PrecomputedFpsCenterModel(
        copy.deepcopy(model), use_onnx_loop=args.export
    ).to(device).eval()
    sample = torch.randn(1, 3, 1024, device=device)
    fps_indices = precompute_fps_indices(sample, args.seed)

    torch.manual_seed(args.seed)
    with torch.inference_mode():
        original = model(sample)
    started = time.perf_counter()
    with torch.inference_mode():
        precomputed = wrapper(sample, *fps_indices)
    elapsed = time.perf_counter() - started
    max_abs_error = torch.max(torch.abs(original - precomputed)).item()
    allclose = torch.allclose(original, precomputed, rtol=1e-3, atol=1e-4)
    print(
        json.dumps(
            {
                "status": "verified" if allclose else "mismatch",
                "input_shape": list(sample.shape),
                "fps_shapes": [list(value.shape) for value in fps_indices],
                "output_shape": list(original.shape),
                "wrapper_seconds": elapsed,
                "max_abs_error": max_abs_error,
                "allclose": bool(allclose),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if not allclose or not args.export:
        return 0 if allclose else 2

    output_path = (PROJECT_ROOT / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[export] writing {output_path}", flush=True)
    print("[export] scripting complete wrapper to preserve ONNX Loop", flush=True)
    export_model = torch.jit.script(wrapper)
    export_started = time.perf_counter()
    torch.onnx.export(
        export_model,
        (sample, *fps_indices),
        str(output_path),
        input_names=["events", "fps0", "fps1", "fps2"],
        output_names=["prediction"],
        opset_version=17,
        do_constant_folding=True,
    )
    export_seconds = time.perf_counter() - export_started
    print(f"[export] completed in {export_seconds:.2f}s", flush=True)

    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(str(output_path)))
    available_providers = ort.get_available_providers()
    if args.ort_provider not in available_providers:
        raise RuntimeError(
            f"Requested {args.ort_provider}, available: {available_providers}"
        )
    session = ort.InferenceSession(
        str(output_path), providers=[args.ort_provider]
    )
    if session.get_providers()[0] != args.ort_provider:
        raise RuntimeError(
            f"Provider initialization fell back to {session.get_providers()[0]}"
        )
    ort_inputs = {
        "events": sample.detach().cpu().numpy(),
        "fps0": fps_indices[0].detach().cpu().numpy(),
        "fps1": fps_indices[1].detach().cpu().numpy(),
        "fps2": fps_indices[2].detach().cpu().numpy(),
    }
    session.run(None, ort_inputs)
    ort_times = []
    actual = None
    for _ in range(args.ort_repeats):
        ort_started = time.perf_counter()
        actual = session.run(None, ort_inputs)[0]
        ort_times.append(time.perf_counter() - ort_started)
    ort_seconds = sum(ort_times) / len(ort_times)
    expected = original.detach().cpu().numpy()
    ort_error = float(np.max(np.abs(actual - expected)))
    ort_allclose = bool(np.allclose(actual, expected, rtol=1e-3, atol=1e-4))
    print(
        json.dumps(
            {
                "status": "verified" if ort_allclose else "mismatch",
                "onnx_path": str(output_path),
                "onnx_bytes": output_path.stat().st_size,
                "export_seconds": export_seconds,
                "onnxruntime_provider": session.get_providers()[0],
                "onnxruntime_seconds": ort_seconds,
                "onnxruntime_min_seconds": min(ort_times),
                "onnx_max_abs_error": ort_error,
                "allclose": ort_allclose,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0 if ort_allclose else 3


if __name__ == "__main__":
    raise SystemExit(main())
