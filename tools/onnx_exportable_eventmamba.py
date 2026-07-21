"""Pure-PyTorch EventMamba graph used to export checkpoints without mamba_ssm."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.models.modules import LocalGrouper, furthest_point_sample, index_points
from tools.onnx_selective_scan_loop_probe import ScriptedSelectiveScan


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, output):
        return torch.softmax(self.linear(output).squeeze(-1), dim=1)


class Linear1Layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.act = nn.ReLU(inplace=True)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            self.act,
        )

    def forward(self, values):
        return self.net(values)


class Linear2Layer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.act = nn.ReLU(inplace=True)
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // 2, 1),
            nn.BatchNorm1d(in_channels // 2),
            self.act,
        )
        self.net2 = nn.Sequential(
            nn.Conv1d(in_channels // 2, in_channels, 1),
            nn.BatchNorm1d(in_channels),
        )

    def forward(self, values):
        return self.act(self.net2(self.net1(values)) + values)


class CheckpointMambaMixer(nn.Module):
    """Parameter holder whose names and shapes match mamba_ssm.Mamba."""

    def __init__(self, dimension, state_size=16, convolution_size=4):
        super().__init__()
        inner = dimension * 2
        dt_rank = (dimension + 15) // 16
        self.A_log = nn.Parameter(torch.empty(inner, state_size))
        self.D = nn.Parameter(torch.empty(inner))
        self.A_b_log = nn.Parameter(torch.empty(inner, state_size))
        self.D_b = nn.Parameter(torch.empty(inner))
        self.in_proj = nn.Linear(dimension, inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            inner,
            inner,
            convolution_size,
            groups=inner,
            padding=convolution_size - 1,
        )
        self.x_proj = nn.Linear(inner, dt_rank + state_size * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, inner)
        self.conv1d_b = nn.Conv1d(
            inner,
            inner,
            convolution_size,
            groups=inner,
            padding=convolution_size - 1,
        )
        self.x_proj_b = nn.Linear(inner, dt_rank + state_size * 2, bias=False)
        self.dt_proj_b = nn.Linear(dt_rank, inner)
        self.out_proj = nn.Linear(inner, dimension, bias=False)


class CheckpointMambaBlock(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.mixer = CheckpointMambaMixer(dimension)
        self.norm = nn.LayerNorm(dimension)


class EllipseCheckpointModel(nn.Module):
    """Ellipse architecture used only to load the original state_dict strictly."""

    def __init__(self):
        super().__init__()
        self.group = LocalGrouper(3, 512, 24, False, "anchor")
        self.group_1 = LocalGrouper(64, 256, 24, False, "anchor")
        self.group_2 = LocalGrouper(128, 128, 24, False, "anchor")
        self.embed_dim = Linear1Layer(6, 64)
        self.conv1 = Linear2Layer(64)
        self.conv1_1 = Linear2Layer(64)
        self.conv2 = Linear2Layer(128)
        self.conv2_1 = Linear2Layer(128)
        self.conv3 = Linear2Layer(256)
        self.conv3_1 = Linear2Layer(256)
        self.mamba1 = CheckpointMambaBlock(64)
        self.mamba2 = CheckpointMambaBlock(128)
        self.mamba3 = CheckpointMambaBlock(256)
        self.attention_1 = Attention(64)
        self.attention_2 = Attention(128)
        self.attention_3 = Attention(256)
        self.attention_4 = Attention(256)
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
        )


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
        batch = xyz.shape[0]
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


class ExportableScanBranch(nn.Module):
    def __init__(self, conv, x_proj, dt_proj, a_log, d_skip):
        super().__init__()
        self.conv = conv
        self.x_proj = x_proj
        self.dt_proj = dt_proj
        self.a_log = a_log
        self.d_skip = d_skip
        self.scan = torch.jit.script(ScriptedSelectiveScan())

    def forward(self, xz):
        sequence_length = xz.shape[2]
        x, z = xz.chunk(2, dim=1)
        x = F.silu(self.conv(x)[..., :sequence_length])
        x_dbl = x.transpose(1, 2).reshape(-1, x.shape[1])
        x_dbl = F.linear(x_dbl, self.x_proj.weight)
        dt_rank = self.dt_proj.weight.shape[1]
        state_size = self.a_log.shape[1]
        dt, b_term, c_term = torch.split(
            x_dbl,
            [dt_rank, state_size, state_size],
            dim=-1,
        )
        batch = x.shape[0]
        dt = F.linear(dt, self.dt_proj.weight).transpose(0, 1)
        dt = dt.reshape(x.shape[1], batch, sequence_length).permute(1, 0, 2)
        b_term = b_term.reshape(batch, sequence_length, state_size).transpose(1, 2)
        c_term = c_term.reshape(batch, sequence_length, state_size).transpose(1, 2)
        return self.scan(
            x,
            dt,
            -torch.exp(self.a_log.float()),
            b_term,
            c_term,
            self.d_skip.float(),
            z,
            self.dt_proj.bias.float(),
        )


class ExportableBiMamba(nn.Module):
    def __init__(self, block):
        super().__init__()
        mixer = block.mixer
        self.norm = block.norm
        self.in_proj = mixer.in_proj
        self.forward_branch = ExportableScanBranch(
            mixer.conv1d,
            mixer.x_proj,
            mixer.dt_proj,
            mixer.A_log,
            mixer.D,
        )
        self.backward_branch = ExportableScanBranch(
            mixer.conv1d_b,
            mixer.x_proj_b,
            mixer.dt_proj_b,
            mixer.A_b_log,
            mixer.D_b,
        )
        self.out_proj = mixer.out_proj

    def forward(self, hidden_states):
        residual = hidden_states
        normalized = self.norm(residual.to(self.norm.weight.dtype))
        xz = self.in_proj(normalized).transpose(1, 2)
        forward = self.forward_branch(xz)
        backward = self.backward_branch(xz.flip(-1)).flip(-1)
        output = self.out_proj((forward + backward).transpose(1, 2))
        return output, residual


class PrecomputedFpsEllipseModel(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.group0 = IndexedLocalGrouper(source.group)
        self.group1 = IndexedLocalGrouper(source.group_1)
        self.group2 = IndexedLocalGrouper(source.group_2)
        for name in (
            "embed_dim",
            "conv1",
            "conv1_1",
            "conv2",
            "conv2_1",
            "conv3",
            "conv3_1",
            "attention_1",
            "attention_2",
            "attention_3",
            "attention_4",
            "classifier",
        ):
            setattr(self, name, getattr(source, name))
        self.mamba1 = ExportableBiMamba(source.mamba1)
        self.mamba2 = ExportableBiMamba(source.mamba2)
        self.mamba3 = ExportableBiMamba(source.mamba3)

    def forward(self, events, fps0, fps1, fps2):
        xyz = events.permute(0, 2, 1)
        xyz, features = self.group0(xyz, events.permute(0, 2, 1), fps0)
        features = self._encode_group0(features)
        batch = events.shape[0]
        features = features.reshape(batch, self.group0.groups, -1)
        features, _ = self.mamba1(features)
        features = self.conv1_1(features.permute(0, 2, 1)).permute(0, 2, 1)

        xyz, features = self.group1(xyz, features, fps1)
        features = self._encode_group1(features)
        features = features.reshape(batch, self.group1.groups, -1)
        features, _ = self.mamba2(features)
        features = self.conv2_1(features.permute(0, 2, 1)).permute(0, 2, 1)

        xyz, features = self.group2(xyz, features, fps2)
        features = self._encode_group2(features)
        features = features.reshape(batch, self.group2.groups, -1)
        features, _ = self.mamba3(features)
        features = self.conv3_1(features.permute(0, 2, 1)).permute(0, 2, 1)
        weights = self.attention_4(features)
        features = torch.bmm(weights.unsqueeze(1), features).squeeze(1)
        return self.classifier(features)

    def _encode_group0(self, features):
        features = features.permute(0, 1, 3, 2)
        channels = features.shape[2]
        neighbors = features.shape[3]
        features = features.reshape(-1, channels, neighbors)
        features = self.conv1(self.embed_dim(features)).permute(0, 2, 1)
        weights = self.attention_1(features)
        return torch.bmm(weights.unsqueeze(1), features).squeeze(1)

    def _encode_group1(self, features):
        features = features.permute(0, 1, 3, 2)
        channels = features.shape[2]
        neighbors = features.shape[3]
        features = self.conv2(
            features.reshape(-1, channels, neighbors)
        ).permute(0, 2, 1)
        weights = self.attention_2(features)
        return torch.bmm(weights.unsqueeze(1), features).squeeze(1)

    def _encode_group2(self, features):
        features = features.permute(0, 1, 3, 2)
        channels = features.shape[2]
        neighbors = features.shape[3]
        features = self.conv3(
            features.reshape(-1, channels, neighbors)
        ).permute(0, 2, 1)
        weights = self.attention_3(features)
        return torch.bmm(weights.unsqueeze(1), features).squeeze(1)


def precompute_fps_indices(events, seed=7):
    torch.manual_seed(seed)
    xyz0 = events.permute(0, 2, 1)
    fps0 = furthest_point_sample(xyz0, 512).long()
    xyz1 = index_points(xyz0, torch.sort(fps0, dim=1)[0])
    fps1 = furthest_point_sample(xyz1, 256).long()
    xyz2 = index_points(xyz1, torch.sort(fps1, dim=1)[0])
    fps2 = furthest_point_sample(xyz2, 128).long()
    return fps0, fps1, fps2
