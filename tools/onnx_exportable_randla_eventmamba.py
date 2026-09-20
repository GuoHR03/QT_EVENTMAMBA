"""Exportable RandLA random-sample EventMamba architecture."""

import numpy as np
import torch
import torch.nn as nn

from tools.onnx_exportable_eventmamba import (
    Attention,
    CheckpointMambaBlock,
    ExportableBiMamba,
    Linear1Layer,
    Linear2Layer,
    exportable_index_points,
    exportable_square_distance,
)


class FirstRandLACheckpointGrouper(nn.Module):
    def __init__(self, groups=512, neighbors=24, hidden_channel=16):
        super().__init__()
        self.groups = groups
        self.kneighbors = neighbors
        self.channel = 3
        self.affine_alpha = nn.Parameter(torch.ones(1, 1, 1, self.channel))
        self.affine_beta = nn.Parameter(torch.zeros(1, 1, 1, self.channel))
        self.geo_mlp = nn.Sequential(
            nn.Conv1d(10, hidden_channel, 1, bias=False),
            nn.BatchNorm1d(hidden_channel),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channel, self.channel, 1, bias=False),
            nn.BatchNorm1d(self.channel),
            nn.ReLU(inplace=True),
        )
        self.feature_mlp = nn.Sequential(
            nn.Conv1d(9, hidden_channel, 1, bias=False),
            nn.BatchNorm1d(hidden_channel),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channel, self.channel, 1, bias=False),
            nn.BatchNorm1d(self.channel),
            nn.ReLU(inplace=True),
        )


class RandLACheckpointGrouper(nn.Module):
    def __init__(self, channel, groups, neighbors=24):
        super().__init__()
        self.groups = groups
        self.kneighbors = neighbors
        self.affine_alpha = nn.Parameter(torch.ones(1, 1, 1, channel))
        self.affine_beta = nn.Parameter(torch.zeros(1, 1, 1, channel))
        self.geo_mlp = nn.Sequential(
            nn.Conv1d(10, channel, 1, bias=False),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
        )
        self.feature_mlp = nn.Sequential(
            nn.Conv1d(channel * 2 + 3, channel, 1, bias=False),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
        )


class RandLAEllipseCheckpointModel(nn.Module):
    """Parameter holder that strictly matches the training checkpoint."""

    def __init__(self):
        super().__init__()
        self.group = FirstRandLACheckpointGrouper()
        self.group_1 = RandLACheckpointGrouper(64, 256)
        self.group_2 = RandLACheckpointGrouper(128, 128)
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


class IndexedRandLAGrouper(nn.Module):
    """RandLA LocFE with externally supplied random sample indices."""

    def __init__(self, source):
        super().__init__()
        self.groups = source.groups
        self.kneighbors = source.kneighbors
        self.affine_alpha = source.affine_alpha
        self.affine_beta = source.affine_beta
        self.geo_mlp = source.geo_mlp
        self.feature_mlp = source.feature_mlp

    def forward(self, xyz, points, sample_idx):
        batch = xyz.shape[0]
        sample_idx, _ = torch.sort(sample_idx.long(), dim=1)
        new_xyz = exportable_index_points(xyz, sample_idx)
        new_points = exportable_index_points(points, sample_idx)

        distances = exportable_square_distance(new_xyz, xyz)
        _, neighbor_idx = torch.topk(
            distances,
            self.kneighbors,
            dim=-1,
            largest=False,
            sorted=False,
        )
        neighbor_idx, _ = torch.sort(neighbor_idx, dim=-1)
        grouped_xyz = exportable_index_points(xyz, neighbor_idx)
        grouped_points = exportable_index_points(points, neighbor_idx)

        center_xyz = new_xyz.unsqueeze(2).expand(
            -1, -1, self.kneighbors, -1
        )
        relative_xyz = grouped_xyz - center_xyz
        distance = torch.linalg.vector_norm(relative_xyz, dim=-1, keepdim=True)
        locse = torch.cat(
            (center_xyz, grouped_xyz, relative_xyz, distance),
            dim=-1,
        )
        geo = locse.reshape(-1, self.kneighbors, 10).permute(0, 2, 1)
        geo = self.geo_mlp(geo).permute(0, 2, 1)
        geo = geo.reshape(batch, self.groups, self.kneighbors, -1)

        anchored = grouped_points - new_points.unsqueeze(2)
        std = torch.std(anchored.reshape(batch, -1), dim=-1, keepdim=True)
        std = std.unsqueeze(-1).unsqueeze(-1)
        anchored = self.affine_alpha * (anchored / (std + 1e-5))
        anchored = anchored + self.affine_beta
        centers = new_points.unsqueeze(2).expand(
            -1, -1, self.kneighbors, -1
        )
        point_feature = torch.cat((grouped_xyz, anchored, centers), dim=-1)
        point_feature = point_feature.reshape(
            -1, self.kneighbors, point_feature.shape[-1]
        ).permute(0, 2, 1)
        point_feature = self.feature_mlp(point_feature).permute(0, 2, 1)
        point_feature = point_feature.reshape(
            batch, self.groups, self.kneighbors, -1
        )
        return new_xyz, torch.cat((point_feature, geo), dim=-1)


class PrecomputedRandomSampleEllipseModel(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.group0 = IndexedRandLAGrouper(source.group)
        self.group1 = IndexedRandLAGrouper(source.group_1)
        self.group2 = IndexedRandLAGrouper(source.group_2)
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

    def forward(self, events, sample0, sample1, sample2):
        batch = events.shape[0]
        xyz = events.permute(0, 2, 1)
        xyz, features = self.group0(xyz, xyz, sample0)
        features = self._encode_group0(features)
        features = features.reshape(batch, self.group0.groups, -1)
        features, _ = self.mamba1(features)
        features = self.conv1_1(features.permute(0, 2, 1)).permute(0, 2, 1)

        xyz, features = self.group1(xyz, features, sample1)
        features = self._encode_group1(features)
        features = features.reshape(batch, self.group1.groups, -1)
        features, _ = self.mamba2(features)
        features = self.conv2_1(features.permute(0, 2, 1)).permute(0, 2, 1)

        xyz, features = self.group2(xyz, features, sample2)
        features = self._encode_group2(features)
        features = features.reshape(batch, self.group2.groups, -1)
        features, _ = self.mamba3(features)
        features = self.conv3_1(features.permute(0, 2, 1)).permute(0, 2, 1)
        weights = self.attention_4(features)
        features = torch.bmm(weights.unsqueeze(1), features).squeeze(1)
        return self.classifier(features)

    def _encode_group0(self, features):
        features = features.permute(0, 1, 3, 2)
        features = features.reshape(-1, features.shape[2], features.shape[3])
        features = self.conv1(self.embed_dim(features)).permute(0, 2, 1)
        weights = self.attention_1(features)
        return torch.bmm(weights.unsqueeze(1), features).squeeze(1)

    def _encode_group1(self, features):
        features = features.permute(0, 1, 3, 2)
        features = features.reshape(-1, features.shape[2], features.shape[3])
        features = self.conv2(features).permute(0, 2, 1)
        weights = self.attention_2(features)
        return torch.bmm(weights.unsqueeze(1), features).squeeze(1)

    def _encode_group2(self, features):
        features = features.permute(0, 1, 3, 2)
        features = features.reshape(-1, features.shape[2], features.shape[3])
        features = self.conv3(features).permute(0, 2, 1)
        weights = self.attention_3(features)
        return torch.bmm(weights.unsqueeze(1), features).squeeze(1)


def precompute_random_sample_indices(batch_size=1, seed=2026):
    rng = np.random.default_rng(seed)
    counts = ((1024, 512), (512, 256), (256, 128))
    result = []
    for population, count in counts:
        indices = np.stack(
            [
                np.sort(rng.choice(population, count, replace=False))
                for _unused in range(batch_size)
            ]
        ).astype(np.int64)
        result.append(torch.from_numpy(indices))
    return tuple(result)
