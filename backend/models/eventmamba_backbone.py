import torch
import torch.nn as nn

from .mamba_layer import MambaBlock
from .modules import LocalGrouper


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, output):
        weights = self.linear(output).squeeze(-1)
        return torch.softmax(weights, dim=1)


class Linear1Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True):
        super().__init__()
        self.act = nn.ReLU(inplace=True)
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                bias=bias,
            ),
            nn.BatchNorm1d(out_channels),
            self.act,
        )

    def forward(self, x):
        return self.net(x)


class Linear2Layer(nn.Module):
    def __init__(self, in_channels, kernel_size=1, groups=1, bias=True):
        super().__init__()
        hidden_channels = in_channels // 2
        self.act = nn.ReLU(inplace=True)
        self.net1 = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm1d(hidden_channels),
            self.act,
        )
        self.net2 = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                bias=bias,
            ),
            nn.BatchNorm1d(in_channels),
        )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


class EventMambaBackbone(nn.Module):
    """Shared EventMamba feature extractor with checkpoint-stable names."""

    def __init__(self):
        super().__init__()
        bimamba_type = "v2"
        self.feature_list = [6, 64, 128, 256]
        self.group = LocalGrouper(3, 512, 24, False, "anchor")
        self.group_1 = LocalGrouper(
            self.feature_list[1], 256, 24, False, "anchor"
        )
        self.group_2 = LocalGrouper(
            self.feature_list[2], 128, 24, False, "anchor"
        )
        self.embed_dim = Linear1Layer(self.feature_list[0], self.feature_list[1])
        self.conv1 = Linear2Layer(self.feature_list[1])
        self.conv1_1 = Linear2Layer(self.feature_list[1])
        self.conv2 = Linear2Layer(self.feature_list[2])
        self.conv2_1 = Linear2Layer(self.feature_list[2])
        self.conv3 = Linear2Layer(self.feature_list[3])
        self.conv3_1 = Linear2Layer(self.feature_list[3])
        self.mamba1 = MambaBlock(
            dim=self.feature_list[1], layer_idx=0, bimamba_type=bimamba_type
        )
        self.mamba2 = MambaBlock(
            dim=self.feature_list[2], layer_idx=1, bimamba_type=bimamba_type
        )
        self.mamba3 = MambaBlock(
            dim=self.feature_list[3], layer_idx=2, bimamba_type=bimamba_type
        )
        self.attention_1 = Attention(self.feature_list[1])
        self.attention_2 = Attention(self.feature_list[2])
        self.attention_3 = Attention(self.feature_list[3])
        self.attention_4 = Attention(self.feature_list[3])

    def forward_features(self, x):
        xyz = x.permute(0, 2, 1)
        xyz, x = self.group(xyz, x.permute(0, 2, 1))
        x = x.permute(0, 1, 3, 2)
        batch_groups, group_count, channels, neighbors = x.size()
        x = x.reshape(-1, channels, neighbors)
        x = self.conv1(self.embed_dim(x))
        x = x.permute(0, 2, 1)
        weights = self.attention_1(x)
        x = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        x = x.reshape(batch_groups, group_count, -1)
        x, _ = self.mamba1(x)
        x = self.conv1_1(x.permute(0, 2, 1)).permute(0, 2, 1)

        xyz, x = self.group_1(xyz, x)
        x = x.permute(0, 1, 3, 2)
        batch_groups, group_count, channels, neighbors = x.size()
        x = self.conv2(x.reshape(-1, channels, neighbors)).permute(0, 2, 1)
        weights = self.attention_2(x)
        x = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        x = x.reshape(batch_groups, group_count, -1)
        x, _ = self.mamba2(x)
        x = self.conv2_1(x.permute(0, 2, 1)).permute(0, 2, 1)

        _, x = self.group_2(xyz, x)
        x = x.permute(0, 1, 3, 2)
        batch_groups, group_count, channels, neighbors = x.size()
        x = self.conv3(x.reshape(-1, channels, neighbors)).permute(0, 2, 1)
        weights = self.attention_3(x)
        x = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        x = x.reshape(batch_groups, group_count, -1)
        x, _ = self.mamba3(x)
        x = self.conv3_1(x.permute(0, 2, 1)).permute(0, 2, 1)

        weights = self.attention_4(x)
        return torch.bmm(weights.unsqueeze(1), x).squeeze(1)

    def forward(self, x):
        return self.classifier(self.forward_features(x))
