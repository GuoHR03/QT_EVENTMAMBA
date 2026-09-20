# VSA-EVENTMAMBA with EventMamba pipeline and RandLA-style LocFE grouper.
# V3-RandSample: keep the v3 LocFE structure, but replace FPS center
# sampling with RandLA-style random sampling.

import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import index_points, square_distance
from .mamba_layer import MambaBlock


def random_point_sample(xyz, npoint):
    B, N, _ = xyz.shape
    scores = torch.rand(B, N, device=xyz.device)
    return scores.argsort(dim=1)[:, :npoint]


#add 
def sort_tokens_by_time(self, xyz, x):
    order = xyz[:, :, 0].argsort(dim=1)
    order_xyz = order.unsqueeze(-1).expand(-1, -1, xyz.size(-1))
    order_x = order.unsqueeze(-1).expand(-1, -1, x.size(-1))
    xyz = torch.gather(xyz, dim=1, index=order_xyz)
    x = torch.gather(x, dim=1, index=order_x)
    return xyz, x



class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, output):
        attn_weights = self.linear(output).squeeze(-1)
        attn_probs = torch.softmax(attn_weights, dim=1)
        return attn_probs
    
class Linear1Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True):
        super(Linear1Layer, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)

class Linear2Layer(nn.Module):
    def __init__(self, in_channels, kernel_size=1, groups=1, bias=True):
        super(Linear2Layer, self).__init__()

        self.act = nn.ReLU(inplace=True)
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=int(in_channels/2),
                    kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(in_channels/2)),
            self.act
        )
        self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(in_channels/2), out_channels=in_channels,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(in_channels)
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


class RandLALocFEGrouper(nn.Module):
    def __init__(self, channel, groups, kneighbors):
        super(RandLALocFEGrouper, self).__init__()
        self.groups = groups
        self.kneighbors = kneighbors
        self.geo_dim = 10
        self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, channel]))
        self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channel]))

        self.geo_mlp = nn.Sequential(
            nn.Conv1d(self.geo_dim, channel, 1, bias=False),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
        )
        self.feature_mlp = nn.Sequential(
            nn.Conv1d(channel * 2 + 3, channel, 1, bias=False),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, xyz, points):
        # xyz: [B, N, 3], points: [B, N, C]
        B, N, _ = xyz.shape
        S = min(self.groups, N)
        K = min(self.kneighbors, N)
        xyz = xyz.contiguous()
        points = points.contiguous()

        sample_idx = random_point_sample(xyz, S).long()  # [B, S]
        sample_idx, _ = torch.sort(sample_idx, dim=1)  # [B, S]
        new_xyz = index_points(xyz, sample_idx)  # [B, S, 3]
        new_points = index_points(points, sample_idx)  # [B, S, C]

        dists = square_distance(new_xyz, xyz)  # [B, S, N]
        idx = dists.argsort()[:, :, :K]  # [B, S, K]
        idx = idx.sort(dim=-1)[0]  # [B, S, K]
        grouped_xyz = index_points(xyz, idx)  # [B, S, K, 3]
        grouped_points = index_points(points, idx)  # [B, S, K, C]

        center_xyz = new_xyz.unsqueeze(2).repeat(1, 1, K, 1)  # [B, S, K, 3]
        relative_xyz = grouped_xyz - center_xyz  # [B, S, K, 3]
        distance = torch.norm(relative_xyz, dim=-1, keepdim=True)  # [B, S, K, 1]
        locse = torch.cat([center_xyz, grouped_xyz, relative_xyz, distance], dim=-1)  # [B, S, K, 10]

        geo = locse.reshape(B * S, K, self.geo_dim).permute(0, 2, 1)  # [B*S, 10, K]
        geo = self.geo_mlp(geo).permute(0, 2, 1).reshape(B, S, K, -1)  # [B, S, K, C]

        anchor = new_points.unsqueeze(dim=-2)  # [B, S, 1, C]
        anchored_points = grouped_points - anchor  # [B, S, K, C]
        std = torch.std(anchored_points.reshape(B, -1), dim=-1, keepdim=True)  # [B, 1]
        std = std.unsqueeze(dim=-1).unsqueeze(dim=-1)  # [B, 1, 1, 1]
        anchored_points = anchored_points / (std + 1e-5)  # [B, S, K, C]
        anchored_points = self.affine_alpha * anchored_points + self.affine_beta  # [B, S, K, C]
        center_points = new_points.view(B, S, 1, -1).repeat(1, 1, K, 1)  # [B, S, K, C]

        point_feature = torch.cat([grouped_xyz, anchored_points, center_points], dim=-1)  # [B, S, K, 2C+3]
        point_feature = point_feature.reshape(B * S, K, -1).permute(0, 2, 1)  # [B*S, 2C+3, K]
        point_feature = self.feature_mlp(point_feature).permute(0, 2, 1).reshape(B, S, K, -1)  # [B, S, K, C]

        new_points = torch.cat([point_feature, geo], dim=-1)  # [B, S, K, 2C]
        return new_xyz, new_points


class FirstLayerRandLALocFEGrouperHidden(nn.Module):
    def __init__(self, groups, kneighbors, hidden_channel=16):
        super(FirstLayerRandLALocFEGrouperHidden, self).__init__()
        self.groups = groups
        self.kneighbors = kneighbors
        self.geo_dim = 10
        self.channel = 3
        self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, self.channel]))
        self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, self.channel]))

        self.geo_mlp = nn.Sequential(
            nn.Conv1d(self.geo_dim, hidden_channel, 1, bias=False),
            nn.BatchNorm1d(hidden_channel),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channel, self.channel, 1, bias=False),
            nn.BatchNorm1d(self.channel),
            nn.ReLU(inplace=True),
        )
        self.feature_mlp = nn.Sequential(
            nn.Conv1d(self.channel * 2 + 3, hidden_channel, 1, bias=False),
            nn.BatchNorm1d(hidden_channel),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channel, self.channel, 1, bias=False),
            nn.BatchNorm1d(self.channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, xyz, points):
        # xyz: [B, N, 3], points: [B, N, 3]
        B, N, _ = xyz.shape
        S = min(self.groups, N)
        K = min(self.kneighbors, N)
        xyz = xyz.contiguous()
        points = points.contiguous()

        sample_idx = random_point_sample(xyz, S).long()  # [B, S]
        sample_idx, _ = torch.sort(sample_idx, dim=1)  # [B, S]
        new_xyz = index_points(xyz, sample_idx)  # [B, S, 3]
        new_points = index_points(points, sample_idx)  # [B, S, 3]

        dists = square_distance(new_xyz, xyz)  # [B, S, N]
        idx = dists.argsort()[:, :, :K]  # [B, S, K]
        idx = idx.sort(dim=-1)[0]  # [B, S, K]
        grouped_xyz = index_points(xyz, idx)  # [B, S, K, 3]
        grouped_points = index_points(points, idx)  # [B, S, K, 3]

        center_xyz = new_xyz.unsqueeze(2).repeat(1, 1, K, 1)  # [B, S, K, 3]
        relative_xyz = grouped_xyz - center_xyz  # [B, S, K, 3]
        distance = torch.norm(relative_xyz, dim=-1, keepdim=True)  # [B, S, K, 1]
        locse = torch.cat([center_xyz, grouped_xyz, relative_xyz, distance], dim=-1)  # [B, S, K, 10]

        geo = locse.reshape(B * S, K, self.geo_dim).permute(0, 2, 1)  # [B*S, 10, K]
        geo = self.geo_mlp(geo).permute(0, 2, 1).reshape(B, S, K, -1)  # [B, S, K, 3]

        anchor = new_points.unsqueeze(dim=-2)  # [B, S, 1, 3]
        anchored_points = grouped_points - anchor  # [B, S, K, 3]
        std = torch.std(anchored_points.reshape(B, -1), dim=-1, keepdim=True)  # [B, 1]
        std = std.unsqueeze(dim=-1).unsqueeze(dim=-1)  # [B, 1, 1, 1]
        anchored_points = anchored_points / (std + 1e-5)  # [B, S, K, 3]
        anchored_points = self.affine_alpha * anchored_points + self.affine_beta  # [B, S, K, 3]
        center_points = new_points.view(B, S, 1, -1).repeat(1, 1, K, 1)  # [B, S, K, 3]

        point_feature = torch.cat([grouped_xyz, anchored_points, center_points], dim=-1)  # [B, S, K, 9]
        point_feature = point_feature.reshape(B * S, K, -1).permute(0, 2, 1)  # [B*S, 9, K]
        point_feature = self.feature_mlp(point_feature).permute(0, 2, 1).reshape(B, S, K, -1)  # [B, S, K, 3]

        new_points = torch.cat([point_feature, geo], dim=-1)  # [B, S, K, 6]
        return new_xyz, new_points


class EventMamba(nn.Module):
    def __init__(self,num_classes=6,num=1024):
        super().__init__()
        self.n = num
        bimamba_type = "v2"
        # bimamba_type = None
        # self.feature_list = [6,16,32,64]
        # self.feature_list = [6,32,64,128]
        self.feature_list = [6,64,128,256]
        # self.feature_list = [6,128,256,512]
        self.group = FirstLayerRandLALocFEGrouperHidden(512, 24, hidden_channel=16)
        self.group_1 = RandLALocFEGrouper(self.feature_list[1], 256, 24)
        self.group_2 = RandLALocFEGrouper(self.feature_list[2], 128, 24)
        # self.group = LocalGrouper(3, 1024, 24, False, "anchor")
        # self.group_1 =LocalGrouper(self.feature_list[1], 512, 24, False, "anchor")
        # self.group_2 =LocalGrouper(self.feature_list[2], 256, 24, False, "anchor")
        self.embed_dim = Linear1Layer(self.feature_list[0],self.feature_list[1],1)
        self.conv1 = Linear2Layer(self.feature_list[1],1,1)
        self.conv1_1 = Linear2Layer(self.feature_list[1],1,1)
        self.conv2 = Linear2Layer(self.feature_list[2],1,1)
        self.conv2_1 = Linear2Layer(self.feature_list[2],1,1)
        self.conv3 = Linear2Layer(self.feature_list[3],1,1)
        self.conv3_1 = Linear2Layer(self.feature_list[3],1,1)
        self.mamba1 = MambaBlock(dim = self.feature_list[1], layer_idx = 0, bimamba_type = bimamba_type)
        self.mamba2 = MambaBlock(dim = self.feature_list[2], layer_idx = 1,bimamba_type = bimamba_type)
        self.mamba3 = MambaBlock(dim = self.feature_list[3], layer_idx = 2,bimamba_type = bimamba_type)
        self.attention_1 = Attention(self.feature_list[1])
        self.attention_2 = Attention(self.feature_list[2])
        self.attention_3 = Attention(self.feature_list[3])
        self.attention_4 = Attention(self.feature_list[3])
        # self.classifier = nn.Sequential(
        #     nn.Linear(self.feature_list[3], 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.2),
        #     nn.Linear(512, 1024),
        #     #nn.Sigmoid()
        # )
    

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_list[3], 512), 
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            #nn.Linear(512, 5),
            nn.Linear(512, 1024),
        )
    def forward(self, x: torch.Tensor):
        # Input x: [B, 3, N]
        xyz = x.permute(0,2,1)  # [B, N, 3]
        batch_size, _, _ = x.size()
        xyz, x = self.group(xyz, x.permute(0, 2, 1))  # xyz: [B, 512, 3], x: [B, 512, 24, 6]
        x= x.permute(0, 1, 3, 2)  # [B, 512, 6, 24]
        b, n, d, s = x.size()
        x = x.reshape(-1,d,s)  # [B*512, 6, 24]
        x = self.embed_dim(x)  # [B*512, 64, 24]
        x = self.conv1(x)  # [B*512, 64, 24]
       
        x = x.permute(0,2,1)  # [B*512, 24, 64]
        att = self.attention_1(x)  # [B*512, 24]
        x = torch.bmm(att.unsqueeze(1), x).squeeze(1)  # [B*512, 64]
        x = x.reshape(b, n, -1)  # [B, 512, 64]

        x , _= self.mamba1(x)  # [B, 512, 64]
        x = x.permute(0,2,1)  # [B, 64, 512]
        x = self.conv1_1(x)  # [B, 64, 512]
        x = x.permute(0,2,1)  # [B, 512, 64]
       
        xyz,x = self.group_1(xyz, x)  # xyz: [B, 256, 3], x: [B, 256, 24, 128]
        x= x.permute(0, 1, 3, 2)  # [B, 256, 128, 24]
        b, n, d, s = x.size()
        x = x.reshape(-1,d,s)  # [B*256, 128, 24]
        x = self.conv2(x)  # [B*256, 128, 24]
        x = x.permute(0,2,1)  # [B*256, 24, 128]
        att = self.attention_2(x)  # [B*256, 24]
        x = torch.bmm(att.unsqueeze(1), x).squeeze(1)  # [B*256, 128]
        x = x.reshape(b, n, -1)  # [B, 256, 128]


        x , _= self.mamba2(x)  # [B, 256, 128]
        x = x.permute(0,2,1)  # [B, 128, 256]
        x = self.conv2_1(x)  # [B, 128, 256]
        x = x.permute(0,2,1)  # [B, 256, 128]

        xyz,x = self.group_2(xyz, x)  # xyz: [B, 128, 3], x: [B, 128, 24, 256]
        x= x.permute(0, 1, 3, 2)  # [B, 128, 256, 24]
        b, n, d, s = x.size()
        x = x.reshape(-1,d,s)  # [B*128, 256, 24]
        x = self.conv3(x)  # [B*128, 256, 24]
        x = x.permute(0,2,1)  # [B*128, 24, 256]
        att = self.attention_3(x)  # [B*128, 24]
        x = torch.bmm(att.unsqueeze(1), x).squeeze(1)  # [B*128, 256]
        x = x.reshape(b, n, -1)  # [B, 128, 256]

        
        x,_= self.mamba3(x)  # [B, 128, 256]
        x = x.permute(0,2,1)  # [B, 256, 128]
        x = self.conv3_1(x)  # [B, 256, 128]
        x = x.permute(0,2,1)  # [B, 128, 256]

        attn = self.attention_4(x)  # [B, 128]
        x = torch.bmm(attn.unsqueeze(1), x).squeeze(1)  # [B, 256]
        x = self.classifier(x)  # [B, 1024]

        # x = torch.max(x, 2, keepdim=True)[0]
        # x = x.view(-1, self.feature_list[-1])
        # x = self.classifier(x)
        return x
        # # 1. 得到裸输出 (值域在负无穷到正无穷) [B, 5]
        # pred_raw = self.classifier(x)

        # # ================= 核心修改区：物理参数约束 =================
        # # 约束 x, y 到 [0, 1]
        # pred_x = torch.sigmoid(pred_raw[:, 0])
        # pred_y = torch.sigmoid(pred_raw[:, 1])
        
        # # 约束 a, b 到 [0, 1]，并且严格保证 a >= b
        # pred_a = torch.sigmoid(pred_raw[:, 2])
        # pred_b = pred_a * torch.sigmoid(pred_raw[:, 3]) 
        
        # # 约束 angle 到 [-0.5pi, 0.5pi)
        # pred_angle = torch.tanh(pred_raw[:, 4]) * (torch.pi / 2.0)
        
        # # 组合成最终的物理参数预测结果 [B, 5]
        # pred_params = torch.stack([pred_x, pred_y, pred_a, pred_b, pred_angle], dim=1)
        # # ============================================================

        # # 直接返回这 5 个受约束的参数
        # return pred_params
