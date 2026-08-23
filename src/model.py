# src/model.py
# 模型定义（EdgeAttNet简化版）

import torch
import torch.nn as nn
import torch.nn.functional as F


class EG_MHSA(nn.Module):
    """边缘引导的多头自注意力模块"""

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, edge_map):
        B, N, C = x.shape

        # 生成 Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 边缘引导：将边缘信息注入 Q 和 K
        # edge_map 形状: (B, N, C)，直接加到 q 和 k 上
        edge_q = edge_map.view(B, N, C)
        edge_k = edge_map.view(B, N, C)

        # 扩展维度以匹配 q 和 k 的形状 (B, num_heads, N, head_dim)
        # 需要将 edge 投影到相同的 head_dim
        # 简单做法：对 edge 做线性变换
        if not hasattr(self, 'edge_proj_q'):
            self.edge_proj_q = nn.Linear(C, C).to(x.device)
            self.edge_proj_k = nn.Linear(C, C).to(x.device)

        edge_q = self.edge_proj_q(edge_q).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        edge_k = self.edge_proj_k(edge_k).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        q = q + edge_q
        k = k + edge_k

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class ConvBlock(nn.Module):
    """标准卷积块：Conv -> BN -> ReLU"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class SimpleEdgeAttNet(nn.Module):
    """简化版EdgeAttNet，用于日珥分割"""

    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super().__init__()

        # ========== 编码器 ==========
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)

        # ========== 瓶颈 ==========
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)
        self.eg_mhsa = EG_MHSA(dim=base_channels * 8)

        # ========== 解码器 ==========
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)

        self.final = nn.Conv2d(base_channels, out_channels, 1)

        # ========== 边缘先验提取器 ==========
        self.edge_prior = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 边缘先验
        edge = self.edge_prior(x)  # (B, 1, H, W)

        # ========== 编码路径 ==========
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        # ========== 瓶颈 ==========
        bottleneck = self.bottleneck(p3)
        B, C, H, W = bottleneck.shape

        # 展平用于 Transformer
        flat = bottleneck.flatten(2).transpose(1, 2)  # (B, N, C)

        # 边缘下采样并投影到与 bottleneck 相同的通道数
        edge_down = F.interpolate(edge, size=(H, W), mode='bilinear')
        # 将 1 通道边缘投影到 C 通道
        edge_proj = torch.cat([edge_down] * C, dim=1)  # (B, C, H, W)
        edge_flat = edge_proj.flatten(2).transpose(1, 2)  # (B, N, C)

        # 应用边缘引导注意力
        out = self.eg_mhsa(flat, edge_flat)
        out = out.transpose(1, 2).reshape(B, C, H, W)

        # ========== 解码路径（带跳连接） ==========
        d3 = self.up3(out)
        d3 = torch.cat([e3, d3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)

        # 输出
        return torch.sigmoid(self.final(d1))