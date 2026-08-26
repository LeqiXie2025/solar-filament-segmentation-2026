# src/model_edgeattnet.py
# EdgeAttNet - 论文官方架构实现

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class EG_MHSA(nn.Module):
    """边缘引导多头自注意力"""
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, edge_prior):
        B, C, H, W = x.shape
        N = H * W

        x_flat = x.flatten(2).transpose(1, 2)
        qkv = self.qkv(x_flat).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        edge_flat = edge_prior.flatten(2).transpose(1, 2)
        edge_expanded = edge_flat.unsqueeze(1).expand(-1, self.num_heads, -1, self.head_dim)

        q = q + edge_expanded
        k = k + edge_expanded

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)

        return out.transpose(1, 2).reshape(B, C, H, W)


class EdgeAttNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=32):
        super().__init__()

        # ===== 编码器 =====
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool2d(2)

        # ===== 瓶颈 =====
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 8)
        self.eg_mhsa1 = EG_MHSA(dim=base_channels * 8, num_heads=4)
        self.eg_mhsa2 = EG_MHSA(dim=base_channels * 8, num_heads=4)
        self.bottleneck_out = ConvBlock(base_channels * 8, base_channels * 8)

        # ===== 边缘先验 =====
        self.edge_prior = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )

        # ===== 解码器 =====
        self.up4 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec4 = ConvBlock(base_channels * 4 + base_channels * 8, base_channels * 4)

        self.up3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec3 = ConvBlock(base_channels * 2 + base_channels * 4, base_channels * 2)

        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec2 = ConvBlock(base_channels + base_channels * 2, base_channels)

        self.up1 = nn.ConvTranspose2d(base_channels, base_channels, 2, stride=2)
        self.dec1 = ConvBlock(base_channels + base_channels, base_channels)

        self.final = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x):
        edge = self.edge_prior(x)

        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        bottleneck = self.bottleneck(p4)
        edge_down = F.interpolate(edge, size=bottleneck.shape[2:], mode='bilinear')

        bottleneck = self.eg_mhsa1(bottleneck, edge_down)
        bottleneck = self.eg_mhsa2(bottleneck, edge_down)
        bottleneck = self.bottleneck_out(bottleneck)

        d4 = self.up4(bottleneck)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.final(d1)