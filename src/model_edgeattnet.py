import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
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


class EdgeGuidedMHSA(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, edge_prior):
        batch, channels, height, width = x.shape
        tokens = height * width
        x_flat = x.flatten(2).transpose(1, 2)
        qkv = self.qkv(x_flat).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        edge_flat = edge_prior.flatten(2).transpose(1, 2)
        edge_flat = edge_flat.unsqueeze(1).expand(-1, self.num_heads, -1, self.head_dim)
        q = q + edge_flat
        k = k + edge_flat

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(batch, tokens, channels)
        out = self.proj(out)
        return out.transpose(1, 2).reshape(batch, channels, height, width)


class EdgeAttNet(nn.Module):
    """Teammate EdgeAttNet branch used as the high-SQ semantic signal."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 8)
        self.eg_mhsa1 = EdgeGuidedMHSA(base_channels * 8, num_heads=4)
        self.eg_mhsa2 = EdgeGuidedMHSA(base_channels * 8, num_heads=4)
        self.bottleneck_out = ConvBlock(base_channels * 8, base_channels * 8)

        self.edge_prior = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid(),
        )

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
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        bottleneck = self.bottleneck(self.pool4(e4))
        edge_down = F.interpolate(edge, size=bottleneck.shape[2:], mode="bilinear", align_corners=False)
        bottleneck = self.eg_mhsa1(bottleneck, edge_down)
        bottleneck = self.eg_mhsa2(bottleneck, edge_down)
        bottleneck = self.bottleneck_out(bottleneck)

        d4 = self.dec4(torch.cat([self.up4(bottleneck), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)
