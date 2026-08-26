import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import cv2
from pathlib import Path
from src.dataset import read_gray_image
from src.preprocessing import make_input_channels, solar_disk_mask
from src.model_edgeattnet import EdgeAttNet
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")

# 用绝对路径加载模型
project_root = Path(r"C:\Users\love5\Desktop\solar-filament-segmentation-2026-master")
checkpoint_path = project_root / "models" / "edgeattnet_best.pth"
print(f"模型路径: {checkpoint_path}")

if not checkpoint_path.exists():
    print(f"❌ 模型不存在: {checkpoint_path}")
    exit()

checkpoint = torch.load(checkpoint_path, map_location=device)
model = EdgeAttNet(in_channels=3, out_channels=1, base_channels=32).to(device)
model.load_state_dict(checkpoint["model"])
model.eval()
print(f"✅ 模型加载成功: {checkpoint_path.name}")
print(f"   val_dice: {checkpoint.get('val_dice', 'unknown')}")

# 读取第一张测试图
test_dir = project_root / "data" / "raw" / "test_images"
test_files = list(test_dir.glob("*.jpeg"))[:1]

for path in test_files:
    print(f"\n处理: {path.name}")
    image = read_gray_image(path)
    print(f"  图片尺寸: {image.shape}")

    channels = make_input_channels(image)
    print(f"  输入通道: {channels.shape}")

    tensor = torch.from_numpy(channels).float().unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

    print(f"  概率图 - max: {probs.max():.6f}")
    print(f"  概率图 - min: {probs.min():.6f}")
    print(f"  概率图 - mean: {probs.mean():.6f}")
    print(f"  概率图 > 0.1: {(probs > 0.1).sum()} pixels")
    print(f"  概率图 > 0.01: {(probs > 0.01).sum()} pixels")

    for thresh in [0.01, 0.05, 0.10, 0.15, 0.20, 0.30]:
        binary = (probs > thresh).astype(np.uint8)
        print(f"  阈值 {thresh}: {binary.sum()} 像素")