# src/predict.py
# 推理脚本 - 使用 EdgeAttNet 模型

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import config
from src.dataset import read_gray_image
from src.masks import encode_binary_mask, split_instances
from src.model_edgeattnet import EdgeAttNet  # ← EdgeAttNet
from src.preprocessing import make_input_channels, solar_disk_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, default=config.MODEL_DIR / "edgeattnet_best.pth")
    parser.add_argument("--output", type=Path, default=config.SUBMISSION_DIR / "submission_edgeattnet.csv")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 张测试图，用于快速测试")
    parser.add_argument("--tile-size", type=int, default=config.INFER_TILE_SIZE)
    parser.add_argument("--stride", type=int, default=config.INFER_STRIDE)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


@torch.no_grad()
def sliding_window_predict(
    model,
    image: np.ndarray,
    device,
    tile_size: int,
    stride: int,
    batch_size: int,
) -> np.ndarray:
    """滑窗推理，处理大图"""
    height, width = image.shape
    channels = make_input_channels(image)
    prob_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)

    # 计算滑窗起始位置
    y_starts = list(range(0, max(height - tile_size, 0) + 1, stride))
    x_starts = list(range(0, max(width - tile_size, 0) + 1, stride))
    if y_starts[-1] != height - tile_size:
        y_starts.append(height - tile_size)
    if x_starts[-1] != width - tile_size:
        x_starts.append(width - tile_size)

    # 汉宁窗口，减少拼接痕迹
    window = np.outer(np.hanning(tile_size), np.hanning(tile_size)).astype(np.float32)
    window = np.maximum(window, 1e-3)

    coords = [(y0, x0) for y0 in y_starts for x0 in x_starts]
    for start in range(0, len(coords), batch_size):
        batch_coords = coords[start:start + batch_size]
        tiles = [
            channels[:, y0:y0 + tile_size, x0:x0 + tile_size]
            for y0, x0 in batch_coords
        ]
        tensor = torch.from_numpy(np.stack(tiles)).float().to(device)
        logits = model(tensor)
        probs_batch = torch.sigmoid(logits)[:, 0].detach().cpu().numpy()
        for (y0, x0), probs in zip(batch_coords, probs_batch):
            prob_sum[y0:y0 + tile_size, x0:x0 + tile_size] += probs * window
            weight_sum[y0:y0 + tile_size, x0:x0 + tile_size] += window

    return prob_sum / np.maximum(weight_sum, 1e-6)


def main() -> None:
    args = parse_args()

    # 检查测试集目录
    test_dir = args.data_dir / "test_images"
    if not test_dir.exists():
        raise FileNotFoundError(f"测试集目录不存在: {test_dir}")

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 加载模型
    checkpoint_path = args.checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"模型不存在: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    threshold = args.threshold if args.threshold is not None else config.PRED_THRESHOLD

    # ===== 使用 EdgeAttNet =====
    model = EdgeAttNet(
        in_channels=config.IN_CHANNELS,
        out_channels=1,
        base_channels=config.BASE_CHANNELS
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"模型: EdgeAttNet")
    print(f"阈值: {threshold}")

    # 测试集文件列表
    test_files = sorted([
        p for p in test_dir.iterdir()
        if p.suffix.lower() in {".jpeg", ".jpg", ".png"}
    ])
    if args.limit:
        test_files = test_files[:args.limit]
    print(f"测试图片数: {len(test_files)}")

    rows = []

    for path in tqdm(test_files, desc="推理中"):
        image = read_gray_image(path)

        # 滑窗预测
        probs = sliding_window_predict(
            model,
            image,
            device,
            tile_size=args.tile_size,
            stride=args.stride,
            batch_size=args.batch_size,
        )

        # 二值化 + 太阳圆盘掩膜
        disk = solar_disk_mask(*image.shape)
        binary = (probs > threshold).astype(np.uint8)
        binary[~disk] = 0

        # 形态学闭运算，填补空洞
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 拆分实例并编码
        image_id = path.stem
        instances = split_instances(binary, config.MIN_COMPONENT_AREA)
        for idx, instance in enumerate(instances, start=1):
            rows.append({
                "filament_id": f"{image_id}_{idx}",
                "segmentation_rle": encode_binary_mask(instance),
            })

    # 保存提交文件
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    df.to_csv(args.output, index=False)
    print(f"\n✅ 提交文件已保存: {args.output}")
    print(f"   共 {len(rows)} 个日珥实例")
    print(f"   阈值: {threshold}")


if __name__ == "__main__":
    main()