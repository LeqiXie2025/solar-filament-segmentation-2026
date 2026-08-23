# src/train.py
# 训练脚本

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from src.dataset import FilamentDataset, load_annotations
from src.model import SimpleEdgeAttNet
from src.utils import dice_score


def train():
    global DEVICE  # ← 放在函数最开头

    print("=" * 50)
    print("检查数据目录...")
    print(f"训练图片目录: {TRAIN_IMG_DIR}")
    print(f"标注文件: {ANNO_PATH}")

    if not os.path.exists(TRAIN_IMG_DIR):
        print(f"❌ 错误: 训练图片目录不存在: {TRAIN_IMG_DIR}")
        return
    if not os.path.exists(ANNO_PATH):
        print(f"❌ 错误: 标注文件不存在: {ANNO_PATH}")
        return

    train_images = [f for f in os.listdir(TRAIN_IMG_DIR) if f.endswith(('.jpeg', '.jpg', '.png'))]
    print(f"✅ 找到 {len(train_images)} 张训练图片")
    if len(train_images) > 0:
        print(f"   前5张: {train_images[:5]}")

    print("\n加载标注数据...")
    try:
        img_info, img_masks = load_annotations(ANNO_PATH)
        all_ids = list(img_info.keys())
        print(f"✅ 加载了 {len(all_ids)} 张图片的标注")
        if len(all_ids) > 0:
            first_id = all_ids[0]
            first_file = img_info[first_id]['file_name']
            print(f"   标注中第一张: {first_file}")
    except Exception as e:
        print(f"❌ 加载标注失败: {e}")
        return

    split_idx = int(len(all_ids) * 0.8)
    train_ids, valid_ids = all_ids[:split_idx], all_ids[split_idx:]
    print(f"\n训练集: {len(train_ids)} 张, 验证集: {len(valid_ids)} 张")

    print("\n创建数据集...")
    train_dataset = FilamentDataset(train_ids, img_info, img_masks, IMAGE_SIZE, augment=True)
    valid_dataset = FilamentDataset(valid_ids, img_info, img_masks, IMAGE_SIZE, augment=False)

    print("测试数据加载...")
    try:
        test_img, test_mask, test_name = train_dataset[0]
        print(f"✅ 测试成功: {test_name}, 图片形状: {test_img.shape}, 掩膜形状: {test_mask.shape}")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"\n创建模型...")
    print(f"设备: {DEVICE}")
    model = SimpleEdgeAttNet(in_channels=1, out_channels=1, base_channels=BASE_CHANNELS)

    try:
        model = model.to(DEVICE)
        print(f"✅ 模型已加载到 {DEVICE}")
    except Exception as e:
        print(f"⚠️ 模型加载到 {DEVICE} 失败: {e}")
        DEVICE = 'cpu'
        model = model.to(DEVICE)
        print(f"✅ 模型已加载到 CPU")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCELoss()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_dice = 0.0

    print(f"\n开始训练...")
    print(f"轮数: {EPOCHS}, 批次大小: {BATCH_SIZE}")
    print("=" * 50)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for images, masks, _ in loop:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            preds = model(images)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        model.eval()
        val_dice = 0
        val_count = 0

        with torch.no_grad():
            for images, masks, _ in valid_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                preds = model(images)
                pred_binary = (preds > 0.5).cpu().numpy()
                mask_np = masks.cpu().numpy()

                for i in range(len(pred_binary)):
                    val_dice += dice_score(pred_binary[i], mask_np[i])
                    val_count += 1

        val_dice = val_dice / val_count if val_count > 0 else 0
        avg_loss = train_loss / len(train_loader)

        print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}, Val Dice = {val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
            print(f"✅ 最佳模型已保存! Dice = {best_dice:.4f}")

    print("\n" + "=" * 50)
    print("🎉 训练完成!")
    print(f"最佳 Dice 分数: {best_dice:.4f}")


if __name__ == '__main__':
    train()