# src/train.py
# 目标 0.85+ 完整配置

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from src.dataset import FilamentPatchDataset, sorted_time_split
from src.losses import DiceBCELoss
from src.masks import dice_score, load_coco_binary_masks
from src.model_edgeattnet import EdgeAttNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--image-size", type=int, default=config.IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--debug-limit", type=int, default=0)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, threshold: float) -> float:
    model.eval()
    scores = []
    for images, masks, _ in loader:
        images = images.to(device)
        logits = model(images)
        preds = (torch.sigmoid(logits).cpu().numpy() > threshold).astype(np.uint8)
        masks_np = masks.numpy().astype(np.uint8)
        for pred, target in zip(preds, masks_np):
            scores.append(dice_score(pred[0], target[0]))
    return float(np.mean(scores)) if scores else 0.0


def main():
    args = parse_args()
    seed_everything(args.seed)

    print("=" * 50)
    print(f"🎯 目标: 0.85+")
    print(f"随机种子: {args.seed}")
    print(f"学习率: {args.lr}")
    print(f"训练轮数: {args.epochs}")
    print("=" * 50)

    train_dir = args.data_dir / "train_images"
    annotation_path = args.data_dir / "train_annotations.json"
    if not train_dir.exists() or not annotation_path.exists():
        raise FileNotFoundError(f"数据不存在: {train_dir} 或 {annotation_path}")

    image_info, masks_by_image = load_coco_binary_masks(annotation_path)
    existing_files = {p.name for p in train_dir.iterdir() if p.suffix.lower() in {".jpeg", ".jpg", ".png"}}
    image_info = {image_id: info for image_id, info in image_info.items() if info["file_name"] in existing_files}
    masks_by_image = {image_id: masks_by_image[image_id] for image_id in image_info}
    print(f"使用 {len(image_info)} 张图片")

    train_ids, val_ids = sorted_time_split(image_info, config.VAL_FRACTION)
    if args.debug_limit:
        train_ids = train_ids[:args.debug_limit]
        val_ids = val_ids[:max(1, args.debug_limit // 4)]

    train_ds = FilamentPatchDataset(train_ids, image_info, masks_by_image, train_dir, args.image_size, True)
    val_ds = FilamentPatchDataset(val_ids, image_info, masks_by_image, train_dir, args.image_size, False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=config.NUM_WORKERS, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    model = EdgeAttNet(in_channels=config.IN_CHANNELS, out_channels=1,
                       base_channels=config.BASE_CHANNELS).to(device)
    print(f"模型: EdgeAttNet, 参数量: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    criterion = DiceBCELoss(bce_weight=0.3)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ===== 关键：预热 3 轮，快速进入状态 =====
    warmup_epochs = 5

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_score = 0.0
    best_path = config.MODEL_DIR / "edgeattnet_best.pth"
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    print(f"\n开始训练 {args.epochs} 轮...")
    print(f"预热轮数: {warmup_epochs}")
    print(f"梯度裁剪: max_norm=1.0")
    print("=" * 50)

    for epoch in range(1, args.epochs + 1):
        # ===== 预热 =====
        if epoch <= warmup_epochs:
            lr = args.lr * (epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        model.train()
        losses = []
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for images, masks, _ in loop:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, masks)

            # ===== 梯度裁剪（防止 nan） =====
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            loss_val = float(loss.detach().cpu())
            losses.append(loss_val)

            # ===== nan 检测 =====
            if torch.isnan(loss):
                print(f"\n❌ 检测到 loss=nan，停止训练！")
                return

            loop.set_postfix(loss=loss_val)

        if epoch > warmup_epochs:
            scheduler.step()

        val_dice = evaluate(model, val_loader, device, config.PRED_THRESHOLD)
        current_lr = optimizer.param_groups[0]['lr']
        avg_loss = np.mean(losses)

        print(f"Epoch {epoch}: loss={avg_loss:.5f}, val_dice={val_dice:.5f}, lr={current_lr:.2e}")

        if val_dice > best_score and not np.isnan(val_dice):
            best_score = val_dice
            torch.save({
                "model": model.state_dict(),
                "val_dice": best_score,
                "seed": args.seed,
                "epoch": epoch
            }, best_path)
            print(f"✅ 最佳模型已保存! val_dice={best_score:.5f}")

    print(f"\n🎉 训练完成!")
    print(f"🏆 最佳 val_dice: {best_score:.5f}")
    print(f"📁 模型保存在: {best_path}")


if __name__ == "__main__":
    main()