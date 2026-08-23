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
from src.model import UNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--image-size", type=int, default=config.IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
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


def main() -> None:
    args = parse_args()
    seed_everything(config.SEED)

    train_dir = args.data_dir / "train_images"
    annotation_path = args.data_dir / "train_annotations.json"
    if not train_dir.exists() or not annotation_path.exists():
        raise FileNotFoundError(
            f"Expected {train_dir} and {annotation_path}. Download Kaggle data into data/raw first."
        )

    image_info, masks_by_image = load_coco_binary_masks(annotation_path)
    existing_files = {p.name for p in train_dir.iterdir() if p.suffix.lower() in {".jpeg", ".jpg", ".png"}}
    image_info = {
        image_id: info
        for image_id, info in image_info.items()
        if info["file_name"] in existing_files
    }
    masks_by_image = {
        image_id: masks_by_image[image_id]
        for image_id in image_info
    }
    print(f"Using {len(image_info)} annotated images present in the local train_images directory")

    train_ids, val_ids = sorted_time_split(image_info, config.VAL_FRACTION)
    if args.debug_limit:
        train_ids = train_ids[: args.debug_limit]
        val_ids = val_ids[: max(1, args.debug_limit // 4)]

    train_ds = FilamentPatchDataset(train_ids, image_info, masks_by_image, train_dir, args.image_size, True)
    val_ds = FilamentPatchDataset(val_ids, image_info, masks_by_image, train_dir, args.image_size, False)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = UNet(in_channels=config.IN_CHANNELS, base_channels=config.BASE_CHANNELS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    criterion = DiceBCELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_score = -1.0
    best_path = config.MODEL_DIR / "unet_baseline_best.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for images, masks, _ in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

        val_dice = evaluate(model, val_loader, device, config.PRED_THRESHOLD)
        print(f"epoch={epoch} loss={np.mean(losses):.5f} val_dice={val_dice:.5f}")

        if val_dice > best_score:
            best_score = val_dice
            torch.save(
                {
                    "model": model.state_dict(),
                    "val_dice": best_score,
                    "image_size": args.image_size,
                    "threshold": config.PRED_THRESHOLD,
                    "in_channels": config.IN_CHANNELS,
                    "base_channels": config.BASE_CHANNELS,
                },
                best_path,
            )
            print(f"saved {best_path.name}")

    print(f"best_val_dice={best_score:.5f}")


if __name__ == "__main__":
    main()
