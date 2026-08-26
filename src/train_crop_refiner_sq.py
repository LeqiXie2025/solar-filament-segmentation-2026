import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import config
from src.crop_refiner_utils import (
    crop_bounds_from_bbox,
    crop_with_padding,
    decode_record_mask,
    load_instance_records,
    make_refiner_channels,
    seed_from_bbox,
)
from src.dataset import read_gray_image
from src.losses import build_loss
from src.masks import dice_score
from src.model_factory import create_model, create_model_from_checkpoint
from src.preprocessing import make_input_channels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--sq-checkpoint", type=Path, required=True)
    parser.add_argument("--proposal-json", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--samples-per-epoch", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--encoder-name", default="resnet34")
    parser.add_argument("--encoder-weights", default="imagenet")
    parser.add_argument("--loss", choices=["dice_bce", "tversky", "focal_tversky", "dice_focal"], default="dice_bce")
    parser.add_argument("--output", type=Path, default=config.MODEL_DIR / "crop_refiner_sq_resnet34_512.pth")
    parser.add_argument("--patience", type=int, default=2)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def time_split_files(file_names: list[str], val_fraction: float) -> tuple[set[str], set[str]]:
    ordered = sorted(file_names)
    val_count = max(1, int(round(len(ordered) * val_fraction)))
    return set(ordered[:-val_count]), set(ordered[-val_count:])


def load_detector_proposals(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return list(payload.get("proposals", payload))


def load_sq_model(path: Path, device):
    checkpoint = torch.load(path, map_location=device)
    model = create_model_from_checkpoint(checkpoint).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


class SQRefinerDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        image_dir: Path,
        image_size: int,
        augment: bool,
        samples_per_epoch: int | None = None,
    ) -> None:
        self.records = list(records)
        self.image_dir = Path(image_dir)
        self.image_size = int(image_size)
        self.augment = augment
        self.samples_per_epoch = samples_per_epoch

    def __len__(self) -> int:
        return self.samples_per_epoch or len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx % len(self.records)]
        if self.augment:
            record = random.choice(self.records)
        image = read_gray_image(self.image_dir / record["file_name"])
        mask = decode_record_mask(record)
        source_bbox = record.get("proposal_bbox", record["bbox"])
        bounds = crop_bounds_from_bbox(source_bbox, image.shape[0], image.shape[1], pad=0.50, jitter=self.augment)
        image_crop = crop_with_padding(image, bounds)
        mask_crop = crop_with_padding(mask, bounds)
        image_crop = cv2.resize(image_crop, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask_crop = cv2.resize(mask_crop, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        seed = seed_from_bbox(source_bbox, bounds, self.image_size)

        if self.augment and random.random() < 0.5:
            image_crop = image_crop[:, ::-1]
            mask_crop = mask_crop[:, ::-1]
            seed = seed[:, ::-1]
        if self.augment and random.random() < 0.35:
            image_crop = np.clip(image_crop * random.uniform(0.85, 1.18) + random.uniform(-0.04, 0.04), 0.0, 1.0)

        refiner_channels = make_refiner_channels(image_crop, seed)
        sq_channels = make_input_channels(image_crop)
        return (
            torch.from_numpy(np.ascontiguousarray(refiner_channels)).float(),
            torch.from_numpy(np.ascontiguousarray(sq_channels)).float(),
            torch.from_numpy(np.ascontiguousarray(mask_crop[None, ...])).float(),
            record["file_name"],
        )


@torch.no_grad()
def sq_refiner_input(sq_model, refiner_channels: torch.Tensor, sq_channels: torch.Tensor) -> torch.Tensor:
    sq_prob = torch.sigmoid(sq_model(sq_channels))[:, :1]
    return torch.cat([refiner_channels, sq_prob], dim=1)


@torch.no_grad()
def evaluate(model, sq_model, loader, device, threshold: float = 0.5) -> float:
    model.eval()
    scores = []
    for ref_channels, sq_channels, masks, _ in loader:
        ref_channels = ref_channels.to(device, non_blocking=True)
        sq_channels = sq_channels.to(device, non_blocking=True)
        logits = model(sq_refiner_input(sq_model, ref_channels, sq_channels))
        preds = (torch.sigmoid(logits).cpu().numpy() > threshold).astype(np.uint8)
        targets = masks.numpy().astype(np.uint8)
        for pred, target in zip(preds, targets):
            scores.append(dice_score(pred[0], target[0]))
    return float(np.mean(scores)) if scores else 0.0


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    train_dir = args.data_dir / "train_images"
    annotation_path = args.data_dir / "train_annotations.json"
    existing_files = {p.name for p in train_dir.iterdir() if p.suffix.lower() in {".jpeg", ".jpg", ".png"}}
    _, records = load_instance_records(annotation_path, existing_files)
    if args.proposal_json is not None:
        proposals = [r for r in load_detector_proposals(args.proposal_json) if r["file_name"] in existing_files]
        records = records + proposals

    train_files, val_files = time_split_files(sorted({r["file_name"] for r in records}), config.VAL_FRACTION)
    train_records = [r for r in records if r["file_name"] in train_files]
    val_records = [r for r in records if r["file_name"] in val_files]
    print(f"sq_crop_refiner train_instances={len(train_records)} val_instances={len(val_records)}")

    train_ds = SQRefinerDataset(train_records, train_dir, args.image_size, augment=True, samples_per_epoch=args.samples_per_epoch)
    val_ds = SQRefinerDataset(val_records, train_dir, args.image_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sq_model = load_sq_model(args.sq_checkpoint, device)
    model = create_model(
        arch="smp_unet",
        in_channels=4,
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
    ).to(device)
    criterion = build_loss(args.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_score = -1.0
    stale = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for ref_channels, sq_channels, masks, _ in tqdm(train_loader, desc=f"sq-crop-refiner {epoch}/{args.epochs}"):
            ref_channels = ref_channels.to(device, non_blocking=True)
            sq_channels = sq_channels.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                inputs = sq_refiner_input(sq_model, ref_channels, sq_channels)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = criterion(model(inputs), masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

        val_dice = evaluate(model, sq_model, val_loader, device)
        print(f"epoch={epoch} loss={np.mean(losses):.5f} val_crop_dice={val_dice:.5f}")
        if val_dice > best_score + 1e-4:
            best_score = val_dice
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "arch": "smp_unet",
                    "encoder_name": args.encoder_name,
                    "encoder_weights": None,
                    "in_channels": 4,
                    "image_size": args.image_size,
                    "threshold": 0.5,
                    "mode": "sq_crop_refiner",
                    "sq_checkpoint": str(args.sq_checkpoint),
                    "val_crop_dice": val_dice,
                },
                args.output,
            )
            print(f"saved {args.output.name}")
        else:
            stale += 1
            if args.patience and stale >= args.patience:
                print(f"early_stop epoch={epoch}")
                break
    print(f"best_val_crop_dice={best_score:.5f}")


if __name__ == "__main__":
    main()
