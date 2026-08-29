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
    decode_consensus_mask,
    decode_record_mask,
    load_instance_records,
    make_refiner_channels,
    seed_from_bbox,
)
from src.dataset import read_gray_image
from src.losses import build_loss
from src.masks import dice_score
from src.model_factory import create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--annotation-path", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--samples-per-epoch", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--encoder-name", default="resnet34")
    parser.add_argument("--encoder-weights", default="imagenet")
    parser.add_argument("--arch", choices=["smp_unet", "smp_unetplusplus", "smp_deeplabv3plus"], default="smp_unet")
    parser.add_argument("--loss", choices=["dice_bce", "tversky", "focal_tversky", "dice_focal", "boundary_dice_focal"], default="dice_bce")
    parser.add_argument("--output", type=Path, default=config.MODEL_DIR / "crop_refiner_resnet34_512.pth")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--proposal-json", type=Path, default=None)
    parser.add_argument("--positive-proposal-weight", type=float, default=1.0)
    parser.add_argument("--small-instance-power", type=float, default=0.5)
    parser.add_argument(
        "--consensus-targets",
        action="store_true",
        help="Train on soft multi-annotator agreement masks while validating against canonical masks.",
    )
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


class RefinerDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        image_dir: Path,
        image_size: int,
        augment: bool,
        samples_per_epoch: int | None = None,
        small_instance_power: float = 0.0,
        consensus_targets: bool = False,
    ) -> None:
        self.records = list(records)
        self.image_dir = Path(image_dir)
        self.image_size = int(image_size)
        self.augment = augment
        self.samples_per_epoch = samples_per_epoch
        self.consensus_targets = consensus_targets
        positive_areas = [float(record.get("area", 0.0)) for record in self.records if float(record.get("area", 0.0)) > 0]
        reference_area = float(np.median(positive_areas)) if positive_areas else 1.0
        weights = []
        for record in self.records:
            area = float(record.get("area", 0.0))
            if area > 0:
                weights.append(area ** -float(small_instance_power))
            else:
                weights.append(0.25 * reference_area ** -float(small_instance_power))
        weights = np.asarray(weights, dtype=np.float64)
        self.sample_probabilities = weights / weights.sum()

    def __len__(self) -> int:
        return self.samples_per_epoch or len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx % len(self.records)]
        if self.augment:
            record = self.records[int(np.random.choice(len(self.records), p=self.sample_probabilities))]
        image = read_gray_image(self.image_dir / record["file_name"])
        mask = decode_consensus_mask(record) if self.consensus_targets else decode_record_mask(record).astype(np.float32)
        source_bbox = record.get("proposal_bbox", record["bbox"])
        bounds = crop_bounds_from_bbox(source_bbox, image.shape[0], image.shape[1], pad=0.50, jitter=self.augment)
        image_crop = crop_with_padding(image, bounds)
        mask_crop = crop_with_padding(mask, bounds)
        image_crop = cv2.resize(image_crop, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask_crop = cv2.resize(
            mask_crop,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR if self.consensus_targets else cv2.INTER_NEAREST,
        )
        seed = seed_from_bbox(source_bbox, bounds, self.image_size)

        if self.augment and random.random() < 0.5:
            image_crop = image_crop[:, ::-1]
            mask_crop = mask_crop[:, ::-1]
            seed = seed[:, ::-1]
        if self.augment and random.random() < 0.5:
            image_crop = image_crop[::-1, :]
            mask_crop = mask_crop[::-1, :]
            seed = seed[::-1, :]
        if self.augment and random.random() < 0.5:
            k = random.randint(1, 3)
            image_crop = np.rot90(image_crop, k)
            mask_crop = np.rot90(mask_crop, k)
            seed = np.rot90(seed, k)
        if self.augment and random.random() < 0.35:
            image_crop = np.clip(image_crop * random.uniform(0.85, 1.18) + random.uniform(-0.04, 0.04), 0.0, 1.0)

        channels = make_refiner_channels(image_crop, seed)
        return (
            torch.from_numpy(np.ascontiguousarray(channels)).float(),
            torch.from_numpy(np.ascontiguousarray(mask_crop[None, ...])).float(),
            record["file_name"],
        )


@torch.no_grad()
def evaluate(model, loader, device, threshold: float = 0.5) -> float:
    model.eval()
    scores = []
    for images, masks, _ in loader:
        logits = model(images.to(device, non_blocking=True))
        preds = (torch.sigmoid(logits).cpu().numpy() > threshold).astype(np.uint8)
        targets = masks.numpy().astype(np.uint8)
        for pred, target in zip(preds, targets):
            scores.append(dice_score(pred[0], target[0]))
    return float(np.mean(scores)) if scores else 0.0


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    train_dir = args.data_dir / "train_images"
    annotation_path = args.annotation_path or (args.data_dir / "train_annotations.json")
    existing_files = {p.name for p in train_dir.iterdir() if p.suffix.lower() in {".jpeg", ".jpg", ".png"}}
    _, records = load_instance_records(annotation_path, existing_files)
    if args.proposal_json is not None:
        proposal_records = load_detector_proposals(args.proposal_json)
        by_file = {file_name: [] for file_name in existing_files}
        for record in proposal_records:
            if record["file_name"] in by_file:
                by_file[record["file_name"]].append(record)
        # Keep every GT instance available as a fallback, then add detector-like
        # crops. This prevents the refiner from forgetting clean GT boxes while
        # learning the noisy boxes it sees at inference time.
        records = records + [r for rows in by_file.values() for r in rows]
    train_files, val_files = time_split_files(sorted({r["file_name"] for r in records}), config.VAL_FRACTION)
    train_records = [r for r in records if r["file_name"] in train_files]
    val_records = [r for r in records if r["file_name"] in val_files]
    print(f"crop_refiner train_instances={len(train_records)} val_instances={len(val_records)}")

    train_ds = RefinerDataset(
        train_records,
        train_dir,
        args.image_size,
        augment=True,
        samples_per_epoch=args.samples_per_epoch,
        small_instance_power=args.small_instance_power,
        consensus_targets=args.consensus_targets,
    )
    val_ds = RefinerDataset(val_records, train_dir, args.image_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(
        arch=args.arch,
        in_channels=3,
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
    ).to(device)
    if args.init_checkpoint is not None:
        initial = torch.load(args.init_checkpoint, map_location=device)
        model.load_state_dict(initial["model"])
    criterion = build_loss(args.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_score = -1.0
    stale = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for images, masks, _ in tqdm(train_loader, desc=f"crop-refiner {epoch}/{args.epochs}"):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = criterion(model(images), masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

        val_dice = evaluate(model, val_loader, device)
        print(f"epoch={epoch} loss={np.mean(losses):.5f} val_crop_dice={val_dice:.5f}")
        if val_dice > best_score + 1e-4:
            best_score = val_dice
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "arch": args.arch,
                    "encoder_name": args.encoder_name,
                    "encoder_weights": None,
                    "in_channels": 3,
                    "image_size": args.image_size,
                    "threshold": 0.5,
                    "mode": "crop_refiner",
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
