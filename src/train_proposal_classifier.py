import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

import config
from src.crop_refiner_utils import crop_bounds_from_bbox, crop_with_padding, make_refiner_channels, seed_from_bbox
from src.dataset import read_gray_image
from src.proposal_classifier import create_proposal_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--proposal-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=config.MODEL_DIR / "proposal_classifier_resnet18.pth")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--pad", type=float, default=0.50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=config.SEED)
    return parser.parse_args()


class ProposalDataset(Dataset):
    def __init__(self, records, image_dir: Path, image_size: int, pad: float, augment: bool):
        self.records = list(records)
        self.image_dir = image_dir
        self.image_size = int(image_size)
        self.pad = float(pad)
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = read_gray_image(self.image_dir / record["file_name"])
        x, y, w, h = [float(value) for value in record["proposal_bbox"]]
        bbox = (x, y, w, h)
        bounds = crop_bounds_from_bbox(bbox, image.shape[0], image.shape[1], pad=self.pad, jitter=self.augment)
        crop = crop_with_padding(image, bounds)
        crop = cv2.resize(crop, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        seed = seed_from_bbox(bbox, bounds, self.image_size)
        if self.augment and random.random() < 0.5:
            crop = crop[:, ::-1]
            seed = seed[:, ::-1]
        if self.augment and random.random() < 0.5:
            crop = crop[::-1, :]
            seed = seed[::-1, :]
        if self.augment and random.random() < 0.5:
            k = random.randint(1, 3)
            crop = np.rot90(crop, k)
            seed = np.rot90(seed, k)
        channels = make_refiner_channels(crop, seed)
        return torch.from_numpy(np.ascontiguousarray(channels)).float(), float(record["proposal_label"])


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probabilities = []
    labels = []
    for images, targets in loader:
        probabilities.extend(torch.sigmoid(model(images.to(device))[:, 0]).cpu().numpy().tolist())
        labels.extend(targets.numpy().tolist())
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels, dtype=np.uint8)
    best = (0.5, -1.0, 0.0, 0.0)
    for threshold in np.linspace(0.05, 0.95, 37):
        pred = probabilities >= threshold
        tp = int(np.sum(pred & (labels == 1)))
        fp = int(np.sum(pred & (labels == 0)))
        fn = int(np.sum(~pred & (labels == 1)))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        if f1 > best[1]:
            best = (float(threshold), float(f1), float(precision), float(recall))
    return best


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    payload = json.loads(args.proposal_json.read_text(encoding="utf-8"))
    records = list(payload.get("proposals", payload))
    files = sorted({record["file_name"] for record in records})
    val_count = max(1, int(round(len(files) * config.VAL_FRACTION)))
    val_files = set(files[-val_count:])
    train_records = [record for record in records if record["file_name"] not in val_files]
    val_records = [record for record in records if record["file_name"] in val_files]

    train_ds = ProposalDataset(train_records, args.data_dir / "train_images", args.image_size, args.pad, True)
    val_ds = ProposalDataset(val_records, args.data_dir / "train_images", args.image_size, args.pad, False)
    labels = np.asarray([int(record["proposal_label"]) for record in train_records])
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    weights = np.asarray([1.0 / max(1.0, counts[label]) for label in labels])
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_proposal_classifier(pretrained=True).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_f1 = -1.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for images, targets in tqdm(train_loader, desc=f"proposal-classifier {epoch}/{args.epochs}"):
            images = images.to(device, non_blocking=True)
            targets = targets.float().to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images)[:, 0]
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        threshold, f1, precision, recall = evaluate(model, val_loader, device)
        print(
            f"epoch={epoch} loss={np.mean(losses):.5f} threshold={threshold:.3f} "
            f"val_f1={f1:.5f} precision={precision:.5f} recall={recall:.5f}"
        )
        if f1 > best_f1:
            best_f1 = f1
            torch.save(
                {
                    "model": model.state_dict(),
                    "image_size": args.image_size,
                    "pad": args.pad,
                    "threshold": threshold,
                    "val_f1": f1,
                    "val_precision": precision,
                    "val_recall": recall,
                },
                args.output,
            )
    print(f"saved={args.output} best_val_f1={best_f1:.5f}")


if __name__ == "__main__":
    main()
