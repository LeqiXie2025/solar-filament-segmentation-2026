import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from src.dataset import FilamentPatchDataset, merge_duplicate_file_records, random_split, sorted_time_split
from src.losses import build_loss
from src.masks import dice_score, load_coco_binary_masks
from src.model_factory import create_model
from src.validation import evaluate_full_image_metric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--image-size", type=int, default=config.IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--debug-limit", type=int, default=0)
    parser.add_argument("--samples-per-epoch", type=int, default=0)
    parser.add_argument("--positive-patch-prob", type=float, default=0.65)
    parser.add_argument(
        "--small-component-prob",
        type=float,
        default=0.0,
        help="Within positive patches, probability of sampling from the smaller half of mask components.",
    )
    parser.add_argument(
        "--scale-jitter",
        type=float,
        default=0.0,
        help="Randomly crop larger/smaller context before resizing to image_size; 0 disables multiscale training.",
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=config.MODEL_DIR / "unet_baseline_best.pth")
    parser.add_argument("--split", choices=["time", "random", "all"], default="time")
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument(
        "--arch",
        choices=["custom_unet", "smp_unet", "smp_unetplusplus", "smp_deeplabv3plus"],
        default="custom_unet",
    )
    parser.add_argument("--encoder-name", default="resnet34")
    parser.add_argument("--encoder-weights", default=None)
    parser.add_argument(
        "--loss",
        choices=["dice_bce", "tversky", "focal_tversky", "dice_focal"],
        default="dice_bce",
    )
    parser.add_argument(
        "--save-metric",
        choices=["patch_dice", "full_dice", "mean_dice_matched", "pq"],
        default="patch_dice",
        help="Metric used for checkpoint selection.",
    )
    parser.add_argument("--full-val-interval", type=int, default=1)
    parser.add_argument("--full-val-limit", type=int, default=8)
    parser.add_argument("--full-val-thresholds", type=float, nargs="+", default=[0.35, 0.40, 0.45, 0.50])
    parser.add_argument("--full-val-min-areas", type=int, nargs="+", default=[40, 80, 160])
    parser.add_argument("--full-val-tile-size", type=int, default=512)
    parser.add_argument("--full-val-stride", type=int, default=256)
    parser.add_argument("--full-val-batch-size", type=int, default=8)
    parser.add_argument("--full-val-morphology", choices=["none", "close", "openclose"], default="close")
    parser.add_argument("--full-val-instance-method", choices=["connected", "watershed"], default="connected")
    parser.add_argument("--full-val-tta", choices=["none", "hflip", "hvflip"], default="none")
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
    seed_everything(args.seed)

    train_dir = args.data_dir / "train_images"
    annotation_path = args.data_dir / "train_annotations.json"
    if not train_dir.exists() or not annotation_path.exists():
        raise FileNotFoundError(
            f"Expected {train_dir} and {annotation_path}. Download Kaggle data into data/raw first."
        )

    image_info, masks_by_image = load_coco_binary_masks(annotation_path)
    image_info, masks_by_image = merge_duplicate_file_records(image_info, masks_by_image)
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

    if args.split == "random":
        train_ids, val_ids = random_split(image_info, config.VAL_FRACTION, args.seed)
    elif args.split == "all":
        train_ids = sorted(image_info.keys(), key=lambda image_id: image_info[image_id]["file_name"])
        _, val_ids = sorted_time_split(image_info, config.VAL_FRACTION)
    else:
        train_ids, val_ids = sorted_time_split(image_info, config.VAL_FRACTION)
    print(f"Split={args.split}: train={len(train_ids)} val={len(val_ids)}")
    if args.debug_limit:
        train_ids = train_ids[: args.debug_limit]
        val_ids = val_ids[: max(1, args.debug_limit // 4)]

    samples_per_epoch = args.samples_per_epoch or len(train_ids) * 4
    train_ds = FilamentPatchDataset(
        train_ids,
        image_info,
        masks_by_image,
        train_dir,
        args.image_size,
        True,
        positive_patch_prob=args.positive_patch_prob,
        small_component_prob=args.small_component_prob,
        scale_jitter=args.scale_jitter,
        samples_per_epoch=samples_per_epoch,
    )
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

    model = create_model(
        arch=args.arch,
        in_channels=config.IN_CHANNELS,
        base_channels=config.BASE_CHANNELS,
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
    ).to(device)
    if args.resume is not None and args.resume.exists():
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        print(f"Resumed model weights from {args.resume.name}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )
    criterion = build_loss(args.loss)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_score = -1.0
    best_path = args.output
    stale_epochs = 0

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
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

        val_dice = evaluate(model, val_loader, device, config.PRED_THRESHOLD)
        selection_score = val_dice
        full_val_summary = None
        if args.save_metric != "patch_dice" and (epoch % max(args.full_val_interval, 1) == 0):
            if args.full_val_limit:
                start = ((epoch - 1) * args.full_val_limit) % len(val_ids)
                full_val_ids = [val_ids[(start + i) % len(val_ids)] for i in range(args.full_val_limit)]
            else:
                full_val_ids = val_ids
            full_val_summary = evaluate_full_image_metric(
                model=model,
                image_ids=full_val_ids,
                image_info=image_info,
                masks_by_image=masks_by_image,
                image_dir=train_dir,
                annotation_path=annotation_path,
                device=device,
                thresholds=args.full_val_thresholds,
                min_areas=args.full_val_min_areas,
                tile_size=args.full_val_tile_size,
                stride=args.full_val_stride,
                batch_size=args.full_val_batch_size,
                morphology=args.full_val_morphology,
                instance_method=args.full_val_instance_method,
                tta=args.full_val_tta,
            )
            best_full = full_val_summary["best"]
            metric_key = "full_dice" if args.save_metric == "full_dice" else args.save_metric
            selection_score = float(best_full[metric_key])

        scheduler.step(selection_score)
        current_lr = optimizer.param_groups[0]["lr"]
        message = (
            f"epoch={epoch} loss={np.mean(losses):.5f} patch_dice={val_dice:.5f} "
            f"select_{args.save_metric}={selection_score:.5f} lr={current_lr:.2e}"
        )
        if full_val_summary is not None:
            best_full = full_val_summary["best"]
            message += (
                f" full_best_thr={best_full['threshold']:.2f}"
                f" full_best_area={best_full['min_area']}"
                f" full_dice={best_full['full_dice']:.5f}"
                f" pq={best_full['pq']:.5f}"
                f" fp={best_full['fp']} fn={best_full['fn']}"
            )
        print(message)

        if selection_score > best_score + args.min_delta:
            best_score = selection_score
            stale_epochs = 0
            checkpoint_threshold = config.PRED_THRESHOLD
            checkpoint_min_area = config.MIN_COMPONENT_AREA
            if full_val_summary is not None:
                checkpoint_threshold = float(full_val_summary["best"]["threshold"])
                checkpoint_min_area = int(full_val_summary["best"]["min_area"])
            torch.save(
                {
                    "model": model.state_dict(),
                    "arch": args.arch,
                    "val_dice": val_dice,
                    "selection_metric": args.save_metric,
                    "selection_score": selection_score,
                    "full_val_best": full_val_summary["best"] if full_val_summary is not None else None,
                    "image_size": args.image_size,
                    "threshold": checkpoint_threshold,
                    "min_area": checkpoint_min_area,
                    "in_channels": config.IN_CHANNELS,
                    "base_channels": config.BASE_CHANNELS,
                    "encoder_name": args.encoder_name,
                    "loss": args.loss,
                    "seed": args.seed,
                },
                best_path,
            )
            print(f"saved {best_path.name}")
        else:
            stale_epochs += 1
            if args.patience and stale_epochs >= args.patience:
                print(f"early_stop epoch={epoch} stale_epochs={stale_epochs}")
                break

    print(f"best_{args.save_metric}={best_score:.5f}")


if __name__ == "__main__":
    main()
