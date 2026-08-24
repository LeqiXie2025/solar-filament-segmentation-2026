import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import config
from src.dataset import merge_duplicate_file_records, read_gray_image, sorted_time_split
from src.masks import dice_score, load_coco_binary_masks, postprocess_binary_mask
from src.model_factory import create_model_from_checkpoint
from src.predict import sliding_window_predict
from src.preprocessing import solar_disk_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, default=config.MODEL_DIR / "unet_baseline_best.pth")
    parser.add_argument("--tile-size", type=int, default=config.INFER_TILE_SIZE)
    parser.add_argument("--stride", type=int, default=config.INFER_STRIDE)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.35, 0.4, 0.45, 0.5, 0.55])
    parser.add_argument("--morphology", choices=["none", "close", "openclose"], default="close")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    train_dir = args.data_dir / "train_images"
    annotation_path = args.data_dir / "train_annotations.json"

    image_info, masks_by_image = load_coco_binary_masks(annotation_path)
    image_info, masks_by_image = merge_duplicate_file_records(image_info, masks_by_image)
    existing_files = {p.name for p in train_dir.iterdir() if p.suffix.lower() in {".jpeg", ".jpg", ".png"}}
    image_info = {image_id: info for image_id, info in image_info.items() if info["file_name"] in existing_files}
    masks_by_image = {image_id: masks_by_image[image_id] for image_id in image_info}
    _, val_ids = sorted_time_split(image_info, config.VAL_FRACTION)
    if args.limit:
        val_ids = val_ids[: args.limit]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = create_model_from_checkpoint(checkpoint).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    scores_by_threshold = {threshold: [] for threshold in args.thresholds}
    pred_area_by_threshold = {threshold: [] for threshold in args.thresholds}
    target_areas = []

    for image_id in tqdm(val_ids, desc="full-val"):
        info = image_info[image_id]
        image = read_gray_image(train_dir / info["file_name"])
        target = masks_by_image[image_id].astype(np.uint8)
        probs = sliding_window_predict(
            model,
            image,
            device,
            tile_size=args.tile_size,
            stride=args.stride,
            batch_size=args.batch_size,
        )
        disk = solar_disk_mask(*image.shape)
        target_areas.append(int(target.sum()))
        for threshold in args.thresholds:
            pred = (probs > threshold).astype(np.uint8)
            pred[~disk] = 0
            pred = postprocess_binary_mask(pred, args.morphology)
            scores_by_threshold[threshold].append(dice_score(pred, target))
            pred_area_by_threshold[threshold].append(int(pred.sum()))

    print(f"images={len(val_ids)} checkpoint_val_patch_dice={checkpoint.get('val_dice')}")
    print(f"target_area median={np.median(target_areas):.1f} mean={np.mean(target_areas):.1f}")
    for threshold in args.thresholds:
        scores = scores_by_threshold[threshold]
        areas = pred_area_by_threshold[threshold]
        print(
            f"threshold={threshold:.3f} "
            f"full_image_dice={np.mean(scores):.5f} "
            f"median_pred_area={np.median(areas):.1f} "
            f"empty_preds={sum(area == 0 for area in areas)}"
        )


if __name__ == "__main__":
    main()
