import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import config
from src.dataset import merge_duplicate_file_records, read_gray_image, sorted_time_split
from src.masks import _decode_coco_rle, _decode_polygon, load_coco_binary_masks, postprocess_binary_mask, split_instances
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
    parser.add_argument("--min-areas", type=int, nargs="+", default=[80, 250, 500, 1000, 2000])
    parser.add_argument("--morphology", choices=["none", "close", "openclose"], default="close")
    parser.add_argument("--instance-method", choices=["connected", "watershed"], default="connected")
    parser.add_argument("--watershed-peak-rel", type=float, default=0.45)
    parser.add_argument("--watershed-peak-abs", type=float, default=3.0)
    parser.add_argument("--watershed-min-component-area", type=int, default=0)
    parser.add_argument("--tta", choices=["none", "hflip", "hvflip"], default="none")
    return parser.parse_args()


def load_gt_instances(annotation_path: Path, target_file_names: set[str] | None = None) -> tuple[dict, dict]:
    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_info = {img["id"]: img for img in data["images"]}
    instances_by_file = {}
    for ann in data.get("annotations", []):
        image_id = ann["image_id"]
        info = raw_info[image_id]
        if target_file_names is not None and info["file_name"] not in target_file_names:
            continue
        height, width = int(info["height"]), int(info["width"])
        segmentation = ann.get("segmentation")
        if isinstance(segmentation, list):
            mask = _decode_polygon(segmentation, height, width)
        elif isinstance(segmentation, dict):
            mask = _decode_coco_rle(segmentation, height, width)
        else:
            continue
        if mask.sum() == 0:
            continue
        instances_by_file.setdefault(info["file_name"], []).append(mask.astype(np.uint8))

    return raw_info, instances_by_file


def panoptic_quality(pred_instances: list[np.ndarray], gt_instances: list[np.ndarray], iou_threshold: float = 0.5) -> tuple[float, int, int, int]:
    if not pred_instances and not gt_instances:
        return 1.0, 0, 0, 0
    if not pred_instances:
        return 0.0, 0, 0, len(gt_instances)
    if not gt_instances:
        return 0.0, 0, len(pred_instances), 0

    candidates = []
    for pred_idx, pred in enumerate(pred_instances):
        pred_bool = pred.astype(bool)
        pred_area = int(pred_bool.sum())
        if pred_area == 0:
            continue
        for gt_idx, gt in enumerate(gt_instances):
            gt_bool = gt.astype(bool)
            intersection = np.logical_and(pred_bool, gt_bool).sum()
            if intersection == 0:
                continue
            union = pred_area + int(gt_bool.sum()) - int(intersection)
            iou = float(intersection / max(union, 1))
            if iou >= iou_threshold:
                candidates.append((iou, pred_idx, gt_idx))

    matched_preds = set()
    matched_gts = set()
    iou_sum = 0.0
    for iou, pred_idx, gt_idx in sorted(candidates, reverse=True):
        if pred_idx in matched_preds or gt_idx in matched_gts:
            continue
        matched_preds.add(pred_idx)
        matched_gts.add(gt_idx)
        iou_sum += iou

    tp = len(matched_preds)
    fp = len(pred_instances) - tp
    fn = len(gt_instances) - tp
    pq = iou_sum / (tp + 0.5 * fp + 0.5 * fn + 1e-6)
    return float(pq), tp, fp, fn


@torch.no_grad()
def main() -> None:
    args = parse_args()
    train_dir = args.data_dir / "train_images"
    annotation_path = args.data_dir / "train_annotations.json"

    image_info, masks_by_image = load_coco_binary_masks(annotation_path)
    image_info, masks_by_image = merge_duplicate_file_records(image_info, masks_by_image)
    existing_files = {p.name for p in train_dir.iterdir() if p.suffix.lower() in {".jpeg", ".jpg", ".png"}}
    image_info = {image_id: info for image_id, info in image_info.items() if info["file_name"] in existing_files}
    _, val_ids = sorted_time_split(image_info, config.VAL_FRACTION)
    if args.limit:
        val_ids = val_ids[: args.limit]

    val_file_names = {image_info[image_id]["file_name"] for image_id in val_ids}
    _, gt_instances_by_file = load_gt_instances(annotation_path, val_file_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = create_model_from_checkpoint(checkpoint).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    results = {
        (threshold, min_area): {"pq": [], "tp": 0, "fp": 0, "fn": 0, "pred_count": []}
        for threshold in args.thresholds
        for min_area in args.min_areas
    }

    for image_id in tqdm(val_ids, desc="instance-val"):
        info = image_info[image_id]
        image = read_gray_image(train_dir / info["file_name"])
        gt_instances = gt_instances_by_file.get(info["file_name"], [])
        probs = sliding_window_predict(
            model,
            image,
            device,
            tile_size=args.tile_size,
            stride=args.stride,
            batch_size=args.batch_size,
            tta=args.tta,
        )
        disk = solar_disk_mask(*image.shape)
        for threshold in args.thresholds:
            binary = (probs > threshold).astype(np.uint8)
            binary[~disk] = 0
            binary = postprocess_binary_mask(binary, args.morphology)
            for min_area in args.min_areas:
                pred_instances = split_instances(
                    binary,
                    min_area,
                    method=args.instance_method,
                    watershed_peak_rel=args.watershed_peak_rel,
                    watershed_peak_abs=args.watershed_peak_abs,
                    watershed_min_component_area=args.watershed_min_component_area or None,
                )
                pq, tp, fp, fn = panoptic_quality(pred_instances, gt_instances)
                row = results[(threshold, min_area)]
                row["pq"].append(pq)
                row["tp"] += tp
                row["fp"] += fp
                row["fn"] += fn
                row["pred_count"].append(len(pred_instances))

    print(f"images={len(val_ids)} checkpoint_patch_dice={checkpoint.get('val_dice')}")
    ranked = []
    for (threshold, min_area), row in results.items():
        mean_pq = float(np.mean(row["pq"]))
        ranked.append((mean_pq, threshold, min_area, row))
    for mean_pq, threshold, min_area, row in sorted(ranked, reverse=True):
        print(
            f"threshold={threshold:.3f} min_area={min_area:5d} "
            f"pq={mean_pq:.5f} tp={row['tp']} fp={row['fp']} fn={row['fn']} "
            f"median_pred_instances={np.median(row['pred_count']):.1f}"
        )


if __name__ == "__main__":
    main()
