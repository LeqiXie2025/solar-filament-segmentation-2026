from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.dataset import read_gray_image
from src.masks import (
    _decode_coco_rle,
    _decode_polygon,
    postprocess_binary_mask,
    split_instances,
)
from src.metrics import full_image_dice, greedy_match, mean_dice_matched, panoptic_quality
from src.predict import sliding_window_predict
from src.preprocessing import detect_solar_disk


def load_gt_instances(annotation_path: Path, target_file_names: set[str]) -> dict[str, list[np.ndarray]]:
    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_info = {img["id"]: img for img in data["images"]}
    instances_by_file: dict[str, list[np.ndarray]] = defaultdict(list)
    for ann in data.get("annotations", []):
        info = raw_info[ann["image_id"]]
        if info["file_name"] not in target_file_names:
            continue

        height, width = int(info["height"]), int(info["width"])
        segmentation = ann.get("segmentation")
        if isinstance(segmentation, list):
            mask = _decode_polygon(segmentation, height, width)
        elif isinstance(segmentation, dict):
            mask = _decode_coco_rle(segmentation, height, width)
        else:
            continue

        if int(mask.sum()) > 0:
            instances_by_file[info["file_name"]].append(mask.astype(np.uint8))
    return instances_by_file


@torch.no_grad()
def evaluate_full_image_metric(
    model,
    image_ids: list,
    image_info: dict,
    masks_by_image: dict,
    image_dir: Path,
    annotation_path: Path,
    device,
    thresholds: list[float],
    min_areas: list[int],
    tile_size: int,
    stride: int,
    batch_size: int,
    morphology: str = "close",
    instance_method: str = "connected",
    tta: str = "none",
) -> dict:
    """Evaluate whole-image predictions with instance-aware metrics.

    This is intentionally lighter than a full leaderboard simulation: it scans a
    small threshold/min-area grid over a validation subset and returns the best
    setting by honest panoptic quality. That signal is much closer to the CSV
    submission behavior than patch-level Dice.
    """
    model.eval()
    val_names = {image_info[image_id]["file_name"] for image_id in image_ids}
    gt_instances_by_file = load_gt_instances(annotation_path, val_names)

    combos = [(float(t), int(a)) for t in thresholds for a in min_areas]
    agg = {
        combo: {"full": [], "mdc": [], "pq": [], "tp": 0, "fp": 0, "fn": 0, "pred": []}
        for combo in combos
    }

    for image_id in tqdm(image_ids, desc="full-val", leave=False):
        info = image_info[image_id]
        image = read_gray_image(image_dir / info["file_name"])
        gt_union = masks_by_image[image_id].astype(np.uint8)
        gt_instances = gt_instances_by_file.get(info["file_name"], [])
        probs = sliding_window_predict(
            model,
            image,
            device,
            tile_size=tile_size,
            stride=stride,
            batch_size=batch_size,
            tta=tta,
        )
        disk = detect_solar_disk(image)

        for threshold, min_area in combos:
            binary = (probs > threshold).astype(np.uint8)
            binary[~disk] = 0
            binary = postprocess_binary_mask(binary, morphology)
            pred_instances = split_instances(binary, min_area, method=instance_method)
            matched, fp, fn = greedy_match(pred_instances, gt_instances, overlap_threshold=0.0, mode="dice")

            row = agg[(threshold, min_area)]
            row["full"].append(full_image_dice(binary, gt_union))
            row["mdc"].append(mean_dice_matched(matched))
            row["pq"].append(panoptic_quality(matched, len(pred_instances), len(gt_instances), fp, fn))
            row["tp"] += len(matched)
            row["fp"] += fp
            row["fn"] += fn
            row["pred"].append(len(pred_instances))

    rows = []
    for (threshold, min_area), values in agg.items():
        rows.append(
            {
                "threshold": threshold,
                "min_area": min_area,
                "full_dice": float(np.mean(values["full"])),
                "mean_dice_matched": float(np.mean(values["mdc"])),
                "pq": float(np.mean(values["pq"])),
                "tp": int(values["tp"]),
                "fp": int(values["fp"]),
                "fn": int(values["fn"]),
                "median_pred_per_image": float(np.median(values["pred"])),
            }
        )

    rows.sort(key=lambda row: row["pq"], reverse=True)
    return {"best": rows[0] if rows else {}, "rows": rows}
