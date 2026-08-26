from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np

from src.masks import _decode_coco_rle, _decode_polygon


def crop_bounds_from_bbox(
    bbox: tuple[float, float, float, float],
    height: int,
    width: int,
    pad: float = 0.45,
    jitter: bool = False,
) -> tuple[int, int, int, int]:
    x, y, w, h = [float(v) for v in bbox]
    cx = x + w / 2.0
    cy = y + h / 2.0
    side = max(w, h) * (1.0 + 2.0 * pad)
    if jitter:
        cx += random.uniform(-0.12, 0.12) * side
        cy += random.uniform(-0.12, 0.12) * side
        side *= random.uniform(0.82, 1.28)
    side = max(32.0, side)

    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    x1 = int(round(cx + side / 2.0))
    y1 = int(round(cy + side / 2.0))
    return x0, y0, x1, y1


def crop_with_padding(array: np.ndarray, bounds: tuple[int, int, int, int], fill: float = 0.0) -> np.ndarray:
    x0, y0, x1, y1 = bounds
    height, width = array.shape[:2]
    out_h = max(1, y1 - y0)
    out_w = max(1, x1 - x0)
    if array.ndim == 2:
        crop = np.full((out_h, out_w), fill, dtype=array.dtype)
    else:
        crop = np.full((out_h, out_w, array.shape[2]), fill, dtype=array.dtype)

    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(width, x1), min(height, y1)
    if sx1 <= sx0 or sy1 <= sy0:
        return crop

    dx0, dy0 = sx0 - x0, sy0 - y0
    crop[dy0 : dy0 + (sy1 - sy0), dx0 : dx0 + (sx1 - sx0)] = array[sy0:sy1, sx0:sx1]
    return crop


def seed_from_bbox(
    bbox: tuple[float, float, float, float],
    bounds: tuple[int, int, int, int],
    out_size: int,
) -> np.ndarray:
    x, y, w, h = [float(v) for v in bbox]
    x0, y0, x1, y1 = bounds
    crop_w = max(1, x1 - x0)
    crop_h = max(1, y1 - y0)
    sx0 = int(round((x - x0) / crop_w * out_size))
    sy0 = int(round((y - y0) / crop_h * out_size))
    sx1 = int(round((x + w - x0) / crop_w * out_size))
    sy1 = int(round((y + h - y0) / crop_h * out_size))
    seed = np.zeros((out_size, out_size), dtype=np.float32)
    sx0, sy0 = max(0, sx0), max(0, sy0)
    sx1, sy1 = min(out_size, sx1), min(out_size, sy1)
    if sx1 > sx0 and sy1 > sy0:
        seed[sy0:sy1, sx0:sx1] = 1.0
    return seed


def make_refiner_channels(crop: np.ndarray, seed: np.ndarray) -> np.ndarray:
    crop = crop.astype(np.float32)
    if crop.max() > 1.0:
        crop = crop / 255.0
    crop_u8 = np.clip(crop * 255.0, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(crop_u8).astype(np.float32) / 255.0
    return np.stack([crop, clahe, seed.astype(np.float32)], axis=0).astype(np.float32)


def load_instance_records(annotation_path: Path, allowed_files: set[str] | None = None) -> tuple[dict, list[dict]]:
    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    image_info = {img["id"]: img for img in data["images"]}
    records = []
    for ann in data.get("annotations", []):
        info = image_info[ann["image_id"]]
        if allowed_files is not None and info["file_name"] not in allowed_files:
            continue
        if float(ann.get("area", 0)) <= 0:
            continue
        records.append(
            {
                "image_id": ann["image_id"],
                "file_name": info["file_name"],
                "height": int(info["height"]),
                "width": int(info["width"]),
                "bbox": tuple(float(v) for v in ann["bbox"]),
                "segmentation": ann.get("segmentation"),
                "area": float(ann.get("area", 0)),
            }
        )
    return image_info, records


def decode_record_mask(record: dict) -> np.ndarray:
    segmentation = record["segmentation"]
    height, width = int(record["height"]), int(record["width"])
    if isinstance(segmentation, list):
        return _decode_polygon(segmentation, height, width)
    if isinstance(segmentation, dict):
        return _decode_coco_rle(segmentation, height, width)
    return np.zeros((height, width), dtype=np.uint8)
