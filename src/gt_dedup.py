"""Deduplicate multi-annotator ground-truth filaments.

The challenge provides multiple annotation records for the *same* physical image
(same ``file_name`` + ``date_captured``, different ``image_id``) produced by
different annotators/observations. For both training and evaluation we collapse
those to one representative mask per filament: union across annotators, then
greedy IoU NMS so a filament drawn by several annotators counts once.

This matters because ``load_gt_instances`` appends every record's masks keyed by
``file_name``, so a filament seen by 3 annotators shows up 3x in the GT and
inflates fn / deflates PQ.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from src.masks import _decode_coco_rle, _decode_polygon


def _decode_segmentation(segmentation, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, list):
        return _decode_polygon(segmentation, height, width)
    if isinstance(segmentation, dict):
        return _decode_coco_rle(segmentation, height, width)
    return np.zeros((height, width), dtype=np.uint8)


def _decode_polygon_scaled(segmentation, out_h: int, out_w: int, src_h: int, src_w: int) -> np.ndarray:
    """Decode a polygon list directly at ``out_h``x``out_w`` (scaling points)."""
    mask = np.zeros((out_h, out_w), dtype=np.uint8)
    sx, sy = out_w / src_w, out_h / src_h
    polygons = segmentation if isinstance(segmentation, list) else []
    for polygon in polygons:
        pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] *= sx
        pts[:, 1] *= sy
        if len(pts) >= 3:
            cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    return mask


def load_anns_by_file(annotation_path: Path) -> tuple[dict, dict]:
    """Return (image_info_by_id, anns_by_file) without decoding masks.

    ``anns_by_file`` maps file_name -> list of raw annotation dicts (polygon
    coordinate lists are tiny), so callers decode one image at a time.
    """
    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    image_info = {img["id"]: img for img in data["images"]}
    anns_by_file: dict[str, list[dict]] = defaultdict(list)
    for ann in data.get("annotations", []):
        info = image_info[ann["image_id"]]
        anns_by_file[info["file_name"]].append(ann)
    return image_info, anns_by_file


def image_masks_from_anns(
    anns: list[dict],
    height: int,
    width: int,
    out_size: tuple[int, int] | None = None,
    iou_thresh: float = 0.5,
) -> list[np.ndarray]:
    """Decode + dedup one image's multi-annotator filaments.

    Returns deduped instance masks, optionally resized to ``out_size`` (nearest).
    """
    raw: list[np.ndarray] = []
    for ann in anns:
        seg = ann.get("segmentation")
        if out_size is not None:
            oh, ow = out_size[0], out_size[1]
            if isinstance(seg, list):
                mask = _decode_polygon_scaled(seg, oh, ow, height, width)
            elif isinstance(seg, dict):
                mask = _decode_segmentation(seg, height, width)
                mask = cv2.resize(mask, (ow, oh), interpolation=cv2.INTER_NEAREST)
            else:
                continue
        else:
            mask = _decode_segmentation(seg, height, width)
        if int(mask.sum()) <= 0:
            continue
        raw.append(mask.astype(np.uint8))
    return dedup_masks(raw, iou_thresh)


def masks_to_boxes(masks: list[np.ndarray]) -> np.ndarray:
    """Axis-aligned bbox (x0, y0, x1, y1) of each binary mask."""
    boxes = np.zeros((len(masks), 4), dtype=np.float32)
    for i, m in enumerate(masks):
        ys, xs = np.where(m)
        if len(xs):
            boxes[i] = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
    return boxes


def dedup_masks(masks: list[np.ndarray], iou_thresh: float = 0.5) -> list[np.ndarray]:
    """Greedy IoU NMS: keep the largest-area mask per overlapping cluster."""
    if not masks:
        return []
    order = sorted(range(len(masks)), key=lambda i: -int(masks[i].sum()))
    kept: list[np.ndarray] = []
    kept_boxes: list[np.ndarray] = []
    for i in order:
        m = masks[i]
        area = int(m.sum())
        if area <= 0:
            continue
        b = masks_to_boxes([m])[0]
        dup = False
        for j, k in enumerate(kept):
            kb = kept_boxes[j]
            if b[2] <= kb[0] or kb[2] <= b[0] or b[3] <= kb[1] or kb[3] <= b[1]:
                continue
            inter = int(np.logical_and(m, k).sum())
            if inter == 0:
                continue
            union = area + int(k.sum()) - inter
            if inter / union >= iou_thresh:
                dup = True
                break
        if not dup:
            kept.append(m)
            kept_boxes.append(b)
    return kept
