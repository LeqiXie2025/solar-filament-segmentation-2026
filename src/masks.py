import json
from pathlib import Path

import cv2
import numpy as np


def _decode_polygon(segmentation, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    polygons = segmentation if isinstance(segmentation, list) else []
    for polygon in polygons:
        pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if len(pts) >= 3:
            cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    return mask


def _decode_coco_rle(segmentation, height: int, width: int) -> np.ndarray:
    try:
        from pycocotools import mask as coco_mask
    except ImportError as exc:
        raise ImportError(
            "Compressed COCO RLE annotations need pycocotools. "
            "Install it if train_annotations.json uses RLE instead of polygons."
        ) from exc

    rle = dict(segmentation)
    if isinstance(rle.get("counts"), str):
        rle["counts"] = rle["counts"].encode("utf-8")
    if "size" not in rle:
        rle["size"] = [height, width]
    return coco_mask.decode(rle).astype(np.uint8)


def load_coco_binary_masks(annotation_path: Path) -> tuple[dict, dict]:
    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_info = {img["id"]: img for img in data["images"]}
    masks_by_image = {
        image_id: np.zeros((int(info["height"]), int(info["width"])), dtype=np.uint8)
        for image_id, info in image_info.items()
    }

    for ann in data.get("annotations", []):
        image_id = ann["image_id"]
        info = image_info[image_id]
        height, width = int(info["height"]), int(info["width"])
        segmentation = ann.get("segmentation")

        if isinstance(segmentation, list):
            mask = _decode_polygon(segmentation, height, width)
        elif isinstance(segmentation, dict):
            mask = _decode_coco_rle(segmentation, height, width)
        else:
            continue

        masks_by_image[image_id] = np.maximum(masks_by_image[image_id], mask)

    return image_info, masks_by_image


def binary_mask_to_counts(mask: np.ndarray) -> list[int]:
    pixels = np.asarray(mask, dtype=np.uint8).flatten(order="F")
    pixels = (pixels > 0).astype(np.uint8)
    change_points = np.flatnonzero(pixels[1:] != pixels[:-1]) + 1
    run_lengths = np.diff(np.concatenate(([0], change_points, [pixels.size]))).astype(int)
    if pixels.size and pixels[0] == 1:
        run_lengths = np.concatenate(([0], run_lengths))
    return run_lengths.tolist()


def encode_coco_counts(counts: list[int]) -> str:
    encoded = []
    for idx, value in enumerate(counts):
        x = int(value)
        if idx > 2:
            x -= int(counts[idx - 2])

        more = True
        while more:
            c = x & 0x1F
            x >>= 5
            more = x != -1 if (c & 0x10) else x != 0
            if more:
                c |= 0x20
            encoded.append(chr(c + 48))
    return "".join(encoded)


def encode_binary_mask(mask: np.ndarray) -> str:
    return encode_coco_counts(binary_mask_to_counts(mask))


def _connected_component_instances(binary_mask: np.ndarray, min_area: int) -> list[np.ndarray]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask.astype(np.uint8), connectivity=8
    )
    instances = []
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area >= min_area:
            instances.append((labels == label_id).astype(np.uint8))
    return instances


def _watershed_split_component(
    component: np.ndarray,
    min_area: int,
    peak_rel: float,
    peak_abs: float,
) -> list[np.ndarray]:
    distance = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 5)
    max_distance = float(distance.max())
    if max_distance <= 0:
        return []

    peak_threshold = max(float(peak_abs), max_distance * float(peak_rel))
    peak_mask = (distance >= peak_threshold).astype(np.uint8)
    num_markers, markers = cv2.connectedComponents(peak_mask, connectivity=8)
    if num_markers <= 2:
        return [component.astype(np.uint8)] if int(component.sum()) >= min_area else []

    elevation = cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    elevation = 255 - elevation
    markers = markers.astype(np.int32)
    markers[component == 0] = 0
    cv2.watershed(cv2.cvtColor(elevation, cv2.COLOR_GRAY2BGR), markers)

    split = []
    for marker_id in range(1, num_markers):
        instance = ((markers == marker_id) & (component > 0)).astype(np.uint8)
        if int(instance.sum()) >= min_area:
            split.append(instance)

    if not split and int(component.sum()) >= min_area:
        split.append(component.astype(np.uint8))
    return split


def split_instances(
    binary_mask: np.ndarray,
    min_area: int,
    method: str = "connected",
    watershed_peak_rel: float = 0.45,
    watershed_peak_abs: float = 3.0,
    watershed_min_component_area: int | None = None,
) -> list[np.ndarray]:
    if method == "connected":
        return _connected_component_instances(binary_mask, min_area)
    if method != "watershed":
        raise ValueError(f"Unknown instance split method: {method}")

    split_min_area = watershed_min_component_area or max(min_area * 2, min_area + 1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask.astype(np.uint8), connectivity=8
    )
    instances = []
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        component = (labels == label_id).astype(np.uint8)
        if area < split_min_area:
            instances.append(component)
            continue
        instances.extend(
            _watershed_split_component(
                component,
                min_area=min_area,
                peak_rel=watershed_peak_rel,
                peak_abs=watershed_peak_abs,
            )
        )
    return instances


def postprocess_binary_mask(binary_mask: np.ndarray, mode: str = "close") -> np.ndarray:
    if mode == "none":
        return binary_mask.astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if mode == "close":
        return cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    if mode == "openclose":
        opened = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    raise ValueError(f"Unknown morphology mode: {mode}")


def dice_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(pred, target).sum()
    return float((2.0 * intersection + eps) / (pred.sum() + target.sum() + eps))
