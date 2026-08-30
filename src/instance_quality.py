from __future__ import annotations

from pathlib import Path

import cv2
import joblib
import numpy as np


FEATURE_NAMES = [
    "detector_score",
    "log_area",
    "log_bbox_area",
    "fill_ratio",
    "aspect_log",
    "compactness",
    "solidity",
    "elongation_log",
    "inside_mean",
    "inside_std",
    "inside_p10",
    "inside_p90",
    "ring_mean",
    "ring_std",
    "ring_minus_inside",
    "center_x",
    "center_y",
    "center_radius",
]


def instance_features(image: np.ndarray, mask: np.ndarray, detector_score: float) -> np.ndarray:
    mask_bool = mask.astype(bool)
    ys, xs = np.nonzero(mask_bool)
    if len(xs) == 0:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    height, width = image.shape
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    box_w, box_h = max(1, x1 - x0), max(1, y1 - y0)
    area = float(len(xs))
    bbox_area = float(box_w * box_h)

    crop_mask = mask_bool[y0:y1, x0:x1].astype(np.uint8)
    contours, _ = cv2.findContours(crop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
    hull_area = 0.0
    for contour in contours:
        if len(contour) >= 3:
            hull_area += float(cv2.contourArea(cv2.convexHull(contour)))

    centered = np.column_stack((xs - xs.mean(), ys - ys.mean())).astype(np.float64)
    if len(centered) >= 3:
        eigenvalues = np.linalg.eigvalsh(np.cov(centered, rowvar=False))
        elongation = float((eigenvalues[-1] + 1e-6) / (eigenvalues[0] + 1e-6))
    else:
        elongation = 1.0

    inside = image[mask_bool].astype(np.float32)
    kernel_size = max(5, int(round(np.sqrt(area) * 0.20)) | 1)
    kernel_size = min(kernel_size, 51)
    dilated = cv2.dilate(mask_bool.astype(np.uint8), np.ones((kernel_size, kernel_size), np.uint8)) > 0
    ring = image[dilated & ~mask_bool].astype(np.float32)
    ring_mean = float(ring.mean()) if ring.size else float(inside.mean())
    ring_std = float(ring.std()) if ring.size else 0.0

    center_x = float(xs.mean() / max(width - 1, 1))
    center_y = float(ys.mean() / max(height - 1, 1))
    center_radius = float(np.hypot(center_x - 0.5, center_y - 0.5) / np.sqrt(0.5))
    inside_mean = float(inside.mean())

    return np.asarray(
        [
            float(detector_score),
            np.log1p(area),
            np.log1p(bbox_area),
            area / bbox_area,
            abs(np.log(max(box_w, 1) / max(box_h, 1))),
            4.0 * np.pi * area / max(perimeter * perimeter, 1e-6),
            area / max(hull_area, area),
            np.log1p(elongation),
            inside_mean,
            float(inside.std()),
            float(np.quantile(inside, 0.10)),
            float(np.quantile(inside, 0.90)),
            ring_mean,
            ring_std,
            ring_mean - inside_mean,
            center_x,
            center_y,
            center_radius,
        ],
        dtype=np.float32,
    )


def load_instance_quality_model(path: Path):
    payload = joblib.load(path)
    if payload.get("feature_names") != FEATURE_NAMES:
        raise ValueError("Instance-quality feature schema does not match this code version")
    return payload


def predict_instance_quality(payload: dict, features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    models = payload.get("models") or [payload["model"]]
    probabilities = [model.predict_proba(features)[:, 1] for model in models]
    return np.mean(probabilities, axis=0).astype(np.float32)
