from __future__ import annotations

import random

import cv2
import numpy as np


def order_quad_long_edge_first(points: np.ndarray) -> np.ndarray:
    """Return four cyclic points with the first edge on the rectangle's long axis."""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    edge_lengths = np.linalg.norm(np.roll(ordered, -1, axis=0) - ordered, axis=1)
    start = int(np.argmax(edge_lengths))
    ordered = np.roll(ordered, -start, axis=0)
    # Keep a deterministic direction so inference/TTA do not randomly reverse crops.
    axis = ordered[1] - ordered[0]
    if axis[0] < 0 or (abs(float(axis[0])) < 1e-6 and axis[1] < 0):
        ordered = np.asarray([ordered[1], ordered[0], ordered[3], ordered[2]], dtype=np.float32)
    return ordered.astype(np.float32)


def mask_to_obb(mask: np.ndarray) -> np.ndarray | None:
    """Fit a minimum-area rectangle to a binary instance mask."""
    ys, xs = np.where(mask.astype(bool))
    if len(xs) < 3:
        return None
    points = np.column_stack([xs, ys]).astype(np.float32)
    rect = cv2.minAreaRect(points)
    if min(rect[1]) < 1.0:
        return None
    return order_quad_long_edge_first(cv2.boxPoints(rect))


def quad_dimensions(quad: np.ndarray) -> tuple[float, float]:
    quad = order_quad_long_edge_first(quad)
    long_side = float(np.linalg.norm(quad[1] - quad[0]))
    short_side = float(np.linalg.norm(quad[3] - quad[0]))
    return max(long_side, 1.0), max(short_side, 1.0)


def expand_obb(
    quad: np.ndarray,
    pad_long: float = 0.18,
    pad_short: float = 0.65,
    jitter: bool = False,
) -> np.ndarray:
    """Expand an oriented rectangle while optionally simulating detector noise."""
    quad = order_quad_long_edge_first(quad)
    center = quad.mean(axis=0)
    long_vector = quad[1] - quad[0]
    short_vector = quad[3] - quad[0]
    long_side = max(float(np.linalg.norm(long_vector)), 1.0)
    short_side = max(float(np.linalg.norm(short_vector)), 1.0)
    long_axis = long_vector / long_side
    short_axis = short_vector / short_side

    long_scale = 1.0 + 2.0 * float(pad_long)
    short_scale = 1.0 + 2.0 * float(pad_short)
    if jitter:
        center = (
            center
            + long_axis * random.uniform(-0.10, 0.10) * long_side
            + short_axis * random.uniform(-0.18, 0.18) * short_side
        )
        long_scale *= random.uniform(0.90, 1.16)
        short_scale *= random.uniform(0.85, 1.22)
        angle = random.uniform(-5.0, 5.0)
        radians = np.deg2rad(angle)
        rotation = np.asarray(
            [[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]],
            dtype=np.float32,
        )
        long_axis = rotation @ long_axis
        short_axis = rotation @ short_axis

    half_long = 0.5 * long_side * long_scale
    half_short = 0.5 * short_side * short_scale
    return np.asarray(
        [
            center - long_axis * half_long - short_axis * half_short,
            center + long_axis * half_long - short_axis * half_short,
            center + long_axis * half_long + short_axis * half_short,
            center - long_axis * half_long + short_axis * half_short,
        ],
        dtype=np.float32,
    )


def perspective_matrix(quad: np.ndarray, out_width: int, out_height: int) -> tuple[np.ndarray, np.ndarray]:
    source = order_quad_long_edge_first(quad).astype(np.float32)
    destination = np.asarray(
        [[0.0, 0.0], [out_width - 1.0, 0.0], [out_width - 1.0, out_height - 1.0], [0.0, out_height - 1.0]],
        dtype=np.float32,
    )
    forward = cv2.getPerspectiveTransform(source, destination)
    inverse = cv2.getPerspectiveTransform(destination, source)
    return forward, inverse


def warp_obb(
    array: np.ndarray,
    quad: np.ndarray,
    out_width: int,
    out_height: int,
    interpolation: int = cv2.INTER_LINEAR,
) -> tuple[np.ndarray, np.ndarray]:
    forward, inverse = perspective_matrix(quad, out_width, out_height)
    warped = cv2.warpPerspective(
        array,
        forward,
        (int(out_width), int(out_height)),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped, inverse


def unwarp_mask(mask: np.ndarray, inverse: np.ndarray, out_width: int, out_height: int) -> np.ndarray:
    return cv2.warpPerspective(
        mask.astype(np.uint8),
        inverse,
        (int(out_width), int(out_height)),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def seed_from_inner_obb(inner_quad: np.ndarray, crop_quad: np.ndarray, out_width: int, out_height: int) -> np.ndarray:
    forward, _ = perspective_matrix(crop_quad, out_width, out_height)
    transformed = cv2.perspectiveTransform(
        order_quad_long_edge_first(inner_quad)[None, :, :].astype(np.float32), forward
    )[0]
    seed = np.zeros((int(out_height), int(out_width)), dtype=np.float32)
    cv2.fillPoly(seed, [np.round(transformed).astype(np.int32)], 1.0)
    return seed


def make_obb_refiner_channels(crop: np.ndarray, seed: np.ndarray) -> np.ndarray:
    crop = crop.astype(np.float32)
    if crop.max() > 1.0:
        crop = crop / 255.0
    crop = np.clip(crop, 0.0, 1.0)
    crop_u8 = np.round(crop * 255.0).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 6)).apply(crop_u8).astype(np.float32) / 255.0

    local_mean = cv2.GaussianBlur(crop, (0, 0), sigmaX=9.0, sigmaY=9.0)
    darkness = np.clip(local_mean - crop, 0.0, None)
    scale = float(np.percentile(darkness, 99.0))
    if scale > 1e-6:
        darkness = np.clip(darkness / scale, 0.0, 1.0)

    grad_x = cv2.Sobel(clahe, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(clahe, cv2.CV_32F, 0, 1, ksize=3)
    edges = cv2.magnitude(grad_x, grad_y)
    edge_scale = float(np.percentile(edges, 99.0))
    if edge_scale > 1e-6:
        edges = np.clip(edges / edge_scale, 0.0, 1.0)

    return np.stack([crop, clahe, darkness, seed.astype(np.float32), edges], axis=0).astype(np.float32)
