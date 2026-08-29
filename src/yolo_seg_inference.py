from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def result_masks(result, height: int, width: int, threshold: float) -> list[tuple[np.ndarray, float]]:
    if result.masks is None or result.boxes is None:
        return []
    masks = result.masks.data.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
    output = []
    for mask, score in zip(masks, scores):
        if mask.shape != (height, width):
            mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
        output.append(((mask >= threshold).astype(np.uint8), float(score)))
    return output


def tile_starts(length: int, tile_size: int, overlap: float) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = max(1, int(round(tile_size * (1.0 - overlap))))
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def predict_yolo_seg_multiscale(
    model,
    image_path: Path,
    image: np.ndarray,
    device: str,
    imgsz: list[int],
    conf: float,
    iou: float,
    max_det: int,
    threshold: float,
    tile_size: int = 0,
    tile_overlap: float = 0.25,
    tile_score_weight: float = 0.95,
) -> list[tuple[np.ndarray, float]]:
    height, width = image.shape
    candidates: list[tuple[np.ndarray, float]] = []
    for size in imgsz:
        result = model.predict(
            source=str(image_path),
            imgsz=int(size),
            conf=conf,
            iou=iou,
            max_det=max_det,
            device=device,
            verbose=False,
        )[0]
        candidates.extend(result_masks(result, height, width, threshold))

    if tile_size <= 0:
        return candidates

    gray_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    bgr = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    for y0 in tile_starts(height, tile_size, tile_overlap):
        for x0 in tile_starts(width, tile_size, tile_overlap):
            y1, x1 = min(height, y0 + tile_size), min(width, x0 + tile_size)
            tile = bgr[y0:y1, x0:x1]
            result = model.predict(
                source=tile,
                imgsz=tile_size,
                conf=conf,
                iou=iou,
                max_det=max_det,
                device=device,
                verbose=False,
            )[0]
            for local_mask, score in result_masks(result, y1 - y0, x1 - x0, threshold):
                # Internal-edge masks are usually truncated duplicates. An
                # overlapping neighbour or the full-image pass can retain them.
                touches_internal_edge = (
                    (x0 > 0 and local_mask[:, :2].any())
                    or (x1 < width and local_mask[:, -2:].any())
                    or (y0 > 0 and local_mask[:2, :].any())
                    or (y1 < height and local_mask[-2:, :].any())
                )
                if touches_internal_edge:
                    continue
                full_mask = np.zeros((height, width), dtype=np.uint8)
                full_mask[y0:y1, x0:x1] = local_mask
                candidates.append((full_mask, float(score) * tile_score_weight))
    return candidates
