from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as coco_mask

import config
from src.dataset import read_gray_image
from src.metrics import official_pq, overlap_matrix
from src.predict_detect_refine import time_val_names
from src.validation import load_gt_instances_by_annotator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    return parser.parse_args()


def decode_rows(frame: pd.DataFrame, file_name: str, height: int, width: int) -> list[np.ndarray]:
    rows = frame.loc[frame["file_name"] == file_name, "segmentation_rle"]
    return [
        coco_mask.decode({"size": [height, width], "counts": value.encode("utf-8")}).astype(np.uint8)
        for value in rows
    ]


def greedy_pairs(a: list[np.ndarray], b: list[np.ndarray], threshold: float):
    matrix = overlap_matrix(a, b, mode="iou")
    candidates = sorted(
        [(float(matrix[i, j]), i, j) for i in range(len(a)) for j in range(len(b)) if matrix[i, j] >= threshold],
        reverse=True,
    )
    used_a, used_b, pairs = set(), set(), []
    for iou, i, j in candidates:
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            pairs.append((i, j, iou))
    return pairs, used_a, used_b


def combine(
    a: list[np.ndarray],
    b: list[np.ndarray],
    match_iou: float,
    source: str,
    fusion: str,
    min_area: int,
    add_extras: bool = True,
):
    if not a and not b:
        return []
    pairs, used_a, used_b = greedy_pairs(a, b, match_iou)
    pair_by_a = {i: j for i, j, _iou in pairs}
    outputs = []
    if source == "a":
        for i, mask in enumerate(a):
            if i in pair_by_a and fusion != "a":
                other = b[pair_by_a[i]].astype(bool)
                if fusion == "intersection":
                    mask = (mask.astype(bool) & other).astype(np.uint8)
                elif fusion == "union":
                    mask = (mask.astype(bool) | other).astype(np.uint8)
                elif fusion == "b":
                    mask = other.astype(np.uint8)
            if int(mask.sum()) >= min_area:
                outputs.append(mask)
        extras = [b[j] for j in range(len(b)) if j not in used_b]
    else:
        pair_by_b = {j: i for i, j, _iou in pairs}
        for j, mask in enumerate(b):
            if j in pair_by_b and fusion != "b":
                other = a[pair_by_b[j]].astype(bool)
                if fusion == "intersection":
                    mask = (mask.astype(bool) & other).astype(np.uint8)
                elif fusion == "union":
                    mask = (mask.astype(bool) | other).astype(np.uint8)
                elif fusion == "a":
                    mask = other.astype(np.uint8)
            if int(mask.sum()) >= min_area:
                outputs.append(mask)
        extras = [a[i] for i in range(len(a)) if i not in used_a]

    claimed = np.zeros_like(a[0] if a else b[0], dtype=bool)
    clean_outputs = []
    for mask in sorted(outputs, key=lambda item: int(item.sum()), reverse=True):
        clean = (mask.astype(bool) & ~claimed).astype(np.uint8)
        if int(clean.sum()) >= min_area:
            clean_outputs.append(clean)
            claimed |= clean.astype(bool)
    outputs = clean_outputs
    if add_extras:
        for extra in sorted(extras, key=lambda mask: int(mask.sum()), reverse=True):
            clean = (extra.astype(bool) & ~claimed).astype(np.uint8)
            if int(clean.sum()) >= min_area:
                outputs.append(clean)
                claimed |= clean.astype(bool)
    return outputs


def fast_official_counts(preds: list[np.ndarray], gts: list[np.ndarray]):
    if not gts:
        return 0.0, 0, len(preds), 0
    if not preds:
        return 0.0, 0, 0, len(gts)
    pred_rles = [coco_mask.encode(np.asfortranarray(mask.astype(np.uint8))) for mask in preds]
    gt_rles = [coco_mask.encode(np.asfortranarray(mask.astype(np.uint8))) for mask in gts]
    matrix = coco_mask.iou(pred_rles, gt_rles, [0] * len(gt_rles))
    hits = matrix > 0.5
    return (
        float(matrix[hits].sum()),
        int(hits.sum()),
        int((hits.sum(axis=1) == 0).sum()),
        int((hits.sum(axis=0) == 0).sum()),
    )


def main() -> None:
    args = parse_args()
    a_frame, b_frame = pd.read_csv(args.a), pd.read_csv(args.b)
    existing_files = {
        path.name
        for path in config.TRAIN_IMG_DIR.iterdir()
        if path.suffix.lower() in {".jpeg", ".jpg", ".png"}
    }
    file_names = time_val_names(config.ANNOTATION_PATH, existing_files)
    gt = load_gt_instances_by_annotator(config.ANNOTATION_PATH, set(file_names))
    decoded = {}
    for file_name in file_names:
        image = read_gray_image(config.TRAIN_IMG_DIR / file_name)
        decoded[file_name] = (
            decode_rows(a_frame, file_name, *image.shape),
            decode_rows(b_frame, file_name, *image.shape),
        )

    configs = [
        (0.10, 120, "a", "a", False),
        (0.35, 120, "a", "b", False),
        (0.35, 120, "a", "intersection", False),
        (0.35, 120, "a", "union", False),
        (0.70, 120, "a", "intersection", False),
        (0.70, 120, "a", "union", False),
    ]

    results = []
    for match_iou, min_area, source, fusion, add_extras in configs:
        iou_sum = 0.0
        tp = fp = fn = instances = 0
        for file_name in file_names:
            preds = combine(*decoded[file_name], match_iou, source, fusion, min_area, add_extras)
            instances += len(preds)
            for annotator_gts in gt.get(file_name, []):
                values = fast_official_counts(preds, annotator_gts)
                iou_sum += values[0]
                tp += values[1]
                fp += values[2]
                fn += values[3]
        results.append(
            {
                "pq": official_pq(iou_sum, tp, fp, fn),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "instances": instances,
                "match_iou": match_iou,
                "min_area": min_area,
                "source": source,
                "fusion": fusion,
                "add_extras": add_extras,
            }
        )
    output = pd.DataFrame(results).sort_values("pq", ascending=False)
    output.to_csv("experiments/detector_mask_ensemble_sweep.csv", index=False)
    print(output.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
