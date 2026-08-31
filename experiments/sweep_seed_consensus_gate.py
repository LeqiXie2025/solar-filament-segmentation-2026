from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as coco_mask

import config
from experiments.sweep_detector_mask_ensemble import decode_rows, greedy_pairs
from src.dataset import read_gray_image
from src.metrics import official_pq
from src.predict_detect_refine import time_val_names
from src.validation import load_gt_instances_by_annotator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--peer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_frame = pd.read_csv(args.base)
    peer_frame = pd.read_csv(args.peer)
    existing_files = {
        path.name
        for path in config.TRAIN_IMG_DIR.iterdir()
        if path.suffix.lower() in {".jpeg", ".jpg", ".png"}
    }
    file_names = time_val_names(config.ANNOTATION_PATH, existing_files)
    gt = load_gt_instances_by_annotator(config.ANNOTATION_PATH, set(file_names))
    decoded: dict[str, tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]] = {}
    for file_name in file_names:
        height, width = read_gray_image(config.TRAIN_IMG_DIR / file_name).shape
        base = decode_rows(base_frame, file_name, height, width)
        peer = decode_rows(peer_frame, file_name, height, width)
        base_rles = [coco_mask.encode(np.asfortranarray(mask)) for mask in base]
        matrices = []
        for annotator_gts in gt.get(file_name, []):
            gt_rles = [coco_mask.encode(np.asfortranarray(mask)) for mask in annotator_gts]
            if base_rles and gt_rles:
                matrices.append(coco_mask.iou(base_rles, gt_rles, [0] * len(gt_rles)))
            else:
                matrices.append(np.zeros((len(base), len(annotator_gts)), dtype=np.float32))
        decoded[file_name] = (base, peer, matrices)

    results = []
    for match_iou in (0.10, 0.20, 0.35, 0.50):
        for unmatched_min_area in (0, 120, 250, 500, 750, 1000, 1500, 2000):
            iou_sum = 0.0
            tp = fp = fn = instances = removed = 0
            for file_name in file_names:
                base, peer, matrices = decoded[file_name]
                _pairs, matched_base, _matched_peer = greedy_pairs(base, peer, match_iou)
                kept_indices = []
                for index, mask in enumerate(base):
                    if index in matched_base or int(mask.sum()) >= unmatched_min_area:
                        kept_indices.append(index)
                    else:
                        removed += 1
                instances += len(kept_indices)
                for matrix in matrices:
                    selected = matrix[kept_indices, :] if kept_indices else matrix[:0, :]
                    hits = selected > 0.5
                    iou_sum += float(selected[hits].sum())
                    tp += int(hits.sum())
                    fp += int((hits.sum(axis=1) == 0).sum())
                    fn += int((hits.sum(axis=0) == 0).sum())
            results.append(
                {
                    "pq": official_pq(iou_sum, tp, fp, fn),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "instances": instances,
                    "removed": removed,
                    "match_iou": match_iou,
                    "unmatched_min_area": unmatched_min_area,
                }
            )
    output = pd.DataFrame(results).sort_values("pq", ascending=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(output.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
