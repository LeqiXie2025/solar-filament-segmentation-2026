from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as coco_mask
from sklearn.model_selection import GroupKFold

import config
from experiments.sweep_detector_mask_ensemble import decode_rows, greedy_pairs
from src.instance_quality import FEATURE_NAMES
from src.metrics import official_pq
from src.predict_detect_refine import time_val_names
from src.train_instance_quality import make_model
from src.validation import load_gt_instances_by_annotator
from src.dataset import read_gray_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--peer", type=Path, required=True)
    parser.add_argument("--quality-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quality = json.loads(args.quality_dataset.read_text(encoding="utf-8"))
    if quality["feature_names"] != FEATURE_NAMES:
        raise ValueError("Quality feature schema mismatch")
    images = quality["images"]
    features, labels, groups = [], [], []
    for group_idx, image in enumerate(images):
        for candidate in image["candidates"]:
            features.append(candidate["features"])
            labels.append(float(np.mean([bool(hits) for hits in candidate["hits"]])))
            groups.append(group_idx)
    x = np.asarray(features, dtype=np.float32)
    y_soft = np.asarray(labels, dtype=np.float32)
    y = (y_soft >= 0.50).astype(np.uint8)
    sample_weight = 0.5 + np.abs(y_soft - 0.5)
    groups = np.asarray(groups)
    oof = np.zeros(len(x), dtype=np.float32)
    for fold, (train_idx, val_idx) in enumerate(GroupKFold(n_splits=5).split(x, y, groups)):
        model = make_model(42 + fold)
        model.fit(x[train_idx], y[train_idx], sample_weight=sample_weight[train_idx])
        oof[val_idx] = model.predict_proba(x[val_idx])[:, 1]

    base_frame = pd.read_csv(args.base)
    peer_frame = pd.read_csv(args.peer)
    existing_files = {p.name for p in config.TRAIN_IMG_DIR.iterdir() if p.suffix.lower() in {".jpeg", ".jpg", ".png"}}
    file_names = time_val_names(config.ANNOTATION_PATH, existing_files)
    if [image["file_name"] for image in images] != file_names:
        raise ValueError("Quality dataset image order does not match chronological validation")
    gt = load_gt_instances_by_annotator(config.ANNOTATION_PATH, set(file_names))

    decoded = {}
    offset = 0
    for image in images:
        file_name = image["file_name"]
        height, width = read_gray_image(config.TRAIN_IMG_DIR / file_name).shape
        base = decode_rows(base_frame, file_name, height, width)
        peer = decode_rows(peer_frame, file_name, height, width)
        count = len(image["candidates"])
        if count != len(base):
            raise ValueError(f"Candidate mismatch for {file_name}: {count} vs {len(base)}")
        base_rles = [coco_mask.encode(np.asfortranarray(mask)) for mask in base]
        matrices = []
        for annotator_gts in gt.get(file_name, []):
            gt_rles = [coco_mask.encode(np.asfortranarray(mask)) for mask in annotator_gts]
            matrices.append(coco_mask.iou(base_rles, gt_rles, [0] * len(gt_rles)) if base_rles and gt_rles else np.zeros((len(base), len(annotator_gts))))
        decoded[file_name] = (base, peer, oof[offset : offset + count], matrices)
        offset += count

    rows = []
    for quality_threshold in (0.10, 0.15, 0.20, 0.26, 0.30, 0.35, 0.40):
        for match_iou in (0.0, 0.10, 0.20, 0.35, 0.50):
            iou_sum = 0.0
            tp = fp = fn = instances = 0
            for file_name in file_names:
                base, peer, probabilities, matrices = decoded[file_name]
                if match_iou <= 0.0:
                    confirmed = set()
                else:
                    _pairs, confirmed, _ = greedy_pairs(base, peer, match_iou)
                keep = [idx for idx, p in enumerate(probabilities) if p >= quality_threshold or idx in confirmed]
                instances += len(keep)
                for matrix in matrices:
                    selected = matrix[keep, :] if keep else matrix[:0, :]
                    hits = selected > 0.5
                    iou_sum += float(selected[hits].sum())
                    tp += int(hits.sum())
                    fp += int((hits.sum(axis=1) == 0).sum())
                    fn += int((hits.sum(axis=0) == 0).sum())
            rows.append({
                "pq": official_pq(iou_sum, tp, fp, fn), "tp": tp, "fp": fp, "fn": fn,
                "instances": instances, "quality_threshold": quality_threshold, "match_iou": match_iou,
            })
    output = pd.DataFrame(rows).sort_values("pq", ascending=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(output.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
