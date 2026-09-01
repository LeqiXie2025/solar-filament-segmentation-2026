from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from pycocotools import mask as coco_mask
from sklearn.model_selection import GroupKFold

from experiments.sweep_detector_mask_ensemble import decode_rows
from src.dataset import read_gray_image
from src.metrics import overlap_matrix
from src.train_instance_quality import evaluate_threshold
import config


EXTRA_FEATURES = ["peer_best_iou", "peer_matches_iou10", "peer_area_ratio", "peer_overlap_fraction"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality-dataset", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--peer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def make_model(seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary", n_estimators=250, learning_rate=0.02, num_leaves=7,
        max_depth=3, min_child_samples=28, subsample=0.9, colsample_bytree=0.9,
        reg_alpha=0.7, reg_lambda=4.0, random_state=seed, verbosity=-1,
    )


def peer_features(base_masks: list[np.ndarray], peer_masks: list[np.ndarray]) -> np.ndarray:
    if not base_masks:
        return np.empty((0, len(EXTRA_FEATURES)), dtype=np.float32)
    if not peer_masks:
        return np.zeros((len(base_masks), len(EXTRA_FEATURES)), dtype=np.float32)
    matrix = overlap_matrix(base_masks, peer_masks, mode="iou")
    rows = []
    for index, mask in enumerate(base_masks):
        values = matrix[index]
        best_idx = int(np.argmax(values))
        best_iou = float(values[best_idx])
        peer = peer_masks[best_idx]
        inter = float(np.logical_and(mask, peer).sum())
        area = max(float(mask.sum()), 1.0)
        rows.append((best_iou, float((values >= 0.10).sum()), float(peer.sum()) / area, inter / area))
    return np.asarray(rows, dtype=np.float32)


def main() -> None:
    args = parse_args()
    payload = json.loads(args.quality_dataset.read_text(encoding="utf-8"))
    images = payload["images"]
    base_frame, peer_frame = pd.read_csv(args.base), pd.read_csv(args.peer)
    x_rows, labels, groups = [], [], []
    for group_idx, image in enumerate(images):
        file_name = image["file_name"]
        height, width = read_gray_image(config.TRAIN_IMG_DIR / file_name).shape
        base_masks = decode_rows(base_frame, file_name, height, width)
        peer_masks = decode_rows(peer_frame, file_name, height, width)
        candidates = image["candidates"]
        if len(base_masks) != len(candidates):
            raise ValueError(f"Base/quality mismatch in {file_name}")
        extras = peer_features(base_masks, peer_masks)
        for candidate, extra in zip(candidates, extras):
            x_rows.append(list(candidate["features"]) + list(extra))
            labels.append(float(np.mean([bool(hits) for hits in candidate["hits"]])))
            groups.append(group_idx)
    x = np.asarray(x_rows, dtype=np.float32)
    soft_y = np.asarray(labels, dtype=np.float32)
    y = (soft_y >= 0.50).astype(np.uint8)
    groups = np.asarray(groups)
    sample_weight = 0.5 + np.abs(soft_y - 0.5)
    oof = np.zeros(len(x), dtype=np.float32)
    for fold, (train_idx, val_idx) in enumerate(GroupKFold(n_splits=5).split(x, y, groups)):
        model = make_model(3407 + fold)
        model.fit(x[train_idx], y[train_idx], sample_weight=sample_weight[train_idx])
        oof[val_idx] = model.predict_proba(x[val_idx])[:, 1]

    best = (0.0, 0.0, 0, 0, 0)
    for threshold in np.linspace(0.05, 0.80, 151):
        pq, tp, fp, fn = evaluate_threshold(images, oof, float(threshold))
        if pq > best[0]:
            best = (pq, float(threshold), tp, fp, fn)
    final_models = []
    for seed in (3507, 3508, 3509):
        model = make_model(seed)
        model.fit(x, y, sample_weight=sample_weight)
        final_models.append(model)
    model_payload = {
        "models": final_models,
        "feature_names": payload["feature_names"] + EXTRA_FEATURES,
        "threshold": best[1], "oof_pq": best[0], "oof_tp": best[2], "oof_fp": best[3], "oof_fn": best[4],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_payload, args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        f"oof_pq={best[0]:.6f}\nthreshold={best[1]:.3f}\ntp={best[2]}\nfp={best[3]}\nfn={best[4]}\n",
        encoding="utf-8",
    )
    print(args.report.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
