from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

from src.instance_quality import FEATURE_NAMES
from src.metrics import official_pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_model(seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.02,
        num_leaves=7,
        max_depth=3,
        min_child_samples=30,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_alpha=0.5,
        reg_lambda=3.0,
        random_state=seed,
        verbosity=-1,
    )


def evaluate_threshold(images: list[dict], probabilities: np.ndarray, threshold: float) -> tuple[float, int, int, int]:
    iou_sum = 0.0
    tp = fp = fn = 0
    offset = 0
    for image in images:
        candidates = image["candidates"]
        selected = probabilities[offset : offset + len(candidates)] >= threshold
        offset += len(candidates)
        for annotator_idx, gt_count in enumerate(image["gt_counts"]):
            hit_gt = set()
            for keep, candidate in zip(selected, candidates):
                if not keep:
                    continue
                hits = candidate["hits"][annotator_idx]
                if hits:
                    for gt_idx, iou in hits:
                        tp += 1
                        iou_sum += float(iou)
                        hit_gt.add(int(gt_idx))
                else:
                    fp += 1
            fn += int(gt_count) - len(hit_gt)
    return official_pq(iou_sum, tp, fp, fn), tp, fp, fn


def main() -> None:
    args = parse_args()
    payload = json.loads(args.dataset.read_text(encoding="utf-8"))
    if payload["feature_names"] != FEATURE_NAMES:
        raise ValueError("Quality dataset feature schema does not match")
    images = payload["images"]
    features = []
    groups = []
    labels = []
    candidate_slices = []
    for group_idx, image in enumerate(images):
        start = len(features)
        for candidate in image["candidates"]:
            features.append(candidate["features"])
            groups.append(group_idx)
            annotator_labels = [1 if hits else 0 for hits in candidate["hits"]]
            labels.append(float(np.mean(annotator_labels)) if annotator_labels else 0.0)
        candidate_slices.append((start, len(features)))

    x = np.asarray(features, dtype=np.float32)
    soft_y = np.asarray(labels, dtype=np.float32)
    y = (soft_y >= 0.50).astype(np.uint8)
    sample_weight = 0.5 + np.abs(soft_y - 0.5)
    groups_array = np.asarray(groups)
    oof = np.zeros(len(x), dtype=np.float32)
    splitter = GroupKFold(n_splits=min(args.folds, len(images)))
    fold_models = []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(x, y, groups_array)):
        model = make_model(args.seed + fold)
        model.fit(x[train_idx], y[train_idx], sample_weight=sample_weight[train_idx])
        oof[val_idx] = model.predict_proba(x[val_idx])[:, 1]
        fold_models.append(model)

    baseline = evaluate_threshold(images, np.ones(len(x), dtype=np.float32), 0.0)
    best = (baseline[0], 0.0, baseline[1], baseline[2], baseline[3])
    for threshold in np.linspace(0.05, 0.95, 181):
        pq, tp, fp, fn = evaluate_threshold(images, oof, float(threshold))
        if pq > best[0]:
            best = (pq, float(threshold), tp, fp, fn)

    final_models = []
    for seed_offset in range(3):
        model = make_model(args.seed + 100 + seed_offset)
        model.fit(x, y, sample_weight=sample_weight)
        final_models.append(model)

    output = {
        "models": final_models,
        "feature_names": FEATURE_NAMES,
        "threshold": best[1],
        "oof_pq": best[0],
        "oof_tp": best[2],
        "oof_fp": best[3],
        "oof_fn": best[4],
        "baseline_pq": baseline[0],
        "baseline_tp": baseline[1],
        "baseline_fp": baseline[2],
        "baseline_fn": baseline[3],
        "training_candidates": len(x),
        "training_images": len(images),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(output, args.output)
    print(
        f"baseline={baseline[0]:.6f} ({baseline[1]}/{baseline[2]}/{baseline[3]}) "
        f"oof={best[0]:.6f} ({best[2]}/{best[3]}/{best[4]}) threshold={best[1]:.3f}"
    )


if __name__ == "__main__":
    main()
