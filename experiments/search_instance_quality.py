from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.train_instance_quality import evaluate_threshold


DATASET = Path("data/clean/instance_quality_official_time.json")


def main() -> None:
    payload = json.loads(DATASET.read_text(encoding="utf-8"))
    images = payload["images"]
    x, soft_y, groups = [], [], []
    for group_idx, image in enumerate(images):
        for candidate in image["candidates"]:
            x.append(candidate["features"])
            soft_y.append(np.mean([1.0 if hits else 0.0 for hits in candidate["hits"]]))
            groups.append(group_idx)
    x = np.asarray(x, dtype=np.float32)
    soft_y = np.asarray(soft_y, dtype=np.float32)
    y = (soft_y >= 0.5).astype(np.uint8)
    groups = np.asarray(groups)
    weight = 0.5 + np.abs(soft_y - 0.5)

    factories = {
        "lgb7": lambda seed: LGBMClassifier(objective="binary", n_estimators=300, learning_rate=0.02, num_leaves=7, max_depth=3, min_child_samples=30, reg_alpha=0.5, reg_lambda=3.0, colsample_bytree=0.9, random_state=seed, verbosity=-1),
        "lgb15": lambda seed: LGBMClassifier(objective="binary", n_estimators=220, learning_rate=0.025, num_leaves=15, max_depth=5, min_child_samples=25, reg_alpha=0.7, reg_lambda=4.0, colsample_bytree=0.85, random_state=seed, verbosity=-1),
        "lgb_soft": lambda seed: LGBMRegressor(objective="regression_l1", n_estimators=260, learning_rate=0.02, num_leaves=9, max_depth=4, min_child_samples=30, reg_alpha=0.5, reg_lambda=3.0, colsample_bytree=0.9, random_state=seed, verbosity=-1),
        "extra": lambda seed: ExtraTreesClassifier(n_estimators=500, max_depth=6, min_samples_leaf=12, max_features=0.8, class_weight="balanced", random_state=seed, n_jobs=-1),
        "rf": lambda seed: RandomForestClassifier(n_estimators=500, max_depth=6, min_samples_leaf=12, max_features=0.8, class_weight="balanced", random_state=seed, n_jobs=-1),
        "hist": lambda seed: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.04, max_leaf_nodes=9, l2_regularization=3.0, min_samples_leaf=25, random_state=seed),
        "logit": lambda seed: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000, random_state=seed)),
    }
    folds = list(GroupKFold(n_splits=5).split(x, y, groups))
    predictions = {}
    for name, factory in factories.items():
        oof = np.zeros(len(x), dtype=np.float32)
        for fold, (train_idx, val_idx) in enumerate(folds):
            model = factory(100 + fold)
            target = soft_y[train_idx] if name == "lgb_soft" else y[train_idx]
            try:
                model.fit(x[train_idx], target, sample_weight=weight[train_idx])
            except (TypeError, ValueError):
                model.fit(x[train_idx], target)
            if name == "lgb_soft":
                oof[val_idx] = np.clip(model.predict(x[val_idx]), 0.0, 1.0)
            else:
                oof[val_idx] = model.predict_proba(x[val_idx])[:, 1]
        predictions[name] = oof

    ensembles = {
        "lgb_mean": np.mean([predictions["lgb7"], predictions["lgb15"], predictions["lgb_soft"]], axis=0),
        "all_tree_mean": np.mean([predictions[name] for name in ("lgb7", "lgb15", "lgb_soft", "extra", "rf", "hist")], axis=0),
    }
    predictions.update(ensembles)
    for name, probabilities in predictions.items():
        best = (0.0, 0.0, 0, 0, 0)
        for threshold in np.linspace(0.02, 0.98, 193):
            pq, tp, fp, fn = evaluate_threshold(images, probabilities, float(threshold))
            if pq > best[0]:
                best = (pq, float(threshold), tp, fp, fn)
        print(f"{name:14s} pq={best[0]:.6f} threshold={best[1]:.3f} tp/fp/fn={best[2]}/{best[3]}/{best[4]}")


if __name__ == "__main__":
    main()
