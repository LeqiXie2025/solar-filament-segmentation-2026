from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as coco_mask

import config
from experiments.sweep_detector_mask_ensemble import combine
from src.dataset import read_gray_image
from src.masks import encode_binary_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--match-iou", type=float, default=0.10)
    parser.add_argument("--min-area", type=int, default=120)
    parser.add_argument("--fusion", choices=["a", "b", "intersection", "union"], default="union")
    return parser.parse_args()


def add_stem(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["stem"] = frame["filament_id"].str.rsplit("_", n=1).str[0]
    return frame


def decode(frame: pd.DataFrame, stem: str, height: int, width: int) -> list[np.ndarray]:
    return [
        coco_mask.decode({"size": [height, width], "counts": value.encode("utf-8")}).astype(np.uint8)
        for value in frame.loc[frame["stem"] == stem, "segmentation_rle"]
    ]


def main() -> None:
    args = parse_args()
    a_frame, b_frame = add_stem(pd.read_csv(args.a)), add_stem(pd.read_csv(args.b))
    image_paths = {
        path.stem: path
        for path in config.TEST_IMG_DIR.iterdir()
        if path.suffix.lower() in {".jpeg", ".jpg", ".png"}
    }
    rows = []
    for stem in sorted(image_paths):
        height, width = read_gray_image(image_paths[stem]).shape
        masks = combine(
            decode(a_frame, stem, height, width),
            decode(b_frame, stem, height, width),
            args.match_iou,
            "a",
            args.fusion,
            args.min_area,
        )
        for index, mask in enumerate(sorted(masks, key=lambda item: int(item.sum()), reverse=True), start=1):
            rows.append({"filament_id": f"{stem}_{index}", "segmentation_rle": encode_binary_mask(mask)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(args.output, index=False)
    print(f"saved={args.output} rows={len(rows)}")


if __name__ == "__main__":
    main()
