from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as coco_mask

import config
from experiments.sweep_detector_mask_ensemble import greedy_pairs
from src.dataset import read_gray_image
from src.masks import encode_binary_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--peer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--match-iou", type=float, default=0.10)
    parser.add_argument("--unmatched-min-area", type=int, default=2000)
    return parser.parse_args()


def masks_for(frame: pd.DataFrame, stem: str, height: int, width: int) -> list[np.ndarray]:
    rows = frame.loc[frame["stem"] == stem, "segmentation_rle"]
    return [
        coco_mask.decode({"size": [height, width], "counts": rle.encode("utf-8")}).astype(np.uint8)
        for rle in rows
    ]


def main() -> None:
    args = parse_args()
    base = pd.read_csv(args.base)
    peer = pd.read_csv(args.peer)
    for frame in (base, peer):
        frame["stem"] = frame["filament_id"].str.rsplit("_", n=1).str[0]
    image_paths = {
        path.stem: path
        for path in config.TEST_IMG_DIR.iterdir()
        if path.suffix.lower() in {".jpeg", ".jpg", ".png"}
    }
    rows = []
    retained = removed = 0
    for stem, image_path in sorted(image_paths.items()):
        height, width = read_gray_image(image_path).shape
        base_masks = masks_for(base, stem, height, width)
        peer_masks = masks_for(peer, stem, height, width)
        _pairs, matched_base, _matched_peer = greedy_pairs(base_masks, peer_masks, args.match_iou)
        kept = [
            mask for index, mask in enumerate(base_masks)
            if index in matched_base or int(mask.sum()) >= args.unmatched_min_area
        ]
        retained += len(kept)
        removed += len(base_masks) - len(kept)
        for index, mask in enumerate(kept, start=1):
            rows.append({"filament_id": f"{stem}_{index}", "segmentation_rle": encode_binary_mask(mask)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(args.output, index=False)
    print(f"retained={retained} removed={removed} output={args.output}")


if __name__ == "__main__":
    main()
