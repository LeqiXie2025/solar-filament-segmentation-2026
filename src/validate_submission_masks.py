import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.check_submissions import decode_coco_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("--image-dir", type=Path, default=Path("data/raw/test_images"))
    return parser.parse_args()


def decode_mask(rle: str, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    flat = np.zeros(height * width, dtype=np.uint8)
    value = 0
    pos = 0
    for count in decode_coco_counts(rle):
        if value and count:
            flat[pos : pos + count] = 1
        pos += count
        value = 1 - value
    return flat


def read_shape(path: Path) -> tuple[int, int]:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image.shape


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.submission)
    if df["filament_id"].duplicated().any():
        raise SystemExit("duplicate filament_id")
    if df.isna().any().any():
        raise SystemExit("missing values")

    by_image: dict[str, list[str]] = defaultdict(list)
    for _, row in df.iterrows():
        image_id = str(row["filament_id"]).rsplit("_", 1)[0]
        by_image[image_id].append(str(row["segmentation_rle"]))

    violations = []
    for image_id, rles in by_image.items():
        shape = read_shape(args.image_dir / f"{image_id}.jpeg")
        occupied = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        for idx, rle in enumerate(rles, start=1):
            mask = decode_mask(rle, shape)
            overlap = int(((occupied > 0) & (mask > 0)).sum())
            if overlap:
                violations.append((image_id, idx, overlap))
                break
            occupied |= mask

    if violations:
        raise SystemExit(f"overlapping masks: {violations[:5]}")
    print(f"OK rows={len(df)} images={len(by_image)} no_overlaps=True")


if __name__ == "__main__":
    main()
