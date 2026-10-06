from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as coco_mask

import config
from src.dataset import read_gray_image
from src.masks import encode_binary_mask


def greedy_pairs(a: list[np.ndarray], b: list[np.ndarray], threshold: float):
    """Same float32 IoU/tie ordering as the dense matcher, using binary RLEs."""
    if not a or not b:
        return [], set(), set()
    encoded_a = [coco_mask.encode(np.asfortranarray(mask.astype(np.uint8))) for mask in a]
    encoded_b = [coco_mask.encode(np.asfortranarray(mask.astype(np.uint8))) for mask in b]
    matrix = coco_mask.iou(encoded_a, encoded_b, [0] * len(b)).astype(np.float32)
    candidates = sorted(
        [(float(matrix[i, j]), i, j) for i in range(len(a)) for j in range(len(b))
         if matrix[i, j] >= threshold], reverse=True)
    used_a, used_b, pairs = set(), set(), []
    for iou, i, j in candidates:
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            pairs.append((i, j, iou))
    return pairs, used_a, used_b


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate the established submission with OBB confirmations and add conservative OBB-only instances."
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--confirm", type=Path, required=True)
    parser.add_argument("--extras", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, default=config.TEST_IMG_DIR)
    parser.add_argument("--confirm-iou", type=float, default=0.10)
    parser.add_argument("--unmatched-min-area", type=int, default=3000)
    parser.add_argument("--extra-match-iou", type=float, default=0.10)
    parser.add_argument("--extra-min-area", type=int, default=500)
    return parser.parse_args()


def image_id_from_filament_id(value: str) -> str:
    image_id, separator, instance_index = str(value).rpartition("_")
    if not separator or not instance_index.isdigit():
        raise ValueError(f"invalid filament_id: {value}")
    return image_id


def decode_submission_rows(frame: pd.DataFrame, image_id: str, shape: tuple[int, int]) -> list[np.ndarray]:
    height, width = shape
    row_image_ids = frame["filament_id"].astype(str).map(image_id_from_filament_id)
    rows = frame.loc[row_image_ids == image_id, "segmentation_rle"]
    return [
        coco_mask.decode({"size": [height, width], "counts": str(value).encode("utf-8")}).astype(np.uint8)
        for value in rows
    ]


def assemble_image(
    base: list[np.ndarray],
    confirm: list[np.ndarray],
    extras: list[np.ndarray],
    args: argparse.Namespace,
) -> tuple[list[np.ndarray], int, int]:
    _pairs, confirmed_base, _confirmed_obbs = greedy_pairs(base, confirm, args.confirm_iou)
    kept = [
        mask.copy()
        for index, mask in enumerate(base)
        if index in confirmed_base or int(mask.sum()) >= args.unmatched_min_area
    ]

    # Match extras against the original base set so an OBB refinement cannot
    # re-enter merely because its corresponding base mask was rejected.
    _pairs, _matched_base, matched_extras = greedy_pairs(base, extras, args.extra_match_iou)
    reference = base[0] if base else (extras[0] if extras else None)
    if reference is None:
        return [], 0, 0
    claimed = np.zeros_like(reference, dtype=bool)
    outputs = []
    for mask in sorted(kept, key=lambda item: int(item.sum()), reverse=True):
        clean = mask.astype(bool) & ~claimed
        if int(clean.sum()) == 0:
            continue
        outputs.append(clean.astype(np.uint8))
        claimed |= clean

    extras_added = 0
    for index in sorted(range(len(extras)), key=lambda item: -int(extras[item].sum())):
        if index in matched_extras:
            continue
        clean = extras[index].astype(bool) & ~claimed
        if int(clean.sum()) < args.extra_min_area:
            continue
        outputs.append(clean.astype(np.uint8))
        claimed |= clean
        extras_added += 1
    return outputs, len(kept), extras_added


def main() -> None:
    args = parse_args()
    frames = [pd.read_csv(path) for path in (args.base, args.confirm, args.extras)]
    expected_columns = ["filament_id", "segmentation_rle"]
    for path, frame in zip((args.base, args.confirm, args.extras), frames):
        if list(frame.columns) != expected_columns:
            raise ValueError(f"{path} must contain exactly {expected_columns}")
        if frame.isna().any().any() or frame["filament_id"].duplicated().any():
            raise ValueError(f"invalid rows in {path}")

    base_frame, confirm_frame, extras_frame = frames
    image_paths = sorted(
        path for path in args.test_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    rows = []
    base_kept = extras_added = 0
    for path in image_paths:
        shape = read_gray_image(path).shape
        base = decode_submission_rows(base_frame, path.stem, shape)
        confirm = decode_submission_rows(confirm_frame, path.stem, shape)
        extras = decode_submission_rows(extras_frame, path.stem, shape)
        masks, kept_count, added_count = assemble_image(base, confirm, extras, args)
        base_kept += kept_count
        extras_added += added_count
        for index, mask in enumerate(masks, start=1):
            rows.append(
                {
                    "filament_id": f"{path.stem}_{index}",
                    "segmentation_rle": encode_binary_mask(mask),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=expected_columns).to_csv(args.output, index=False)
    print(
        f"saved {args.output} rows={len(rows)} images={len(image_paths)} "
        f"base_kept={base_kept} extras_added={extras_added}"
    )


if __name__ == "__main__":
    main()
