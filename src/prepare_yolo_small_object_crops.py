"""Build object-centred YOLO crops that oversample small filament instances."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2

import config
from src.dataset import read_gray_image, sorted_time_split
from src.prepare_yolo_tiled_boxes import clipped_yolo_box


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--annotation-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(r"D:\solar_filament_small_crops"))
    parser.add_argument("--crop-size", type=int, default=768)
    parser.add_argument("--max-instance-area", type=float, default=1000.0)
    parser.add_argument("--min-visible", type=float, default=0.50)
    parser.add_argument("--jitter", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=config.VAL_FRACTION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    coco = json.loads(args.annotation_path.read_text(encoding="utf-8"))
    images = {int(info["id"]): info for info in coco["images"]}
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[int(ann["image_id"])].append(ann)
    train_ids, val_ids = sorted_time_split(images, args.val_fraction)
    split_by_id = {image_id: "train" for image_id in train_ids}
    split_by_id.update({image_id: "val" for image_id in val_ids})

    for split in ("train", "val"):
        (args.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    crop_count = box_count = 0
    image_dir = args.data_dir / "train_images"
    size = int(args.crop_size)
    for image_id, info in images.items():
        annotations = anns_by_image.get(image_id, [])
        small = [ann for ann in annotations if 0 < float(ann.get("area", 0)) <= args.max_instance_area]
        if not small:
            continue
        image = (read_gray_image(image_dir / info["file_name"]) * 255.0).clip(0, 255).astype("uint8")
        height, width = image.shape
        split = split_by_id[image_id]
        for index, target in enumerate(small):
            x, y, w, h = [float(value) for value in target["bbox"]]
            cx, cy = x + w * 0.5, y + h * 0.5
            cx += rng.uniform(-args.jitter, args.jitter) * size
            cy += rng.uniform(-args.jitter, args.jitter) * size
            x0 = min(max(0, int(round(cx - size * 0.5))), max(0, width - size))
            y0 = min(max(0, int(round(cy - size * 0.5))), max(0, height - size))
            labels = []
            for ann in annotations:
                box = clipped_yolo_box(ann["bbox"], x0, y0, size, args.min_visible)
                if box is not None:
                    labels.append("0 " + " ".join(f"{value:.8f}" for value in box))
            if not labels:
                continue
            stem = f"{Path(info['file_name']).stem}_small{index:03d}_x{x0}_y{y0}"
            crop = image[y0 : y0 + size, x0 : x0 + size]
            cv2.imwrite(str(args.output_dir / "images" / split / f"{stem}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 94])
            (args.output_dir / "labels" / split / f"{stem}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
            crop_count += 1
            box_count += len(labels)

    yaml_path = args.output_dir / "filament_small_crops.yaml"
    yaml_path.write_text(
        f"path: {args.output_dir.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\n"
        "names:\n  0: filament\n",
        encoding="utf-8",
    )
    print(f"crops={crop_count} boxes={box_count}")
    print(yaml_path)


if __name__ == "__main__":
    main()
