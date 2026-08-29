import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2

import config
from src.dataset import sorted_time_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--annotation-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=config.PROJECT_ROOT / "data" / "yolo_tiled_boxes")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--min-visible", type=float, default=0.50)
    parser.add_argument("--negative-fraction", type=float, default=0.20)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--val-fraction", type=float, default=config.VAL_FRACTION)
    return parser.parse_args()


def tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    stride = max(1, tile_size - overlap)
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    last = max(0, length - tile_size)
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


def keep_negative(file_name: str, x0: int, y0: int, fraction: float) -> bool:
    digest = hashlib.sha1(f"{file_name}:{x0}:{y0}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:4], "little") / float(2**32)
    return value < fraction


def clipped_yolo_box(bbox, x0: int, y0: int, tile_size: int, min_visible: float):
    bx, by, bw, bh = [float(value) for value in bbox]
    if bw <= 0 or bh <= 0:
        return None
    cx, cy = bx + 0.5 * bw, by + 0.5 * bh
    if not (x0 <= cx < x0 + tile_size and y0 <= cy < y0 + tile_size):
        return None
    x1 = max(bx, x0)
    y1 = max(by, y0)
    x2 = min(bx + bw, x0 + tile_size)
    y2 = min(by + bh, y0 + tile_size)
    clipped_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if clipped_area / (bw * bh) < min_visible:
        return None
    cx_tile = ((x1 + x2) * 0.5 - x0) / tile_size
    cy_tile = ((y1 + y2) * 0.5 - y0) / tile_size
    w_tile = (x2 - x1) / tile_size
    h_tile = (y2 - y1) / tile_size
    return cx_tile, cy_tile, w_tile, h_tile


def main() -> None:
    args = parse_args()
    coco = json.loads(args.annotation_path.read_text(encoding="utf-8"))
    images = {int(info["id"]): info for info in coco["images"]}
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[int(ann["image_id"])].append(ann)

    train_ids, val_ids = sorted_time_split(images, args.val_fraction)
    split_by_name = {str(images[image_id]["file_name"]): "train" for image_id in train_ids}
    split_by_name.update({str(images[image_id]["file_name"]): "val" for image_id in val_ids})

    for split in ("train", "val"):
        (args.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    tile_count = box_count = negative_count = 0
    image_dir = args.data_dir / "train_images"
    for image_id, info in images.items():
        file_name = str(info["file_name"])
        split = split_by_name[file_name]
        image = cv2.imread(str(image_dir / file_name), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(image_dir / file_name)
        height, width = image.shape
        tile_size = int(args.tile_size)
        for y0 in tile_starts(height, tile_size, int(args.overlap)):
            for x0 in tile_starts(width, tile_size, int(args.overlap)):
                labels = []
                for ann in anns_by_image.get(image_id, []):
                    box = clipped_yolo_box(ann["bbox"], x0, y0, tile_size, float(args.min_visible))
                    if box is not None:
                        labels.append("0 " + " ".join(f"{value:.8f}" for value in box))
                if not labels and not keep_negative(file_name, x0, y0, float(args.negative_fraction)):
                    continue
                stem = f"{Path(file_name).stem}_x{x0}_y{y0}"
                tile = image[y0 : y0 + tile_size, x0 : x0 + tile_size]
                cv2.imwrite(
                    str(args.output_dir / "images" / split / f"{stem}.jpg"),
                    tile,
                    [cv2.IMWRITE_JPEG_QUALITY, int(args.jpeg_quality)],
                )
                (args.output_dir / "labels" / split / f"{stem}.txt").write_text(
                    "\n".join(labels) + ("\n" if labels else ""), encoding="utf-8"
                )
                tile_count += 1
                box_count += len(labels)
                negative_count += int(not labels)

    yaml_path = args.output_dir / "filament_tiled_boxes.yaml"
    yaml_path.write_text(
        f"path: {args.output_dir.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n  0: filament\n",
        encoding="utf-8",
    )
    print(f"tiles={tile_count} boxes={box_count} negatives={negative_count}")
    print(yaml_path)


if __name__ == "__main__":
    main()
