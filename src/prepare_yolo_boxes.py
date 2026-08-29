import argparse
import json
import shutil
from pathlib import Path

import config
from src.dataset import sorted_time_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--annotation-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=config.PROJECT_ROOT / "data" / "yolo_boxes")
    parser.add_argument("--val-fraction", type=float, default=config.VAL_FRACTION)
    parser.add_argument("--copy-images", action="store_true")
    return parser.parse_args()


def link_or_copy(src: Path, dst: Path, copy_images: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if copy_images:
        shutil.copy2(src, dst)
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    train_dir = args.data_dir / "train_images"
    annotation_path = args.annotation_path or (args.data_dir / "train_annotations.json")
    with open(annotation_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    image_info_raw = {img["id"]: img for img in coco["images"]}
    image_info = {}
    seen_files = set()
    for image_id, info in image_info_raw.items():
        if info["file_name"] not in seen_files:
            image_info[image_id] = info
            seen_files.add(info["file_name"])
    existing_files = {p.name for p in train_dir.iterdir() if p.suffix.lower() in {".jpeg", ".jpg", ".png"}}
    image_info = {image_id: info for image_id, info in image_info.items() if info["file_name"] in existing_files}
    train_ids, val_ids = sorted_time_split(image_info, args.val_fraction)
    split_by_file = {image_info[i]["file_name"]: "train" for i in train_ids}
    split_by_file.update({image_info[i]["file_name"]: "val" for i in val_ids})

    boxes_by_file: dict[str, list[tuple[float, float, float, float]]] = {}
    for ann in coco.get("annotations", []):
        info = image_info_raw[ann["image_id"]]
        file_name = info["file_name"]
        if file_name not in split_by_file:
            continue
        x, y, w, h = [float(v) for v in ann["bbox"]]
        if w <= 1 or h <= 1:
            continue
        width, height = float(info["width"]), float(info["height"])
        cx = (x + w / 2.0) / width
        cy = (y + h / 2.0) / height
        boxes_by_file.setdefault(file_name, []).append((cx, cy, w / width, h / height))

    for file_name, split in split_by_file.items():
        src = train_dir / file_name
        dst_img = args.output_dir / "images" / split / file_name
        dst_label = args.output_dir / "labels" / split / f"{Path(file_name).stem}.txt"
        link_or_copy(src, dst_img, args.copy_images)
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        labels = boxes_by_file.get(file_name, [])
        with open(dst_label, "w", encoding="utf-8") as f:
            for cx, cy, w, h in labels:
                f.write(f"0 {cx:.8f} {cy:.8f} {w:.8f} {h:.8f}\n")

    yaml_path = args.output_dir / "filament_boxes.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {args.output_dir.as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: filament",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"saved {yaml_path}")
    print(f"train_images={len(train_ids)} val_images={len(val_ids)} boxes={sum(len(v) for v in boxes_by_file.values())}")


if __name__ == "__main__":
    main()
