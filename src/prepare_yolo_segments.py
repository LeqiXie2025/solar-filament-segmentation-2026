import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

import config
from src.dataset import sorted_time_split
from src.gt_dedup import image_masks_from_anns, load_anns_by_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=config.PROJECT_ROOT / "data" / "yolo_segments")
    parser.add_argument("--val-fraction", type=float, default=config.VAL_FRACTION)
    parser.add_argument("--split", choices=["time", "random"], default="time")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--dedup-iou", type=float, default=0.5)
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--min-area", type=int, default=5)
    parser.add_argument("--epsilon-ratio", type=float, default=0.002)
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


def masks_to_yolo_segments(masks: list[np.ndarray], width: int, height: int, min_area: int, epsilon_ratio: float) -> list[str]:
    labels: list[str] = []
    for mask in masks:
        if int(mask.sum()) < min_area:
            continue
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        if contour.shape[0] < 3:
            continue
        epsilon = max(0.5, epsilon_ratio * cv2.arcLength(contour, True))
        approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2).astype(np.float32)
        if approx.shape[0] < 3:
            continue
        approx[:, 0] = np.clip(approx[:, 0] / max(width, 1), 0.0, 1.0)
        approx[:, 1] = np.clip(approx[:, 1] / max(height, 1), 0.0, 1.0)
        coords = " ".join(f"{value:.8f}" for value in approx.reshape(-1))
        labels.append(f"0 {coords}")
    return labels


def main() -> None:
    args = parse_args()
    train_dir = args.data_dir / "train_images"
    image_info, anns_by_file = load_anns_by_file(args.data_dir / "train_annotations.json")
    existing_files = {p.name for p in train_dir.iterdir() if p.suffix.lower() in {".jpeg", ".jpg", ".png"}}

    dedup_info = {}
    seen_files = set()
    for image_id, info in image_info.items():
        file_name = info["file_name"]
        if file_name in existing_files and file_name not in seen_files:
            dedup_info[image_id] = info
            seen_files.add(file_name)

    ids = list(dedup_info)
    if args.split == "time":
        train_ids, val_ids = sorted_time_split(dedup_info, args.val_fraction)
    else:
        rng = np.random.default_rng(args.seed)
        shuffled = ids[:]
        rng.shuffle(shuffled)
        val_count = max(1, int(round(len(shuffled) * args.val_fraction)))
        val_ids = shuffled[:val_count]
        train_ids = shuffled[val_count:]

    split_by_file = {dedup_info[i]["file_name"]: "train" for i in train_ids}
    split_by_file.update({dedup_info[i]["file_name"]: "val" for i in val_ids})

    total_instances = 0
    for image_id in ids:
        info = dedup_info[image_id]
        file_name = info["file_name"]
        split = split_by_file[file_name]
        src = train_dir / file_name
        dst_img = args.output_dir / "images" / split / file_name
        dst_label = args.output_dir / "labels" / split / f"{Path(file_name).stem}.txt"
        link_or_copy(src, dst_img, args.copy_images)
        dst_label.parent.mkdir(parents=True, exist_ok=True)

        masks = image_masks_from_anns(
            anns_by_file.get(file_name, []),
            int(info["height"]),
            int(info["width"]),
            iou_thresh=args.dedup_iou,
        )
        labels = masks_to_yolo_segments(masks, int(info["width"]), int(info["height"]), args.min_area, args.epsilon_ratio)
        total_instances += len(labels)
        dst_label.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")

    yaml_path = args.output_dir / "filament_segments.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {args.output_dir.resolve().as_posix()}",
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
    print(f"train_images={len(train_ids)} val_images={len(val_ids)} instances={total_instances}")


if __name__ == "__main__":
    main()
