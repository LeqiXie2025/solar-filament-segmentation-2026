"""Build a canonical, auditable instance-label file from multi-annotator COCO data."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import config
from src.gt_dedup import _decode_polygon_scaled, _decode_segmentation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=config.ANNOTATION_PATH)
    parser.add_argument("--output", type=Path, default=config.PROJECT_ROOT / "data" / "clean" / "train_annotations_medoid.json")
    parser.add_argument("--report", type=Path, default=config.PROJECT_ROOT / "data" / "label_audit.json")
    parser.add_argument("--dedup-iou", type=float, default=0.5)
    parser.add_argument("--min-area", type=int, default=5)
    parser.add_argument("--cluster-size", type=int, default=1024)
    return parser.parse_args()


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    if inter == 0:
        return 0.0
    return inter / max(1, int(a.sum()) + int(b.sum()) - inter)


def cluster_masks(items: list[tuple[dict, np.ndarray]], threshold: float) -> list[list[int]]:
    order = sorted(range(len(items)), key=lambda i: -int(items[i][1].sum()))
    clusters: list[list[int]] = []
    for idx in order:
        best_cluster = -1
        best_iou = threshold
        for cluster_idx, members in enumerate(clusters):
            # Complete-link clustering prevents a chain of partially overlapping
            # neighbouring filaments from being collapsed into one instance.
            overlap = min(mask_iou(items[idx][1], items[j][1]) for j in members)
            if overlap >= best_iou:
                best_iou = overlap
                best_cluster = cluster_idx
        if best_cluster < 0:
            clusters.append([idx])
        else:
            clusters[best_cluster].append(idx)
    return clusters


def medoid_index(items: list[tuple[dict, np.ndarray]], members: list[int]) -> int:
    if len(members) == 1:
        return members[0]
    scores = []
    for i in members:
        scores.append((sum(mask_iou(items[i][1], items[j][1]) for j in members if j != i), i))
    return max(scores)[1]


def main() -> None:
    args = parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    image_by_id = {image["id"]: image for image in data["images"]}
    images_by_file: dict[str, list[dict]] = defaultdict(list)
    anns_by_file: dict[str, list[dict]] = defaultdict(list)
    for image in data["images"]:
        images_by_file[image["file_name"]].append(image)
    for ann in data.get("annotations", []):
        info = image_by_id[ann["image_id"]]
        anns_by_file[info["file_name"]].append(ann)

    clean_images = []
    clean_annotations = []
    invalid = 0
    duplicate_clusters = 0
    cluster_sizes: list[int] = []
    areas: list[int] = []
    ann_id = 1

    for file_name in sorted(images_by_file):
        source_images = images_by_file[file_name]
        canonical = dict(source_images[0])
        canonical["id"] = len(clean_images) + 1
        clean_images.append(canonical)
        height, width = int(canonical["height"]), int(canonical["width"])
        decoded: list[tuple[dict, np.ndarray]] = []
        for ann in anns_by_file.get(file_name, []):
            segmentation = ann.get("segmentation")
            if isinstance(segmentation, list):
                mask = _decode_polygon_scaled(
                    segmentation,
                    args.cluster_size,
                    args.cluster_size,
                    height,
                    width,
                )
            else:
                full_mask = _decode_segmentation(segmentation, height, width)
                mask = cv2.resize(full_mask, (args.cluster_size, args.cluster_size), interpolation=cv2.INTER_NEAREST)
            if float(ann.get("area", 0)) < args.min_area or int(mask.sum()) == 0:
                invalid += 1
                continue
            decoded.append((ann, mask.astype(np.uint8)))

        for members in cluster_masks(decoded, args.dedup_iou):
            selected = medoid_index(decoded, members)
            source_ann, _ = decoded[selected]
            area = int(round(float(source_ann.get("area", 0))))
            annotation = dict(source_ann)
            annotation.update(
                {
                    "id": ann_id,
                    "image_id": canonical["id"],
                    "area": area,
                    "bbox": [float(value) for value in source_ann["bbox"]],
                    "iscrowd": 0,
                }
            )
            clean_annotations.append(annotation)
            ann_id += 1
            areas.append(area)
            cluster_sizes.append(len(members))
            duplicate_clusters += int(len(members) > 1)

    output = {key: value for key, value in data.items() if key not in {"images", "annotations"}}
    output["images"] = clean_images
    output["annotations"] = clean_annotations
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")

    area_array = np.asarray(areas, dtype=np.float64)
    report = {
        "source": str(args.input),
        "clean_output": str(args.output),
        "raw_image_records": len(data["images"]),
        "unique_images": len(clean_images),
        "raw_annotations": len(data.get("annotations", [])),
        "clean_instances": len(clean_annotations),
        "invalid_removed": invalid,
        "duplicate_clusters": duplicate_clusters,
        "multi_annotator_files": sum(len(rows) > 1 for rows in images_by_file.values()),
        "dedup_iou": args.dedup_iou,
        "selection": "IoU-cluster medoid annotation",
        "area": {
            "min": int(area_array.min()) if len(area_array) else 0,
            "p10": float(np.percentile(area_array, 10)) if len(area_array) else 0.0,
            "median": float(np.median(area_array)) if len(area_array) else 0.0,
            "p90": float(np.percentile(area_array, 90)) if len(area_array) else 0.0,
            "max": int(area_array.max()) if len(area_array) else 0,
            "below_500": int((area_array < 500).sum()),
        },
        "cluster_size_histogram": {str(size): cluster_sizes.count(size) for size in sorted(set(cluster_sizes))},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
