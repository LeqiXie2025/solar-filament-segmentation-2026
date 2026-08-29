import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

import config
from src.crop_refiner_utils import load_instance_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--annotation-path", type=Path, default=None)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=config.DATA_DIR / "detector_proposals.json")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--bbox-iou-threshold", type=float, default=0.05)
    parser.add_argument("--max-negatives-per-image", type=int, default=2)
    return parser.parse_args()


def patch_torchvision_nms() -> None:
    import torchvision

    original_nms = torchvision.ops.nms

    def cpu_safe_nms(boxes, scores, iou_threshold):
        if boxes.is_cuda:
            keep = original_nms(boxes.detach().cpu(), scores.detach().cpu(), iou_threshold)
            return keep.to(boxes.device)
        return original_nms(boxes, scores, iou_threshold)

    torchvision.ops.nms = cpu_safe_nms


def bbox_xywh_to_xyxy(bbox: tuple[float, float, float, float]) -> np.ndarray:
    x, y, w, h = bbox
    return np.asarray([x, y, x + w, y + h], dtype=np.float32)


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    x0 = max(float(a[0]), float(b[0]))
    y0 = max(float(a[1]), float(b[1]))
    x1 = min(float(a[2]), float(b[2]))
    y1 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return float(inter / max(area_a + area_b - inter, 1e-6))


def xyxy_to_xywh(box: np.ndarray) -> tuple[float, float, float, float]:
    return (float(box[0]), float(box[1]), float(max(1.0, box[2] - box[0])), float(max(1.0, box[3] - box[1])))


def main() -> None:
    args = parse_args()
    patch_torchvision_nms()
    from ultralytics import YOLO

    train_dir = args.data_dir / "train_images"
    existing_files = {p.name for p in train_dir.iterdir() if p.suffix.lower() in {".jpeg", ".jpg", ".png"}}
    annotation_path = args.annotation_path or (args.data_dir / "train_annotations.json")
    _, records = load_instance_records(annotation_path, existing_files)
    records_by_file: dict[str, list[dict]] = {}
    for record in records:
        records_by_file.setdefault(record["file_name"], []).append(record)

    detector = YOLO(str(args.detector))
    proposals = []
    positive = 0
    negative = 0
    files = sorted(train_dir / name for name in records_by_file)
    for path in tqdm(files, desc="detector-proposals"):
        det = detector.predict(
            source=str(path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=0,
            verbose=False,
        )[0]
        boxes = det.boxes.xyxy.detach().cpu().numpy() if det.boxes is not None else np.empty((0, 4), dtype=np.float32)
        scores = det.boxes.conf.detach().cpu().numpy() if det.boxes is not None else np.empty((0,), dtype=np.float32)
        gt_records = records_by_file[path.name]
        gt_boxes = [bbox_xywh_to_xyxy(record["bbox"]) for record in gt_records]
        negs_for_image = 0
        for box, score in zip(boxes, scores):
            if gt_boxes:
                ious = [bbox_iou(box, gt_box) for gt_box in gt_boxes]
                best_idx = int(np.argmax(ious))
                best_iou = float(ious[best_idx])
            else:
                best_idx = -1
                best_iou = 0.0

            if best_iou >= args.bbox_iou_threshold:
                base = dict(gt_records[best_idx])
                base["proposal_bbox"] = xyxy_to_xywh(box)
                base["proposal_score"] = float(score)
                base["proposal_iou"] = best_iou
                base["proposal_label"] = 1
                proposals.append(base)
                positive += 1
            elif negs_for_image < args.max_negatives_per_image:
                info = gt_records[0]
                proposals.append(
                    {
                        "image_id": info["image_id"],
                        "file_name": path.name,
                        "height": int(info["height"]),
                        "width": int(info["width"]),
                        "bbox": xyxy_to_xywh(box),
                        "proposal_bbox": xyxy_to_xywh(box),
                        "proposal_score": float(score),
                        "proposal_iou": best_iou,
                        "proposal_label": 0,
                        "segmentation": None,
                        "area": 0.0,
                    }
                )
                negative += 1
                negs_for_image += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"proposals": proposals, "positive": positive, "negative": negative}, f)
    print(f"saved {args.output} positive={positive} negative={negative} total={len(proposals)}")


if __name__ == "__main__":
    main()
