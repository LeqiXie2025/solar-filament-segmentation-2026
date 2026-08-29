"""YOLO proposal detection followed by MobileSAM box-prompt instance masks."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import config
from src.dataset import read_gray_image
from src.masks import encode_binary_mask
from src.metrics import greedy_match, mean_dice_matched, panoptic_quality
from src.predict_detect_refine import make_masks_disjoint, mask_nms, patch_torchvision_nms, time_val_names
from src.validation import load_gt_instances


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--annotation-path", type=Path, default=None)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--mobile-sam-root", type=Path, default=Path(r"D:\codex_models\MobileSAM"))
    parser.add_argument("--mobile-sam-checkpoint", type=Path, default=Path(r"D:\codex_models\MobileSAM\weights\mobile_sam.pt"))
    parser.add_argument("--mode", choices=["val", "test"], default="val")
    parser.add_argument("--output", type=Path, default=config.SUBMISSION_DIR / "submission_detect_mobilesam.csv")
    parser.add_argument("--conf", type=float, default=0.18)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--min-area", type=int, default=120)
    parser.add_argument("--mask-nms-iou", type=float, default=0.12)
    parser.add_argument("--box-pad", type=float, default=0.05)
    parser.add_argument("--sam-iou-threshold", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def padded_boxes(boxes: np.ndarray, shape: tuple[int, int], pad: float) -> np.ndarray:
    height, width = shape
    out = boxes.astype(np.float32).copy()
    box_w = out[:, 2] - out[:, 0]
    box_h = out[:, 3] - out[:, 1]
    out[:, 0] = np.maximum(0.0, out[:, 0] - box_w * pad)
    out[:, 1] = np.maximum(0.0, out[:, 1] - box_h * pad)
    out[:, 2] = np.minimum(float(width), out[:, 2] + box_w * pad)
    out[:, 3] = np.minimum(float(height), out[:, 3] + box_h * pad)
    return out


def main() -> None:
    args = parse_args()
    patch_torchvision_nms()
    from ultralytics import YOLO

    sys.path.insert(0, str(args.mobile_sam_root.resolve()))
    from mobile_sam import SamPredictor, sam_model_registry

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = YOLO(str(args.detector))
    sam = sam_model_registry["vit_t"](checkpoint=str(args.mobile_sam_checkpoint)).to(device)
    sam.eval()
    predictor = SamPredictor(sam)

    image_dir = args.data_dir / ("test_images" if args.mode == "test" else "train_images")
    files = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".jpeg", ".jpg", ".png"})
    if args.mode == "val":
        annotation_path = args.annotation_path or (args.data_dir / "train_annotations.json")
        names = set(time_val_names(annotation_path, {path.name for path in files}))
        files = [path for path in files if path.name in names]
    if args.limit:
        files = files[: args.limit]

    gt_by_file = {}
    if args.mode == "val":
        gt_by_file = load_gt_instances(args.annotation_path or (args.data_dir / "train_annotations.json"), {p.name for p in files})
    rows = []
    pq_scores, dice_scores = [], []
    total_tp = total_fp = total_fn = 0

    for path in tqdm(files, desc=f"detect-mobilesam-{args.mode}"):
        gray = read_gray_image(path)
        rgb = cv2.cvtColor(np.clip(gray * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        result = detector.predict(
            source=str(path), imgsz=args.imgsz, conf=args.conf, iou=args.iou,
            device=0 if device.type == "cuda" else "cpu", verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            instances = []
        else:
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            detector_scores = result.boxes.conf.detach().cpu().numpy()
            boxes = padded_boxes(boxes, gray.shape, args.box_pad)
            predictor.set_image(rgb)
            transformed = predictor.transform.apply_boxes_torch(torch.as_tensor(boxes, device=device), rgb.shape[:2])
            masks, sam_scores, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed,
                multimask_output=False,
            )
            masks = masks[:, 0].detach().cpu().numpy().astype(np.uint8)
            sam_scores = sam_scores[:, 0].detach().cpu().numpy()
            scored = []
            for mask, det_score, sam_score in zip(masks, detector_scores, sam_scores):
                if int(mask.sum()) < args.min_area or float(sam_score) < args.sam_iou_threshold:
                    continue
                scored.append((mask, float(det_score) * max(0.0, float(sam_score))))
            instances = make_masks_disjoint(mask_nms(scored, args.mask_nms_iou), args.min_area)

        if args.mode == "val":
            gts = gt_by_file.get(path.name, [])
            matched, fp, fn = greedy_match(instances, gts, overlap_threshold=0.0, mode="dice")
            pq_scores.append(panoptic_quality(matched, len(instances), len(gts), fp, fn))
            dice_scores.append(mean_dice_matched(matched))
            total_tp += len(matched)
            total_fp += fp
            total_fn += fn
        else:
            image_id = path.stem
            for index, instance in enumerate(sorted(instances, key=lambda x: int(x.sum()), reverse=True), start=1):
                rows.append({"filament_id": f"{image_id}_{index}", "segmentation_rle": encode_binary_mask(instance)})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "val":
        summary = pd.DataFrame([{
            "conf": args.conf,
            "imgsz": args.imgsz,
            "min_area": args.min_area,
            "box_pad": args.box_pad,
            "pq": float(np.mean(pq_scores)) if pq_scores else 0.0,
            "mean_dice_matched": float(np.mean(dice_scores)) if dice_scores else 0.0,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "images": len(files),
        }])
        summary.to_csv(args.output, index=False)
        print(summary.to_string(index=False))
    else:
        pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(args.output, index=False)
        print(f"saved {args.output} rows={len(rows)}")


if __name__ == "__main__":
    main()
