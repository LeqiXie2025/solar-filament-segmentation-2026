import argparse
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
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mode", choices=["val", "test"], default="val")
    parser.add_argument("--output", type=Path, default=config.SUBMISSION_DIR / "submission_yolo_seg.csv")
    parser.add_argument("--imgsz", type=int, default=2048)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.0)
    parser.add_argument("--max-det", type=int, default=120)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-area", type=int, default=5)
    parser.add_argument("--mask-nms-iou", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def result_masks(result, height: int, width: int, threshold: float) -> list[tuple[np.ndarray, float]]:
    if result.masks is None or result.boxes is None:
        return []
    masks = result.masks.data.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
    out = []
    for mask, score in zip(masks, scores):
        if mask.shape != (height, width):
            mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
        binary = (mask >= threshold).astype(np.uint8)
        out.append((binary, float(score)))
    return out


def main() -> None:
    args = parse_args()
    patch_torchvision_nms()
    from ultralytics import YOLO

    device = "0" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(args.model))
    image_dir = args.data_dir / ("test_images" if args.mode == "test" else "train_images")
    files = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpeg", ".jpg", ".png"}])
    if args.mode == "val":
        names = set(time_val_names(args.data_dir / "train_annotations.json", {p.name for p in files}))
        files = [p for p in files if p.name in names]
    if args.limit:
        files = files[: args.limit]

    gt_by_file = load_gt_instances(args.data_dir / "train_annotations.json", {p.name for p in files}) if args.mode == "val" else {}
    rows = []
    pq_scores = []
    mean_dice_scores = []
    total_tp = total_fp = total_fn = 0

    for path in tqdm(files, desc=f"yolo-seg-{args.mode}"):
        image = read_gray_image(path)
        height, width = image.shape
        result = model.predict(
            source=str(path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=device,
            verbose=False,
        )[0]
        scored = [(m, s) for m, s in result_masks(result, height, width, args.threshold) if int(m.sum()) >= args.min_area]
        pred_instances = mask_nms(scored, args.mask_nms_iou)
        pred_instances = make_masks_disjoint(pred_instances, args.min_area)

        if args.mode == "val":
            gts = gt_by_file.get(path.name, [])
            matched, fp, fn = greedy_match(pred_instances, gts, overlap_threshold=0.0, mode="dice")
            pq_scores.append(panoptic_quality(matched, len(pred_instances), len(gts), fp, fn))
            mean_dice_scores.append(mean_dice_matched(matched))
            total_tp += len(matched)
            total_fp += fp
            total_fn += fn
        else:
            image_id = path.stem
            pred_instances = sorted(pred_instances, key=lambda instance: int(instance.sum()), reverse=True)
            pred_instances = make_masks_disjoint(pred_instances, args.min_area)
            for idx, instance in enumerate(pred_instances, start=1):
                rows.append({"filament_id": f"{image_id}_{idx}", "segmentation_rle": encode_binary_mask(instance)})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "val":
        out = pd.DataFrame(
            [
                {
                    "conf": args.conf,
                    "iou": args.iou,
                    "threshold": args.threshold,
                    "min_area": args.min_area,
                    "mask_nms_iou": args.mask_nms_iou,
                    "imgsz": args.imgsz,
                    "pq": float(np.mean(pq_scores)) if pq_scores else 0.0,
                    "mean_dice_matched": float(np.mean(mean_dice_scores)) if mean_dice_scores else 0.0,
                    "tp": total_tp,
                    "fp": total_fp,
                    "fn": total_fn,
                    "images": len(files),
                }
            ]
        )
        out.to_csv(args.output, index=False)
        print(out.to_string(index=False))
    else:
        pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(args.output, index=False)
        print(f"saved {args.output} rows={len(rows)}")


if __name__ == "__main__":
    main()
