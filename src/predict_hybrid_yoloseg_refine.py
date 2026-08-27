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
from src.model_factory import create_model_from_checkpoint
from src.predict_detect_refine import make_masks_disjoint, mask_nms, patch_torchvision_nms, refine_boxes, time_val_names
from src.validation import load_gt_instances


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--box-detector", type=Path, required=True)
    parser.add_argument("--refiner", type=Path, required=True)
    parser.add_argument("--seg-model", type=Path, required=True)
    parser.add_argument("--mode", choices=["val", "test"], default="val")
    parser.add_argument("--output", type=Path, default=config.SUBMISSION_DIR / "submission_hybrid_yoloseg_refine.csv")
    parser.add_argument("--box-conf", type=float, default=0.18)
    parser.add_argument("--box-iou", type=float, default=0.45)
    parser.add_argument("--box-imgsz", type=int, nargs="+", default=[1024, 1280])
    parser.add_argument("--box-pad", type=float, default=0.50)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--seg-conf", type=float, default=0.20)
    parser.add_argument("--seg-iou", type=float, default=0.0)
    parser.add_argument("--seg-imgsz", type=int, default=1024)
    parser.add_argument("--seg-threshold", type=float, default=0.50)
    parser.add_argument("--seg-score-weight", type=float, default=0.65)
    parser.add_argument("--min-area", type=int, default=160)
    parser.add_argument("--mask-nms-iou", type=float, default=0.12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def yolo_seg_instances(result, height: int, width: int, threshold: float, score_weight: float) -> list[tuple[np.ndarray, float]]:
    if result.masks is None or result.boxes is None:
        return []
    masks = result.masks.data.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
    out = []
    for mask, score in zip(masks, scores):
        if mask.shape != (height, width):
            mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
        binary = (mask >= threshold).astype(np.uint8)
        out.append((binary, float(score) * score_weight))
    return out


def main() -> None:
    args = parse_args()
    patch_torchvision_nms()
    from ultralytics import YOLO

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo_device = "0" if device.type == "cuda" else "cpu"
    box_detector = YOLO(str(args.box_detector))
    seg_model = YOLO(str(args.seg_model))

    checkpoint = torch.load(args.refiner, map_location=device)
    refiner = create_model_from_checkpoint(checkpoint).to(device)
    refiner.load_state_dict(checkpoint["model"])
    refiner.eval()
    image_size = int(checkpoint.get("image_size", 512))

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

    for path in tqdm(files, desc=f"hybrid-yoloseg-refine-{args.mode}"):
        image = read_gray_image(path)
        height, width = image.shape
        all_boxes = []
        all_scores = []
        for size in args.box_imgsz:
            det = box_detector.predict(
                source=str(path),
                imgsz=int(size),
                conf=args.box_conf,
                iou=args.box_iou,
                device=yolo_device,
                verbose=False,
            )[0]
            if det.boxes is not None:
                all_boxes.append(det.boxes.xyxy.detach().cpu().numpy())
                all_scores.append(det.boxes.conf.detach().cpu().numpy())
        boxes = np.concatenate(all_boxes, axis=0) if all_boxes else np.empty((0, 4), dtype=np.float32)
        scores = np.concatenate(all_scores, axis=0) if all_scores else np.empty((0,), dtype=np.float32)
        scored = refine_boxes(refiner, image, boxes, scores, device, args, image_size) if len(boxes) else []

        seg_result = seg_model.predict(
            source=str(path),
            imgsz=args.seg_imgsz,
            conf=args.seg_conf,
            iou=args.seg_iou,
            max_det=120,
            device=yolo_device,
            verbose=False,
        )[0]
        scored.extend(
            (m, s)
            for m, s in yolo_seg_instances(seg_result, height, width, args.seg_threshold, args.seg_score_weight)
            if int(m.sum()) >= args.min_area
        )
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
                    "box_conf": args.box_conf,
                    "seg_conf": args.seg_conf,
                    "seg_score_weight": args.seg_score_weight,
                    "threshold": args.threshold,
                    "min_area": args.min_area,
                    "mask_nms_iou": args.mask_nms_iou,
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
