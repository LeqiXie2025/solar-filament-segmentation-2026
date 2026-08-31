"""Decisive metric-calibration run.

Local PQ (Dice, overlap_threshold=0.0) reports 0.498 for detect-refine, but the
public LB came back 0.33. This script re-runs detect-refine inference on the
time-split val set ONCE and computes every candidate metric in parallel so we can
see which one actually maps to the LB:

  (a) PQ, Dice score, threshold 0.0      <- current local metric (0.498)
  (b) PQ, Dice score, threshold 0.5
  (c) PQ, IoU  score, threshold 0.5      <- standard COCO panoptic PQ
  (d) grouped (image-level) Dice         <- union all pred vs union all gt, per image
  (e) buggy mean_dice_matched (empty->1)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

import config
from src.crop_refiner_utils import crop_bounds_from_bbox, crop_with_padding, make_refiner_channels, seed_from_bbox
from src.dataset import read_gray_image
from src.masks import postprocess_binary_mask
from src.metrics import full_image_dice, greedy_match, mean_dice_matched, panoptic_quality
from src.model_factory import create_model_from_checkpoint
from src.predict_detect_refine import make_masks_disjoint, mask_nms, patch_torchvision_nms, refine_boxes, time_val_names
from src.validation import load_gt_instances


def union_mask(instances: list[np.ndarray]) -> np.ndarray:
    if not instances:
        return np.zeros_like(instances[0]) if False else None
    u = np.zeros_like(instances[0], dtype=np.uint8)
    for inst in instances:
        u = np.maximum(u, inst)
    return u


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    ap.add_argument("--detector", type=Path, required=True)
    ap.add_argument("--refiner", type=Path, required=True)
    ap.add_argument("--conf", type=float, default=0.14)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--min-area", type=int, default=140)
    ap.add_argument("--mask-nms-iou", type=float, default=0.20)
    ap.add_argument("--box-pad", type=float, default=0.50)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    patch_torchvision_nms()
    from ultralytics import YOLO

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = YOLO(str(args.detector))
    ckpt = torch.load(args.refiner, map_location=device)
    refiner = create_model_from_checkpoint(ckpt).to(device)
    refiner.load_state_dict(ckpt["model"])
    refiner.eval()
    image_size = int(ckpt.get("image_size", 512))

    image_dir = args.data_dir / "train_images"
    files = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpeg", ".jpg", ".png"}])
    names = set(time_val_names(args.data_dir / "train_annotations.json", {p.name for p in files}))
    files = [p for p in files if p.name in names]
    if args.limit:
        files = files[: args.limit]
    gt_by_file = load_gt_instances(args.data_dir / "train_annotations.json", {p.name for p in files})

    agg = {
        "pq_dice0": [], "pq_dice05": [], "pq_iou05": [],
        "grouped_dice": [], "mean_dice_matched": [],
    }
    tp_dice0 = fp_dice0 = fn_dice0 = 0
    tp_iou05 = fp_iou05 = fn_iou05 = 0
    global_pred_sum = global_gt_sum = global_inter_sum = 0

    for path in files:
        image = read_gray_image(path)
        det = detector.predict(source=str(path), imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                               device=0 if device.type == "cuda" else "cpu", verbose=False)[0]
        boxes = det.boxes.xyxy.detach().cpu().numpy() if det.boxes is not None else np.empty((0, 4), dtype=np.float32)
        scores = det.boxes.conf.detach().cpu().numpy() if det.boxes is not None else np.empty((0,), dtype=np.float32)
        pred_scored = refine_boxes(refiner, image, boxes, scores, device, args, image_size) if len(boxes) else []
        preds = mask_nms(pred_scored, args.mask_nms_iou)
        preds = make_masks_disjoint(preds, args.min_area)
        gts = gt_by_file.get(path.name, [])

        # candidate instance metrics
        m0, fp0, fn0 = greedy_match(preds, gts, overlap_threshold=0.0, mode="dice")
        agg["pq_dice0"].append(panoptic_quality(m0, len(preds), len(gts), fp0, fn0))
        agg["mean_dice_matched"].append(mean_dice_matched(m0))
        tp_dice0 += len(m0); fp_dice0 += fp0; fn_dice0 += fn0

        m05, fp05, fn05 = greedy_match(preds, gts, overlap_threshold=0.5, mode="dice")
        agg["pq_dice05"].append(panoptic_quality(m05, len(preds), len(gts), fp05, fn05))

        mi, fpi, fni = greedy_match(preds, gts, overlap_threshold=0.5, mode="iou")
        agg["pq_iou05"].append(panoptic_quality(mi, len(preds), len(gts), fpi, fni))
        tp_iou05 += len(mi); fp_iou05 += fpi; fn_iou05 += fni

        # grouped (image-level) dice: union all preds vs union all gts
        h, w = image.shape
        pred_union = np.zeros((h, w), dtype=np.uint8)
        for inst in preds:
            pred_union = np.maximum(pred_union, inst)
        gt_union = np.zeros((h, w), dtype=np.uint8)
        for inst in gts:
            gt_union = np.maximum(gt_union, inst)
        gd = full_image_dice(pred_union, gt_union)
        agg["grouped_dice"].append(gd)
        global_pred_sum += int(pred_union.sum())
        global_gt_sum += int(gt_union.sum())
        global_inter_sum += int(np.logical_and(pred_union.astype(bool), gt_union.astype(bool)).sum())

    print("=" * 72)
    print(f"images={len(files)}  config: conf={args.conf} thr={args.threshold} min_area={args.min_area} nms={args.mask_nms_iou} imgsz={args.imgsz}")
    print("=" * 72)
    print(f"PQ Dice thr0.0   : {np.mean(agg['pq_dice0']):.4f}   (tp={tp_dice0} fp={fp_dice0} fn={fn_dice0})")
    print(f"PQ Dice thr0.5   : {np.mean(agg['pq_dice05']):.4f}")
    print(f"PQ IoU  thr0.5   : {np.mean(agg['pq_iou05']):.4f}   (tp={tp_iou05} fp={fp_iou05} fn={fn_iou05})")
    print(f"mean_dice_matched: {np.mean(agg['mean_dice_matched']):.4f}")
    print(f"grouped dice     : mean={np.mean(agg['grouped_dice']):.4f}  median={np.median(agg['grouped_dice']):.4f}")
    print(f"global pooled    : {2*global_inter_sum/(global_pred_sum+global_gt_sum):.4f}")
    print(f"grouped dice per-image (first 20): {[round(x,3) for x in agg['grouped_dice'][:20]]}")


if __name__ == "__main__":
    main()
