"""Detector recall/precision evaluation on the DENSE (test-representative) split.

Runs a detector (Mask R-CNN or YOLO) over the seed-42 observation-grouped random
split (the same 141 images as ``eval_dense_rebaseline``) and reports detection
recall (box IoU>=0.5) and, for Mask R-CNN, segmentation recall (mask IoU>=0.5)
against **deduped** ground truth. Dedup matters: the challenge ships multiple
annotator records per image, and un-deduped GT over-counts filaments ~1.28x.

This is the apples-to-apples detector comparison vs YOLO's R=52.6% (ultralytics
mAP50 recall): run once with --mode maskrcnn (each epoch) and once with
--mode yolo to get the head-to-head.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch

import config
from src.dataset import read_gray_image
from src.gt_dedup import image_masks_from_anns, load_anns_by_file, masks_to_boxes
from src.preprocessing import make_input_channels
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def build_maskrcnn(ckpt_path: Path, device):
    model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
    in_feat_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat_box, 2)
    in_feat_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_feat_mask, 256, 2)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


def box_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of xyxy boxes. Returns (len(a), len(b))."""
    A, B = len(boxes_a), len(boxes_b)
    if A == 0 or B == 0:
        return np.zeros((A, B), dtype=np.float32)
    a = boxes_a[:, None, :]  # A,1,4
    b = boxes_b[None, :, :]  # 1,B,4
    ix = np.maximum(0, np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]))
    iy = np.maximum(0, np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]))
    inter = ix * iy
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-6), 0.0)


def mask_iou(masks_a: list[np.ndarray], masks_b: list[np.ndarray]) -> np.ndarray:
    A, B = len(masks_a), len(masks_b)
    M = np.zeros((A, B), dtype=np.float32)
    if A == 0 or B == 0:
        return M
    for i in range(A):
        for j in range(B):
            inter = int(np.logical_and(masks_a[i], masks_b[j]).sum())
            if inter == 0:
                continue
            union = int(masks_a[i].sum()) + int(masks_b[j].sum()) - inter
            M[i, j] = inter / union
    return M


def greedy_match(ov: np.ndarray, thr: float):
    P, G = ov.shape
    cands = sorted(((ov[i, j], i, j) for i in range(P) for j in range(G) if ov[i, j] >= thr), reverse=True)
    used_p, used_g = set(), set()
    matched = 0
    for _, i, j in cands:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        matched += 1
    return matched, P - len(used_p), G - len(used_g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    ap.add_argument("--mode", choices=["maskrcnn", "yolo"], required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--score-thrs", type=float, nargs="+", default=[0.0, 0.3, 0.5, 0.7])
    ap.add_argument("--mask-thr", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_info, anns_by_file = load_anns_by_file(args.data_dir / "train_annotations.json")
    image_dir = args.data_dir / "train_images"
    disk = sorted(p.name for p in image_dir.iterdir() if p.suffix.lower() in {".jpeg", ".jpg", ".png"})
    files = sorted(f for f in disk if f in anns_by_file)
    rng = random.Random(args.seed)
    rng.shuffle(files)
    n_val = max(1, int(round(len(files) * config.VAL_FRACTION)))
    files = files[:n_val]
    if args.limit:
        files = files[: args.limit]

    if args.mode == "maskrcnn":
        model = build_maskrcnn(args.ckpt, device)
    else:
        from ultralytics import YOLO
        model = YOLO(str(args.ckpt))

    # accumulate per-score-threshold stats
    stats = {s: {"box_tp": 0, "box_fp": 0, "box_fn": 0, "mask_tp": 0, "mask_fp": 0, "mask_fn": 0} for s in args.score_thrs}
    total_gt = 0

    for fname in files:
        gray = read_gray_image(image_dir / fname)
        h, w = gray.shape
        gray_s = cv2.resize(gray, (args.imgsz, args.imgsz), interpolation=cv2.INTER_AREA)

        if args.mode == "maskrcnn":
            img_t = torch.from_numpy(np.ascontiguousarray(make_input_channels(gray_s))).float().to(device)
            with torch.no_grad():
                out = model([img_t])[0]
            boxes = out["boxes"].detach().cpu().numpy()  # imgsz space
            scores = out["scores"].detach().cpu().numpy()
            masks_full = out["masks"].detach().cpu().numpy()[:, 0]  # (N, imgsz, imgsz) in [0,1]
            pred_masks_all = [(masks_full[i] >= args.mask_thr) for i in range(len(masks_full))]
        else:
            det = model.predict(source=str(image_dir / fname), imgsz=args.imgsz, conf=0.01,
                               device=0 if device.type == "cuda" else "cpu", verbose=False)[0]
            boxes = det.boxes.xyxy.detach().cpu().numpy() if det.boxes is not None else np.empty((0, 4), np.float32)
            scores = det.boxes.conf.detach().cpu().numpy() if det.boxes is not None else np.empty((0,), np.float32)
            # ultralytics returns boxes in the ORIGINAL image space (2048); scale to imgsz
            boxes = boxes * (args.imgsz / h)
            pred_masks_all = []

        # deduped GT at imgsz resolution
        gt_masks = image_masks_from_anns(anns_by_file[fname], h, w, out_size=(args.imgsz, args.imgsz), iou_thresh=0.5)
        gt_boxes = masks_to_boxes(gt_masks)
        total_gt += len(gt_masks)

        ov_box = box_iou(boxes, gt_boxes)
        ov_mask = mask_iou(pred_masks_all, gt_masks) if pred_masks_all else np.zeros((0, len(gt_masks)), np.float32)

        for s in args.score_thrs:
            keep = scores >= s
            st = stats[s]
            m, fp, fn = greedy_match(ov_box[keep], 0.5)
            st["box_tp"] += m; st["box_fp"] += fp; st["box_fn"] += fn
            if pred_masks_all:
                m2, fp2, fn2 = greedy_match(ov_mask[np.where(keep)[0]], 0.5)
                st["mask_tp"] += m2; st["mask_fp"] += fp2; st["mask_fn"] += fn2

    print("=" * 74)
    print(f"mode={args.mode}  imgsz={args.imgsz}  seed={args.seed}  images={len(files)}  GT(dedup)={total_gt}")
    print(f"ckpt={args.ckpt}")
    print("=" * 74)
    print(f"{'score_thr':>10} | {'boxR@.5':>8} {'boxP@.5':>8} | {'maskR@.5':>9} {'maskP@.5':>9} | {'box_tp':>6} {'box_fp':>6} {'box_fn':>6}")
    for s in args.score_thrs:
        st = stats[s]
        box_r = st["box_tp"] / max(1, st["box_tp"] + st["box_fn"])
        box_p = st["box_tp"] / max(1, st["box_tp"] + st["box_fp"])
        has_mask = st["mask_tp"] + st["mask_fn"] > 0
        mask_r = st["mask_tp"] / max(1, st["mask_tp"] + st["mask_fn"]) if has_mask else float("nan")
        mask_p = st["mask_tp"] / max(1, st["mask_tp"] + st["mask_fp"]) if has_mask else float("nan")
        print(f"{s:>10.2f} | {box_r:>8.3f} {box_p:>8.3f} | {mask_r:>9.3f} {mask_p:>9.3f} | {st['box_tp']:>6} {st['box_fp']:>6} {st['box_fn']:>6}")


if __name__ == "__main__":
    main()
