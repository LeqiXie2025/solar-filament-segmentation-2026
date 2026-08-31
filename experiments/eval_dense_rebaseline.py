"""Re-baseline detect-refine on a DENSE (test-representative) split + miss analysis.

Why: the time-split val is the sparse 2017-2022 tail (4.8 filaments/image), but the
test set is 83% dense 2011-2017 era (~9.5/image). Local PQ 0.50 on the sparse tail
does not map to LB 0.33 on the dense test. This script evaluates on a random
(observation-grouped) split that matches test's year distribution, and bins the
missed (fn) ground-truth instances by area in the same pass.

Metrics: PQ(Dice thr0), PQ(IoU thr0.5), grouped image-level Dice, + fn-by-area.
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

import config
from src.dataset import read_gray_image
from src.masks import postprocess_binary_mask
from src.metrics import full_image_dice, panoptic_quality
from src.model_factory import create_model_from_checkpoint
from src.predict_detect_refine import make_masks_disjoint, mask_nms, patch_torchvision_nms, refine_boxes, time_val_names
from src.validation import load_gt_instances


def overlap_matrices(preds, gts):
    """Compute dice and iou matrices in one pass over pairwise intersections."""
    P, G = len(preds), len(gts)
    Md = np.zeros((P, G), dtype=np.float32)
    Mi = np.zeros((P, G), dtype=np.float32)
    if P == 0 or G == 0:
        return Md, Mi
    p_areas = np.array([int(p.sum()) for p in preds], dtype=np.float32)
    g_areas = np.array([int(g.sum()) for g in gts], dtype=np.float32)
    for i, p in enumerate(preds):
        if p_areas[i] == 0:
            continue
        p_bool = p.astype(bool)
        for j, g in enumerate(gts):
            if g_areas[j] == 0:
                continue
            inter = int(np.logical_and(p_bool, g).sum())
            if inter == 0:
                continue
            s = p_areas[i] + g_areas[j]
            Md[i, j] = 2.0 * inter / s
            Mi[i, j] = inter / (s - inter)
    return Md, Mi


def greedy_from_matrix(M, threshold):
    P, G = M.shape
    candidates = [(M[i, j], i, j) for i in range(P) for j in range(G) if M[i, j] >= threshold]
    candidates.sort(reverse=True)
    used_p, used_g = set(), set()
    matched = []
    for ov, i, j in candidates:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        matched.append((i, j, float(ov)))
    return matched, P - len(used_p), G - len(used_g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    ap.add_argument("--detector", type=Path, required=True)
    ap.add_argument("--refiner", type=Path, required=True)
    ap.add_argument("--split", choices=["time", "random"], default="random")
    ap.add_argument("--seed", type=int, default=42)
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
    if args.split == "time":
        names = set(time_val_names(args.data_dir / "train_annotations.json", {p.name for p in files}))
        files = [p for p in files if p.name in names]
    else:
        random.seed(args.seed)
        random.shuffle(files)
        n_val = max(1, int(round(len(files) * config.VAL_FRACTION)))
        files = files[:n_val]
    if args.limit:
        files = files[: args.limit]
    gt_by_file = load_gt_instances(args.data_dir / "train_annotations.json", {p.name for p in files})

    pq_dice0, pq_iou05, grouped = [], [], []
    tp0 = fp0 = fn0 = 0
    tpi = fpi = fni = 0
    fn_area_bins = {"<500": 0, "500-1500": 0, "1500-4000": 0, ">=4000": 0}
    gt_total_bins = {"<500": 0, "500-1500": 0, "1500-4000": 0, ">=4000": 0}
    global_ps = global_gs = global_inter = 0

    def bin_of(area):
        if area < 500: return "<500"
        if area < 1500: return "500-1500"
        if area < 4000: return "1500-4000"
        return ">=4000"

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

        Md, Mi = overlap_matrices(preds, gts)

        m0, fp, fn = greedy_from_matrix(Md, 0.0)
        pq_dice0.append(panoptic_quality(m0, len(preds), len(gts), fp, fn))
        tp0 += len(m0); fp0 += fp; fn0 += fn

        mi, fpi_, fni_ = greedy_from_matrix(Mi, 0.5)
        pq_iou05.append(panoptic_quality(mi, len(preds), len(gts), fpi_, fni_))
        tpi += len(mi); fpi += fpi_; fni += fni_

        # miss analysis: GT area of all GT, and of unmatched GT (fn at Dice>0)
        used_g = {j for _, j, _ in m0}
        for j, g in enumerate(gts):
            a = int(g.sum())
            b = bin_of(a)
            gt_total_bins[b] += 1
            if j not in used_g:
                fn_area_bins[b] += 1

        h, w = image.shape
        pu = np.zeros((h, w), np.uint8)
        for inst in preds:
            pu = np.maximum(pu, inst)
        gu = np.zeros((h, w), np.uint8)
        for inst in gts:
            gu = np.maximum(gu, inst)
        grouped.append(full_image_dice(pu, gu))
        global_ps += int(pu.sum()); global_gs += int(gu.sum()); global_inter += int(np.logical_and(pu.astype(bool), gu.astype(bool)).sum())

    print("=" * 74)
    print(f"split={args.split} seed={args.seed}  images={len(files)}  conf={args.conf} thr={args.threshold} min_area={args.min_area} nms={args.mask_nms_iou} imgsz={args.imgsz}")
    print("=" * 74)
    print(f"PQ Dice thr0.0 : {np.mean(pq_dice0):.4f}   (tp={tp0} fp={fp0} fn={fn0})")
    print(f"PQ IoU  thr0.5 : {np.mean(pq_iou05):.4f}   (tp={tpi} fp={fpi} fn={fni})")
    print(f"grouped dice    : mean={np.mean(grouped):.4f}  median={np.median(grouped):.4f}  pooled={2*global_inter/(global_ps+global_gs):.4f}")
    print("-" * 74)
    print("miss analysis (fn by GT area, Dice>0 matching):")
    for b in ["<500", "500-1500", "1500-4000", ">=4000"]:
        tot = gt_total_bins[b]
        miss = fn_area_bins[b]
        rate = miss / tot if tot else 0
        print(f"  {b:>9}: GT={tot:4d}  fn={miss:4d}  miss_rate={rate:.1%}")
    print(f"  total GT={sum(gt_total_bins.values())}  fn={sum(fn_area_bins.values())}")


if __name__ == "__main__":
    main()
