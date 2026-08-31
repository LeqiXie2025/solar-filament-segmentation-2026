"""Dense-split post-processing sweep for YOLO+refiner (deduped GT, fast matching).

Answers: does lowering YOLO ``conf`` below 0.14 raise end-to-end recall/PQ on the
dense (test-representative) split? Uses deduped GT (multi-annotator collapse) and
bbox-prefiltered matching (only computes overlap inside the box intersection, so
it no longer costs O(P*G*H*W)).

Efficiency: YOLO runs ONCE per image at conf=0.05 (all boxes + scores), and
``refine_boxes`` runs ONCE per image (its outputs are tagged with the original
YOLO score). Each ``conf`` is then just a cheap score filter + mask NMS + disjoint.
Note ``detect_conf`` defaults to the lowest swept conf (0.05); YOLO NMS sorts by
score descending so high-score boxes are never suppressed by lower ones, making
the score>=0.05 box set identical whether detected at 0.01 or 0.05.

Chunking: ``--start``/``--count`` slice the val files so long runs can be split
under the 10-min task cap; each run dumps *additive* accumulators (PQ numerator/
denominator, grouped-dice sum, tp/fp/fn) to ``--out-json`` so chunks combine by
summing. Aggregate PQ@IoU = sum(num)/sum(den); grouped dice = sum(grouped)/N.

Primary metric: PQ@IoU0.5 (standard COCO PQ). Secondary: PQ@Dice0 (lenient) and
grouped image Dice.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch

import config
from src.dataset import read_gray_image
from src.gt_dedup import image_masks_from_anns, load_anns_by_file
from src.metrics import full_image_dice
from src.model_factory import create_model_from_checkpoint
from src.predict_detect_refine import make_masks_disjoint, mask_nms, patch_torchvision_nms, refine_boxes


def bbox_of(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def overlap_fast(preds, gts):
    """Dice + IoU matrices, computing overlap only inside box intersections."""
    P, G = len(preds), len(gts)
    Md = np.zeros((P, G), np.float32)
    Mi = np.zeros((P, G), np.float32)
    if P == 0 or G == 0:
        return Md, Mi
    p_area = np.array([int(p.sum()) for p in preds], np.float32)
    g_area = np.array([int(g.sum()) for g in gts], np.float32)
    p_box = [bbox_of(p) for p in preds]
    g_box = [bbox_of(g) for g in gts]
    for i in range(P):
        if p_area[i] == 0:
            continue
        px0, py0, px1, py1 = p_box[i]
        for j in range(G):
            if g_area[j] == 0:
                continue
            gx0, gy0, gx1, gy1 = g_box[j]
            ix0, iy0 = max(px0, gx0), max(py0, gy0)
            ix1, iy1 = min(px1, gx1), min(py1, gy1)
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            inter = int(np.logical_and(preds[i][iy0:iy1, ix0:ix1], gts[j][iy0:iy1, ix0:ix1]).sum())
            if inter == 0:
                continue
            s = p_area[i] + g_area[j]
            Md[i, j] = 2.0 * inter / s
            Mi[i, j] = inter / (s - inter)
    return Md, Mi


def greedy(M, thr):
    P, G = M.shape
    cands = sorted(((M[i, j], i, j) for i in range(P) for j in range(G) if M[i, j] >= thr), reverse=True)
    up, ug = set(), set()
    matched = 0
    ov_sum = 0.0
    for ov, i, j in cands:
        if i in up or j in ug:
            continue
        up.add(i)
        ug.add(j)
        matched += 1
        ov_sum += float(ov)
    return matched, P - len(up), G - len(ug), ov_sum


class Args:
    def __init__(self, threshold, min_area, mask_nms_iou, box_pad, batch_size, imgsz):
        self.threshold = threshold
        self.min_area = min_area
        self.mask_nms_iou = mask_nms_iou
        self.box_pad = box_pad
        self.batch_size = batch_size
        self.imgsz = imgsz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    ap.add_argument("--detector", type=Path, default=config.MODEL_DIR / "yolo/yolo11s_boxes_1280_e25/weights/best.pt")
    ap.add_argument("--refiner", type=Path, default=config.MODEL_DIR / "crop_refiner_resnet34_512_detectorbox_e8.pth")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--confs", type=float, nargs="+", default=[0.05, 0.08, 0.10, 0.12, 0.14, 0.18, 0.22, 0.26, 0.30])
    ap.add_argument("--detect-conf", type=float, default=0.05, help="YOLO run once at this floor; confs filter by score")
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--min-area", type=int, default=140)
    ap.add_argument("--mask-nms-iou", type=float, default=0.20)
    ap.add_argument("--box-pad", type=float, default=0.50)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--iou-thr", type=float, default=0.45, help="YOLO NMS iou")
    ap.add_argument("--start", type=int, default=0, help="offset into the shuffled val list")
    ap.add_argument("--count", type=int, default=0, help="how many val images (0=all after start)")
    ap.add_argument("--out-json", type=Path, default=None, help="dump additive accumulators for chunk combining")
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

    _, anns_by_file = load_anns_by_file(args.data_dir / "train_annotations.json")
    image_dir = args.data_dir / "train_images"
    files = sorted(p.name for p in image_dir.iterdir() if p.suffix.lower() in {".jpeg", ".jpg", ".png"})
    rng = random.Random(args.seed)
    rng.shuffle(files)
    n_val = max(1, int(round(len(files) * config.VAL_FRACTION)))
    files = files[:n_val]
    files = files[args.start : args.start + args.count] if args.count else files[args.start :]

    det_args = Args(args.threshold, args.min_area, args.mask_nms_iou, args.box_pad, args.batch_size, args.imgsz)

    acc = {c: {"tp": 0, "fp": 0, "fn": 0, "n_pred": 0,
               "pq_iou_num": 0.0, "pq_iou_den": 0.0,
               "pq_dice_num": 0.0, "pq_dice_den": 0.0, "grouped_sum": 0.0} for c in args.confs}

    print("=" * 78)
    print(f"seed={args.seed}  images={len(files)}  imgsz={args.imgsz}  thr={args.threshold} min_area={args.min_area} nms={args.mask_nms_iou}")
    print(f"{'conf':>6} | {'PQ_iou05':>9} {'PQ_dice0':>9} {'grp_dice':>8} | {'tp':>5} {'fp':>6} {'fn':>5} {'pred':>6}")
    print("-" * 78, flush=True)

    for idx, fname in enumerate(files):
        image = read_gray_image(image_dir / fname)
        h, w = image.shape
        gts = image_masks_from_anns(anns_by_file[fname], h, w)

        det = detector.predict(source=str(image_dir / fname), imgsz=args.imgsz, conf=args.detect_conf, iou=args.iou_thr,
                               device=0 if device.type == "cuda" else "cpu", verbose=False)[0]
        boxes = det.boxes.xyxy.detach().cpu().numpy() if det.boxes is not None else np.empty((0, 4), np.float32)
        scores = det.boxes.conf.detach().cpu().numpy() if det.boxes is not None else np.empty((0,), np.float32)
        scored = refine_boxes(refiner, image, boxes, scores, device, det_args, image_size) if len(boxes) else []

        gu = np.zeros((h, w), np.uint8)
        for inst in gts:
            gu = np.maximum(gu, inst)

        for conf in args.confs:
            subset = [(m, s) for m, s in scored if s >= conf]
            preds = mask_nms(subset, args.mask_nms_iou)
            preds = make_masks_disjoint(preds, args.min_area)

            Md, Mi = overlap_fast(preds, gts)
            m_i, fpi, fni, ov_i = greedy(Mi, 0.5)
            a = acc[conf]
            a["tp"] += m_i; a["fp"] += fpi; a["fn"] += fni
            a["pq_iou_num"] += ov_i
            a["pq_iou_den"] += m_i + 0.5 * fpi + 0.5 * fni
            m_d, fpd, fnd, ov_d = greedy(Md, 0.0)
            a["pq_dice_num"] += ov_d
            a["pq_dice_den"] += m_d + 0.5 * fpd + 0.5 * fnd

            pu = np.zeros((h, w), np.uint8)
            for inst in preds:
                pu = np.maximum(pu, inst)
            a["grouped_sum"] += full_image_dice(pu, gu)
            a["n_pred"] += len(preds)

        if (idx + 1) % 10 == 0 or idx + 1 == len(files):
            print(f"[{args.start + idx + 1}/{args.start + len(files)}] processed", flush=True)

    print("=" * 78)
    for conf in args.confs:
        a = acc[conf]
        pq_iou = a["pq_iou_num"] / max(a["pq_iou_den"], 1e-6)
        pq_dice = a["pq_dice_num"] / max(a["pq_dice_den"], 1e-6)
        grp = a["grouped_sum"] / max(len(files), 1)
        print(f"{conf:>6.2f} | {pq_iou:>9.4f} {pq_dice:>9.4f} {grp:>8.4f} | "
              f"{a['tp']:>5} {a['fp']:>6} {a['fn']:>5} {a['n_pred']:>6}")
    print("=" * 78)
    print("PQ_iou05 uses greedy IoU>=0.5 matching (standard COCO PQ); PQ_dice0 uses Dice>0 (lenient).")

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "seed": args.seed, "imgsz": args.imgsz, "threshold": args.threshold,
            "min_area": args.min_area, "mask_nms_iou": args.mask_nms_iou,
            "detect_conf": args.detect_conf, "start": args.start, "count": len(files),
            "n_images": len(files),
            "confs": {f"{c:.2f}": acc[c] for c in args.confs},
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"saved {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
