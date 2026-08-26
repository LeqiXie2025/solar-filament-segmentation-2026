import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import config
from src.crop_refiner_utils import crop_bounds_from_bbox, crop_with_padding, make_refiner_channels, seed_from_bbox
from src.dataset import read_gray_image
from src.masks import encode_binary_mask, postprocess_binary_mask
from src.metrics import greedy_match, panoptic_quality
from src.model_factory import create_model_from_checkpoint
from src.predict import sliding_window_predict
from src.predict_detect_refine import make_masks_disjoint, mask_nms, patch_torchvision_nms, time_val_names
from src.validation import load_gt_instances


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--refiner", type=Path, required=True)
    parser.add_argument("--sq-checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=["val", "test"], default="val")
    parser.add_argument("--output", type=Path, default=config.SUBMISSION_DIR / "submission_fused_dq_sq.csv")
    parser.add_argument("--conf", type=float, default=0.18)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--min-area", type=int, default=120)
    parser.add_argument("--mask-nms-iou", type=float, default=0.35)
    parser.add_argument("--box-pad", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sq-tile-size", type=int, default=512)
    parser.add_argument("--sq-stride", type=int, default=384)
    parser.add_argument("--sq-weight", type=float, default=0.35)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def load_sq_model(path: Path, device):
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("arch") == "edgeattnet":
        from src.model_edgeattnet import EdgeAttNet

        model = EdgeAttNet(in_channels=config.IN_CHANNELS, out_channels=1, base_channels=config.BASE_CHANNELS).to(device)
    else:
        model = create_model_from_checkpoint(checkpoint).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


@torch.no_grad()
def refine_fused(
    refiner,
    image: np.ndarray,
    semantic_prob: np.ndarray,
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    device,
    args,
    image_size: int,
) -> list[tuple[np.ndarray, float]]:
    height, width = image.shape
    tensors = []
    metas = []
    for box in boxes_xyxy:
        x0, y0, x1, y1 = [float(v) for v in box[:4]]
        bbox = (x0, y0, max(1.0, x1 - x0), max(1.0, y1 - y0))
        bounds = crop_bounds_from_bbox(bbox, height, width, pad=args.box_pad, jitter=False)
        crop = crop_with_padding(image, bounds)
        crop_rs = cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        seed = seed_from_bbox(bbox, bounds, image_size)
        tensors.append(torch.from_numpy(make_refiner_channels(crop_rs, seed)).float())
        metas.append(bounds)

    instances = []
    for start in range(0, len(tensors), args.batch_size):
        batch = torch.stack(tensors[start : start + args.batch_size]).to(device)
        probs = torch.sigmoid(refiner(batch))[:, 0].detach().cpu().numpy()
        batch_scores = scores[start : start + args.batch_size]
        for ref_prob, bounds, score in zip(probs, metas[start : start + args.batch_size], batch_scores):
            x0, y0, x1, y1 = bounds
            crop_h = max(1, y1 - y0)
            crop_w = max(1, x1 - x0)
            sx0, sy0 = max(0, x0), max(0, y0)
            sx1, sy1 = min(width, x1), min(height, y1)
            if sx1 <= sx0 or sy1 <= sy0:
                continue

            ref_full = cv2.resize(ref_prob, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
            sem_crop = crop_with_padding(semantic_prob, bounds)
            sem_full = cv2.resize(sem_crop, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
            fused = (1.0 - args.sq_weight) * ref_full + args.sq_weight * sem_full
            binary_crop = (fused > args.threshold).astype(np.uint8)
            binary_crop = postprocess_binary_mask(binary_crop, "close")

            dx0, dy0 = sx0 - x0, sy0 - y0
            instance = np.zeros((height, width), dtype=np.uint8)
            instance[sy0:sy1, sx0:sx1] = binary_crop[dy0 : dy0 + (sy1 - sy0), dx0 : dx0 + (sx1 - sx0)]
            if int(instance.sum()) >= args.min_area:
                score_fused = float(score) * float(fused[binary_crop.astype(bool)].mean() if binary_crop.any() else 0.0)
                instances.append((instance, score_fused))
    return instances


def pq_parts(matched: list[tuple[int, int, float]], n_pred: int, n_gt: int, fp: int, fn: int) -> tuple[float, float]:
    if not matched:
        return 0.0, 0.0
    sq = float(np.mean([ov for _, _, ov in matched]))
    rq = float(len(matched) / max(len(matched) + 0.5 * fp + 0.5 * fn, 1e-6))
    return sq, rq


def main() -> None:
    args = parse_args()
    patch_torchvision_nms()
    from ultralytics import YOLO

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = YOLO(str(args.detector))

    ref_ckpt = torch.load(args.refiner, map_location=device)
    refiner = create_model_from_checkpoint(ref_ckpt).to(device)
    refiner.load_state_dict(ref_ckpt["model"])
    refiner.eval()
    refiner_size = int(ref_ckpt.get("image_size", 512))
    sq_model = load_sq_model(args.sq_checkpoint, device)

    image_dir = args.data_dir / ("test_images" if args.mode == "test" else "train_images")
    files = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpeg", ".jpg", ".png"}])
    if args.mode == "val":
        names = set(time_val_names(args.data_dir / "train_annotations.json", {p.name for p in files}))
        files = [p for p in files if p.name in names]
    if args.limit:
        files = files[: args.limit]

    rows = []
    pq_scores = []
    sq_scores = []
    rq_scores = []
    total_tp = total_fp = total_fn = 0
    gt_by_file = load_gt_instances(args.data_dir / "train_annotations.json", {p.name for p in files}) if args.mode == "val" else {}

    for path in tqdm(files, desc=f"fused-dq-sq-{args.mode}"):
        image = read_gray_image(path)
        semantic_prob = sliding_window_predict(
            sq_model,
            image,
            device,
            tile_size=args.sq_tile_size,
            stride=args.sq_stride,
            batch_size=max(1, args.batch_size // 2),
            tta=False,
        )
        det = detector.predict(
            source=str(path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=0 if device.type == "cuda" else "cpu",
            verbose=False,
        )[0]
        boxes = det.boxes.xyxy.detach().cpu().numpy() if det.boxes is not None else np.empty((0, 4), dtype=np.float32)
        scores = det.boxes.conf.detach().cpu().numpy() if det.boxes is not None else np.empty((0,), dtype=np.float32)
        scored = refine_fused(refiner, image, semantic_prob, boxes, scores, device, args, refiner_size) if len(boxes) else []
        pred_instances = mask_nms(scored, args.mask_nms_iou)
        pred_instances = make_masks_disjoint(pred_instances, args.min_area)

        if args.mode == "val":
            gt_instances = gt_by_file.get(path.name, [])
            matched, fp, fn = greedy_match(pred_instances, gt_instances, overlap_threshold=0.0, mode="dice")
            pq_scores.append(panoptic_quality(matched, len(pred_instances), len(gt_instances), fp, fn))
            sq, rq = pq_parts(matched, len(pred_instances), len(gt_instances), fp, fn)
            sq_scores.append(sq)
            rq_scores.append(rq)
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
                    "threshold": args.threshold,
                    "min_area": args.min_area,
                    "sq_weight": args.sq_weight,
                    "pq": float(np.mean(pq_scores)) if pq_scores else 0.0,
                    "sq": float(np.mean(sq_scores)) if sq_scores else 0.0,
                    "rq": float(np.mean(rq_scores)) if rq_scores else 0.0,
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
