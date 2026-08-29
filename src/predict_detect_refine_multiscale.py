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
from src.metrics import greedy_match, mean_dice_matched, panoptic_quality
from src.model_factory import create_model_from_checkpoint
from src.predict_detect_refine import make_masks_disjoint, mask_nms, patch_torchvision_nms, refine_boxes, time_val_names
from src.validation import load_gt_instances


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--annotation-path", type=Path, default=None)
    parser.add_argument("--detector", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--detector-conf",
        type=float,
        nargs="+",
        default=None,
        help="Per-detector confidence thresholds; defaults to --conf for every detector.",
    )
    parser.add_argument("--refiner", type=Path, required=True)
    parser.add_argument("--mode", choices=["val", "test"], default="val")
    parser.add_argument("--output", type=Path, default=config.SUBMISSION_DIR / "submission_detect_refine_multiscale.csv")
    parser.add_argument("--conf", type=float, default=0.14)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--min-area", type=int, default=140)
    parser.add_argument("--mask-nms-iou", type=float, default=0.20)
    parser.add_argument("--box-pad", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, nargs="+", default=[1024, 1280])
    parser.add_argument("--tile-size", type=int, default=0)
    parser.add_argument("--tile-imgsz", type=int, default=0)
    parser.add_argument("--tile-detector", type=Path, nargs="+", default=None)
    parser.add_argument("--tile-detector-conf", type=float, nargs="+", default=None)
    parser.add_argument("--tile-overlap", type=int, default=256)
    parser.add_argument("--tile-conf", type=float, default=None)
    parser.add_argument("--tile-border", type=int, default=24)
    parser.add_argument("--tile-score-weight", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--refiner-tta", action="store_true")
    parser.add_argument(
        "--score-mode",
        choices=["detector", "meanprob", "detector_meanprob"],
        default="detector",
    )
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def _flip_tensor(batch: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "identity":
        return batch
    if mode == "hflip":
        return torch.flip(batch, dims=[3])
    if mode == "vflip":
        return torch.flip(batch, dims=[2])
    if mode == "hvflip":
        return torch.flip(batch, dims=[2, 3])
    raise ValueError(mode)


def _unflip_array(batch: np.ndarray, mode: str) -> np.ndarray:
    if mode == "identity":
        return batch
    if mode == "hflip":
        return batch[:, :, ::-1]
    if mode == "vflip":
        return batch[:, ::-1, :]
    if mode == "hvflip":
        return batch[:, ::-1, ::-1]
    raise ValueError(mode)


def tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if tile_size <= 0 or tile_size >= length:
        return [0]
    stride = max(1, tile_size - overlap)
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def tiled_detect(detector, image: np.ndarray, detector_conf: float, device, args):
    tile_size = int(args.tile_size)
    if tile_size <= 0:
        return [], []
    height, width = image.shape
    boxes_out = []
    scores_out = []
    x_starts = tile_starts(width, tile_size, int(args.tile_overlap))
    y_starts = tile_starts(height, tile_size, int(args.tile_overlap))
    tile_conf = float(args.tile_conf if args.tile_conf is not None else detector_conf)
    for y0 in y_starts:
        for x0 in x_starts:
            tile = image[y0 : min(height, y0 + tile_size), x0 : min(width, x0 + tile_size)]
            tile_source = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR) if tile.ndim == 2 else tile
            det = detector.predict(
                source=tile_source,
                imgsz=int(args.tile_imgsz or tile_size),
                conf=tile_conf,
                iou=args.iou,
                device=0 if device.type == "cuda" else "cpu",
                verbose=False,
            )[0]
            if det.boxes is None or len(det.boxes) == 0:
                continue
            boxes = det.boxes.xyxy.detach().cpu().numpy()
            scores = det.boxes.conf.detach().cpu().numpy()
            centers_x = (boxes[:, 0] + boxes[:, 2]) * 0.5
            centers_y = (boxes[:, 1] + boxes[:, 3]) * 0.5
            border = int(args.tile_border)
            keep = np.ones(len(boxes), dtype=bool)
            if x0 > 0:
                keep &= centers_x >= border
            if x0 + tile.shape[1] < width:
                keep &= centers_x < tile.shape[1] - border
            if y0 > 0:
                keep &= centers_y >= border
            if y0 + tile.shape[0] < height:
                keep &= centers_y < tile.shape[0] - border
            boxes = boxes[keep]
            scores = scores[keep]
            if not len(boxes):
                continue
            boxes[:, [0, 2]] += x0
            boxes[:, [1, 3]] += y0
            boxes_out.append(boxes)
            scores_out.append(scores * float(args.tile_score_weight))
    return boxes_out, scores_out


@torch.no_grad()
def refine_boxes_tta(refiner, image: np.ndarray, boxes_xyxy: np.ndarray, scores: np.ndarray, device, args, image_size: int):
    height, width = image.shape
    instances = []
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

    transforms = ("identity", "hflip", "vflip", "hvflip")
    for start in range(0, len(tensors), args.batch_size):
        batch = torch.stack(tensors[start : start + args.batch_size]).to(device)
        prob_sum = None
        for transform in transforms:
            transformed = _flip_tensor(batch, transform)
            prob = torch.sigmoid(refiner(transformed))[:, 0].detach().cpu().numpy()
            prob = _unflip_array(prob, transform)
            prob_sum = prob if prob_sum is None else prob_sum + prob
        probs = prob_sum / float(len(transforms))
        batch_scores = scores[start : start + args.batch_size]
        for prob, bounds, score in zip(probs, metas[start : start + args.batch_size], batch_scores):
            x0, y0, x1, y1 = bounds
            crop_h = max(1, y1 - y0)
            crop_w = max(1, x1 - x0)
            prob_full = cv2.resize(prob, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
            binary_crop = (prob_full > args.threshold).astype(np.uint8)
            binary_crop = postprocess_binary_mask(binary_crop, "close")
            sx0, sy0 = max(0, x0), max(0, y0)
            sx1, sy1 = min(width, x1), min(height, y1)
            if sx1 <= sx0 or sy1 <= sy0:
                continue
            dx0, dy0 = sx0 - x0, sy0 - y0
            instance = np.zeros((height, width), dtype=np.uint8)
            instance[sy0:sy1, sx0:sx1] = binary_crop[dy0 : dy0 + (sy1 - sy0), dx0 : dx0 + (sx1 - sx0)]
            if int(instance.sum()) >= args.min_area:
                mask_probs = prob_full[binary_crop.astype(bool)]
                mean_prob = float(mask_probs.mean()) if mask_probs.size else 0.0
                if args.score_mode == "meanprob":
                    keep_score = mean_prob
                elif args.score_mode == "detector_meanprob":
                    keep_score = float(score) * mean_prob
                else:
                    keep_score = float(score)
                instances.append((instance, keep_score))
    return instances


def main() -> None:
    args = parse_args()
    patch_torchvision_nms()
    from ultralytics import YOLO

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detectors = [YOLO(str(path)) for path in args.detector]
    detector_confs = args.detector_conf or [args.conf] * len(detectors)
    if len(detector_confs) == 1 and len(detectors) > 1:
        detector_confs *= len(detectors)
    if len(detector_confs) != len(detectors):
        raise ValueError("--detector-conf must contain one value or one value per --detector")
    tile_detectors = [YOLO(str(path)) for path in args.tile_detector] if args.tile_detector else detectors
    tile_detector_confs = args.tile_detector_conf or [
        args.tile_conf if args.tile_conf is not None else args.conf
    ] * len(tile_detectors)
    if len(tile_detector_confs) == 1 and len(tile_detectors) > 1:
        tile_detector_confs *= len(tile_detectors)
    if len(tile_detector_confs) != len(tile_detectors):
        raise ValueError("--tile-detector-conf must contain one value or one value per tile detector")
    checkpoint = torch.load(args.refiner, map_location=device)
    refiner = create_model_from_checkpoint(checkpoint).to(device)
    refiner.load_state_dict(checkpoint["model"])
    refiner.eval()
    image_size = int(checkpoint.get("image_size", 512))

    image_dir = args.data_dir / ("test_images" if args.mode == "test" else "train_images")
    files = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpeg", ".jpg", ".png"}])
    annotation_path = args.annotation_path or (args.data_dir / "train_annotations.json")
    if args.mode == "val":
        names = set(time_val_names(annotation_path, {p.name for p in files}))
        files = [p for p in files if p.name in names]
    if args.limit:
        files = files[: args.limit]

    rows = []
    pq_scores = []
    mean_dice_scores = []
    total_tp = total_fp = total_fn = 0
    gt_by_file = load_gt_instances(annotation_path, {p.name for p in files}) if args.mode == "val" else {}

    for path in tqdm(files, desc=f"detect-refine-multiscale-{args.mode}"):
        image = read_gray_image(path)
        all_boxes = []
        all_scores = []
        for detector, detector_conf in zip(detectors, detector_confs):
            for size in args.imgsz:
                det = detector.predict(
                    source=str(path),
                    imgsz=int(size),
                    conf=float(detector_conf),
                    iou=args.iou,
                    device=0 if device.type == "cuda" else "cpu",
                    verbose=False,
                )[0]
                if det.boxes is None:
                    continue
                all_boxes.append(det.boxes.xyxy.detach().cpu().numpy())
                all_scores.append(det.boxes.conf.detach().cpu().numpy())
            if not args.tile_detector:
                tile_boxes, tile_scores = tiled_detect(detector, image, detector_conf, device, args)
                all_boxes.extend(tile_boxes)
                all_scores.extend(tile_scores)
        if args.tile_detector:
            for tile_detector, tile_conf in zip(tile_detectors, tile_detector_confs):
                tile_boxes, tile_scores = tiled_detect(tile_detector, image, tile_conf, device, args)
                all_boxes.extend(tile_boxes)
                all_scores.extend(tile_scores)

        boxes = np.concatenate(all_boxes, axis=0) if all_boxes else np.empty((0, 4), dtype=np.float32)
        scores = np.concatenate(all_scores, axis=0) if all_scores else np.empty((0,), dtype=np.float32)
        if len(boxes) and args.refiner_tta:
            pred_scored = refine_boxes_tta(refiner, image, boxes, scores, device, args, image_size)
        else:
            pred_scored = refine_boxes(refiner, image, boxes, scores, device, args, image_size) if len(boxes) else []
        pred_instances = mask_nms(pred_scored, args.mask_nms_iou)
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
                    "threshold": args.threshold,
                    "min_area": args.min_area,
                    "mask_nms_iou": args.mask_nms_iou,
                    "imgsz": " ".join(str(x) for x in args.imgsz),
                    "tile_size": int(args.tile_size),
                    "refiner_tta": bool(args.refiner_tta),
                    "score_mode": args.score_mode,
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
