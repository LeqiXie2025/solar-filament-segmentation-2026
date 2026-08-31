import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import config
from src.crop_refiner_utils import crop_bounds_from_bbox, crop_with_padding, make_refiner_channels, seed_from_bbox
from src.dataset import read_gray_image
from src.instance_quality import (
    FEATURE_NAMES,
    instance_features,
    load_instance_quality_model,
    predict_instance_quality,
)
from src.masks import encode_binary_mask, postprocess_binary_mask
from src.metrics import greedy_match, mean_dice_matched, official_pq, official_pq_counts, overlap_matrix, panoptic_quality
from src.model_factory import create_model_from_checkpoint
from src.predict_detect_refine import make_masks_disjoint, mask_nms, patch_torchvision_nms, refine_boxes, time_val_names
from src.proposal_classifier import classify_boxes, load_proposal_classifier
from src.validation import load_gt_instances, load_gt_instances_by_annotator


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
    parser.add_argument(
        "--shape-refiner",
        type=Path,
        default=None,
        help="Optional consensus-trained refiner blended only after the primary refiner accepts a proposal.",
    )
    parser.add_argument("--shape-weight", type=float, default=0.25)
    parser.add_argument("--mobile-sam-fallback", action="store_true")
    parser.add_argument("--mobile-sam-root", type=Path, default=Path(r"D:\codex_models\MobileSAM"))
    parser.add_argument(
        "--mobile-sam-checkpoint",
        type=Path,
        default=Path(r"D:\codex_models\MobileSAM\weights\mobile_sam.pt"),
    )
    parser.add_argument("--mobile-sam-iou-threshold", type=float, default=0.80)
    parser.add_argument("--mobile-sam-box-pad", type=float, default=0.05)
    parser.add_argument("--mobile-sam-score-weight", type=float, default=0.25)
    parser.add_argument("--proposal-classifier", type=Path, default=None)
    parser.add_argument("--proposal-classifier-threshold", type=float, default=None)
    parser.add_argument("--proposal-classifier-score-weight", type=float, default=0.50)
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
    parser.add_argument(
        "--min-instance-score",
        type=float,
        default=0.0,
        help="Reject refined instances below the selected score-mode value before mask NMS.",
    )
    parser.add_argument("--instance-quality-model", type=Path, default=None)
    parser.add_argument("--instance-quality-threshold", type=float, default=None)
    parser.add_argument(
        "--save-val-predictions",
        type=Path,
        default=None,
        help="Save validation masks as submission-style RLE for detector-ensemble analysis.",
    )
    parser.add_argument(
        "--quality-dataset-output",
        type=Path,
        default=None,
        help="Save post-NMS validation instance features and official-IoU links for OOF quality training.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--official-metric", action="store_true")
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


def mask_nms_scored(instances: list[tuple[np.ndarray, float]], iou_threshold: float):
    if iou_threshold <= 0 or len(instances) < 2:
        return list(instances)
    scores = np.asarray([score for _instance, score in instances], dtype=np.float32)
    order = np.argsort(-scores)
    keep = []
    for idx in order:
        candidate = instances[int(idx)]
        if all(
            int(np.logical_and(candidate[0], kept[0]).sum())
            / max(int(np.logical_or(candidate[0], kept[0]).sum()), 1)
            < iou_threshold
            for kept in keep
        ):
            keep.append(candidate)
    return keep


def make_masks_disjoint_scored(instances: list[tuple[np.ndarray, float]], min_area: int):
    if not instances:
        return []
    claimed = np.zeros_like(instances[0][0], dtype=bool)
    disjoint = []
    for instance, score in instances:
        clean = (instance.astype(bool) & ~claimed).astype(np.uint8)
        if int(clean.sum()) >= min_area:
            disjoint.append((clean, float(score)))
            claimed |= clean.astype(bool)
    return disjoint


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
def mobile_sam_fallback_masks(predictor, image, boxes, scores, args, device):
    if predictor is None or not len(boxes):
        return []
    height, width = image.shape
    padded = boxes.astype(np.float32).copy()
    box_w = padded[:, 2] - padded[:, 0]
    box_h = padded[:, 3] - padded[:, 1]
    pad = float(args.mobile_sam_box_pad)
    padded[:, 0] = np.maximum(0.0, padded[:, 0] - box_w * pad)
    padded[:, 1] = np.maximum(0.0, padded[:, 1] - box_h * pad)
    padded[:, 2] = np.minimum(float(width), padded[:, 2] + box_w * pad)
    padded[:, 3] = np.minimum(float(height), padded[:, 3] + box_h * pad)
    rgb = cv2.cvtColor(np.clip(image * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    predictor.set_image(rgb)
    transformed = predictor.transform.apply_boxes_torch(torch.as_tensor(padded, device=device), rgb.shape[:2])
    masks, sam_scores, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed,
        multimask_output=False,
    )
    masks = masks[:, 0].detach().cpu().numpy().astype(np.uint8)
    sam_scores = sam_scores[:, 0].detach().cpu().numpy()
    fallback = []
    for mask, detector_score, sam_score in zip(masks, scores, sam_scores):
        if float(sam_score) < float(args.mobile_sam_iou_threshold) or int(mask.sum()) < args.min_area:
            continue
        score = float(detector_score) * float(sam_score) * float(args.mobile_sam_score_weight)
        fallback.append((mask, score))
    return fallback


@torch.no_grad()
def refine_boxes_tta(
    refiner,
    image: np.ndarray,
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    device,
    args,
    image_size: int,
    shape_refiner=None,
):
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
        shape_prob_sum = None
        for transform in transforms:
            transformed = _flip_tensor(batch, transform)
            prob = torch.sigmoid(refiner(transformed))[:, 0].detach().cpu().numpy()
            prob = _unflip_array(prob, transform)
            prob_sum = prob if prob_sum is None else prob_sum + prob
            if shape_refiner is not None:
                shape_prob = torch.sigmoid(shape_refiner(transformed))[:, 0].detach().cpu().numpy()
                shape_prob = _unflip_array(shape_prob, transform)
                shape_prob_sum = shape_prob if shape_prob_sum is None else shape_prob_sum + shape_prob
        primary_probs = prob_sum / float(len(transforms))
        if shape_prob_sum is None:
            probs = primary_probs
        else:
            shape_probs = shape_prob_sum / float(len(transforms))
            weight = float(args.shape_weight)
            probs = primary_probs * (1.0 - weight) + shape_probs * weight
        batch_scores = scores[start : start + args.batch_size]
        batch_primary_probs = primary_probs[: len(batch_scores)]
        for prob, primary_prob, bounds, score in zip(
            probs,
            batch_primary_probs,
            metas[start : start + args.batch_size],
            batch_scores,
        ):
            x0, y0, x1, y1 = bounds
            crop_h = max(1, y1 - y0)
            crop_w = max(1, x1 - x0)
            prob_full = cv2.resize(prob, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
            primary_prob_full = cv2.resize(primary_prob, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
            # The production refiner remains the recognition gate. The consensus
            # model may adjust boundaries but cannot turn a rejected box into a
            # new instance, which preserves the primary model's low FP rate.
            if int((primary_prob_full > args.threshold).sum()) < args.min_area:
                continue
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
    if args.shape_refiner is not None and not args.refiner_tta:
        raise ValueError("--shape-refiner currently requires --refiner-tta")
    if not 0.0 <= float(args.shape_weight) <= 1.0:
        raise ValueError("--shape-weight must be between 0 and 1")
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
    shape_refiner = None
    if args.shape_refiner is not None:
        shape_checkpoint = torch.load(args.shape_refiner, map_location=device)
        if int(shape_checkpoint.get("image_size", image_size)) != image_size:
            raise ValueError("Primary and shape refiners must use the same image size")
        shape_refiner = create_model_from_checkpoint(shape_checkpoint).to(device)
        shape_refiner.load_state_dict(shape_checkpoint["model"])
        shape_refiner.eval()
    mobile_sam_predictor = None
    if args.mobile_sam_fallback:
        sys.path.insert(0, str(args.mobile_sam_root.resolve()))
        from mobile_sam import SamPredictor, sam_model_registry

        mobile_sam = sam_model_registry["vit_t"](checkpoint=str(args.mobile_sam_checkpoint)).to(device)
        mobile_sam.eval()
        mobile_sam_predictor = SamPredictor(mobile_sam)
    proposal_classifier = None
    proposal_classifier_image_size = 0
    proposal_classifier_pad = 0.0
    proposal_classifier_threshold = 0.0
    if args.proposal_classifier is not None:
        proposal_checkpoint = torch.load(args.proposal_classifier, map_location="cpu")
        proposal_classifier_threshold = float(
            args.proposal_classifier_threshold
            if args.proposal_classifier_threshold is not None
            else proposal_checkpoint.get("threshold", 0.50)
        )
        proposal_classifier, proposal_classifier_image_size, proposal_classifier_pad = load_proposal_classifier(
            args.proposal_classifier, device
        )
    instance_quality = load_instance_quality_model(args.instance_quality_model) if args.instance_quality_model else None
    instance_quality_threshold = float(
        args.instance_quality_threshold
        if args.instance_quality_threshold is not None
        else (instance_quality.get("threshold", 0.50) if instance_quality else 0.0)
    )

    image_dir = args.data_dir / ("test_images" if args.mode == "test" else "train_images")
    files = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpeg", ".jpg", ".png"}])
    annotation_path = args.annotation_path or (args.data_dir / "train_annotations.json")
    if args.mode == "val":
        names = set(time_val_names(annotation_path, {p.name for p in files}))
        files = [p for p in files if p.name in names]
    if args.limit:
        files = files[: args.limit]

    rows = []
    val_prediction_rows = []
    pq_scores = []
    mean_dice_scores = []
    total_tp = total_fp = total_fn = 0
    official_iou_sum = 0.0
    quality_images = []
    if args.mode == "val" and args.official_metric:
        gt_by_file = load_gt_instances_by_annotator(annotation_path, {p.name for p in files})
    elif args.mode == "val":
        gt_by_file = load_gt_instances(annotation_path, {p.name for p in files})
    else:
        gt_by_file = {}

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
        if proposal_classifier is not None and len(boxes):
            classifier_scores = classify_boxes(
                proposal_classifier,
                image,
                boxes,
                device,
                proposal_classifier_image_size,
                proposal_classifier_pad,
                args.batch_size,
            )
            keep = classifier_scores >= proposal_classifier_threshold
            boxes = boxes[keep]
            detector_scores = scores[keep]
            classifier_scores = classifier_scores[keep]
            weight = float(args.proposal_classifier_score_weight)
            scores = detector_scores ** (1.0 - weight) * classifier_scores**weight
        if len(boxes) and args.refiner_tta:
            pred_scored = refine_boxes_tta(
                refiner,
                image,
                boxes,
                scores,
                device,
                args,
                image_size,
                shape_refiner=shape_refiner,
            )
        else:
            pred_scored = refine_boxes(refiner, image, boxes, scores, device, args, image_size) if len(boxes) else []
        pred_scored.extend(mobile_sam_fallback_masks(mobile_sam_predictor, image, boxes, scores, args, device))
        if args.min_instance_score > 0:
            pred_scored = [item for item in pred_scored if float(item[1]) >= args.min_instance_score]
        pred_scored = mask_nms_scored(pred_scored, args.mask_nms_iou)
        pred_scored = make_masks_disjoint_scored(pred_scored, args.min_area)
        quality_features = [instance_features(image, mask, score) for mask, score in pred_scored]
        if args.quality_dataset_output is not None:
            if args.mode != "val" or not args.official_metric:
                raise ValueError("--quality-dataset-output requires --mode val --official-metric")
            annotator_sets = gt_by_file.get(path.name, [])
            candidates = []
            for candidate_idx, ((mask, _score), features) in enumerate(zip(pred_scored, quality_features)):
                candidate_hits = []
                for gts in annotator_sets:
                    overlaps = overlap_matrix([mask], gts, mode="iou")[0] if gts else np.empty((0,))
                    candidate_hits.append(
                        [[int(gt_idx), float(iou)] for gt_idx, iou in enumerate(overlaps) if float(iou) > 0.5]
                    )
                candidates.append(
                    {
                        "candidate_index": candidate_idx,
                        "features": features.astype(float).tolist(),
                        "hits": candidate_hits,
                    }
                )
            quality_images.append(
                {
                    "file_name": path.name,
                    "gt_counts": [len(gts) for gts in annotator_sets],
                    "candidates": candidates,
                }
            )
        if instance_quality is not None and quality_features:
            quality_scores = predict_instance_quality(instance_quality, np.stack(quality_features))
            pred_scored = [
                item for item, quality_score in zip(pred_scored, quality_scores)
                if float(quality_score) >= instance_quality_threshold
            ]
        pred_instances = [mask for mask, _score in pred_scored]

        if args.mode == "val" and args.save_val_predictions is not None:
            for idx, instance in enumerate(pred_instances, start=1):
                val_prediction_rows.append(
                    {
                        "file_name": path.name,
                        "filament_id": f"{path.stem}_{idx}",
                        "segmentation_rle": encode_binary_mask(instance),
                    }
                )

        if args.mode == "val":
            if args.official_metric:
                for gts in gt_by_file.get(path.name, []):
                    iou_sum, tp, fp, fn = official_pq_counts(pred_instances, gts)
                    official_iou_sum += iou_sum
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn
            else:
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
                    "proposal_classifier_threshold": (
                        proposal_classifier_threshold if proposal_classifier is not None else None
                    ),
                    "refiner_tta": bool(args.refiner_tta),
                    "score_mode": args.score_mode,
                    "min_instance_score": float(args.min_instance_score),
                    "instance_quality_threshold": (
                        instance_quality_threshold if instance_quality is not None else None
                    ),
                    "shape_weight": float(args.shape_weight) if shape_refiner is not None else 0.0,
                    "mobile_sam_fallback": bool(args.mobile_sam_fallback),
                    "mobile_sam_iou_threshold": (
                        float(args.mobile_sam_iou_threshold) if args.mobile_sam_fallback else None
                    ),
                    "metric": "official_iou_pq" if args.official_metric else "legacy_dice_pq",
                    "pq": (
                        official_pq(official_iou_sum, total_tp, total_fp, total_fn)
                        if args.official_metric
                        else (float(np.mean(pq_scores)) if pq_scores else 0.0)
                    ),
                    "mean_dice_matched": (
                        None if args.official_metric else (float(np.mean(mean_dice_scores)) if mean_dice_scores else 0.0)
                    ),
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
    if args.quality_dataset_output is not None:
        args.quality_dataset_output.parent.mkdir(parents=True, exist_ok=True)
        args.quality_dataset_output.write_text(
            json.dumps({"feature_names": FEATURE_NAMES, "images": quality_images}, ensure_ascii=True),
            encoding="utf-8",
        )
        print(f"saved quality dataset={args.quality_dataset_output}")
    if args.save_val_predictions is not None:
        args.save_val_predictions.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            val_prediction_rows,
            columns=["file_name", "filament_id", "segmentation_rle"],
        ).to_csv(args.save_val_predictions, index=False)
        print(f"saved validation predictions={args.save_val_predictions}")


if __name__ == "__main__":
    main()
