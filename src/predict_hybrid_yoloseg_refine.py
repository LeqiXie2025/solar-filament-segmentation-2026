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
from src.yolo_seg_inference import predict_yolo_seg_multiscale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--annotation-path", type=Path, default=None)
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
    parser.add_argument("--seg-imgsz", type=int, nargs="+", default=[1536])
    parser.add_argument("--seg-tile-size", type=int, default=0)
    parser.add_argument("--seg-tile-overlap", type=float, default=0.25)
    parser.add_argument("--seg-threshold", type=float, default=0.50)
    parser.add_argument("--seg-score-weight", type=float, default=0.65)
    parser.add_argument("--refine-seg-boxes", action="store_true")
    parser.add_argument("--seg-refine-agreement", type=float, default=0.0)
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
    annotation_path = args.annotation_path or (args.data_dir / "train_annotations.json")
    if args.mode == "val":
        names = set(time_val_names(annotation_path, {p.name for p in files}))
        files = [p for p in files if p.name in names]
    if args.limit:
        files = files[: args.limit]

    gt_by_file = load_gt_instances(annotation_path, {p.name for p in files}) if args.mode == "val" else {}
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

        seg_candidates = predict_yolo_seg_multiscale(
            seg_model,
            path,
            image,
            yolo_device,
            args.seg_imgsz,
            args.seg_conf,
            args.seg_iou,
            120,
            args.seg_threshold,
            args.seg_tile_size,
            args.seg_tile_overlap,
        )
        if args.refine_seg_boxes and seg_candidates:
            seg_boxes = []
            seg_scores = []
            for mask, score in seg_candidates:
                ys, xs = np.where(mask > 0)
                if len(xs) == 0:
                    continue
                seg_boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
                seg_scores.append(float(score) * args.seg_score_weight)
            if seg_boxes:
                refined_seg = refine_boxes(
                    refiner,
                    image,
                    np.asarray(seg_boxes, dtype=np.float32),
                    np.asarray(seg_scores, dtype=np.float32),
                    device,
                    args,
                    image_size,
                )
                if args.seg_refine_agreement > 0:
                    raw_masks = [mask.astype(bool) for mask, _ in seg_candidates]
                    for refined_mask, score in refined_seg:
                        refined_bool = refined_mask.astype(bool)
                        refined_area = int(refined_bool.sum())
                        best_iou = 0.0
                        for raw_mask in raw_masks:
                            inter = int(np.logical_and(refined_bool, raw_mask).sum())
                            if inter == 0:
                                continue
                            union = refined_area + int(raw_mask.sum()) - inter
                            best_iou = max(best_iou, inter / max(union, 1))
                        if best_iou >= args.seg_refine_agreement:
                            scored.append((refined_mask, float(score) * best_iou))
                else:
                    scored.extend(refined_seg)
        else:
            scored.extend(
                (mask, score * args.seg_score_weight)
                for mask, score in seg_candidates
                if int(mask.sum()) >= args.min_area
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
