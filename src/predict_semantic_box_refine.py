import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import config
from src.dataset import read_gray_image
from src.masks import encode_binary_mask, postprocess_binary_mask, split_instances
from src.metrics import greedy_match, mean_dice_matched, panoptic_quality
from src.model_factory import create_model_from_checkpoint
from src.predict import sliding_window_predict
from src.predict_detect_refine import (
    make_masks_disjoint,
    mask_nms,
    patch_torchvision_nms,
    refine_boxes,
    time_val_names,
)
from src.preprocessing import detect_solar_disk
from src.validation import load_gt_instances


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--detector", type=Path, default=None)
    parser.add_argument("--semantic-checkpoint", type=Path, required=True)
    parser.add_argument("--refiner", type=Path, required=True)
    parser.add_argument("--mode", choices=["val", "test"], default="val")
    parser.add_argument("--output", type=Path, default=config.SUBMISSION_DIR / "submission_semantic_box_refine.csv")
    parser.add_argument("--yolo-conf", type=float, default=0.18)
    parser.add_argument("--yolo-iou", type=float, default=0.45)
    parser.add_argument("--semantic-threshold", type=float, default=0.30)
    parser.add_argument("--semantic-min-area", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--min-area", type=int, default=160)
    parser.add_argument("--mask-nms-iou", type=float, default=0.35)
    parser.add_argument("--box-pad", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def load_model(path: Path, device):
    checkpoint = torch.load(path, map_location=device)
    model = create_model_from_checkpoint(checkpoint).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def instances_to_boxes(instances: list[np.ndarray], prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    boxes = []
    scores = []
    for instance in instances:
        ys, xs = np.where(instance > 0)
        if len(xs) == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        boxes.append([x0, y0, x1, y1])
        scores.append(float(prob[instance.astype(bool)].mean()))
    if not boxes:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return np.asarray(boxes, dtype=np.float32), np.asarray(scores, dtype=np.float32)


def main() -> None:
    args = parse_args()
    patch_torchvision_nms()
    from ultralytics import YOLO

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = YOLO(str(args.detector)) if args.detector is not None else None
    semantic_model, _ = load_model(args.semantic_checkpoint, device)
    refiner_ckpt = torch.load(args.refiner, map_location=device)
    refiner = create_model_from_checkpoint(refiner_ckpt).to(device)
    refiner.load_state_dict(refiner_ckpt["model"])
    refiner.eval()
    refiner_size = int(refiner_ckpt.get("image_size", 512))

    image_dir = args.data_dir / ("test_images" if args.mode == "test" else "train_images")
    files = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpeg", ".jpg", ".png"}])
    if args.mode == "val":
        names = set(time_val_names(args.data_dir / "train_annotations.json", {p.name for p in files}))
        files = [p for p in files if p.name in names]
    if args.limit:
        files = files[: args.limit]

    rows = []
    pq_scores = []
    mean_dice_scores = []
    total_tp = total_fp = total_fn = 0
    gt_by_file = load_gt_instances(args.data_dir / "train_annotations.json", {p.name for p in files}) if args.mode == "val" else {}

    for path in tqdm(files, desc=f"semantic-box-refine-{args.mode}"):
        image = read_gray_image(path)
        semantic_prob = sliding_window_predict(
            semantic_model,
            image,
            device,
            tile_size=args.tile_size,
            stride=args.stride,
            batch_size=args.batch_size,
            tta=False,
        )
        binary = (semantic_prob > args.semantic_threshold).astype(np.uint8)
        binary[~detect_solar_disk(image)] = 0
        binary = postprocess_binary_mask(binary, "close")
        sem_instances = split_instances(binary, args.semantic_min_area)
        sem_boxes, sem_scores = instances_to_boxes(sem_instances, semantic_prob)

        all_boxes = sem_boxes
        all_scores = sem_scores
        if detector is not None:
            det = detector.predict(
                source=str(path),
                imgsz=args.imgsz,
                conf=args.yolo_conf,
                iou=args.yolo_iou,
                device=0 if device.type == "cuda" else "cpu",
                verbose=False,
            )[0]
            yolo_boxes = det.boxes.xyxy.detach().cpu().numpy() if det.boxes is not None else np.empty((0, 4), dtype=np.float32)
            yolo_scores = det.boxes.conf.detach().cpu().numpy() if det.boxes is not None else np.empty((0,), dtype=np.float32)
            all_boxes = np.concatenate([yolo_boxes, sem_boxes], axis=0) if len(yolo_boxes) or len(sem_boxes) else all_boxes
            all_scores = np.concatenate([yolo_scores, sem_scores], axis=0) if len(yolo_scores) or len(sem_scores) else all_scores

        pred_instances_scored = refine_boxes(refiner, image, all_boxes, all_scores, device, args, refiner_size) if len(all_boxes) else []
        pred_instances = mask_nms(pred_instances_scored, args.mask_nms_iou)
        pred_instances = make_masks_disjoint(pred_instances, args.min_area)

        if args.mode == "val":
            gt_instances = gt_by_file.get(path.name, [])
            matched, fp, fn = greedy_match(pred_instances, gt_instances, overlap_threshold=0.0, mode="dice")
            pq_scores.append(panoptic_quality(matched, len(pred_instances), len(gt_instances), fp, fn))
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
                    "yolo_conf": args.yolo_conf,
                    "semantic_threshold": args.semantic_threshold,
                    "threshold": args.threshold,
                    "min_area": args.min_area,
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
