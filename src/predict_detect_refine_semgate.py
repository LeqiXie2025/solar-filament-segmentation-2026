import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import config
from src.crop_refiner_utils import crop_bounds_from_bbox, crop_with_padding, seed_from_bbox
from src.dataset import read_gray_image
from src.masks import encode_binary_mask
from src.metrics import greedy_match, mean_dice_matched, panoptic_quality
from src.model_factory import create_model_from_checkpoint
from src.predict_detect_refine import (
    make_masks_disjoint,
    mask_nms,
    patch_torchvision_nms,
    refine_boxes,
    time_val_names,
)
from src.preprocessing import make_input_channels
from src.validation import load_gt_instances


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--refiner", type=Path, required=True)
    parser.add_argument("--semantic-checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=["val", "test"], default="val")
    parser.add_argument("--output", type=Path, default=config.SUBMISSION_DIR / "submission_detect_refine_semgate.csv")
    parser.add_argument("--conf", type=float, default=0.14)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--semantic-gate", type=float, default=0.16)
    parser.add_argument("--semantic-weight", type=float, default=0.35)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--min-area", type=int, default=160)
    parser.add_argument("--mask-nms-iou", type=float, default=0.35)
    parser.add_argument("--box-pad", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def load_model(path: Path, device):
    checkpoint = torch.load(path, map_location=device)
    model = create_model_from_checkpoint(checkpoint).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def semantic_support(
    model,
    image: np.ndarray,
    boxes_xyxy: np.ndarray,
    device,
    image_size: int,
    box_pad: float,
    batch_size: int,
) -> np.ndarray:
    height, width = image.shape
    tensors = []
    seeds = []
    for box in boxes_xyxy:
        x0, y0, x1, y1 = [float(v) for v in box[:4]]
        bbox = (x0, y0, max(1.0, x1 - x0), max(1.0, y1 - y0))
        bounds = crop_bounds_from_bbox(bbox, height, width, pad=box_pad, jitter=False)
        crop = crop_with_padding(image, bounds)
        crop_rs = cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        tensors.append(torch.from_numpy(make_input_channels(crop_rs)).float())
        seeds.append(seed_from_bbox(bbox, bounds, image_size).astype(bool))

    supports = []
    for start in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[start : start + batch_size]).to(device)
        probs = torch.sigmoid(model(batch))[:, 0].detach().cpu().numpy()
        for prob, seed in zip(probs, seeds[start : start + batch_size]):
            if seed.any():
                values = prob[seed]
            else:
                values = prob.reshape(-1)
            if values.size == 0:
                supports.append(0.0)
            else:
                supports.append(float(0.5 * values.mean() + 0.5 * np.percentile(values, 90)))
    return np.asarray(supports, dtype=np.float32)


def main() -> None:
    args = parse_args()
    patch_torchvision_nms()
    from ultralytics import YOLO

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = YOLO(str(args.detector))
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

    for path in tqdm(files, desc=f"detect-refine-semgate-{args.mode}"):
        image = read_gray_image(path)
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
        if len(boxes):
            supports = semantic_support(semantic_model, image, boxes, device, 512, args.box_pad, args.batch_size)
            keep = supports >= args.semantic_gate
            boxes = boxes[keep]
            scores = scores[keep] * (1.0 - args.semantic_weight + args.semantic_weight * supports[keep])

        pred_instances_scored = refine_boxes(refiner, image, boxes, scores, device, args, refiner_size) if len(boxes) else []
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
                    "conf": args.conf,
                    "semantic_gate": args.semantic_gate,
                    "semantic_weight": args.semantic_weight,
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
