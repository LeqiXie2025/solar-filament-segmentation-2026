from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import config
from src.dataset import read_gray_image
from src.masks import encode_binary_mask, postprocess_binary_mask
from src.metrics import official_pq, official_pq_counts
from src.model_factory import create_model_from_checkpoint
from src.obb_utils import (
    expand_obb,
    make_obb_refiner_channels,
    order_quad_long_edge_first,
    seed_from_inner_obb,
    unwarp_mask,
    warp_obb,
)
from src.predict_detect_refine import make_masks_disjoint, patch_torchvision_nms, time_val_names
from src.predict_detect_refine_multiscale import make_masks_disjoint_scored, mask_nms_scored
from src.validation import load_gt_instances_by_annotator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--annotation-path", type=Path, default=None)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--refiner", type=Path, required=True)
    parser.add_argument("--mode", choices=["val", "test"], default="val")
    parser.add_argument("--output", type=Path, default=config.SUBMISSION_DIR / "submission_obb_refine.csv")
    parser.add_argument("--conf", type=float, default=0.18)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--min-area", type=int, default=120)
    parser.add_argument("--mask-nms-iou", type=float, default=0.12)
    parser.add_argument("--pad-long", type=float, default=None)
    parser.add_argument("--pad-short", type=float, default=None)
    parser.add_argument("--imgsz", type=int, nargs="+", default=[1280])
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--refiner-tta", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save-val-predictions", type=Path, default=None)
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


def _unflip(probabilities: np.ndarray, mode: str) -> np.ndarray:
    if mode == "identity":
        return probabilities
    if mode == "hflip":
        return probabilities[:, :, ::-1]
    if mode == "vflip":
        return probabilities[:, ::-1, :]
    if mode == "hvflip":
        return probabilities[:, ::-1, ::-1]
    raise ValueError(mode)


@torch.no_grad()
def refine_obbs(refiner, image, quads, scores, device, args, checkpoint):
    height, width = image.shape
    crop_width = int(checkpoint.get("crop_width", 768))
    crop_height = int(checkpoint.get("crop_height", 256))
    pad_long = float(args.pad_long if args.pad_long is not None else checkpoint.get("pad_long", 0.18))
    pad_short = float(args.pad_short if args.pad_short is not None else checkpoint.get("pad_short", 0.65))
    tensors = []
    inverses = []
    for quad in quads:
        inner = order_quad_long_edge_first(np.asarray(quad, dtype=np.float32))
        crop_quad = expand_obb(inner, pad_long=pad_long, pad_short=pad_short, jitter=False)
        image_crop, inverse = warp_obb(image, crop_quad, crop_width, crop_height, cv2.INTER_LINEAR)
        seed = seed_from_inner_obb(inner, crop_quad, crop_width, crop_height)
        tensors.append(torch.from_numpy(make_obb_refiner_channels(image_crop, seed)).float())
        inverses.append(inverse)

    outputs = []
    transforms = ("identity", "hflip", "vflip", "hvflip") if args.refiner_tta else ("identity",)
    for start in range(0, len(tensors), args.batch_size):
        batch = torch.stack(tensors[start : start + args.batch_size]).to(device)
        probability_sum = None
        for transform in transforms:
            probabilities = torch.sigmoid(refiner(_flip_tensor(batch, transform)))[:, 0].detach().cpu().numpy()
            probabilities = _unflip(probabilities, transform)
            probability_sum = probabilities if probability_sum is None else probability_sum + probabilities
        probabilities = probability_sum / float(len(transforms))
        for probability, inverse, score in zip(
            probabilities,
            inverses[start : start + args.batch_size],
            scores[start : start + args.batch_size],
        ):
            binary = (probability > args.threshold).astype(np.uint8)
            binary = postprocess_binary_mask(binary, "close")
            if int(binary.sum()) == 0:
                continue
            instance = unwarp_mask(binary, inverse, width, height)
            if int(instance.sum()) >= args.min_area:
                outputs.append((instance.astype(np.uint8), float(score)))
    return outputs


def main() -> None:
    args = parse_args()
    patch_torchvision_nms()
    from ultralytics import YOLO

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = YOLO(str(args.detector))
    checkpoint = torch.load(args.refiner, map_location=device)
    refiner = create_model_from_checkpoint(checkpoint).to(device)
    refiner.load_state_dict(checkpoint["model"])
    refiner.eval()

    image_dir = args.data_dir / ("test_images" if args.mode == "test" else "train_images")
    files = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    annotation_path = args.annotation_path or (args.data_dir / "train_annotations.json")
    if args.mode == "val":
        val_names = set(time_val_names(annotation_path, {path.name for path in files}))
        files = [path for path in files if path.name in val_names]
    if args.limit:
        files = files[: args.limit]

    gt_by_file = load_gt_instances_by_annotator(annotation_path, {path.name for path in files}) if args.mode == "val" else {}
    rows = []
    val_rows = []
    total_iou = 0.0
    total_tp = total_fp = total_fn = 0
    for path in tqdm(files, desc=f"obb-refine-{args.mode}"):
        image = read_gray_image(path)
        all_quads = []
        all_scores = []
        for image_size in args.imgsz:
            result = detector.predict(
                source=str(path),
                imgsz=int(image_size),
                conf=float(args.conf),
                iou=float(args.iou),
                device=0 if device.type == "cuda" else "cpu",
                verbose=False,
            )[0]
            if result.obb is None or len(result.obb) == 0:
                continue
            all_quads.append(result.obb.xyxyxyxy.detach().cpu().numpy())
            all_scores.append(result.obb.conf.detach().cpu().numpy())
        quads = np.concatenate(all_quads, axis=0) if all_quads else np.empty((0, 4, 2), dtype=np.float32)
        scores = np.concatenate(all_scores, axis=0) if all_scores else np.empty((0,), dtype=np.float32)
        predictions = refine_obbs(refiner, image, quads, scores, device, args, checkpoint) if len(quads) else []
        predictions = mask_nms_scored(predictions, args.mask_nms_iou)
        predictions = make_masks_disjoint_scored(predictions, args.min_area)
        masks = [mask for mask, _ in predictions]

        if args.mode == "val":
            for ground_truth in gt_by_file.get(path.name, []):
                iou_sum, tp, fp, fn = official_pq_counts(masks, ground_truth)
                total_iou += iou_sum
                total_tp += tp
                total_fp += fp
                total_fn += fn
            if args.save_val_predictions is not None:
                for index, mask in enumerate(masks, start=1):
                    val_rows.append(
                        {
                            "file_name": path.name,
                            "filament_id": f"{path.stem}_{index}",
                            "segmentation_rle": encode_binary_mask(mask),
                        }
                    )
        else:
            masks = sorted(masks, key=lambda mask: int(mask.sum()), reverse=True)
            masks = make_masks_disjoint(masks, args.min_area)
            for index, mask in enumerate(masks, start=1):
                rows.append({"filament_id": f"{path.stem}_{index}", "segmentation_rle": encode_binary_mask(mask)})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "val":
        result = pd.DataFrame(
            [
                {
                    "conf": args.conf,
                    "threshold": args.threshold,
                    "min_area": args.min_area,
                    "mask_nms_iou": args.mask_nms_iou,
                    "imgsz": " ".join(str(size) for size in args.imgsz),
                    "refiner_tta": args.refiner_tta,
                    "pq": official_pq(total_iou, total_tp, total_fp, total_fn),
                    "tp": total_tp,
                    "fp": total_fp,
                    "fn": total_fn,
                    "images": len(files),
                }
            ]
        )
        result.to_csv(args.output, index=False)
        print(result.to_string(index=False))
        if args.save_val_predictions is not None:
            args.save_val_predictions.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(val_rows).to_csv(args.save_val_predictions, index=False)
    else:
        pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(args.output, index=False)
        print(f"saved {args.output} rows={len(rows)}")


if __name__ == "__main__":
    main()
