import argparse
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--detector", type=Path, required=True)
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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


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
    image_size = int(checkpoint.get("image_size", 512))

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

    for path in tqdm(files, desc=f"detect-refine-multiscale-{args.mode}"):
        image = read_gray_image(path)
        all_boxes = []
        all_scores = []
        for size in args.imgsz:
            det = detector.predict(
                source=str(path),
                imgsz=int(size),
                conf=args.conf,
                iou=args.iou,
                device=0 if device.type == "cuda" else "cpu",
                verbose=False,
            )[0]
            if det.boxes is None:
                continue
            all_boxes.append(det.boxes.xyxy.detach().cpu().numpy())
            all_scores.append(det.boxes.conf.detach().cpu().numpy())

        boxes = np.concatenate(all_boxes, axis=0) if all_boxes else np.empty((0, 4), dtype=np.float32)
        scores = np.concatenate(all_scores, axis=0) if all_scores else np.empty((0,), dtype=np.float32)
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
