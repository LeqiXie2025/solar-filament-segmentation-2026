import argparse
from pathlib import Path

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
from src.predict_detect_refine import make_masks_disjoint, mask_nms, patch_torchvision_nms, refine_boxes, time_val_names
from src.preprocessing import detect_solar_disk
from src.validation import load_gt_instances


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--refiner", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--mode", choices=["val", "test"], default="val")
    parser.add_argument("--output", type=Path, default=config.SUBMISSION_DIR / "submission_hybrid_detect_refine.csv")
    parser.add_argument("--det-conf", type=float, default=0.16)
    parser.add_argument("--det-iou", type=float, default=0.45)
    parser.add_argument("--ref-threshold", type=float, default=0.45)
    parser.add_argument("--ref-min-area", type=int, default=80)
    parser.add_argument("--sem-threshold", type=float, default=0.55)
    parser.add_argument("--sem-min-area", type=int, default=500)
    parser.add_argument("--mask-nms-iou", type=float, default=0.35)
    parser.add_argument("--box-pad", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def load_checkpoint_model(path: Path, device):
    checkpoint = torch.load(path, map_location=device)
    model = create_model_from_checkpoint(checkpoint).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def semantic_instances(model, image: np.ndarray, device, args) -> list[np.ndarray]:
    probs = sliding_window_predict(
        model,
        image,
        device,
        tile_size=args.tile_size,
        stride=args.stride,
        batch_size=args.batch_size,
    )
    binary = (probs > args.sem_threshold).astype(np.uint8)
    binary[~detect_solar_disk(image)] = 0
    binary = postprocess_binary_mask(binary, "close")
    return split_instances(binary, args.sem_min_area, method="connected")


def main() -> None:
    args = parse_args()
    args.threshold = args.ref_threshold
    args.min_area = args.ref_min_area
    patch_torchvision_nms()
    from ultralytics import YOLO

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = YOLO(str(args.detector))
    refiner, refiner_ckpt = load_checkpoint_model(args.refiner, device)
    semantic, _ = load_checkpoint_model(args.semantic, device)
    refiner_size = int(refiner_ckpt.get("image_size", 512))

    image_dir = args.data_dir / ("test_images" if args.mode == "test" else "train_images")
    files = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpeg", ".jpg", ".png"}])
    if args.mode == "val":
        names = set(time_val_names(args.data_dir / "train_annotations.json", {p.name for p in files}))
        files = [p for p in files if p.name in names]
    if args.limit:
        files = files[: args.limit]

    gt_by_file = {}
    if args.mode == "val":
        gt_by_file = load_gt_instances(args.data_dir / "train_annotations.json", {p.name for p in files})

    rows = []
    pq_scores = []
    mean_dice_scores = []
    total_tp = total_fp = total_fn = 0
    for path in tqdm(files, desc=f"hybrid-{args.mode}"):
        image = read_gray_image(path)
        det = detector.predict(
            source=str(path),
            imgsz=args.imgsz,
            conf=args.det_conf,
            iou=args.det_iou,
            device=0 if device.type == "cuda" else "cpu",
            verbose=False,
        )[0]
        boxes = det.boxes.xyxy.detach().cpu().numpy() if det.boxes is not None else np.empty((0, 4), dtype=np.float32)
        scores = det.boxes.conf.detach().cpu().numpy() if det.boxes is not None else np.empty((0,), dtype=np.float32)
        refined = (
            refine_boxes(refiner, image, boxes, scores, device, args, refiner_size)
            if len(boxes)
            else []
        )
        semantic_scored = [(instance, 0.10) for instance in semantic_instances(semantic, image, device, args)]
        pred_instances = mask_nms(refined + semantic_scored, args.mask_nms_iou)
        pred_instances = make_masks_disjoint(pred_instances, args.ref_min_area)

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
            pred_instances = make_masks_disjoint(pred_instances, args.ref_min_area)
            for idx, instance in enumerate(pred_instances, start=1):
                rows.append({"filament_id": f"{image_id}_{idx}", "segmentation_rle": encode_binary_mask(instance)})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "val":
        out = pd.DataFrame(
            [
                {
                    "det_conf": args.det_conf,
                    "ref_threshold": args.ref_threshold,
                    "sem_threshold": args.sem_threshold,
                    "sem_min_area": args.sem_min_area,
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
