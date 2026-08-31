"""Locate a hung/slow validation image by timing the full per-image pipeline.

Runs the exact eval loop work (read -> predict -> detect disk -> postprocess ->
split -> decode GT -> greedy_match) one image at a time, printing file_name,
seconds, n_gt and n_pred so a pathological image is easy to spot.

Usage:
  python experiments/probe_slow_image.py --seed 2026 --limit 141
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import config
from src.dataset import random_split, read_gray_image
from src.evaluate_metric import _decode_ann_mask, load_val_annotations_lean
from src.masks import postprocess_binary_mask, split_instances
from src.metrics import greedy_match
from src.model_factory import create_model_from_checkpoint
from src.predict import sliding_window_predict
from src.preprocessing import detect_solar_disk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=config.MODEL_DIR / "smp_unet_resnet34_512_e5.pth")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--min-area", type=int, default=500)
    args = ap.parse_args()

    train_dir = config.DATA_DIR / "train_images"
    annotation_path = config.DATA_DIR / "train_annotations.json"
    image_info, raw_info, anns_by_file = load_val_annotations_lean(annotation_path)
    existing = {p.name for p in train_dir.iterdir() if p.suffix.lower() in {".jpeg", ".jpg", ".png"}}
    image_info = {k: v for k, v in image_info.items() if v["file_name"] in existing}
    _, val_ids = random_split(image_info, config.VAL_FRACTION, args.seed)
    if args.limit:
        val_ids = val_ids[: args.limit]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = create_model_from_checkpoint(ckpt).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    for i, file_name in enumerate(val_ids):
        t0 = time.time()
        info = image_info[file_name]
        image = read_gray_image(train_dir / info["file_name"])
        t_read = time.time() - t0

        probs = sliding_window_predict(model, image, device, tile_size=512, stride=256, batch_size=8)
        t_pred = time.time() - t0

        disk = detect_solar_disk(image)
        binary = (probs > args.threshold).astype("uint8")
        binary[~disk] = 0
        binary = postprocess_binary_mask(binary, "close")
        preds = split_instances(binary, args.min_area, method="connected")
        t_post = time.time() - t0

        gts = []
        for ann in anns_by_file.get(file_name, []):
            m = _decode_ann_mask(ann, raw_info)
            if m is None or m.sum() == 0:
                continue
            gts.append(m.astype("uint8"))
        matched, fp, fn = greedy_match(preds, gts, 0.0, "dice")
        t_match = time.time() - t0

        print(
            f"[{i:3d}] {file_name}  read={t_read:.2f}s pred={t_pred:.2f}s "
            f"post={t_post:.2f}s match={t_match:.2f}s  n_gt={len(gts)} n_pred={len(preds)}  "
            f"tp={len(matched)} fp={fp} fn={fn}",
            flush=True,
        )


if __name__ == "__main__":
    main()
