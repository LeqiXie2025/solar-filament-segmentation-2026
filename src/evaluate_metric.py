"""Faithful local validation mirroring the official instance-matching metric.

Replaces the misleading patch-level Dice (train.py) and the old time-split
evaluate_full_image / evaluate_instances scripts.

Key correctness fixes vs. the earlier version:
  - ``--split time|random`` + ``--seed`` let you evaluate a checkpoint with the
    SAME split it was trained with (the old code hardcoded random_split(42),
    which silently contaminated any time-split checkpoint like e5).
  - Memory-lean: GT masks are decoded one image at a time and released, instead
    of ``load_coco_binary_masks`` (707 full masks) + ``load_gt_instances``
    (all val instance masks) being held in RAM at once (previously >8 GB peak).

For each validation image it computes the probability map once, then sweeps
threshold x min_area and reports three metrics per combo:

- full_image_dice   : semantic union Dice
- mean_dice_matched : replica of the buggy public metric (empty -> 1.0)
- panoptic_quality  : honest instance metric (empty -> 0.0)
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import config
from src.dataset import random_split, read_gray_image, sorted_time_split
from src.masks import (
    _decode_coco_rle,
    _decode_polygon,
    postprocess_binary_mask,
    split_instances,
)
from src.metrics import full_image_dice, greedy_match, mean_dice_matched, panoptic_quality
from src.model_factory import create_model_from_checkpoint
from src.predict import sliding_window_predict
from src.preprocessing import detect_solar_disk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, default=config.MODEL_DIR / "unet_baseline_best.pth")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Eval only first N val images.")
    parser.add_argument("--start", type=int, default=0, help="Skip first N val images (for chunked runs).")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--split", choices=["random", "time"], default="random",
                        help="Must match the split the checkpoint was trained with.")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.35, 0.40, 0.45, 0.50, 0.55])
    parser.add_argument("--min-areas", type=int, nargs="+", default=[80, 250, 500, 1000, 2000])
    parser.add_argument("--morphology", choices=["none", "close", "openclose"], default="close")
    parser.add_argument("--instance-method", choices=["connected", "watershed"], default="connected")
    parser.add_argument("--tta", choices=["none", "hflip", "hvflip"], default="none")
    parser.add_argument("--out", type=Path, default=None, help="Optional CSV of the full sweep.")
    return parser.parse_args()


def load_val_annotations_lean(annotation_path: Path) -> tuple[dict, dict, dict]:
    """Load metadata and group annotations WITHOUT decoding any mask upfront.

    Returns (image_info, raw_info, anns_by_file):
      - image_info   : file_name -> metadata (duplicate file records merged)
      - raw_info     : image_id  -> metadata (for per-annotation height/width)
      - anns_by_file : file_name -> list of undecoded annotation dicts
    """
    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw_info = {img["id"]: img for img in data["images"]}
    image_info: dict = {}
    for image_id, info in raw_info.items():
        image_info.setdefault(info["file_name"], dict(info))
    anns_by_file: dict[str, list] = defaultdict(list)
    for ann in data.get("annotations", []):
        info = raw_info.get(ann["image_id"])
        if info is not None:
            anns_by_file[info["file_name"]].append(ann)
    return image_info, raw_info, anns_by_file


def _decode_ann_mask(ann: dict, raw_info: dict) -> np.ndarray | None:
    info = raw_info[ann["image_id"]]
    height, width = int(info["height"]), int(info["width"])
    seg = ann.get("segmentation")
    if isinstance(seg, list):
        return _decode_polygon(seg, height, width)
    if isinstance(seg, dict):
        return _decode_coco_rle(seg, height, width)
    return None


def _write_sweep_csv(out_path: Path, agg: dict, note: str = "") -> None:
    """Write the accumulated sweep to CSV so partial results survive a crash."""
    import pandas as pd

    rows = []
    for (t, a), r in agg.items():
        rows.append({
            "threshold": t, "min_area": a,
            "full_dice": np.mean(r["full"]), "mean_dice_matched": np.mean(r["mdc"]),
            "pq": np.mean(r["pq"]), "tp": r["tp"], "fp": r["fp"], "fn": r["fn"],
        })
    pd.DataFrame(rows).sort_values("pq", ascending=False).to_csv(out_path, index=False)
    print(f"saved sweep to {out_path} {note}".rstrip())


@torch.no_grad()
def main() -> None:
    args = parse_args()
    train_dir = args.data_dir / "train_images"
    annotation_path = args.data_dir / "train_annotations.json"

    image_info, raw_info, anns_by_file = load_val_annotations_lean(annotation_path)
    existing = {p.name for p in train_dir.iterdir() if p.suffix.lower() in {".jpeg", ".jpg", ".png"}}
    image_info = {k: v for k, v in image_info.items() if v["file_name"] in existing}

    if args.split == "random":
        _, val_ids = random_split(image_info, config.VAL_FRACTION, args.seed)
    else:
        _, val_ids = sorted_time_split(image_info, config.VAL_FRACTION)
    if args.start:
        val_ids = val_ids[args.start:]
    if args.limit:
        val_ids = val_ids[: args.limit]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = create_model_from_checkpoint(checkpoint).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    combos = [(t, a) for t in args.thresholds for a in args.min_areas]
    agg = {
        (t, a): {"full": [], "mdc": [], "pq": [], "tp": 0, "fp": 0, "fn": 0, "pred": []}
        for t, a in combos
    }

    for i, file_name in enumerate(tqdm(val_ids, desc="metric-val")):
        info = image_info[file_name]
        image = read_gray_image(train_dir / info["file_name"])

        # Decode GT for THIS image only (union + instances), released next iter.
        height, width = int(info["height"]), int(info["width"])
        gt_instances: list[np.ndarray] = []
        gt_union: np.ndarray | None = None
        for ann in anns_by_file.get(file_name, []):
            m = _decode_ann_mask(ann, raw_info)
            if m is None or m.sum() == 0:
                continue
            gt_instances.append(m.astype(np.uint8))
            gt_union = m if gt_union is None else np.maximum(gt_union, m)
        if gt_union is None:
            gt_union = np.zeros((height, width), dtype=np.uint8)

        probs = sliding_window_predict(
            model, image, device, tile_size=args.tile_size, stride=args.stride,
            batch_size=args.batch_size, tta=args.tta,
        )
        disk = detect_solar_disk(image)

        for threshold, min_area in combos:
            binary = (probs > threshold).astype(np.uint8)
            binary[~disk] = 0
            binary = postprocess_binary_mask(binary, args.morphology)
            pred_instances = split_instances(
                binary, min_area, method=args.instance_method,
                watershed_peak_rel=0.45, watershed_peak_abs=3.0, watershed_min_component_area=None,
            )
            matched, fp, fn = greedy_match(pred_instances, gt_instances, overlap_threshold=0.0, mode="dice")
            row = agg[(threshold, min_area)]
            row["full"].append(full_image_dice(binary, gt_union))
            row["mdc"].append(mean_dice_matched(matched))
            row["pq"].append(panoptic_quality(matched, len(pred_instances), len(gt_instances), fp, fn))
            row["tp"] += len(matched)
            row["fp"] += fp
            row["fn"] += fn
            row["pred"].append(len(pred_instances))

        if args.out and (i + 1) % 10 == 0:
            _write_sweep_csv(args.out, agg, note=f"[partial {i + 1}/{len(val_ids)}]")

    print(f"images={len(val_ids)}  split={args.split}  seed={args.seed}  "
          f"checkpoint={args.checkpoint.name}  (stored patch val_dice={checkpoint.get('val_dice')})")

    # rank by each metric, print top 5 for each
    for metric, label in [("full", "full_image_dice"), ("mdc", "mean_dice_matched(buggy)"), ("pq", "panoptic_quality")]:
        ranked = sorted(((np.mean(r[metric]), t, a, r) for (t, a), r in agg.items()), reverse=True)
        print(f"\n== top-5 by {label} ==")
        for mean, t, a, r in ranked[:5]:
            print(
                f"  thr={t:.2f} min_area={a:5d}  {label}={mean:.4f}  "
                f"tp={r['tp']} fp={r['fp']} fn={r['fn']}  median_pred/img={np.median(r['pred']):.0f}"
            )

    if args.out:
        _write_sweep_csv(args.out, agg)


if __name__ == "__main__":
    main()
