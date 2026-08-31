"""Area-binned recall comparison: did the new model improve SMALL filaments?

Directive point 4 asks, for old (e5) vs new (dicebce_stable_e3):
  - recall (overall), fp, and specifically "小暗条有没有改善".

This script bins every GT filament by its pixel area and reports, per bin,
how many were matched (recall) by each model at the SAME operating point
(thr=0.45, min_area=500, time split). It also bins unmatched predictions (fp)
by their area so a precision/recall trade-off per size is visible.

Note: min_area=500 in prediction means any GT instance < 500 px is
*unreachable* (guaranteed FN) for BOTH models — the "<500" bin isolates that
ceiling from genuine model recall differences.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

import config
from src.dataset import read_gray_image, sorted_time_split
from src.evaluate_metric import _decode_ann_mask, load_val_annotations_lean
from src.masks import postprocess_binary_mask, split_instances
from src.metrics import greedy_match
from src.model_factory import create_model_from_checkpoint
from src.predict import sliding_window_predict
from src.preprocessing import detect_solar_disk

BINS = [(0, 500), (500, 1500), (1500, 4000), (4000, 10**9)]
BIN_LABELS = ["<500", "500-1500", "1500-4000", ">=4000"]


def bin_of(area: int) -> int:
    for k, (lo, hi) in enumerate(BINS):
        if lo <= area < hi:
            return k
    return len(BINS) - 1


def evaluate_area_bins(checkpoint_path: Path, val_ids, image_info, raw_info, anns_by_file, train_dir) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(checkpoint_path, map_location=device)
    model = create_model_from_checkpoint(ck).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    gt_total = [0] * len(BINS)
    gt_matched = [0] * len(BINS)
    fp_total = [0] * len(BINS)

    for file_name in val_ids:
        info = image_info[file_name]
        image = read_gray_image(train_dir / info["file_name"])

        gts, gt_areas = [], []
        for ann in anns_by_file.get(file_name, []):
            m = _decode_ann_mask(ann, raw_info)
            if m is None or m.sum() == 0:
                continue
            m = m.astype(np.uint8)
            gts.append(m)
            gt_areas.append(int(m.sum()))

        probs = sliding_window_predict(model, image, device, tile_size=512, stride=256, batch_size=8)
        disk = detect_solar_disk(image)
        binary = (probs > 0.45).astype(np.uint8)
        binary[~disk] = 0
        binary = postprocess_binary_mask(binary, "close")
        preds = split_instances(binary, 500, method="connected")

        matched, _, _ = greedy_match(preds, gts, 0.0, "dice")
        matched_gt = {g for _, g, _ in matched}
        matched_pred = {p for p, _, _ in matched}

        for gi, a in enumerate(gt_areas):
            gt_total[bin_of(a)] += 1
            if gi in matched_gt:
                gt_matched[bin_of(a)] += 1

        for pi, p in enumerate(preds):
            if pi not in matched_pred:
                fp_total[bin_of(int(p.sum()))] += 1

    return {"gt_total": gt_total, "gt_matched": gt_matched, "fp_total": fp_total}


def main() -> None:
    train_dir = config.DATA_DIR / "train_images"
    annotation_path = config.DATA_DIR / "train_annotations.json"
    image_info, raw_info, anns_by_file = load_val_annotations_lean(annotation_path)
    existing = {p.name for p in train_dir.iterdir() if p.suffix.lower() in {".jpeg", ".jpg", ".png"}}
    image_info = {k: v for k, v in image_info.items() if v["file_name"] in existing}
    _, val_ids = sorted_time_split(image_info, config.VAL_FRACTION)

    models = {
        "e5 (old)": config.MODEL_DIR / "smp_unet_resnet34_512_e5.pth",
        "stable_e3 (new)": config.MODEL_DIR / "smp_unet_resnet34_512_dicebce_stable_e3.pth",
    }

    results = {}
    for name, ck in models.items():
        t0 = time.time()
        results[name] = evaluate_area_bins(ck, val_ids, image_info, raw_info, anns_by_file, train_dir)
        print(f"[{name}] done in {time.time() - t0:.0f}s", flush=True)

    print("\nGT recall by area bin (time split, thr=0.45 min_area=500):\n")
    print(f"{'bin':<12} {'#GT':>6} | " + " | ".join(f"{n:>22}" for n in results) + "  (recall / fp)")
    print("-" * 100)
    for k, label in enumerate(BIN_LABELS):
        cells = []
        for name, r in results.items():
            tot = r["gt_total"][k]
            mat = r["gt_matched"][k]
            rec = mat / tot if tot else 0.0
            cells.append(f"{mat}/{tot} ({rec*100:4.1f}%) fp={r['fp_total'][k]}")
        print(f"{label:<12} {results['e5 (old)']['gt_total'][k]:>6} | " + " | ".join(f"{c:>22}" for c in cells))

    print()
    for name, r in results.items():
        tot = sum(r["gt_total"])
        mat = sum(r["gt_matched"])
        fpt = sum(r["fp_total"])
        print(f"{name}: overall recall={mat}/{tot} ({mat/tot*100:.1f}%)  total_fp={fpt}")


if __name__ == "__main__":
    main()
