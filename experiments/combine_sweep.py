"""Combine chunked sweep JSONs into one aggregated table.

Each JSON holds *additive* accumulators per conf (tp/fp/fn/n_pred, PQ num/den,
grouped-dice sum). Aggregation = sum across files; PQ = sum(num)/sum(den),
grouped dice = sum(grouped_sum)/sum(n_images).

Usage: python experiments/combine_sweep.py experiments/sweep_s42_0_70.json experiments/sweep_s42_70_141.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main():
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print("pass one or more sweep JSON paths")
        return
    acc = {}
    n_images = 0
    meta = {}
    for p in paths:
        data = json.loads(p.read_text(encoding="utf-8"))
        meta = {k: data[k] for k in ("seed", "imgsz", "threshold", "min_area", "mask_nms_iou") if k in data}
        n_images += data["n_images"]
        for c, v in data["confs"].items():
            a = acc.setdefault(float(c), {"tp": 0, "fp": 0, "fn": 0, "n_pred": 0,
                                          "pq_iou_num": 0.0, "pq_iou_den": 0.0,
                                          "pq_dice_num": 0.0, "pq_dice_den": 0.0, "grouped_sum": 0.0})
            for k in a:
                a[k] += v[k]
    print("=" * 78)
    print(f"combined {len(paths)} chunks  seed={meta.get('seed')}  images={n_images}  "
          f"thr={meta.get('threshold')} min_area={meta.get('min_area')} nms={meta.get('mask_nms_iou')}")
    print(f"{'conf':>6} | {'PQ_iou05':>9} {'PQ_dice0':>9} {'grp_dice':>8} | {'tp':>5} {'fp':>6} {'fn':>5} {'pred':>6}")
    print("-" * 78)
    for c in sorted(acc):
        a = acc[c]
        pq_iou = a["pq_iou_num"] / max(a["pq_iou_den"], 1e-6)
        pq_dice = a["pq_dice_num"] / max(a["pq_dice_den"], 1e-6)
        grp = a["grouped_sum"] / max(n_images, 1)
        print(f"{c:>6.2f} | {pq_iou:>9.4f} {pq_dice:>9.4f} {grp:>8.4f} | "
              f"{a['tp']:>5} {a['fp']:>6} {a['fn']:>5} {a['n_pred']:>6}")
    print("=" * 78)


if __name__ == "__main__":
    main()
