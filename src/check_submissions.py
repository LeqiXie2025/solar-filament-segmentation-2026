"""Per-submission distribution audit.

For each submission CSV it reports, independent of any model:
  - total instance count (rows)
  - per-image instance count distribution (empty / fragmented images flagged)
  - RLE area distribution (pixels), via an in-house decode of our own
    ``encode_coco_counts`` format (pure run lengths, first run is a 0-run).

This is a diagnostic (public-metric probe), NOT a modeling direction. Use it
to spot extreme sparse/dense submissions before they silently drop on private LB.

Usage:
  python -m src.check_submissions \
      --test-dir data/raw/test_images \
      submissions/e5_t045_area500.csv submissions/e5_t045_area1000.csv ...
"""
import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from src.masks import binary_mask_to_counts, encode_coco_counts  # for self-test
import numpy as np


def decode_coco_counts(s: str) -> list[int]:
    """Reverse of ``encode_coco_counts`` (pycocotools-style LEB128 + delta undo)."""
    counts: list[int] = []
    i, n = 0, len(s)
    while i < n:
        x = 0
        k = 0
        more = True
        while more:
            c = ord(s[i]) - 48
            i += 1
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        counts.append(x)
    for idx in range(3, len(counts)):
        counts[idx] += counts[idx - 2]
    return counts


def rle_area(s: str) -> int:
    """Pixel area of an RLE produced by ``encode_binary_mask``.

    ``binary_mask_to_counts`` guarantees run lengths start with a 0-run, so the
    foreground (1) runs sit at odd indices.
    """
    counts = decode_coco_counts(s)
    return int(sum(counts[1::2]))


def _selftest() -> None:
    rng = np.random.RandomState(0)
    for _ in range(200):
        h, w = 32, 48
        mask = (rng.rand(h, w) > 0.85).astype(np.uint8)
        counts = binary_mask_to_counts(mask)
        enc = encode_coco_counts(counts)
        dec = decode_coco_counts(enc)
        assert dec == counts, f"round-trip mismatch:\n{counts}\n{dec}"
        assert rle_area(enc) == int(mask.sum()), f"area mismatch {rle_area(enc)} vs {int(mask.sum())}"
    print("[selftest] RLE decode + area OK (200 random masks)")


def audit(path: Path, total_test_images: int) -> dict:
    df = pd.read_csv(path)
    n_instances = len(df)

    ids = df["filament_id"].astype(str)
    # image_id is everything before the last underscore; idx is the trailing int
    image_ids = ids.str.rsplit("_", n=1).str[0]
    per_image = Counter(image_ids)

    n_images_with_pred = len(per_image)
    n_empty = total_test_images - n_images_with_pred
    counts = np.array(list(per_image.values()))
    hist = Counter()
    for c in counts:
        if c == 0:
            hist["0"] += 1
        elif c == 1:
            hist["1"] += 1
        elif 2 <= c <= 5:
            hist["2-5"] += 1
        elif 6 <= c <= 10:
            hist["6-10"] += 1
        elif 11 <= c <= 20:
            hist["11-20"] += 1
        else:
            hist["21+"] += 1

    areas = np.array([rle_area(str(s)) for s in df["segmentation_rle"]], dtype=np.int64)
    area_hist = Counter()
    for a in areas:
        if a < 50:
            area_hist["<50"] += 1
        elif a < 200:
            area_hist["50-200"] += 1
        elif a < 800:
            area_hist["200-800"] += 1
        elif a < 3000:
            area_hist["800-3k"] += 1
        elif a < 10000:
            area_hist["3k-10k"] += 1
        else:
            area_hist["10k+"] += 1

    fragmented = int((counts > 20).sum()) if counts.size else 0
    return {
        "file": path.name,
        "n_instances": n_instances,
        "n_images": n_images_with_pred,
        "n_empty_images": n_empty,
        "per_img_min": int(counts.min()) if counts.size else 0,
        "per_img_median": int(np.median(counts)) if counts.size else 0,
        "per_img_mean": round(float(counts.mean()), 2) if counts.size else 0.0,
        "per_img_max": int(counts.max()) if counts.size else 0,
        "per_img_hist": dict(sorted(hist.items())),
        "fragmented(>20)": fragmented,
        "area_min": int(areas.min()) if areas.size else 0,
        "area_median": int(np.median(areas)) if areas.size else 0,
        "area_mean": round(float(areas.mean()), 1) if areas.size else 0.0,
        "area_max": int(areas.max()) if areas.size else 0,
        "area_hist": dict(sorted(area_hist.items())),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("submissions", type=Path, nargs="*")
    ap.add_argument("--test-dir", type=Path, default=Path("data/raw/test_images"))
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("experiments/submission_distribution.md"))
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    total = len([p for p in args.test_dir.iterdir() if p.suffix.lower() in {".jpeg", ".jpg", ".png"}])
    print(f"test images on disk: {total}")

    rows = [audit(p, total) for p in args.submissions]

    lines = [
        "# 提交分布检查",
        "",
        f"日期：2026-08-24  测试图总数={total}",
        "",
        "| 提交 | 总实例 | 非空图 | 空图 | 每图min/med/mean/max | 碎图(>20) | 面积 min/med/mean/max |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['file']} | {r['n_instances']} | {r['n_images']} | {r['n_empty_images']} "
            f"| {r['per_img_min']}/{r['per_img_median']}/{r['per_img_mean']}/{r['per_img_max']} "
            f"| {r['fragmented(>20)']} "
            f"| {r['area_min']}/{r['area_median']}/{r['area_mean']}/{r['area_max']} |"
        )

    lines += ["", "## 每图实例数分布"]
    for r in rows:
        lines.append(f"- **{r['file']}**: {r['per_img_hist']}")

    lines += ["", "## RLE 面积分布（像素）"]
    for r in rows:
        lines.append(f"- **{r['file']}**: {r['area_hist']}")

    lines += ["", "## 结论提示", "",
              "- 空图多 = 过度稀疏（易掉私榜）。", "- 碎图多（单图实例 >20）= 过度切碎。",
              "- 面积集中在小值 = 大量碎屑（可能是噪声误检）。"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
