"""One-off data diagnostic. Prints annotation stats + image/disk geometry."""
import json
from pathlib import Path

import cv2
import numpy as np

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

print("=== raw/ top-level ===")
for p in sorted(RAW.iterdir()):
    if p.is_dir():
        n = len(list(p.iterdir()))
        print(f"  [dir ] {p.name}/  ({n} entries)")
    else:
        print(f"  [file] {p.name}")

ann_path = RAW / "train_annotations.json"
if not ann_path.exists():
    print("\n!! train_annotations.json NOT FOUND")
    raise SystemExit(0)

with open(ann_path, "r", encoding="utf-8") as f:
    data = json.load(f)

images = data["images"]
anns = data.get("annotations", [])
print("\n=== annotations ===")
print(f"  n_images     = {len(images)}")
print(f"  n_annotations= {len(anns)}")
print(f"  keys(images[0]) = {list(images[0].keys())}")
print(f"  sample image  = {images[0]}")

# image dims
dims = set((int(i["height"]), int(i["width"])) for i in images)
print(f"  distinct (H,W) sizes = {dims}")

# segmentation type
seg_types = {}
for a in anns:
    s = a.get("segmentation")
    t = "polygon" if isinstance(s, list) else ("rle" if isinstance(s, dict) else "other")
    seg_types[t] = seg_types.get(t, 0) + 1
print(f"  segmentation types = {seg_types}")

# instances per image
from collections import Counter
per_img = Counter()
for a in anns:
    per_img[a["image_id"]] += 1
has_ann = set(per_img.keys())
counts = [per_img[i["id"]] for i in images]
print(f"  images with >=1 instance = {len(has_ann)} / {len(images)}")
print(f"  instances/img: min={min(counts)} max={max(counts)} "
      f"median={np.median(counts):.0f} mean={np.mean(counts):.2f}")

# instance area (for polygons we need to rasterize; for RLE decode counts)
def mask_area(ann, h, w):
    s = ann.get("segmentation")
    if isinstance(s, list):
        m = np.zeros((h, w), np.uint8)
        for poly in s:
            pts = np.asarray(poly, np.float32).reshape(-1, 2)
            if len(pts) >= 3:
                cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
        return int(m.sum())
    if isinstance(s, dict):
        from pycocotools import mask as coco_mask
        rle = dict(s)
        if isinstance(rle.get("counts"), str):
            rle["counts"] = rle["counts"].encode("utf-8")
        if "size" not in rle:
            rle["size"] = [h, w]
        return int(coco_mask.decode(rle).sum())
    return 0

# area stats on a sample (first 200 anns)
areas = []
info_by_id = {i["id"]: i for i in images}
for a in anns[:500]:
    info = info_by_id[a["image_id"]]
    a_ = mask_area(a, int(info["height"]), int(info["width"]))
    if a_ > 0:
        areas.append(a_)
areas = np.array(areas)
print(f"\n=== instance area (px), first 500 anns, n={len(areas)} ===")
for q in [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]:
    print(f"  p{q:>3} = {np.percentile(areas, q):.0f}")

# disk geometry on a few real images
print("\n=== disk geometry ===")
train_dir = RAW / "train_images"
files = sorted(train_dir.glob("*.jpeg")) + sorted(train_dir.glob("*.jpg")) + sorted(train_dir.glob("*.png"))
for p in files[:3]:
    img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    # bright = disk (H-alpha disk is bright vs black sky). threshold at 5th percentile+delta
    bright = img > (img.min() + 0.25 * (img.max() - img.min()))
    ys, xs = np.where(bright)
    cy, cx = ys.mean(), xs.mean()
    # radius estimate: distance from center to ~bright edge
    dist = np.sqrt((xs - cx) ** 2 + (ys - cx) ** 2)  # note cx for both
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    r95 = np.percentile(dist, 98)
    print(f"  {p.name}: shape={img.shape} bright_center=({cy:.0f},{cx:.0f}) "
          f"assumed_center=({h/2:.0f},{w/2:.0f}) r98={r95:.0f} assumed_r={min(h,w)*0.5*0.98:.0f}")
