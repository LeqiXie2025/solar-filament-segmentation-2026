"""One-off: check date coverage of local train vs full annotations vs test."""
import json
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

def stem_date(stem: str) -> str:
    # filenames like 20110109104734Ch / 20140609195854Bh -> YYYYMMDD
    return stem[:8]

def summarize(name, dates):
    dates = sorted(dates)
    if not dates:
        print(f"  {name}: (empty)")
        return
    print(f"  {name}: n={len(dates)} min={dates[0]} max={dates[-1]}")

train_dir = RAW / "train_images"
test_dir = RAW / "test_images"
local_train = [stem_date(p.stem) for p in train_dir.iterdir()
               if p.suffix.lower() in {".jpeg", ".jpg", ".png"}]
test = [stem_date(p.stem) for p in test_dir.iterdir()
        if p.suffix.lower() in {".jpeg", ".jpg", ".png"}]

ann_path = RAW / "train_annotations.json"
data = json.load(open(ann_path, "r", encoding="utf-8"))
all_ann = [stem_date(img["file_name"]) for img in data["images"]]

print("=== date coverage ===")
summarize("local train_images", local_train)
summarize("full annotations (1154)", all_ann)
summarize("test_images", test)

# year distribution
from collections import Counter
print("\n=== year histogram (local train) ===")
for y, c in sorted(Counter(d[:4] for d in local_train).items()):
    print(f"  {y}: {c}")
print("=== year histogram (full annotations) ===")
for y, c in sorted(Counter(d[:4] for d in all_ann).items()):
    print(f"  {y}: {c}")
print("=== year histogram (test) ===")
for y, c in sorted(Counter(d[:4] for d in test).items()):
    print(f"  {y}: {c}")

# overlap between local train and test names
lt = {p.stem for p in train_dir.iterdir() if p.suffix.lower() in {".jpeg", ".jpg", ".png"}}
tt = {p.stem for p in test_dir.iterdir() if p.suffix.lower() in {".jpeg", ".jpg", ".png"}}
print(f"\n=== overlap local_train vs test = {len(lt & tt)} ===")

# what's in _download_tmp_parallel
tmp = RAW / "_download_tmp_parallel"
tmp_files = list(tmp.iterdir()) if tmp.exists() else []
print(f"\n=== _download_tmp_parallel: {len(tmp_files)} entries, sample ===")
for p in tmp_files[:5]:
    print(f"  {p.name}")
