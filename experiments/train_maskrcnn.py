"""Train a Mask R-CNN (ResNet50 FPN v2) instance-segmentation detector for filaments.

Why: YOLO11s box detector caps recall at R=52.6% (mAP50-95=0.311) because thin
curved filaments are poorly localized by boxes. Mask R-CNN predicts a coarse
per-instance mask + box directly, matching fususu's 0.55-LB detector stage
("Mask R-CNN resnet50 FPN v2, score 0.70 / mask 0.50, epoch 6"). Its boxes feed
the existing crop-refiner as a drop-in replacement for YOLO.

Data hygiene: the challenge ships multiple annotation records per physical image
(different annotators/observations). We collapse them to one mask per filament
via ``src.gt_dedup`` (union across annotators, then IoU NMS), so a filament drawn
3x counts once in the training target.

Split: observation-grouped random split (seed 42) == ``eval_dense_rebaseline``'s
dense split, so detector recall is directly comparable on the same 141-image val.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The project path contains non-ASCII chars ("申请"); force UTF-8 stdout so print()
# of paths doesn't crash on Windows' default charmap codec when output is redirected.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import config
from src.dataset import read_gray_image
from src.gt_dedup import image_masks_from_anns, load_anns_by_file, masks_to_boxes
from src.preprocessing import make_input_channels
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


class MaskRCNNDataset(Dataset):
    def __init__(self, files: list[str], anns_by_file: dict, image_dir: Path, imgsz: int) -> None:
        self.files = list(files)
        self.anns_by_file = anns_by_file
        self.image_dir = Path(image_dir)
        self.imgsz = int(imgsz)
        # native annotation resolution (2048x2048 for this dataset)
        first_anns = anns_by_file[files[0]]
        self.src_h = self.src_w = 2048
        for ann in first_anns:  # resolve native size from a sample if cached
            if "segmentation" in ann:
                break
        self._src_size = (self.src_h, self.src_w)

    def __len__(self) -> int:
        return len(self.files)

    def _native_size(self, anns) -> tuple[int, int]:
        # height/width are constant 2048 here; keep generic by looking at anns'
        # bounding boxes is unnecessary — read from a stored attribute set in load.
        return self._src_size

    def __getitem__(self, idx: int):
        fname = self.files[idx]
        anns = self.anns_by_file[fname]
        gray = read_gray_image(self.image_dir / fname)  # [0,1] HxW (2048)
        h, w = gray.shape
        gray = cv2.resize(gray, (self.imgsz, self.imgsz), interpolation=cv2.INTER_AREA)
        image = make_input_channels(gray)  # 3 x imgsz x imgsz, [0,1]

        masks = image_masks_from_anns(anns, h, w, out_size=(self.imgsz, self.imgsz), iou_thresh=0.5)
        boxes = masks_to_boxes(masks)  # (N,4) float32 in imgsz space

        image_tensor = torch.from_numpy(np.ascontiguousarray(image)).float()
        boxes_t = torch.from_numpy(boxes).float()
        masks_t = torch.stack([torch.from_numpy(m) for m in masks])  # (N, imgsz, imgsz) uint8
        labels = torch.ones((len(masks),), dtype=torch.int64)
        area = (boxes_t[:, 3] - boxes_t[:, 1]) * (boxes_t[:, 2] - boxes_t[:, 0])
        iscrowd = torch.zeros((len(masks),), dtype=torch.int64)

        target = {
            "boxes": boxes_t,
            "labels": labels,
            "masks": masks_t,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": iscrowd,
        }
        return image_tensor, target


def collate(batch):
    return tuple(zip(*batch))


def build_model(pretrained: bool):
    # Load full COCO weights (backbone+FPN+RPN transfer), then swap the class-specific
    # heads for 2 classes (background + filament). The RPN is class-agnostic.
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None
    model = maskrcnn_resnet50_fpn_v2(weights=weights)

    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes=2)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes=2)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=config.MODEL_DIR / "maskrcnn")
    ap.add_argument("--pretrained", action="store_true", default=True)
    ap.add_argument("--no-pretrained", action="store_false", dest="pretrained")
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", action="store_false", dest="amp")
    ap.add_argument("--smoke", action="store_true", help="run 2 batches then exit (pipeline sanity check)")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)

    image_info, anns_by_file = load_anns_by_file(args.data_dir / "train_annotations.json")
    image_dir = args.data_dir / "train_images"
    disk_files = sorted(p.name for p in image_dir.iterdir() if p.suffix.lower() in {".jpeg", ".jpg", ".png"})
    files = sorted(f for f in disk_files if f in anns_by_file)

    rng = random.Random(args.seed)
    rng.shuffle(files)
    n_val = max(1, int(round(len(files) * config.VAL_FRACTION)))
    val_files = set(files[:n_val])
    train_files = [f for f in files if f not in val_files]
    print(f"files={len(files)}  train={len(train_files)}  val={len(val_files)}  seed={args.seed}")

    model = build_model(args.pretrained)
    model.transform.min_size = (args.imgsz,)
    model.transform.max_size = args.imgsz
    model.to(device)

    train_ds = MaskRCNNDataset(train_files, anns_by_file, image_dir, args.imgsz)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=0)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.wd)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp) if device.type == "cuda" else None

    model.train()
    step = 0
    for epoch in range(args.epochs):
        model.train()
        ep_loss = 0.0
        n_batches = 0
        for images, targets in train_dl:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            if not torch.isfinite(losses):
                print(f"  [warn] non-finite loss at step {step}, skipping")
                continue

            opt.zero_grad()
            if scaler is not None:
                scaler.scale(losses).backward()
                scaler.step(opt)
                scaler.update()
            else:
                losses.backward()
                opt.step()

            ep_loss += float(losses.detach().cpu())
            n_batches += 1
            step += 1

            if args.smoke and step >= 2:
                print("SMOKE TEST OK — pipeline runs end to end.")
                return

        avg = ep_loss / max(1, n_batches)
        print(f"epoch {epoch}  avg_loss={avg:.4f}  steps={n_batches}", flush=True)

        ckpt = {
            "model": model.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "val_files": sorted(val_files),
            "train_files": sorted(train_files),
        }
        torch.save(ckpt, args.out / f"maskrcnn_e{epoch}.pth")
        torch.save(ckpt, args.out / "last.pth")
        print(f"  saved {args.out / f'maskrcnn_e{epoch}.pth'}", flush=True)

    print("done")


if __name__ == "__main__":
    main()
