import argparse
from pathlib import Path

import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-yaml", type=Path, default=config.PROJECT_ROOT / "data" / "yolo_boxes" / "filament_boxes.yaml")
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", type=Path, default=config.PROJECT_ROOT / "models" / "yolo")
    parser.add_argument("--name", default="yolo11s_boxes_1280")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    import torchvision

    from ultralytics import YOLO

    original_nms = torchvision.ops.nms

    def cpu_safe_nms(boxes, scores, iou_threshold):
        if boxes.is_cuda:
            keep = original_nms(boxes.detach().cpu(), scores.detach().cpu(), iou_threshold)
            return keep.to(boxes.device)
        return original_nms(boxes, scores, iou_threshold)

    torchvision.ops.nms = cpu_safe_nms

    model = YOLO(args.model)
    model.train(
        data=str(args.data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=args.name,
        patience=5,
        workers=0,
        exist_ok=True,
        pretrained=True,
        amp=False,
        optimizer="AdamW",
        lr0=2e-4,
        cos_lr=True,
        close_mosaic=5,
    )


if __name__ == "__main__":
    main()
