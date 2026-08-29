import argparse
from pathlib import Path

import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-yaml", type=Path, default=config.PROJECT_ROOT / "data" / "yolo_segments" / "filament_segments.yaml")
    parser.add_argument("--model", default="yolo11s-seg.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", type=Path, default=config.PROJECT_ROOT / "models" / "yolo")
    parser.add_argument("--name", default="yolo11s_seg_1280")
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr0", type=float, default=2e-4)
    parser.add_argument("--copy-paste", type=float, default=0.15)
    parser.add_argument("--close-mosaic", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
        patience=args.patience,
        workers=args.workers,
        exist_ok=True,
        pretrained=True,
        amp=args.amp,
        optimizer="AdamW",
        lr0=args.lr0,
        cos_lr=True,
        close_mosaic=args.close_mosaic,
        copy_paste=args.copy_paste,
        max_det=120,
    )


if __name__ == "__main__":
    main()
