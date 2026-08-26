import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import config
from src.dataset import read_gray_image
from src.masks import encode_binary_mask, postprocess_binary_mask, split_instances
from src.model_factory import create_model_from_checkpoint
from src.preprocessing import detect_solar_disk, make_input_channels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, default=config.MODEL_DIR / "unet_baseline_best.pth")
    parser.add_argument("--output", type=Path, default=config.SUBMISSION_DIR / "submission_unet_baseline.csv")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N test images for a dry run.")
    parser.add_argument("--start", type=int, default=0, help="Skip the first N test images (for chunked runs).")
    parser.add_argument("--tile-size", type=int, default=config.INFER_TILE_SIZE)
    parser.add_argument("--stride", type=int, default=config.INFER_STRIDE)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-area", type=int, default=None)
    parser.add_argument("--min-areas", type=int, nargs="+", default=None,
                        help="Emit one submission per min_area in a single inference pass.")
    parser.add_argument("--max-instances", type=int, default=0)
    parser.add_argument("--morphology", choices=["none", "close", "openclose"], default="close")
    parser.add_argument("--instance-method", choices=["connected", "watershed"], default="connected")
    parser.add_argument("--watershed-peak-rel", type=float, default=0.45)
    parser.add_argument("--watershed-peak-abs", type=float, default=3.0)
    parser.add_argument("--watershed-min-component-area", type=int, default=0)
    parser.add_argument("--tta", choices=["none", "hflip", "hvflip"], default="none")
    return parser.parse_args()


@torch.no_grad()
def sliding_window_predict(
    model,
    image: np.ndarray,
    device,
    tile_size: int,
    stride: int,
    batch_size: int,
    tta: str = "none",
) -> np.ndarray:
    def predict_once(input_image: np.ndarray) -> np.ndarray:
        height, width = input_image.shape
        channels = make_input_channels(input_image)
        prob_sum = np.zeros((height, width), dtype=np.float32)
        weight_sum = np.zeros((height, width), dtype=np.float32)

        y_starts = list(range(0, max(height - tile_size, 0) + 1, stride))
        x_starts = list(range(0, max(width - tile_size, 0) + 1, stride))
        if y_starts[-1] != height - tile_size:
            y_starts.append(height - tile_size)
        if x_starts[-1] != width - tile_size:
            x_starts.append(width - tile_size)

        window = np.outer(np.hanning(tile_size), np.hanning(tile_size)).astype(np.float32)
        window = np.maximum(window, 1e-3)

        coords = [(y0, x0) for y0 in y_starts for x0 in x_starts]
        for start in range(0, len(coords), batch_size):
            batch_coords = coords[start : start + batch_size]
            tiles = [
                channels[:, y0 : y0 + tile_size, x0 : x0 + tile_size]
                for y0, x0 in batch_coords
            ]
            tensor = torch.from_numpy(np.stack(tiles)).float().to(device)
            logits = model(tensor)
            probs_batch = torch.sigmoid(logits)[:, 0].detach().cpu().numpy()
            for (y0, x0), probs in zip(batch_coords, probs_batch):
                prob_sum[y0 : y0 + tile_size, x0 : x0 + tile_size] += probs * window
                weight_sum[y0 : y0 + tile_size, x0 : x0 + tile_size] += window

        return prob_sum / np.maximum(weight_sum, 1e-6)

    predictions = [predict_once(image)]
    if tta in {"hflip", "hvflip"}:
        pred = predict_once(np.ascontiguousarray(np.fliplr(image)))
        predictions.append(np.ascontiguousarray(np.fliplr(pred)))
    if tta == "hvflip":
        pred = predict_once(np.ascontiguousarray(np.flipud(image)))
        predictions.append(np.ascontiguousarray(np.flipud(pred)))
        pred = predict_once(np.ascontiguousarray(np.flipud(np.fliplr(image))))
        predictions.append(np.ascontiguousarray(np.fliplr(np.flipud(pred))))

    return np.mean(predictions, axis=0).astype(np.float32)


@torch.no_grad()
def _legacy_sliding_window_predict(
    model,
    image: np.ndarray,
    device,
    tile_size: int,
    stride: int,
    batch_size: int,
) -> np.ndarray:
    height, width = image.shape
    channels = make_input_channels(image)
    prob_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)

    y_starts = list(range(0, max(height - tile_size, 0) + 1, stride))
    x_starts = list(range(0, max(width - tile_size, 0) + 1, stride))
    if y_starts[-1] != height - tile_size:
        y_starts.append(height - tile_size)
    if x_starts[-1] != width - tile_size:
        x_starts.append(width - tile_size)

    window = np.outer(np.hanning(tile_size), np.hanning(tile_size)).astype(np.float32)
    window = np.maximum(window, 1e-3)

    coords = [(y0, x0) for y0 in y_starts for x0 in x_starts]
    for start in range(0, len(coords), batch_size):
        batch_coords = coords[start : start + batch_size]
        tiles = [
            channels[:, y0 : y0 + tile_size, x0 : x0 + tile_size]
            for y0, x0 in batch_coords
        ]
        tensor = torch.from_numpy(np.stack(tiles)).float().to(device)
        logits = model(tensor)
        probs_batch = torch.sigmoid(logits)[:, 0].detach().cpu().numpy()
        for (y0, x0), probs in zip(batch_coords, probs_batch):
            prob_sum[y0 : y0 + tile_size, x0 : x0 + tile_size] += probs * window
            weight_sum[y0 : y0 + tile_size, x0 : x0 + tile_size] += window

    return prob_sum / np.maximum(weight_sum, 1e-6)


def main() -> None:
    args = parse_args()
    test_dir = args.data_dir / "test_images"
    if not test_dir.exists():
        raise FileNotFoundError(f"Expected test images at {test_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    threshold = args.threshold if args.threshold is not None else float(checkpoint.get("threshold", config.PRED_THRESHOLD))
    min_area = args.min_area if args.min_area is not None else int(checkpoint.get("min_area", config.MIN_COMPONENT_AREA))

    model = create_model_from_checkpoint(checkpoint).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    test_files = sorted(
        [p for p in test_dir.iterdir() if p.suffix.lower() in {".jpeg", ".jpg", ".png"}]
    )
    if args.start:
        test_files = test_files[args.start:]
    if args.limit:
        test_files = test_files[: args.limit]
    min_areas = args.min_areas if args.min_areas else [min_area]
    rows_by_area = {a: [] for a in min_areas}

    for path in tqdm(test_files, desc="predict"):
        image = read_gray_image(path)
        probs = sliding_window_predict(
            model,
            image,
            device,
            tile_size=args.tile_size,
            stride=args.stride,
            batch_size=args.batch_size,
            tta=args.tta,
        )
        disk = detect_solar_disk(image)
        binary = (probs > threshold).astype(np.uint8)
        binary[~disk] = 0

        binary = postprocess_binary_mask(binary, args.morphology)

        image_id = path.stem
        for min_area in min_areas:
            instances = split_instances(
                binary,
                min_area,
                method=args.instance_method,
                watershed_peak_rel=args.watershed_peak_rel,
                watershed_peak_abs=args.watershed_peak_abs,
                watershed_min_component_area=args.watershed_min_component_area or None,
            )
            instances = sorted(instances, key=lambda instance: int(instance.sum()), reverse=True)
            if args.max_instances:
                instances = instances[: args.max_instances]
            for idx, instance in enumerate(instances, start=1):
                rows_by_area[min_area].append(
                    {
                        "filament_id": f"{image_id}_{idx}",
                        "segmentation_rle": encode_binary_mask(instance),
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for min_area in min_areas:
        if len(min_areas) == 1:
            out = args.output
        else:
            out = args.output.with_name(f"{args.output.stem}_area{min_area}{args.output.suffix}")
        rows = rows_by_area[min_area]
        pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(out, index=False)
        print(f"saved {out} rows={len(rows)} threshold={threshold} min_area={min_area}")


if __name__ == "__main__":
    main()
