from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models import resnet18

from src.crop_refiner_utils import crop_bounds_from_bbox, crop_with_padding, make_refiner_channels, seed_from_bbox


def create_proposal_classifier(pretrained: bool = False) -> torch.nn.Module:
    if pretrained:
        from torchvision.models import ResNet18_Weights

        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    else:
        model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    return model


def proposal_tensor(image: np.ndarray, bbox_xyxy, image_size: int, pad: float) -> torch.Tensor:
    x0, y0, x1, y1 = [float(value) for value in bbox_xyxy[:4]]
    bbox = (x0, y0, max(1.0, x1 - x0), max(1.0, y1 - y0))
    bounds = crop_bounds_from_bbox(bbox, image.shape[0], image.shape[1], pad=pad, jitter=False)
    crop = crop_with_padding(image, bounds)
    crop = cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    seed = seed_from_bbox(bbox, bounds, image_size)
    channels = make_refiner_channels(crop, seed)
    return torch.from_numpy(np.ascontiguousarray(channels)).float()


def load_proposal_classifier(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    model = create_proposal_classifier(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, int(checkpoint.get("image_size", 224)), float(checkpoint.get("pad", 0.50))


@torch.no_grad()
def classify_boxes(model, image: np.ndarray, boxes: np.ndarray, device, image_size: int, pad: float, batch_size: int = 64):
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.float32)
    tensors = [proposal_tensor(image, box, image_size, pad) for box in boxes]
    probabilities = []
    for start in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[start : start + batch_size]).to(device, non_blocking=True)
        probabilities.append(torch.sigmoid(model(batch)[:, 0]).cpu().numpy())
    return np.concatenate(probabilities).astype(np.float32)

