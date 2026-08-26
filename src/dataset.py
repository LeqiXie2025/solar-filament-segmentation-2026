import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.preprocessing import make_input_channels


def read_gray_image(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image.astype(np.float32) / 255.0


class FilamentPatchDataset(Dataset):
    def __init__(
        self,
        image_ids: list[int],
        image_info: dict,
        masks_by_image: dict,
        image_dir: Path,
        patch_size: int,
        augment: bool,
        positive_patch_prob: float = 0.6,
        small_component_prob: float = 0.0,
        scale_jitter: float = 0.0,
        samples_per_epoch: int | None = None,
    ) -> None:
        self.image_ids = list(image_ids)
        self.image_info = image_info
        self.masks_by_image = masks_by_image
        self.image_dir = Path(image_dir)
        self.patch_size = int(patch_size)
        self.augment = augment
        self.positive_patch_prob = positive_patch_prob
        self.small_component_prob = small_component_prob
        self.scale_jitter = max(0.0, float(scale_jitter))
        self.samples_per_epoch = samples_per_epoch
        self._component_cache: dict = {}

    def __len__(self) -> int:
        return self.samples_per_epoch or len(self.image_ids)

    def _components(self, image_id, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, int]]:
        """Lazily compute per-filament connected components of the union mask.

        Sampling patch centers from components (instead of uniformly over all
        foreground pixels) keeps small filaments from being starved: a large
        filament has many more pixels and would otherwise be picked orders of
        magnitude more often than a small one.
        """
        if image_id not in self._component_cache:
            num, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
            comps = []
            for c in range(1, num):
                ys, xs = np.where(labels == c)
                if len(xs) > 0:
                    comps.append((ys, xs, len(xs)))
            self._component_cache[image_id] = comps
        return self._component_cache[image_id]

    def _choose_component(self, comps: list[tuple[np.ndarray, np.ndarray, int]]) -> tuple[np.ndarray, np.ndarray, int]:
        if self.small_component_prob <= 0 or len(comps) < 2 or random.random() >= self.small_component_prob:
            return comps[random.randrange(len(comps))]

        # Bias part of positive samples toward the smallest half of components.
        # This targets thin/small filaments without completely ignoring large ones.
        ranked = sorted(comps, key=lambda item: item[2])
        cutoff = max(1, len(ranked) // 2)
        return ranked[random.randrange(cutoff)]

    def _sample_crop_size(self, height: int, width: int) -> int:
        if not self.augment or self.scale_jitter <= 0:
            return min(self.patch_size, height, width)

        lo = max(0.65, 1.0 - self.scale_jitter)
        hi = min(1.75, 1.0 + self.scale_jitter)
        crop = int(round(self.patch_size * random.uniform(lo, hi)))
        return int(np.clip(crop, min(self.patch_size, height, width), min(height, width)))

    def _choose_patch(self, mask: np.ndarray, image_id, crop_size: int) -> tuple[int, int]:
        height, width = mask.shape
        patch = min(crop_size, height, width)

        if self.augment and mask.sum() > 0 and random.random() < self.positive_patch_prob:
            comps = self._components(image_id, mask)
            if comps:
                ys, xs, _area = self._choose_component(comps)
                idx = random.randrange(len(xs))
                center_x, center_y = int(xs[idx]), int(ys[idx])
            else:
                center_x = random.randrange(width)
                center_y = random.randrange(height)
        else:
            center_x = random.randrange(width)
            center_y = random.randrange(height)

        x0 = int(np.clip(center_x - patch // 2, 0, width - patch))
        y0 = int(np.clip(center_y - patch // 2, 0, height - patch))
        return y0, x0

    def _augment(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.augment:
            return image, mask

        if random.random() < 0.5:
            image = image[:, :, ::-1]
            mask = mask[:, ::-1]
        if random.random() < 0.5:
            image = image[:, ::-1, :]
            mask = mask[::-1, :]

        k = random.randrange(4)
        if k:
            image = np.rot90(image, k, axes=(1, 2))
            mask = np.rot90(mask, k, axes=(0, 1))

        if random.random() < 0.35:
            gain = random.uniform(0.85, 1.15)
            bias = random.uniform(-0.04, 0.04)
            image = np.clip(image * gain + bias, 0.0, 1.0)

        return np.ascontiguousarray(image), np.ascontiguousarray(mask)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx % len(self.image_ids)]
        info = self.image_info[image_id]
        image = read_gray_image(self.image_dir / info["file_name"])
        mask = self.masks_by_image[image_id].astype(np.float32)

        patch = self._sample_crop_size(image.shape[0], image.shape[1])
        y0, x0 = self._choose_patch(mask, image_id, patch)
        image = image[y0 : y0 + patch, x0 : x0 + patch]
        mask = mask[y0 : y0 + patch, x0 : x0 + patch]

        if patch != self.patch_size:
            image = cv2.resize(image, (self.patch_size, self.patch_size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST)

        image = make_input_channels(image)
        image, mask = self._augment(image, mask)

        image_tensor = torch.from_numpy(image).float()
        mask_tensor = torch.from_numpy(mask[None, ...]).float()
        return image_tensor, mask_tensor, info["file_name"]


def merge_duplicate_file_records(image_info: dict, masks_by_image: dict) -> tuple[dict, dict]:
    merged_info = {}
    merged_masks = {}
    for image_id, info in image_info.items():
        file_name = info["file_name"]
        if file_name not in merged_info:
            merged_info[file_name] = dict(info)
            merged_info[file_name]["id"] = file_name
            merged_masks[file_name] = masks_by_image[image_id].copy()
        else:
            merged_masks[file_name] = np.maximum(merged_masks[file_name], masks_by_image[image_id])
    return merged_info, merged_masks


def sorted_time_split(image_info: dict, val_fraction: float) -> tuple[list[int], list[int]]:
    image_ids = sorted(image_info.keys(), key=lambda image_id: image_info[image_id]["file_name"])
    val_count = max(1, int(round(len(image_ids) * val_fraction)))
    return image_ids[:-val_count], image_ids[-val_count:]


def random_split(image_info: dict, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    image_ids = sorted(image_info.keys(), key=lambda image_id: image_info[image_id]["file_name"])
    rng = random.Random(seed)
    rng.shuffle(image_ids)
    val_count = max(1, int(round(len(image_ids) * val_fraction)))
    return image_ids[val_count:], image_ids[:val_count]
