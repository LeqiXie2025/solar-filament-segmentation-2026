import cv2
import numpy as np


def solar_disk_mask(height: int, width: int, margin: float = 0.98) -> np.ndarray:
    cy, cx = height / 2.0, width / 2.0
    radius = min(height, width) * 0.5 * margin
    yy, xx = np.ogrid[:height, :width]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius ** 2


def radial_flattening(image: np.ndarray, bins: int = 128) -> np.ndarray:
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0

    height, width = image.shape
    cy, cx = height / 2.0, width / 2.0
    yy, xx = np.ogrid[:height, :width]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    disk = solar_disk_mask(height, width)
    max_radius = dist[disk].max()

    bin_edges = np.linspace(0, max_radius, bins + 1)
    bin_ids = np.digitize(dist[disk], bin_edges) - 1
    bin_ids = np.clip(bin_ids, 0, bins - 1)

    disk_values = image[disk]
    radial_means = np.ones(bins, dtype=np.float32)
    for idx in range(bins):
        values = disk_values[bin_ids == idx]
        if values.size:
            radial_means[idx] = max(float(np.median(values)), 1e-3)

    profile = np.ones_like(image, dtype=np.float32)
    full_bin_ids = np.digitize(dist, bin_edges) - 1
    full_bin_ids = np.clip(full_bin_ids, 0, bins - 1)
    profile[disk] = radial_means[full_bin_ids[disk]]

    corrected = image / profile
    corrected = corrected / max(np.percentile(corrected[disk], 99.5), 1e-3)
    corrected = np.clip(corrected, 0.0, 1.0)
    corrected[~disk] = 0.0
    return corrected.astype(np.float32)


def make_input_channels(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0

    flat = radial_flattening(image)

    flat_u8 = np.clip(flat * 255, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(flat_u8)
    clahe = clahe.astype(np.float32) / 255.0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    blackhat = cv2.morphologyEx(flat_u8, cv2.MORPH_BLACKHAT, kernel)
    blackhat = blackhat.astype(np.float32) / 255.0

    channels = np.stack([image, clahe, blackhat], axis=0)
    return channels.astype(np.float32)
