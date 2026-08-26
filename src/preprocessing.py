import cv2
import numpy as np


def solar_disk_mask(height: int, width: int, margin: float = 0.98) -> np.ndarray:
    cy, cx = height / 2.0, width / 2.0
    radius = min(height, width) * 0.5 * margin
    yy, xx = np.ogrid[:height, :width]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius ** 2


def detect_solar_disk(image: np.ndarray) -> np.ndarray:
    """Fit a circle to the bright solar disk in a full-disk H-alpha image.

    The hardcoded solar_disk_mask assumes a centered disk with radius
    min(H,W)/2, but real GONG images are off-center and the disk radius varies
    (~904-953 px on 2048x2048). Filaments are *dark* and would show up as holes
    in a simple threshold, so we instead fit a circle to the bright pixels,
    which is robust to a small fraction of dark holes.
    """
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0

    height, width = image.shape
    u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    _, thresh = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ys, xs = np.where(thresh > 0)

    min_side = min(height, width)
    if len(xs) < max(100, int(0.01 * height * width)):
        return solar_disk_mask(height, width)

    cx, cy = float(xs.mean()), float(ys.mean())
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    radius = float(np.percentile(dist, 99.5))
    radius = float(np.clip(radius, 0.30 * min_side, 0.55 * min_side))

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
