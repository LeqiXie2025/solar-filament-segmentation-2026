"""Frozen-mask descriptors and per-annotator RLE evaluation helpers."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as coco

from src.gt_dedup import _decode_segmentation
from src.metrics import official_pq


FEATURES = ['detector_score', 'confidence', 'prob_std', 'uncertain_fraction',
            'crop_fraction', 'log_area', 'long_norm', 'short_norm', 'aspect',
            'contrast', 'inside_std', 'fill_ratio']


def encode(mask):
    return coco.encode(np.asfortranarray(mask.astype(np.uint8)))


def matrix(preds, gt):
    if not preds or not gt:
        return np.zeros((len(preds), len(gt)), dtype=np.float32)
    return coco.iou(preds, gt, [0] * len(gt)).astype(np.float32)


def terms(preds, gt):
    overlaps = matrix(preds, gt)
    hits = overlaps > .5
    return np.array([overlaps[hits].sum(), hits.sum(),
                     (~hits.any(axis=1)).sum(), (~hits.any(axis=0)).sum()], dtype=float)


def score(values):
    total = np.asarray(values).sum(axis=0)
    return dict(pq=official_pq(*total), tp=int(total[1]), fp=int(total[2]), fn=int(total[3]))


class Annotations:
    def __init__(self, path):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        self.images = defaultdict(list)
        self.anns = defaultdict(list)
        for info in data['images']:
            self.images[info['file_name']].append(info)
        for ann in data['annotations']:
            self.anns[ann['image_id']].append(ann)

    def get(self, name):
        groups = []
        for info in self.images[name]:
            masks = []
            for ann in self.anns[info['id']]:
                mask = _decode_segmentation(ann['segmentation'], info['height'], info['width'])
                if mask.any():
                    masks.append(encode(mask))
            groups.append(masks)
        return groups


def quality_targets(predictions, groups):
    """Expected matched IoU; empty annotator sets explicitly contribute zero."""
    result = np.zeros(len(predictions))
    for gt in groups:
        overlaps = matrix(predictions, gt)
        if len(gt):
            result += np.where(overlaps > .5, overlaps, 0).sum(axis=1)
    return result / max(1, len(groups))
