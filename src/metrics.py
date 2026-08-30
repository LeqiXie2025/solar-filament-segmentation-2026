"""Instance-level matching and metrics for local validation.

The official metric (as described in the competition and reconstructed by
competitors) matches predicted filament instances to ground-truth instances by
overlap, then reports mean Dice, Dice/IoU distributions, and one-to-many /
many-to-one relations. It is *not* plain pixel Dice.

We provide three numbers so you can see the trade-offs:

- ``full_image_dice``  : union of all pixels (semantic, ignores instances).
- ``mean_dice_matched``: mean Dice over greedy 1-to-1 matched pairs only.
                         Empty prediction -> 1.0. This replicates the buggy
                         public metric that rewards predicting fewer filaments.
- ``panoptic_quality`` : PQ with Dice overlap. Penalizes both missed (FN) and
                         spurious (FP) instances; empty -> 0.0. This is the
                         "honest" instance metric.
"""
from __future__ import annotations

import numpy as np


def overlap_matrix(preds: list[np.ndarray], gts: list[np.ndarray], mode: str = "dice") -> np.ndarray:
    """P x G matrix of pairwise overlap (dice or iou) between pred and gt masks."""
    P, G = len(preds), len(gts)
    M = np.zeros((P, G), dtype=np.float32)
    if P == 0 or G == 0:
        return M
    for i, p in enumerate(preds):
        p_area = int(p.sum())
        if p_area == 0:
            continue
        p_bool = p.astype(bool)
        for j, g in enumerate(gts):
            g_area = int(g.sum())
            if g_area == 0:
                continue
            inter = int(np.logical_and(p_bool, g).sum())
            if inter == 0:
                continue
            if mode == "dice":
                M[i, j] = 2.0 * inter / (p_area + g_area)
            else:
                M[i, j] = inter / (p_area + g_area - inter)
    return M


def greedy_match(
    preds: list[np.ndarray],
    gts: list[np.ndarray],
    overlap_threshold: float = 0.0,
    mode: str = "dice",
) -> tuple[list[tuple[int, int, float]], int, int]:
    """Greedy 1-to-1 matching by overlap.

    Returns ``(matched, fp, fn)`` where matched is a list of
    ``(pred_idx, gt_idx, overlap)`` tuples, fp = unmatched preds, fn = unmatched gts.
    """
    P, G = len(preds), len(gts)
    if P == 0 or G == 0:
        return [], P, G
    M = overlap_matrix(preds, gts, mode)
    candidates = [(M[i, j], i, j) for i in range(P) for j in range(G) if M[i, j] >= overlap_threshold]
    candidates.sort(reverse=True)

    used_p: set[int] = set()
    used_g: set[int] = set()
    matched: list[tuple[int, int, float]] = []
    for ov, i, j in candidates:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        matched.append((i, j, float(ov)))
    return matched, P - len(used_p), G - len(used_g)


def full_image_dice(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    pred_bool = pred.astype(bool)
    target_bool = target.astype(bool)
    inter = int(np.logical_and(pred_bool, target_bool).sum())
    return float((2.0 * inter + eps) / (int(pred_bool.sum()) + int(target_bool.sum()) + eps))


def mean_dice_matched(matched: list[tuple[int, int, float]]) -> float:
    """Replica of the buggy public metric: mean Dice over matched pairs only.

    Empty prediction (no matched pairs) -> 1.0, which is exactly why empty
    submissions score ~0.93 on the public leaderboard.
    """
    if not matched:
        return 1.0
    return float(np.mean([ov for _, _, ov in matched]))


def panoptic_quality(
    matched: list[tuple[int, int, float]], n_pred: int, n_gt: int, fp: int, fn: int
) -> float:
    """Panoptic Quality using Dice as the overlap score.

    Denominator = matched + 0.5*fp + 0.5*fn, so missed and spurious instances
    both cost points; empty prediction scores exactly 0.
    """
    if n_pred == 0 and n_gt == 0:
        return 1.0
    overlap_sum = sum(ov for _, _, ov in matched)
    denom = len(matched) + 0.5 * fp + 0.5 * fn
    return float(overlap_sum / max(denom, 1e-6))


def official_pq_counts(
    preds: list[np.ndarray],
    gts: list[np.ndarray],
    iou_threshold: float = 0.5,
) -> tuple[float, int, int, int]:
    """Return the exact count terms used by the organizer's evaluation notebook.

    The organizer treats every pair with IoU > 0.5 as a hit, then counts a
    prediction/GT as unmatched only when its hit-matrix column/row is empty.
    Submission masks are required to be disjoint, so qualifying many-to-many
    matches are rare and no additional greedy assignment is applied.
    """
    n_pred, n_gt = len(preds), len(gts)
    if n_gt == 0:
        return 0.0, 0, n_pred, 0
    if n_pred == 0:
        return 0.0, 0, 0, n_gt
    matrix = overlap_matrix(preds, gts, mode="iou")
    hits = matrix > float(iou_threshold)
    tp = int(hits.sum())
    fp = int((hits.sum(axis=1) == 0).sum())
    fn = int((hits.sum(axis=0) == 0).sum())
    return float(matrix[hits].sum()), tp, fp, fn


def official_pq(iou_sum: float, tp: int, fp: int, fn: int) -> float:
    denominator = int(tp) + 0.5 * int(fp) + 0.5 * int(fn)
    return float(iou_sum / denominator) if denominator > 0 else 0.0
