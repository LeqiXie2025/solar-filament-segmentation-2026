# Aggressive pipeline experiment (2026-08-29)

## Scope

This experiment tested three model-level changes without using leaderboard labels:

1. Canonical multi-annotator label deduplication using mask-IoU clustering and a medoid mask.
2. A high-resolution YOLO11m detection/segmentation branch for improved small-filament recall.
3. A U-Net++ EfficientNet-B4 crop refiner with boundary-aware loss and size-aware sampling.

The validation set remains the final time-based 20% split. Model selection uses instance-level PQ, with matched Dice and TP/FP/FN reported separately.

## Label audit

- Raw image records: 1154
- Unique images: 707
- Raw annotations: 8199
- Canonical instances: 6356
- Duplicate clusters merged: 1435
- Images with multiple annotation records: 296
- Instances smaller than 500 pixels: 1091

Canonical annotations: `data/clean/train_annotations_medoid.json`.

## Results

| Pipeline | PQ | Matched Dice | TP | FP | FN |
|---|---:|---:|---:|---:|---:|
| YOLO11s boxes + ResNet34 refiner | **0.5631** | **0.7279** | 447 | 106 | 78 |
| YOLO11m boxes + ResNet34 refiner, conf 0.14 | 0.5404 | 0.7162 | 466 | 164 | 59 |
| YOLO11m boxes + ResNet34 refiner, conf 0.22 | 0.5424 | 0.7260 | 433 | 83 | 92 |
| YOLO11s/YOLO11m box ensemble | 0.5560 | 0.7238 | 460 | 127 | 65 |
| YOLO11m-seg standalone, conf 0.10 | 0.4945 | 0.6815 | 419 | 83 | 106 |
| YOLO11m-seg + detect/refine | 0.5597 | 0.7271 | 450 | 113 | 75 |
| YOLO11s boxes + U-Net++ EfficientNet-B4 refiner | 0.5506 | 0.7129 | 447 | 106 | 78 |

## Decision

The larger models increase recall in some configurations but do not improve honest time-split PQ. The production submission therefore remains the YOLO11s detector plus ResNet34 crop refiner with multiscale inference, refiner TTA, mask NMS, and final disjoint-mask enforcement. The YOLO11m models remain optional diversity branches for future stacking, not default submission models.

## 0.40 attack follow-up

A dedicated 1024-pixel tiled detector was trained on 5494 crops containing 18356 boxes. Full-image plus tiled inference was implemented with border guards and independent confidence calibration. On a fixed 40-image time-split probe it did not add true positives over the full-image detector (PQ remained 0.5833), so tiled proposals are disabled in the production candidate rather than included speculatively.

Clean-label selection found a small reproducible gain from refiner TTA at area 120: PQ 0.5631 to 0.5639 with unchanged TP/FP/FN. The resulting no-overlap test candidate is `submissions/submission_stable_cleanselected_tta_c018_t045_a120_20260829.csv`.

## Proposal classifier and transformer refiner

The next model-level branch trained a ResNet18 proposal classifier on 16448 detector crops (10605 positives and 5843 negatives). Its held-out F1 was 0.7740, but using it to rescue low-confidence YOLO boxes reduced the fixed 40-image PQ probe from 0.5833 to 0.5264. At a conservative setting it still reduced PQ to 0.5728. The classifier is therefore retained as an experimental component but disabled in production.

A MiT-B3 transformer crop refiner was also trained with clean labels, noisy detector boxes, small-instance sampling, and boundary-aware loss. Its best crop Dice was 0.6306, below the production ResNet34 refiner at 0.6764. Fine-tuning the production refiner on the same harder proposal distribution reached only 0.6430. Both replacements were rejected, so no lower-quality leaderboard submission was generated.
