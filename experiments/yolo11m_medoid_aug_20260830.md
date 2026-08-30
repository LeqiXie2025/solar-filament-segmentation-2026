# YOLO11m medoid-label detector

## Training change

The detector was upgraded from YOLO11s to YOLO11m and trained for 20 epochs on one medoid annotation per physical image. Training used 1280-pixel inputs, mixed precision, moderate multi-scale sampling, bidirectional flips, small rotations, reduced translation/scale distortion, and cosine AdamW optimization.

The medoid labels remove conflicting duplicate boxes from different annotators while retaining the chronological train/validation boundary. The best detector reached box `mAP50=0.59366` and `mAP50-95=0.27985`.

## Full official chronological validation

| Detector | Confidence | Official PQ | TP / FP / FN |
|---|---:|---:|---:|
| YOLO11s production | 0.22 | 0.424733 | 478 / 364 / 199 |
| **YOLO11m medoid** | **0.22** | **0.433660** | **404 / 175 / 273** |

Official PQ improves by `0.008927`. The larger detector is substantially more selective, removing 189 false-positive terms at the cost of 74 true-positive terms. Confidence `0.18` produced nearly the same PQ (`0.433597`), so `0.22` is retained as the cleaner operating point.

The YOLO11s+YOLO11m union with a five-fold OOF quality model reached only `0.431399`; it is rejected because the extra recall did not compensate for added false positives.

## Submission

`submissions/submission_yolo11m_medoid_aug_c022.csv`

- 1,252 rows
- 177 represented test images; one test image has no accepted instance
- no overlapping masks
