# OOF instance-quality filtering

## Method

Each post-NMS instance is described by detector confidence, geometry, compactness, elongation, grayscale statistics, local-ring contrast, and normalized disk position. A regularized LightGBM classifier is trained with five-fold grouping by image, so the threshold is selected only from out-of-fold probabilities.

The target is whether an instance reaches the organizer's `IoU > 0.5` match condition across the available annotator records. No leaderboard labels or test-set statistics are used.

## Official chronological validation

| Selection | Official PQ | TP / FP / FN |
|---|---:|---:|
| No quality filter | 0.424733 | 478 / 364 / 199 |
| **Five-fold OOF quality filter** | **0.432746** | **460 / 302 / 217** |

OOF PQ improves by `0.008013`. The filter removes 62 false-positive terms at a cost of 18 true-positive terms. The selected quality threshold is `0.26`.

For diagnostic purposes, a model fitted on all chronological validation instances scores `0.443816` on those same records (`467 / 278 / 210`). This number is optimistic and is not used as the expected generalization score; the OOF result is the decision metric.

## Submission

`submissions/submission_quality_lgbm_oof_c022.csv`

- 1,431 rows
- 178 represented test images
- no overlapping masks
- quality filtering is applied after mask NMS and disjoint-mask construction
