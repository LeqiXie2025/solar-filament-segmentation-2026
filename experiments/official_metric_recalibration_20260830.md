# Official metric recalibration

## Correction

The organizer's self-evaluation notebook uses global Panoptic Quality with:

- pairwise **IoU**, not Dice;
- a true-positive condition of `IoU > 0.5`;
- one score over all annotator-image records;
- every original annotator record evaluated separately against the same physical-image prediction.

The previous local compass used greedy Dice matching on a medoid-collapsed label set and averaged per-image PQ. It was useful for model development but was not an exact leaderboard simulation.

## Exact implementation

- `src.metrics.official_pq_counts` reproduces the organizer hit matrix and TP/FP/FN accounting.
- `src.validation.load_gt_instances_by_annotator` preserves the original annotator-image records.
- `predict_detect_refine_multiscale.py --official-metric` reports global official PQ.

## Full chronological validation

| Detector confidence | Mask threshold | Official PQ | TP / FP / FN |
|---:|---:|---:|---:|
| 0.18 | 0.45 | 0.419576 | 496 / 417 / 181 |
| **0.22** | **0.45** | **0.424047** | **478 / 364 / 199** |
| 0.25 | 0.50 | 0.419239 | 458 / 342 / 219 |

Confidence 0.22 improves the complete chronological split by 0.00447 PQ. It removes 53 false positives while sacrificing 18 true positives. The more aggressive 0.25 setting over-prunes and is rejected.

## Submission candidate

`submissions/submission_officialpq_c022_shape_w020.csv`

- 1,534 rows
- 178 represented test images
- no overlapping masks
- parameters selected on the full chronological validation split, not leaderboard feedback
