# Boundary-only union from the confirmed 0.36 base

## Correction after LB feedback

The recall-expanding cross-detector ensemble added YOLO11m-only instances and dropped Public LB to 0.34. That branch is rejected. The confirmed 0.36 YOLO11s submission is restored as the immutable detection base.

The new ensemble never adds a YOLO11m-only instance and never removes an unmatched YOLO11s instance. It changes only YOLO11s masks that have a YOLO11m match above mask IoU 0.35, replacing the matched boundary with the pixel union. Final masks are made disjoint again.

## Full official chronological validation

| Prediction | Official PQ | TP / FP / FN | Physical instances |
|---|---:|---:|---:|
| Confirmed 0.36 base | 0.424733 | 478 / 364 / 199 | 511 |
| **Boundary-only union** | **0.436215** | **491 / 345 / 186** | **509** |

The change improves local official PQ by `0.011482`, adds 13 official matches, removes 19 false-positive terms, and keeps essentially the same detection density. This directly addresses the failed 0.34 experiment: the proposal set remains the proven 0.36 proposal set.

## Submission

`submissions/submission_base036_boundary_union.csv`

- 1,525 rows versus 1,534 in the confirmed 0.36 base
- 178 represented test images
- no overlapping masks
- no YOLO11m-only detections added
