# Seed-3407 consensus-gated submission

## Result

A same-architecture YOLO11s detector was fine-tuned from the validated YOLO11s checkpoint with seed `3407`. On the first chronological 40-image probe it improved official PQ from `0.436660` to `0.441962`, but on the full chronological validation it scored `0.421410`, below the immutable base score of `0.424733`. It is therefore not used as a direct replacement.

## Conservative use of the second detector

The seed-3407 model is used only as a corroborating signal. The confirmed 0.36 submission remains the mask source. A base instance is removed only when both conditions hold:

1. the second detector has no matching instance at mask IoU `0.10`; and
2. the base instance area is below `2000` pixels.

This avoids boundary fusion and never introduces a new detection. On full chronological validation, the gate changed the base prediction set as follows:

| Prediction | Official PQ | TP / FP / FN | Instances |
|---|---:|---:|---:|
| Confirmed 0.36 base | 0.424733 | 478 / 364 / 199 | 511 |
| Seed-confirmed gate | 0.432219 | 465 / 313 / 212 | 469 |

The local gain comes from removing 42 uncorroborated small predictions, trading 13 matched instances for 51 fewer false-positive terms. Since this threshold was selected on the validation split, it is a **submission candidate**, not a guaranteed leaderboard improvement. The original confirmed 0.36 file remains the private-leaderboard safety baseline.

## Deliverable

`submissions/submission_base_seed3407_consensus_a2000.csv`

- 1,438 instances across 178 non-empty test images
- 2 empty images out of 180
- verified with `src.check_submissions`: no overlapping masks
- no public leaderboard labels or test-label inference used
