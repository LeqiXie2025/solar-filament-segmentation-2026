# Multi-annotator consensus refiner

## Purpose

Use repeated annotations of the same filament as a soft agreement target instead of discarding every mask except the cluster medoid. The production refiner remains the instance-recognition gate; the consensus model only adjusts accepted-mask boundaries.

## Implementation

- `clean_instance_annotations.py --include-cluster-members` stores all masks in each IoU cluster.
- `train_crop_refiner.py --consensus-targets` averages those masks into a soft target for training while validation remains against the canonical medoid mask.
- `predict_detect_refine_multiscale.py --shape-refiner ... --shape-weight ...` blends the consensus probability map after the primary refiner accepts the proposal. This prevents the shape model from introducing new false-positive instances.

## Honest time-split result

Fixed detector, proposals, TTA, threshold, NMS and minimum area:

| Variant | PQ | Matched Dice | TP / FP / FN |
|---|---:|---:|---:|
| Production refiner | 0.563863 | 0.728928 | 447 / 106 / 78 |
| Consensus shape blend, weight 0.20 | **0.564275** | **0.729416** | 447 / 106 / 78 |
| Consensus shape blend, weight 0.35 | 0.564276 | 0.729372 | 447 / 106 / 78 |

The gain is small but clean: recognition counts are unchanged and only boundary quality improves. Weight 0.20 is selected as the more conservative private-leaderboard candidate.

## Rejected variant

Using the consensus refiner by itself greatly increased false positives. It is not a submission candidate. Consensus information is retained only as a weak shape branch behind the production gate.

## Submission

`submissions/submission_consensus_shape_blend_w020.csv`: 1667 rows, 178 represented test images, and no overlapping masks.
