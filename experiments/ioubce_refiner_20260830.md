# IoU-BCE refiner fine-tuning

## Change

The production crop refiner was fine-tuned for two epochs with a soft Jaccard plus BCE objective. This aligns boundary optimization with the organizer's IoU-based Panoptic Quality metric while retaining BCE stability.

## Full chronological validation

| Refiner | Official PQ | TP / FP / FN |
|---|---:|---:|
| detector-box baseline | 0.424047 | 478 / 364 / 199 |
| **IoU-BCE fine-tune** | **0.424733** | **478 / 364 / 199** |

The detection counts are unchanged, so the gain comes from improved matched-mask overlap rather than threshold or sparsity tuning. The improvement is small but consistent with the intended SQ-focused objective.

## Submission

`submissions/submission_officialpq_ioubce_c022.csv`

- 1,534 rows
- 178 represented test images
- no overlapping masks
- detector confidence 0.22, mask threshold 0.45, consensus shape weight 0.20

The optional instance-score filter was also tested on the official validation subset. It reduced PQ and remains disabled (`0.0`) in the submission.
