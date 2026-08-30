# YOLO11m recall recovery fine-tune

## Motivation

The medoid-only YOLO11m submission reduced test predictions from 1,534 to 1,252 and dropped Public LB from 0.36 to 0.35. This showed that the chronological validation split over-rewarded selectivity relative to the public test distribution.

The medoid YOLO11m checkpoint was therefore fine-tuned for ten epochs on the original multi-annotator box labels. The low learning rate and reduced geometric distortion preserve the stronger detector while restoring weight to filaments observed by multiple annotators.

## Full official chronological validation

| Model | Confidence | Official PQ | TP / FP / FN |
|---|---:|---:|---:|
| YOLO11s production | 0.22 | 0.424733 | 478 / 364 / 199 |
| YOLO11m medoid | 0.22 | 0.433660 | 404 / 175 / 273 |
| **YOLO11m raw-label fine-tune** | **0.22** | **0.445635** | **422 / 186 / 255** |
| YOLO11m raw-label fine-tune | 0.18 | 0.437831 | 436 / 235 / 241 |

Confidence 0.22 is the local-PQ optimum, but it produces only 1,349 test instances. Confidence 0.18 produces 1,448 instances, covers all 178 test images, and retains an official PQ above the old production baseline. It is selected as the distribution-robust candidate after the observed LB recall failure.

## Recommended submission

`submissions/submission_yolo11m_raw_recall_ft_c018.csv`

- 1,448 rows
- 178 represented test images
- no overlapping masks

The stricter `c022` file is retained as a local-PQ reference, not the recommended upload.
