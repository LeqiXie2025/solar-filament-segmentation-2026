# Cross-detector mask union

## Motivation

The best Public LB remained 0.36 after two more selective YOLO11m submissions scored 0.35. Direct comparison of the test predictions showed substantial complementary coverage:

- YOLO11s submission: 1,534 instances
- recall-fine-tuned YOLO11m submission: 1,448 instances
- one-to-one matches at mask IoU above 0.10: 1,324
- YOLO11s-only instances: 210
- YOLO11m-only instances: 124
- mean IoU of matched masks: 0.876719

The models therefore agree strongly on most instances but retain different detections. The ensemble keeps the YOLO11s coverage, unions matched boundaries, adds non-overlapping YOLO11m-only masks, and makes the final masks disjoint.

## Full official chronological validation

| Prediction | Official PQ |
|---|---:|
| YOLO11s | 0.424733 |
| YOLO11m raw-label recall fine-tune | 0.437831 |
| **Cross-detector union** | **0.437582** |

The ensemble preserves nearly all of the stronger detector's local PQ while increasing physical validation instances to 523 and official TP to 498. It is selected as a recall-oriented Public-LB candidate rather than a replacement justified by local PQ alone.

## Submission

`submissions/submission_mask_union_s_c022_mraw_c018.csv`

- 1,639 rows
- 178 represented test images
- no overlapping masks
- no leaderboard labels used during construction
