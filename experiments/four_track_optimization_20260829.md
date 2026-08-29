# Four-track optimization result

All comparisons use the fixed chronological validation split and the same PQ implementation. A branch is promoted only when it improves the full split without changing the scoring rules.

## 1. Small-object detector

- Built 2,796 object-centred 768 px crops containing 13,038 boxes from canonical labels.
- Fine-tuned YOLO11s for four epochs from the production detector.
- Crop validation: precision 0.606, recall 0.535, mAP50 0.549, mAP50-95 0.280.
- Adding it as a tiled small-object expert at confidence 0.25 or 0.08 produced exactly the same 40-image downstream PQ as the production detector: 0.585398.

Decision: keep the trained expert as an experimental asset, but do not add it to the submission because it contributes no new accepted instances.

## 2. MobileSAM box prompting

- Added a reproducible YOLO-to-MobileSAM box-prompt inference path.
- On the 12-image probe, MobileSAM reached PQ 0.523440 and matched Dice 0.642423; the crop refiner reached PQ 0.544598 and matched Dice 0.715671.
- On the 40-image split, MobileSAM thresholds 0.70/0.80/0.85 produced PQ 0.450880/0.321201/0.193481.
- As a fallback behind the production refiner, thresholds 0.80 and 0.85 added no masks; threshold 0.70 reduced PQ from 0.585398 to 0.578236.

Decision: MobileSAM is rejected for this grayscale domain without task-specific fine-tuning. It is not used in the final submission.

## 3. Multi-annotator targets

- Clustered repeated annotations and trained a soft consensus mask refiner.
- The consensus model alone increased false positives and was rejected.
- Used only as a weak boundary branch after the production refiner accepts a proposal.

## 4. OOF fusion

Full chronological split:

| Model | PQ | Matched Dice | TP / FP / FN |
|---|---:|---:|---:|
| Production detector + refiner | 0.563863 | 0.728928 | 447 / 106 / 78 |
| Consensus boundary blend 0.20 | **0.564275** | **0.729416** | 447 / 106 / 78 |
| Consensus boundary blend 0.35 | 0.564276 | 0.729372 | 447 / 106 / 78 |

Decision: select weight 0.20. The near-identical 0.20 and 0.35 results show that the gain is small and stable rather than a threshold accident. The lower weight is retained to reduce private-leaderboard risk.

## Final candidate

`submissions/submission_consensus_shape_blend_w020.csv`

- 1,667 rows
- 178 represented test images
- no overlapping masks
- no leaderboard-derived tuning
