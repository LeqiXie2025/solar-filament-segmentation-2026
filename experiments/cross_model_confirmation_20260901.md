# Cross-model confirmation candidate

The latest 0.37 submission remains the reference. This experiment applies the high-level ensemble principle of using a genuinely different model for corroboration instead of adding its detections.

The base is the validated YOLO11s detector plus IoU-BCE crop refiner. The peer is the recall-fine-tuned YOLO11m detector. A base mask is retained when it either overlaps the peer by mask IoU at least 0.50 or has area at least 1500 pixels. No peer-only prediction is added, and base boundaries are unchanged.

On the full chronological validation split:

| Prediction | Official PQ | TP / FP / FN | Instances |
|---|---:|---:|---:|
| Base | 0.424733 | 478 / 364 / 199 | 511 |
| Same-family seed gate | 0.432219 | 465 / 313 / 212 | 469 |
| **YOLO11m confirmation gate** | **0.434264** | **433 / 239 / 244** | **404** |

This is a precision-oriented candidate. It produces 1,351 test instances across 178 non-empty images, with no overlapping masks. The stronger local PQ comes from rejecting 125 false-positive terms, at the cost of 45 matched instances; it is a candidate for Public LB validation, not a replacement for the 0.37 reference until Kaggle returns a score.
