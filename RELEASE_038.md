# Release 0.38 code snapshot

This branch preserves the source path associated with the current 0.38 submission. The public score is user-reported and is not independently verified by this repository.

## Method and validation

The release trains an R25 proposal-driven mask refiner. Its candidate arm changes polygon target rasterization to `pycocotools` COCO polygon decoding. The frozen local evaluation uses raw COCO annotation records grouped by annotator, all-hit matching at strict IoU > 0.5, and pooled PQ. The historical frozen gate passed on 141 exposed development images; these images are not independent out-of-fold evaluation.

The validation and export QA records are preserved alongside the code. They establish the historical local gate and CSV integrity checks, not an independent Kaggle score verification.

## Commands

Run from the repository root after installing dependencies and placing the external assets at the paths listed in the release manifest:

```powershell
python experiments/route_recovery_20261004/r25_proposal/repair_01/train_paired_rasterizer.py
python experiments/route_recovery_20261004/r25_proposal/repair_01/export_test_after_gate.py
```

Training requires CUDA. The test exporter requires the validation gate and all registered inputs; it produces a local CSV and never uploads it.

## Required assets and provenance

`experiments/route_recovery_20261004/r25_proposal/repair_01/release_manifest.json` records exact paths, SHA-256 hashes, and whether each artifact is in Git. Official raw images/annotations, derived proposal and consensus records, large trained checkpoints, and the upstream detector/refiner assets remain external. The final 0.38 submission and two modest OBB test-prediction CSV inputs are included in this branch. Some inherited upstream model lineage is incomplete, so this is not a claim of end-to-end reproduction from the public raw dataset alone.

The original run did not preserve a complete package lockfile. `requirements.txt` lists the project dependencies, but exact dependency versions are not asserted.
