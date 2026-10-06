# Solar Filament Segmentation 2026 — release 0.38

This branch records the code and submission artifacts associated with the current **user-reported public score of 0.38**. The repository does not independently verify a Kaggle upload or score.

## Release

- Method: R25 proposal-driven mask refinement with the COCO polygon rasterizer candidate.
- Local evaluation: per-annotator raw COCO annotation records, all-hit matching at strict IoU > 0.5, pooled panoptic quality (PQ).
- The historical frozen gate passed on 141 exposed development images. It is not independent out-of-fold evidence.
- Candidate checkpoint SHA-256: `cf053df4739e8d91504b293e2165b7dbb1c1acd7bf4126087c047689cdcd4162`.
- 0.38 submission SHA-256: `105668985c949fa330dd1bc22992206ad9e0f70088f67d90e7ea216714fe1a3c`.

Read [RELEASE_038.md](RELEASE_038.md) for implementation details, run commands, required external data and weights, and provenance. The full external-input inventory with SHA-256 hashes is in `experiments/route_recovery_20261004/r25_proposal/repair_01/release_manifest.json`.

## Entry points

```powershell
python experiments/route_recovery_20261004/r25_proposal/repair_01/train_paired_rasterizer.py
python experiments/route_recovery_20261004/r25_proposal/repair_01/export_test_after_gate.py
```

Training requires CUDA. Export checks the frozen gate and writes a local CSV; it does not upload. Official images, annotations, derived training records, and large model weights remain external prerequisites and are not stored in Git. The other project documents describe historical planning and baseline work.
