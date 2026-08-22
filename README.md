# Solar Filament Segmentation Challenge 2026

Kaggle competition workspace for the Solar Filament Segmentation Challenge 2026.

Official competition page:
https://www.kaggle.com/competitions/filament-segmentation-2026

## What This Repository Is For

This repository is a planning and documentation workspace. It intentionally does
not contain training code yet.

Use it to keep:

- competition links and rules notes;
- data inventory and file structure notes;
- weekly work plans;
- experiment logs;
- submission records;
- result summaries and final handoff notes.

## What Not To Commit

Do not commit Kaggle credentials, raw competition data, trained model weights,
large generated masks, or leaderboard-probing submissions unless the team has
confirmed the rules and repository visibility.

The official data should be downloaded locally into `data/raw/`.

## Main Task

Given solar H-alpha observation images, predict segmentation masks for solar
filaments. A good prediction marks the pixels belonging to filaments and leaves
the background unmarked.

The main metric is Dice score, which rewards overlap between the predicted mask
and the hidden ground-truth mask.

## Current Status

- Kaggle account has joined the competition.
- GitHub repository is for documentation first.
- No model code has been added.
- Official data should be kept locally under `data/raw/`.

## Repository Map

- `docs/competition_overview.md`: plain-language competition explanation.
- `docs/data_guide.md`: what official data files mean and how to inspect them.
- `docs/project_structure.md`: what each folder is for.
- `docs/weekly_plan.md`: week-by-week execution plan.
- `experiments/experiments.csv`: table for recording model experiments later.
- `reports/`: final summaries, findings, charts, and handoff notes.
- `submissions/`: local Kaggle submission files, ignored by Git by default.
- `data/raw/`: official Kaggle data, ignored by Git by default.
- `data/samples/`: small local visual samples, ignored by Git by default.
