# Project Structure

This repository currently contains documentation only. Training code can be
added later after the data format is fully understood.

## Root Files

`README.md`

High-level project introduction, official link, current status, and repository
map.

`.gitignore`

Prevents accidental upload of raw Kaggle data, generated submissions, model
weights, credentials, caches, and notebooks.

## `docs/`

Put stable written documentation here.

- `competition_overview.md`: what the competition is about.
- `data_guide.md`: what the data folders/files mean.
- `weekly_plan.md`: weekly schedule and deliverables.
- `project_structure.md`: this file.

## `data/raw/`

Local official Kaggle data. Do not commit raw data by default.

## `data/samples/`

Local image-mask screenshots, overlays, or example visual checks. Do not commit
unless rules allow publication.

## `experiments/`

Experiment tracking. This is where model results will be logged later.

`experiments.csv` should contain one row per experiment:

- experiment id;
- date;
- model;
- input size;
- loss;
- augmentation;
- validation Dice;
- public leaderboard score;
- notes.

## `reports/`

Put human-readable conclusions here:

- EDA summaries;
- model comparison notes;
- failure case analysis;
- final submission explanation.

## `submissions/`

Local Kaggle submission files. Do not commit generated submissions by default.

For each real submission, add a row to `experiments/experiments.csv` instead.
