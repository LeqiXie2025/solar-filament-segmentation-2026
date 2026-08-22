# File Purpose Guide

This document explains what each current file or folder in this repository is
for. The repository is currently a planning and documentation workspace only;
training code will be added later after the official data structure is fully
understood.

## Root

### `README.md`

Project entry page.

Put here:

- competition name and official Kaggle link;
- plain project status;
- short explanation of what the repository contains;
- short warning about not committing raw Kaggle data;
- repository map.

Do not put here:

- long experiment details;
- raw data;
- model code;
- private credentials.

### `.gitignore`

Safety file that prevents accidental uploads.

It currently ignores:

- official Kaggle raw data in `data/raw/`;
- local sample images in `data/samples/`;
- generated submissions in `submissions/`;
- model weights and large generated artifacts;
- notebooks and Python caches;
- `.env` files and `kaggle.json`.

Do not remove these ignore rules unless the team has checked the competition
rules and intentionally decided what can be shared.

## `docs/`

Main documentation folder.

### `docs/competition_overview.md`

Plain-language explanation of the competition.

Put here:

- official competition links;
- what the model receives as input;
- what the model must output;
- how Dice score works;
- why solar filament segmentation is difficult;
- first recommended model direction.

Use this file when explaining the project to a teammate or another AI assistant.

### `docs/data_guide.md`

Data understanding and inspection guide.

Put here:

- where the official Kaggle data is stored locally;
- expected dataset folders and file types;
- checklist for image count, mask count, dimensions, alignment, and submission
  format;
- notes discovered during EDA.

After the full data is downloaded, update the inventory table in this file.

### `docs/project_structure.md`

Repository structure guide.

Put here:

- what each folder is supposed to contain;
- which folders are local-only;
- where future code, experiments, reports, and submissions should go;
- team conventions for keeping the project organized.

Use this as the rulebook for where files should be placed.

### `docs/weekly_plan.md`

Week-by-week execution plan.

Put here:

- weekly goals;
- tasks for setup, EDA, baseline, validation, improvement, postprocessing, and
  final submission;
- expected deliverables for each week.

Update it when the schedule changes or when a phase is completed.

### `docs/download_status.md`

Official data download status.

Put here:

- Kaggle competition slug;
- whether the account has joined the competition;
- which official files have been downloaded locally;
- whether full dataset download succeeded or failed;
- exact commands to retry data download.

This file should describe local data state without committing the raw data
itself.

### `docs/file_purpose_guide.md`

This file.

Put here:

- an explanation of every current file and directory;
- what each file should contain later;
- what should not be placed in each file.

Use this as the onboarding document for teammates.

## `data/`

Local data folder. Most of this folder is ignored by Git.

### `data/raw/`

Local official Kaggle data folder.

Put here locally:

- downloaded competition archive;
- extracted official training images;
- extracted official training masks or annotations;
- official test images;
- sample submission files;
- official metadata files.

Do not commit these raw files to GitHub by default.

### `data/raw/README.md`

Short note explaining that official data belongs in `data/raw/`.

Put here:

- the recommended Kaggle download command;
- warning that raw data should not be redistributed through GitHub.

### `data/raw/.gitkeep`

Placeholder file.

It keeps the otherwise empty `data/raw/` directory visible in Git.

Do not edit this unless the directory no longer needs a placeholder.

### `data/raw/file_listing.csv`

Local-only official file list downloaded from Kaggle.

Purpose:

- helps understand the official data tree before downloading the full archive;
- records example file paths and file sizes.

This file is ignored by Git and was not pushed to GitHub.

### `data/samples/`

Local visual sample folder.

Put here locally:

- a few official images for inspection;
- image-mask overlays;
- prediction-mask comparison figures;
- failure-case screenshots.

Do not commit these images unless the team confirms the rules allow publishing
them.

### `data/samples/README.md`

Short note explaining the purpose of local sample visuals.

Put here:

- examples of what belongs in `data/samples/`;
- reminder that sample images are ignored by Git by default.

### `data/samples/.gitkeep`

Placeholder file.

It keeps the otherwise empty `data/samples/` directory visible in Git.

### `data/samples/*.jpeg`

Local-only official sample images downloaded from Kaggle.

Current local examples:

- `20110120105534Ch.jpeg`;
- `20110130110334Ch.jpeg`;
- `20110214175414Mh.jpeg`.

These files are useful for checking image format and image-loading tools, but
they were not pushed to GitHub.

## `experiments/`

Experiment tracking folder.

### `experiments/experiments.csv`

Main experiment log.

Put one row here for every meaningful run or Kaggle submission.

Recommended columns already included:

- experiment id;
- date;
- model;
- input size;
- loss;
- augmentation;
- validation Dice;
- public leaderboard score;
- submission file;
- git commit;
- notes.

This file is more important than leaderboard screenshots because it connects
each score to a reproducible setup.

## `reports/`

Human-readable result and analysis folder.

Put here later:

- EDA summary;
- data quality findings;
- model comparison notes;
- failure-case analysis;
- final method summary;
- final submission explanation.

Current status:

- directory exists locally but has no committed file yet.

## `submissions/`

Local Kaggle submission folder.

Put here locally:

- generated submission CSV files;
- final candidate submission files;
- temporary submissions for validation.

Do not commit generated submissions by default. Instead, record each uploaded
file in `experiments/experiments.csv`.

### `submissions/README.md`

Short note explaining how to handle submission files.

Put here:

- reminder that submission files are local-only;
- instruction to record every real submission in the experiment log.

### `submissions/.gitkeep`

Placeholder file.

It keeps the otherwise empty `submissions/` directory visible in Git.

## Future Files To Add Later

Do not add these yet unless the team is ready to start coding:

- `requirements.txt`: Python dependencies for training and inference;
- `src/dataset.py`: dataset loading and image-mask parsing;
- `src/train.py`: training loop;
- `src/infer.py`: test prediction and submission generation;
- `src/metrics.py`: Dice and related metrics;
- `src/postprocess.py`: thresholding and mask cleanup;
- `configs/*.yaml`: model and training configurations;
- `notebooks/01_eda.ipynb`: exploratory data analysis notebook.

When these files are added, update this guide so teammates know exactly how to
use them.
