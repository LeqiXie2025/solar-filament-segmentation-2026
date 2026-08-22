# Weekly Plan

Current planning date: 2026-08-22.

Competition deadline: 2026-11-15 06:00 UTC, which is 2026-11-15 14:00 Beijing
time.

## Week 1: Setup And Data Understanding

Goal: join the competition and understand the files before modeling.

Tasks:

- confirm both teammates joined Kaggle competition;
- download official data into `data/raw/`;
- inspect folder structure;
- identify train images, masks, test images, and sample submission;
- create 10-20 image-mask visual checks locally.

Deliverables:

- completed data inventory in `docs/data_guide.md`;
- first visual samples stored locally;
- no model training yet.

## Week 2: Baseline Submission

Goal: create the first valid end-to-end submission.

Tasks:

- implement dataset loading later in a code branch;
- train a basic U-Net or U-Net++ model;
- compute local Dice;
- generate a submission file;
- upload one valid Kaggle submission.

Deliverables:

- first public leaderboard score;
- one experiment row in `experiments/experiments.csv`;
- notes about what failed visually.

## Week 3: Stabilize Validation

Goal: make results reproducible.

Tasks:

- fix validation split;
- fix random seed;
- save clear experiment configs;
- compare predictions against validation masks;
- document common failure cases.

Deliverables:

- stable validation Dice;
- validation split note;
- failure case summary in `reports/`.

## Week 4: Improve Data And Loss

Goal: improve baseline without changing too many things at once.

Tasks:

- test image size options;
- test augmentation options;
- test BCE plus Dice loss;
- test threshold values;
- compare local Dice and public leaderboard.

Deliverables:

- at least five recorded experiments;
- best baseline configuration so far.

## Week 5-6: Try Stronger Segmentation Models

Goal: find a model stronger than the initial baseline.

Tasks:

- try U-Net++;
- try DeepLabV3+;
- optionally try SegFormer;
- compare with the same validation split.

Deliverables:

- model comparison table;
- one clear candidate model for deeper tuning.

## Week 7: Postprocessing

Goal: reduce false positives and repair broken filaments.

Tasks:

- tune prediction threshold;
- remove tiny components;
- test morphological opening and closing;
- compare before and after masks visually.

Deliverables:

- postprocessing ablation table;
- before/after examples in local samples.

## Week 8-9: TTA And Ensemble

Goal: improve score with stable, low-risk inference tricks.

Tasks:

- test flip TTA;
- test multi-fold averaging;
- test two-model ensemble;
- record public/private risk.

Deliverables:

- two or three final candidate submissions.

## Week 10: Finalization

Goal: avoid last-minute mistakes.

Tasks:

- freeze final model choices;
- check submission format;
- verify GitHub documentation;
- backup final outputs locally;
- select final Kaggle submissions.

Deliverables:

- final submission selected;
- `reports/final_summary.md`;
- clean GitHub documentation.
