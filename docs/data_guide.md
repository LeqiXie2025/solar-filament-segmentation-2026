# Data Guide

## Where Official Data Goes

Official Kaggle data should be downloaded locally into:

```text
data/raw/
```

This folder is ignored by Git. Do not publish the raw competition data unless
the team has explicitly confirmed the competition rules allow redistribution.

## Expected Data Contents

The official file listing includes a MAGFiLO-based folder structure such as:

```text
MAGFiLO_1.0_Kaggle_2026/
  train/
  test/
  ...
```

Important file types to identify after download:

- training images;
- training masks or annotations;
- test images;
- sample submission file;
- any metadata CSV files.

## What To Check First

Before training, record:

- number of train images;
- number of test images;
- image dimensions;
- mask dimensions;
- whether image and mask names align;
- whether masks are binary or multi-class;
- expected submission format;
- whether there are empty masks;
- whether there are corrupted images.

## Local Visual Samples

Small local visual samples can be placed in:

```text
data/samples/
```

This folder is also ignored by Git by default. Use it for local inspection
screenshots or image-mask overlays. If the team later decides to publish sample
figures, only commit images allowed by the competition rules.

## Data Inventory Checklist

Fill this after downloading:

| Item | Value |
|---|---|
| Download date |  |
| Kaggle competition slug | filament-segmentation-2026 |
| Raw data location | data/raw/ |
| Train image count |  |
| Train mask count |  |
| Test image count |  |
| Sample submission exists |  |
| Submission format understood |  |
| Notes |  |
