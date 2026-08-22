# Competition Overview

## Official Links

- Main page: https://www.kaggle.com/competitions/filament-segmentation-2026
- Data page: https://www.kaggle.com/competitions/filament-segmentation-2026/data
- Evaluation page: https://www.kaggle.com/competitions/filament-segmentation-2026/overview/evaluation
- Rules page: https://www.kaggle.com/competitions/filament-segmentation-2026/rules

## Simple Explanation

This is an image segmentation competition.

Kaggle gives us solar observation images. In each image, there may be dark,
thin, curved solar filaments. Our job is to mark the pixels that belong to
those filaments.

In simple terms:

- Input: a solar image.
- Output: a mask image.
- Mask value 1: this pixel is part of a filament.
- Mask value 0: this pixel is background.

## What Is Being Scored

The competition compares our predicted masks with hidden official masks.

The main metric is Dice score:

```text
Dice = 2 * overlap / (prediction area + true area)
```

Higher Dice means the predicted filament area overlaps the real filament area
better.

## Why The Task Is Hard

Solar filaments can be:

- very thin;
- curved and broken;
- low contrast;
- close to noisy background structures;
- inconsistent in size and shape.

The model must avoid both missing real filaments and marking too much
background.

## First Model Direction

The recommended first training direction is semantic segmentation:

- U-Net or U-Net++;
- PyTorch;
- Dice loss plus BCE loss;
- fixed validation split;
- visual checks of image and mask alignment.

YOLO segmentation can be tested later as a secondary path, but it should not be
the first baseline unless the submission format clearly benefits from instance
segmentation.
