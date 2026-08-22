# Download Status

Last updated: 2026-08-22.

## Kaggle Account Status

The Kaggle account has joined the competition.

Competition slug:

```text
filament-segmentation-2026
```

## Official Data Location

Official data is stored locally under:

```text
data/raw/
data/samples/
```

These folders are ignored by Git to avoid accidental redistribution of Kaggle
competition data.

## Downloaded Locally

The official file listing has been saved locally as:

```text
data/raw/file_listing.csv
```

The following official sample test images have been downloaded locally:

```text
data/samples/20110120105534Ch.jpeg
data/samples/20110130110334Ch.jpeg
data/samples/20110214175414Mh.jpeg
```

These images are useful for checking the image format before the full dataset is
downloaded.

## Full Dataset Download

The full command attempted was:

```text
kaggle competitions download -c filament-segmentation-2026 -p data/raw
```

During this setup pass, the full archive request did not write a completed file
to disk before being stopped. Individual official files downloaded correctly, so
Kaggle authentication and competition access are working.

Recommended next step:

```text
kaggle competitions download -c filament-segmentation-2026 -p data/raw --force
```

If the full archive still stalls, download files in groups or use Kaggle
Notebooks for the first EDA pass.
