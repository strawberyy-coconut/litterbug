# Figures — licence and attribution

The images under `examples-val/` and `examples-test/` are **derived from** the BUU Waste
Occlusion Dataset (BUU-WOD).

> VisionLab, Burapha University. *BUU Waste Occlusion Dataset (BUU-WOD)*. Kaggle.
> <https://www.kaggle.com/datasets/visionlab1buu/buu-waste-occlusion-dataset>
> Licence: Creative Commons Attribution 4.0 International (CC BY 4.0)
> <https://creativecommons.org/licenses/by/4.0/>

CC BY 4.0 permits redistribution and adaptation, including for any purpose, on three conditions:
credit the creator, link the licence, and **indicate whether changes were made**. The third one is
the easy one to miss, so it is stated explicitly below.

## What was changed

Each figure is a two-panel composite built from one source image:

- ground-truth polygons filled and outlined using the dataset's colour convention
- predicted masks and boxes drawn over the right-hand panel
- **red** outlines marking ground-truth instances the model missed
- **magenta** outlines marking predictions that matched no instance
- class and confidence labels rendered on top

The same image appears in both panels, so each figure duplicates its source frame.

## What was not changed

The dataset files themselves — images and annotations — were used unmodified for training and
evaluation. No image was cropped, recoloured, relabelled or re-annotated. The overlays exist only in
these derived figures.

## Regenerating

```bash
uv run litterbug examples --weights runs/<run>/weights/best.pt --split val
uv run litterbug examples --weights runs/<run>/weights/best.pt --split test
```

## Not covered by this licence

The §8 video evaluation uses third-party stock footage (Shutterstock, Dreamstime). Those are
watermarked preview assets with **no redistribution right**, and no frames from them appear here.
They are cited by clip ID in `report/report.md` §11 and excluded from version control.
