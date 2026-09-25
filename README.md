# litterbug

Detection and instance segmentation of waste objects on a conveyor belt, built on the
**BUU Waste Occlusion** dataset with **Ultralytics YOLO**.

Segmentation is the product. Detection is kept as a comparison baseline run on the same split with
the same schedule, so the report can isolate what the segmentation head buys on heavily occluded
objects rather than comparing two things at once.

## Layout

| Path | Purpose |
| --- | --- |
| `src/litterbug/common/` | Shared surface only: the class contract, config, runtime guards. |
| `src/litterbug/training/` | Training, evaluation, error analysis, EDA and dataset validation. |
| `src/litterbug/running/` | Running a trained model over images and video (`predict.py`). |
| `src/litterbug/cli/` | Command line entry point. Argument parsing and dispatch, no pipeline logic. |
| `tests/` | Tests for the frozen config, the class contract and the matching conventions. |
| `dataset/` | Input data. Git-ignored — see below. |
| `runs/` | Training and inference artifacts. Git-ignored. |
| `report/`, `notebooks/` | Written in parallel with training. |

Each module owns one concern and is importable. `training` and `running` never import each other;
anything they genuinely share lives in `common`.

## Setup

There is no `python` or `python3` on `PATH` inside the dev container. Use `uv`:

```bash
uv sync                                     # create .venv and install dependencies
uv run python -c "import torch; print(torch.cuda.is_available())"
nvidia-smi                                  # confirm the GPU is visible before training
```

A missing GPU does not fail loudly on its own — Ultralytics will happily fall back to CPU and turn a
two hour job into several days. The pipeline refuses to start training without CUDA unless you pass
`--device cpu` explicitly.

## Dataset

**Source:** [BUU Waste Occlusion Dataset (BUU-WOD)](https://www.kaggle.com/datasets/visionlab1buu/buu-waste-occlusion-dataset)
— VisionLab, Burapha University. Licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Used in full and unmodified; full citation
in `report/report.md` §11.

Place the data in `dataset/` and keep the `1_Model_Training_Data/` layout intact:

```
dataset/1_Model_Training_Data/
├── data.yaml                       # nc: 4, names: [Glass, Metal, Paper, Plastic]
├── train/images + train/labels     # 1,400 pairs
├── valid/images + valid/labels     # 400 pairs
└── test/images  + test/labels      # 200 pairs
```

The splits are pre-made and must be used as-is. `data.yaml` has no `path:` key, so the split paths
resolve relative to the YAML's own directory.

The class order — `0 Glass`, `1 Metal`, `2 Paper`, `3 Plastic` — is a contract with the annotations
and is declared exactly once, in `src/litterbug/common/constants.py`.

Validate the data before the first run. This checks image/label pairing, polygon validity, coordinate
ranges and class balance, and exits non-zero when something is wrong:

```bash
uv run litterbug validate
```

`--overlay N` additionally renders N random polygon overlays per split for a visual spot-check.

## Usage

```bash
uv run litterbug validate                                          # check the dataset first
uv run litterbug eda                                               # dataset EDA

uv run litterbug train --task segment --dry-run                    # resolved config, no GPU cost
uv run litterbug train --task segment                              # fine-tune
uv run litterbug train --task detect                               # comparison baseline

uv run litterbug val --weights runs/<run>/weights/best.pt --split val
uv run litterbug val --weights runs/<run>/weights/best.pt --split test   # once, at the end

uv run litterbug predict --weights runs/<run>/weights/best.pt --source <image|dir|video>
uv run litterbug predict --weights runs/<run>/weights/best.pt --source <video> --track

uv run litterbug analyze  --weights runs/<run>/weights/best.pt --split val  # per-instance errors
uv run litterbug examples --weights runs/<run>/weights/best.pt --split val  # commented FP/FN figures

uv run pytest                                                      # contracts + config
```

Add `--dry-run` to `train` to print the exact configuration that would be handed to Ultralytics
without touching the GPU. `python -m litterbug.cli <command>` is equivalent to `uv run litterbug <command>`.

Hyperparameters live in `src/litterbug/common/config.py` and are never restated in documentation — if you want
to know what a run used, read the resolved configuration it prints on startup, or the config dump
written into the run directory.

## Artifacts

Weights (`*.pt`), `runs/` and `dataset/` are inputs and outputs, not source. They are git-ignored and
must stay that way.


