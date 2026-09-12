"""Pipeline configuration.

Hyperparameters live here and nowhere else. Documentation and prompts deliberately point at this
module instead of restating numbers, so training behaviour can only drift in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from common.constants import DATA_YAML, MODELS, RUNS_DIR, TASKS
from common.runtime import DEFAULT_SEED, resolve_workers, vram_gb

# --------------------------------------------------------------------------- #
# Frozen defaults
# --------------------------------------------------------------------------- #
EPOCHS = 100
PATIENCE = 20
IMGSZ = 640
SEED = DEFAULT_SEED

# Batch size is VRAM-bound, not compute-bound. Ordered by descending VRAM threshold,
# so the first match wins: >= 11 GiB -> 16, >= 5 GiB -> 8.
BATCH_BY_VRAM: tuple[tuple[float, int], ...] = ((11.0, 16), (5.0, 8))
DEFAULT_BATCH = 8

# copy_paste is enabled specifically because this dataset's failure mode is occlusion.
COPY_PASTE = 0.1

# Segmentation-only arguments, kept as a named tuple so a detection run can never
# silently inherit them.
SEG_ONLY_KEYS: tuple[str, ...] = ("overlap_mask", "mask_ratio")
OVERLAP_MASK = True
MASK_RATIO = 4


def resolve_batch(requested: int | None = None) -> int:
    """Pick a batch size from available VRAM unless one was supplied explicitly."""
    if requested:
        return int(requested)
    total = vram_gb()
    if total is None:
        return DEFAULT_BATCH
    for min_gib, batch in BATCH_BY_VRAM:
        if total >= min_gib:
            return batch
    return DEFAULT_BATCH


def resolve_inference_batch(requested: int | None = None) -> int:
    """Batch size for inference, which needs more memory per image than AMP training did.

    Training ran in mixed precision, so its activations were roughly half the size. Inference
    defaults to fp32, so the training-derived batch has to be halved to fit the same device.
    Measured on the 5.7 GiB RTX 3060: batch 8 OOMs at inference, batch 4 fits.
    """
    if requested:
        return int(requested)
    return max(1, resolve_batch() // 2)


def model_for(task: str) -> str:
    """Return the checkpoint name for a task, rejecting anything unknown."""
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")
    return MODELS[task]


def segmentation_args(task: str) -> dict[str, object]:
    """Segmentation-only training arguments. Refuses to serve a detection task."""
    if task != "segment":
        raise ValueError(f"segmentation arguments requested for task {task!r}")
    return {"overlap_mask": OVERLAP_MASK, "mask_ratio": MASK_RATIO}


@dataclass(frozen=True)
class TrainConfig:
    """Everything a training run needs, with the frozen schedule as defaults."""

    task: str = "segment"
    data: Path = DATA_YAML
    name: str | None = None
    project: Path = RUNS_DIR
    batch: int | None = None
    workers: int | None = None
    epochs: int = EPOCHS
    patience: int = PATIENCE
    imgsz: int = IMGSZ
    seed: int = SEED
    amp: bool = True
    cos_lr: bool = True
    copy_paste: float = COPY_PASTE
    resume: bool = False
    device: str | None = None
    dry_run: bool = False

    @property
    def model(self) -> str:
        return model_for(self.task)

    @property
    def run_name(self) -> str:
        return self.name or f"{self.task}-{Path(self.model).stem}"

    @property
    def batch_size(self) -> int:
        return resolve_batch(self.batch)


def build_train_kwargs(cfg: TrainConfig) -> dict[str, object]:
    """Translate a :class:`TrainConfig` into the kwargs handed to Ultralytics."""
    model_for(cfg.task)  # rejects unknown tasks
    if cfg.epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {cfg.epochs}")

    kwargs: dict[str, object] = {
        "data": str(cfg.data),
        "imgsz": cfg.imgsz,
        "epochs": cfg.epochs,
        "patience": cfg.patience,
        "batch": cfg.batch_size,
        "workers": resolve_workers(cfg.workers),
        "seed": cfg.seed,
        "amp": cfg.amp,
        "cos_lr": cfg.cos_lr,
        "copy_paste": cfg.copy_paste,
        "project": str(cfg.project),
        "name": cfg.run_name,
        "resume": cfg.resume,
    }
    if cfg.device:
        kwargs["device"] = cfg.device
    if cfg.task == "segment":
        kwargs.update(segmentation_args(cfg.task))
    return kwargs


def describe_config(cfg: TrainConfig, kwargs: dict[str, object] | None = None) -> str:
    """Human-readable resolved configuration, printed before every run."""
    resolved = build_train_kwargs(cfg) if kwargs is None else kwargs
    lines = [
        f"task        {cfg.task}",
        f"model       {cfg.model}",
        f"data        {cfg.data}",
        f"run         {cfg.project / cfg.run_name}",
        "arguments",
    ]
    skip = {"data", "project", "name"}
    lines += [f"  {key:<13} {value}" for key, value in sorted(resolved.items()) if key not in skip]
    return "\n".join(lines)
