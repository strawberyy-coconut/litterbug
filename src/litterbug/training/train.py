"""Fine-tune a YOLO model on the BUU Waste Occlusion dataset."""

from __future__ import annotations

import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from litterbug.common.config import TrainConfig, build_train_kwargs, describe_config
from litterbug.common.runtime import (
    describe_environment,
    resolve_device,
    seed_everything,
    setup_logging,
)

log = logging.getLogger("litterbug.train")


def resume_checkpoint(cfg: TrainConfig) -> Path:
    """Locate the checkpoint a resume must load.

    Resuming has to start from the interrupted run's own ``last.pt``, which carries the
    epoch counter, optimizer and EMA state. Pointing at the pretrained base model instead
    makes Ultralytics warn that the checkpoint "is not a resumable training checkpoint",
    quietly disable resume, and start a fresh run — discarding every epoch trained so far.
    """
    last = cfg.project / cfg.run_name / "weights" / "last.pt"
    if not last.is_file():
        raise FileNotFoundError(
            f"cannot resume {cfg.run_name!r}: {last} does not exist.\n"
            "  - Pass --name of an interrupted run, or drop --resume to start a new one."
        )
    return last


def train(cfg: TrainConfig) -> Path:
    """Run fine-tuning and return the path to the best checkpoint.

    ``cfg.dry_run`` resolves and prints the configuration without touching the GPU, so
    the plumbing can be checked for free.
    """
    kwargs = build_train_kwargs(cfg)
    kwargs["device"] = resolve_device(
        cfg.device, require_cuda=not cfg.dry_run, purpose=f"{cfg.task} training"
    )

    print(describe_config(cfg, kwargs))
    print(describe_environment())

    best = cfg.project / cfg.run_name / "weights" / "best.pt"
    if cfg.dry_run:
        if cfg.resume:
            log.info("Dry run: resume target %s", resume_checkpoint(cfg))
        log.info("Dry run: configuration resolved, nothing was trained.")
        return best

    seed_everything(cfg.seed)

    from ultralytics import YOLO  # imported late so --dry-run stays cheap

    if cfg.resume:
        checkpoint = resume_checkpoint(cfg)
        log.info("Resuming %s from %s", cfg.run_name, checkpoint)
        model = YOLO(str(checkpoint))
        # The checkpoint carries the original schedule, and Ultralytics only accepts a
        # whitelist of overrides on resume. Device and workers are both on that list, so
        # the shared-memory-aware worker count still applies.
        model.train(resume=True, device=kwargs["device"], workers=kwargs["workers"])
    else:
        model = YOLO(cfg.model)
        model.train(**kwargs)

    save_dir = Path(getattr(model.trainer, "save_dir", cfg.project / cfg.run_name))
    best = save_dir / "weights" / "best.pt"
    log.info("Best checkpoint: %s", best)
    return best


def main(argv: Sequence[str] | None = None) -> int:
    """Allow ``python -m training.train ...`` without duplicating the argument parser."""
    from litterbug.cli.main import main as cli_main

    return cli_main(["train", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    setup_logging()
    raise SystemExit(main())
