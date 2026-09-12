"""Tests for the training entry point's resume handling.

Resuming is easy to get subtly wrong: pointing at the pretrained base model instead of the
run's own ``last.pt`` makes Ultralytics silently start over, which costs the epochs already
trained. These tests pin the behaviour that prevents it.
"""

import pytest

from litterbug.common.config import TrainConfig
from litterbug.training.train import resume_checkpoint


def test_resume_requires_an_existing_last_checkpoint(tmp_path):
    cfg = TrainConfig(task="segment", project=tmp_path, name="missing", resume=True)
    with pytest.raises(FileNotFoundError):
        resume_checkpoint(cfg)


def test_resume_targets_the_runs_last_checkpoint(tmp_path):
    weights = tmp_path / "segment-yolo26s-seg" / "weights"
    weights.mkdir(parents=True)
    (weights / "last.pt").touch()

    cfg = TrainConfig(task="segment", project=tmp_path, name="segment-yolo26s-seg", resume=True)

    assert resume_checkpoint(cfg) == weights / "last.pt"


def test_resume_ignores_best_checkpoint(tmp_path):
    # best.pt is an inference artefact; only last.pt carries optimizer/epoch state.
    weights = tmp_path / "run" / "weights"
    weights.mkdir(parents=True)
    (weights / "best.pt").touch()

    cfg = TrainConfig(task="segment", project=tmp_path, name="run", resume=True)

    with pytest.raises(FileNotFoundError):
        resume_checkpoint(cfg)
