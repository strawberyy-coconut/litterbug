"""Tests for the frozen schedule and the task/argument contract."""

import pytest

from litterbug.common.config import (
    EPOCHS,
    IMGSZ,
    PATIENCE,
    SEED,
    SEG_ONLY_KEYS,
    TrainConfig,
    build_train_kwargs,
    resolve_batch,
    segmentation_args,
)
from litterbug.common.constants import DATA_YAML
from litterbug.common.runtime import resolve_workers


def test_frozen_schedule_defaults():
    assert (EPOCHS, PATIENCE, IMGSZ, SEED) == (100, 20, 640, 42)


def test_segment_run_includes_mask_arguments():
    kwargs = build_train_kwargs(TrainConfig(task="segment", batch=8))
    assert kwargs["overlap_mask"] is True
    assert kwargs["mask_ratio"] == 4


def test_detect_run_never_carries_segmentation_arguments():
    kwargs = build_train_kwargs(TrainConfig(task="detect", batch=8))
    assert not set(kwargs) & set(SEG_ONLY_KEYS)


def test_segmentation_args_are_rejected_for_detect():
    with pytest.raises(ValueError):
        segmentation_args("detect")


def test_unknown_task_is_rejected():
    with pytest.raises(ValueError):
        build_train_kwargs(TrainConfig(task="classification"))


def test_zero_epochs_is_rejected():
    with pytest.raises(ValueError):
        build_train_kwargs(TrainConfig(task="segment", epochs=0))


def test_run_name_defaults_to_task_and_model():
    assert TrainConfig(task="segment").run_name.startswith("segment-yolo")


def test_explicit_batch_wins_over_the_vram_probe():
    assert resolve_batch(4) == 4


def test_data_yaml_points_at_the_training_split():
    assert DATA_YAML.name == "data.yaml"
    assert DATA_YAML.parent.name == "1_Model_Training_Data"


def test_workers_override_wins_over_the_shm_probe():
    assert resolve_workers(4) == 4


def test_negative_worker_counts_are_clamped_to_zero():
    assert resolve_workers(-3) == 0


def test_train_kwargs_carry_a_usable_worker_count():
    # /dev/shm is small in this container, so the resolved value is expected to be low —
    # but it must always be a number the data loader can accept.
    assert build_train_kwargs(TrainConfig(task="segment", batch=8))["workers"] >= 0
