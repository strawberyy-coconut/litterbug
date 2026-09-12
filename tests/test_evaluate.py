"""Tests for evaluation helpers.

Both of these pin real bugs. Ultralytics returns per-class metrics as numpy arrays, and
``array or []`` raises "the truth value of an array with more than one element is
ambiguous", which aborted evaluation before ``metrics.json`` was written. The run-name
helper previously collapsed to ``-val`` when handed a bare checkpoint filename.
"""

from pathlib import Path

import numpy as np

from training.evaluate import EvalConfig, per_class_map, run_name_for


class _Metric:
    """Stands in for Ultralytics' Metric, which exposes numpy arrays."""

    def __init__(self, maps, ids):
        self.maps = maps
        self.ap_class_index = ids


def test_per_class_map_handles_numpy_arrays():
    metric = _Metric(np.array([0.51, 0.72]), np.array([0, 1]))

    assert per_class_map(metric) == {"Glass": 0.51, "Metal": 0.72}


def test_per_class_map_labels_every_class():
    metric = _Metric(np.array([0.1, 0.2, 0.3, 0.4]), np.array([0, 1, 2, 3]))

    assert list(per_class_map(metric)) == ["Glass", "Metal", "Paper", "Plastic"]


def test_per_class_map_tolerates_missing_attributes():
    assert per_class_map(object()) == {}


def test_run_name_uses_the_parent_run_directory(tmp_path):
    weights = tmp_path / "my-run" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.touch()

    assert run_name_for(weights, "val") == "my-run-val"


def test_run_name_survives_a_bare_filename(tmp_path, monkeypatch):
    # Reproduces running `litterbug val --weights best.pt` from inside the weights dir,
    # which used to produce a directory literally named "-val".
    weights = tmp_path / "seg-run" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.touch()
    monkeypatch.chdir(weights.parent)

    assert run_name_for(Path("best.pt"), "val") == "seg-run-val"


def test_run_name_falls_back_for_a_loose_checkpoint(tmp_path):
    weights = tmp_path / "handmade.pt"
    weights.touch()

    assert run_name_for(weights, "test") == "handmade-test"


def test_eval_config_prefers_an_explicit_name(tmp_path):
    weights = tmp_path / "run" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.touch()

    assert EvalConfig(weights=weights, name="custom", split="val").run_name == "custom"
