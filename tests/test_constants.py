"""Tests for the class contract and the model registry.

The class order is a contract with the annotations; these tests exist so a careless
edit fails loudly instead of silently retraining on mislabelled data.
"""

from litterbug.common.constants import (
    CLASS_COLORS,
    CLASS_IDS,
    CLASS_NAMES,
    MODELS,
    PROJECT_ROOT,
    TASKS,
)


def test_project_root_is_the_repository():
    """PROJECT_ROOT is derived from this module's depth in the package.

    Moving the package one level up or down would silently point data and run paths at the
    wrong place, so pin it to something that only exists at the repository root.
    """
    assert (PROJECT_ROOT / "pyproject.toml").is_file()
    assert (PROJECT_ROOT / "src" / "litterbug" / "common" / "constants.py").is_file()


def test_class_order_is_the_dataset_contract():
    assert CLASS_NAMES == ["Glass", "Metal", "Paper", "Plastic"]
    assert CLASS_IDS == {"Glass": 0, "Metal": 1, "Paper": 2, "Plastic": 3}


def test_every_class_has_a_colour():
    assert set(CLASS_COLORS) == set(range(len(CLASS_NAMES)))


def test_model_registry_covers_every_task():
    assert set(MODELS) == set(TASKS)
    assert MODELS["segment"].endswith("-seg.pt")
    assert "-seg" not in MODELS["detect"]
