"""Tests for the class contract and the model registry.

The class order is a contract with the annotations; these tests exist so a careless
edit fails loudly instead of silently retraining on mislabelled data.
"""

from common.constants import CLASS_COLORS, CLASS_IDS, CLASS_NAMES, MODELS, TASKS


def test_class_order_is_the_dataset_contract():
    assert CLASS_NAMES == ["Glass", "Metal", "Paper", "Plastic"]
    assert CLASS_IDS == {"Glass": 0, "Metal": 1, "Paper": 2, "Plastic": 3}


def test_every_class_has_a_colour():
    assert set(CLASS_COLORS) == set(range(len(CLASS_NAMES)))


def test_model_registry_covers_every_task():
    assert set(MODELS) == set(TASKS)
    assert MODELS["segment"].endswith("-seg.pt")
    assert "-seg" not in MODELS["detect"]
