"""Tests for the error-analysis matching and ground-truth parsing.

This module produces the report's error analysis, so a silent bug here would put wrong
numbers in the write-up. The matching convention and the label parser are pinned.
"""

import numpy as np
import pytest

from training.error_analysis import (
    _box_iou,
    _iou,
    _quartiles,
    _rasterize,
    best_iou,
    iou_summary,
    load_labels,
    match_image,
    match_with_iou,
    split_paths,
)


def _square(x1, y1, size):
    return np.array([[x1, y1], [x1 + size, y1], [x1 + size, y1 + size], [x1, y1 + size]])


class _Truth:
    def __init__(self, class_id, polygon):
        self.class_id = class_id
        self.polygon = polygon


def _prediction(class_id, box, polygon=None, confidence=0.9):
    return {"class_id": class_id, "confidence": confidence, "box": box, "polygon": polygon}


def test_iou_of_identical_masks_is_one():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:6, 2:6] = True

    assert _iou(mask, mask) == 1.0


def test_iou_of_disjoint_masks_is_zero():
    first = np.zeros((10, 10), dtype=bool)
    second = np.zeros((10, 10), dtype=bool)
    first[0:2, 0:2] = True
    second[8:10, 8:10] = True

    assert _iou(first, second) == 0.0


def test_box_iou_of_identical_boxes_is_one():
    assert _box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_box_iou_of_disjoint_boxes_is_zero():
    assert _box_iou((0, 0, 5, 5), (6, 6, 10, 10)) == 0.0


def test_load_labels_scales_normalised_coordinates_to_pixels(tmp_path):
    label = tmp_path / "a.txt"
    label.write_text("1 0.0 0.0 0.5 0.0 0.5 0.5 0.0 0.5\n", encoding="utf-8")

    instances = load_labels(label, width=200, height=100)

    assert len(instances) == 1
    assert instances[0].class_id == 1
    np.testing.assert_allclose(instances[0].polygon[1], [100.0, 0.0])


def test_load_labels_skips_malformed_rows(tmp_path):
    label = tmp_path / "a.txt"
    label.write_text(
        "0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n"  # valid
        "0 0.1 0.1 0.2\n"  # too few tokens
        "0 0.1 0.1 0.2 0.1 0.2\n"  # odd number of coordinates
        "\n",
        encoding="utf-8",
    )

    assert len(load_labels(label, 100, 100)) == 1


def test_load_labels_returns_empty_for_missing_file(tmp_path):
    assert load_labels(tmp_path / "absent.txt", 10, 10) == []


def test_match_requires_overlap_above_threshold():
    truth = [_Truth(0, _square(0, 0, 10))]
    prediction = [_prediction(0, (0, 0, 10, 10), _square(0, 0, 10))]

    assert match_image(truth, prediction, 100, 100, 0.5, "mask") == [(0, 0)]
    assert match_image(truth, prediction, 100, 100, 0.5, "box") == [(0, 0)]


def test_match_is_class_agnostic_so_confusion_is_visible():
    # A Plastic prediction sitting exactly on a Paper object must register as a match,
    # otherwise confusion would be reported as a miss plus a false positive.
    truth = [_Truth(2, _square(0, 0, 10))]  # Paper
    prediction = [_prediction(3, (0, 0, 10, 10), _square(0, 0, 10))]  # Plastic

    assert match_image(truth, prediction, 100, 100, 0.5, "mask") == [(0, 0)]


def test_match_is_one_to_one():
    truth = [_Truth(0, _square(0, 0, 10))]
    predictions = [
        _prediction(0, (0, 0, 10, 10), _square(0, 0, 10), confidence=0.9),
        _prediction(0, (0, 0, 10, 10), _square(0, 0, 10), confidence=0.8),
    ]

    assert match_image(truth, predictions, 100, 100, 0.5, "mask") == [(0, 0)]


def test_match_prefers_the_higher_confidence_prediction():
    truth = [_Truth(0, _square(0, 0, 10))]
    predictions = [
        _prediction(0, (0, 0, 10, 10), _square(0, 0, 10), confidence=0.4),
        _prediction(0, (0, 0, 10, 10), _square(0, 0, 10), confidence=0.95),
    ]

    assert match_image(truth, predictions, 100, 100, 0.5, "mask") == [(0, 1)]


def test_match_with_iou_reports_the_matched_pair():
    truth = [_Truth(0, _square(0, 0, 10))]
    prediction = [_prediction(0, (0, 0, 10, 10), _square(0, 0, 10))]

    (truth_index, prediction_index, _) = match_with_iou(
        truth, prediction, 100, 100, 0.5, "mask"
    )[0]

    assert (truth_index, prediction_index) == (0, 0)


def test_match_with_iou_equals_the_mask_iou_used_for_the_gate():
    # The reported IoU must be the very quantity that decided the match, not a re-derivation
    # that could drift away from the gate it is meant to describe.
    truth_polygon = _square(0, 0, 20)
    prediction_polygon = _square(10, 0, 20)
    truth = [_Truth(0, truth_polygon)]
    prediction = [_prediction(0, (10, 0, 30, 20), prediction_polygon)]

    (_, _, iou) = match_with_iou(truth, prediction, 100, 100, 0.2, "mask")[0]

    expected = _iou(
        _rasterize(truth_polygon, 100, 100), _rasterize(prediction_polygon, 100, 100)
    )
    assert iou == pytest.approx(expected)
    assert 0.0 < iou < 1.0


def test_best_iou_ignores_whether_the_instance_was_already_matched():
    # Two identical objects and one prediction. The point of this helper is that it reports the
    # overlap regardless of matching bookkeeping, which is what lets a duplicate detection be told
    # apart from a prediction covering bare belt.
    truths = [_Truth(0, _square(0, 0, 10)), _Truth(0, _square(0, 0, 10))]
    prediction = _prediction(0, (0, 0, 10, 10), _square(0, 0, 10))

    assert best_iou(truths, prediction, 100, 100, "mask") == pytest.approx(1.0)


def test_best_iou_is_zero_when_the_prediction_overlaps_nothing():
    truths = [_Truth(0, _square(0, 0, 10))]
    prediction = _prediction(0, (50, 50, 60, 60), _square(50, 50, 10))

    assert best_iou(truths, prediction, 100, 100, "mask") == 0.0


def test_iou_summary_uses_true_positives_only_for_per_class_figures():
    # A cross-class match says nothing about the predicted class's mask quality, so it must not
    # be folded into that class's figure.
    edges = np.array([10.0, 20.0, 30.0])
    records = [
        (0, 0, 1.0, 5.0),
        (0, 0, 0.8, 25.0),
        (1, 0, 0.9, 15.0),
    ]

    summary = iou_summary(records, edges)

    assert summary["matched_pairs"] == 3
    assert summary["mean"] == pytest.approx(0.9)
    assert summary["by_class"]["Glass"]["pairs"] == 2
    assert summary["by_class"]["Glass"]["mean"] == pytest.approx(0.9)
    assert summary["by_class"]["Metal"]["pairs"] == 0
    assert summary["by_class"]["Metal"]["mean"] is None


def test_iou_summary_buckets_by_ground_truth_area():
    edges = np.array([10.0, 20.0, 30.0])
    records = [(0, 0, 0.4, 5.0), (0, 0, 0.9, 35.0)]

    summary = iou_summary(records, edges)

    assert summary["by_size_quartile"][0]["mean"] == pytest.approx(0.4)
    assert summary["by_size_quartile"][3]["mean"] == pytest.approx(0.9)


def test_quartiles_split_records_into_four_buckets():
    records = [(float(value), value >= 8) for value in range(16)]

    buckets = _quartiles(records, "area")

    assert [bucket["instances"] for bucket in buckets] == [4, 4, 4, 4]
    assert buckets[3]["recall"] == 1.0


def test_split_paths_resolves_labels_next_to_images():
    images, labels = split_paths("val")

    assert images.name == "images"
    assert labels.name == "labels"
    assert labels.parent == images.parent


def test_split_paths_rejects_an_unknown_split():
    with pytest.raises(ValueError):
        split_paths("nope")
