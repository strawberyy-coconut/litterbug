"""Tests for the tracking statistics added to video inference.

Track length is the one video measurement that does not need ground truth, so it is used to
characterise footage that cannot be labelled. A silent bug here would put a wrong stability figure
in the report, which is the same failure mode the confidence-threshold and matching tests guard.
"""

import numpy as np
import pytest

from running.predict import Detection, _track_ids, _track_summary


def _detection(track_id, class_name="Glass"):
    return Detection(
        class_id=0,
        class_name=class_name,
        confidence=0.9,
        box_xyxy=(0.0, 0.0, 1.0, 1.0),
        track_id=track_id,
    )


class _Boxes:
    """Minimal stand-in for an Ultralytics Boxes object."""

    def __init__(self, ids, count=2):
        self.id = None if ids is None else np.asarray(ids, dtype=float)
        self._count = count

    def __len__(self):
        return self._count


def test_track_ids_are_none_without_a_tracker():
    assert _track_ids(_Boxes(None)) == [None, None]


def test_track_ids_are_plain_ints():
    assert _track_ids(_Boxes([3.0, 7.0])) == [3, 7]


def test_track_summary_measures_persistence():
    # One track seen in three frames and one seen in a single frame.
    detections = [_detection(1), _detection(1), _detection(1), _detection(2)]
    per_frame = [{"frame": number} for number in (1, 2, 3)]

    summary = _track_summary(detections, per_frame)

    assert summary["unique_tracks"] == 2
    assert summary["mean_track_length_frames"] == pytest.approx(2.0)
    assert summary["median_track_length_frames"] == pytest.approx(2.0)
    assert summary["max_track_length_frames"] == 3
    assert summary["single_frame_tracks"] == 1
    assert summary["share_single_frame"] == pytest.approx(0.5)


def test_track_summary_is_empty_when_nothing_was_tracked():
    assert _track_summary([_detection(None)], [{"frame": 1}]) == {}


def test_track_summary_breaks_tracks_down_by_class():
    detections = [_detection(1, "Paper"), _detection(1, "Paper"), _detection(2, "Metal")]

    summary = _track_summary(detections, [{"frame": 1}, {"frame": 2}])

    assert summary["per_class"]["Paper"] == {"tracks": 1, "median_length_frames": 2}
    assert summary["per_class"]["Metal"] == {"tracks": 1, "median_length_frames": 1}
