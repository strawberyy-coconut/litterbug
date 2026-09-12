"""Tests for the dataset validator.

A validator that quietly stops checking is worse than no validator: it reports PASS and the dataset
gets trained on anyway. These pin the checks that matter, and the ``data.yaml`` resolution, which is
what decides whether the right directories are being inspected at all.
"""

from pathlib import Path

from litterbug.training.validate_data import check_split, collect_splits, label_dir_for

GOOD_LABEL = "0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5\n"


def _split(root: Path, images: dict[str, bytes], labels: dict[str, str]) -> tuple[Path, Path]:
    images_dir = root / "train" / "images"
    labels_dir = root / "train" / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in images.items():
        (images_dir / name).write_bytes(payload)
    for name, payload in labels.items():
        (labels_dir / name).write_text(payload, encoding="utf-8")
    return images_dir, labels_dir


def test_label_dir_for_maps_images_to_labels():
    assert label_dir_for(Path("/data/train/images")) == Path("/data/train/labels")


def test_clean_split_passes_and_counts_instances(tmp_path):
    images_dir, labels_dir = _split(
        tmp_path,
        {"a.jpg": b"a"},
        {"a.txt": GOOD_LABEL + "3 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n"},
    )

    report = check_split("train", images_dir, labels_dir, require_labels=False, max_examples=5)

    assert report.ok
    assert report.images == 1
    assert report.instances == 2
    assert report.per_class[0] == 1
    assert report.per_class[3] == 1


def test_coordinate_outside_the_unit_square_is_an_error(tmp_path):
    images_dir, labels_dir = _split(
        tmp_path, {"a.jpg": b"a"}, {"a.txt": "0 0.1 0.1 1.5 0.1 0.5 0.5\n"}
    )

    report = check_split("train", images_dir, labels_dir, require_labels=False, max_examples=5)

    assert not report.ok
    assert any("outside [0,1]" in why for _, why in report.bad_labels)


def test_degenerate_polygon_is_an_error(tmp_path):
    # Four points, but only two distinct: not a polygon.
    images_dir, labels_dir = _split(
        tmp_path, {"a.jpg": b"a"}, {"a.txt": "0 0.1 0.1 0.1 0.1 0.2 0.2 0.2 0.2\n"}
    )

    report = check_split("train", images_dir, labels_dir, require_labels=False, max_examples=5)

    assert not report.ok
    assert report.instances == 0
    assert any("degenerate" in why for _, why in report.bad_labels)


def test_class_id_out_of_range_is_an_error(tmp_path):
    images_dir, labels_dir = _split(
        tmp_path, {"a.jpg": b"a"}, {"a.txt": "9 0.1 0.1 0.5 0.1 0.5 0.5\n"}
    )

    report = check_split("train", images_dir, labels_dir, require_labels=False, max_examples=5)

    assert not report.ok
    assert any("out of range" in why for _, why in report.bad_labels)


def test_orphan_label_is_an_error(tmp_path):
    images_dir, labels_dir = _split(tmp_path, {"a.jpg": b"a"}, {"ghost.txt": GOOD_LABEL})

    report = check_split("train", images_dir, labels_dir, require_labels=False, max_examples=5)

    assert not report.ok
    assert report.orphan_labels == ["ghost.txt"]


def test_duplicate_images_are_reported(tmp_path):
    images_dir, labels_dir = _split(
        tmp_path, {"a.jpg": b"same", "b.jpg": b"same"}, {"a.txt": GOOD_LABEL, "b.txt": GOOD_LABEL}
    )

    report = check_split("train", images_dir, labels_dir, require_labels=False, max_examples=5)

    assert not report.ok
    assert report.duplicate_images == [("a.jpg", "b.jpg")]


def test_missing_label_is_a_note_unless_labels_are_required(tmp_path):
    images_dir, labels_dir = _split(tmp_path, {"a.jpg": b"a"}, {})

    lenient = check_split("train", images_dir, labels_dir, require_labels=False, max_examples=5)
    strict = check_split("train", images_dir, labels_dir, require_labels=True, max_examples=5)

    # An image with no label may legitimately be a blank; it should not fail the check by default.
    assert lenient.ok
    assert lenient.missing_labels == ["a.jpg"]
    assert not strict.ok


def test_collect_splits_resolves_entries_relative_to_data_yaml(tmp_path):
    # No `path:` key, so `train: train/images` is relative to the yaml's own directory - the shape
    # this project's data.yaml has, and the thing the validator would silently get wrong.
    for split in ("train", "valid", "test"):
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "labels").mkdir(parents=True)
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        "train: train/images\nval: valid/images\ntest: test/images\n"
        "nc: 4\nnames: ['Glass', 'Metal', 'Paper', 'Plastic']\n",
        encoding="utf-8",
    )

    splits = collect_splits(data_yaml=data_yaml)

    assert set(splits) == {"train", "val", "test"}
    images_dir, labels_dir = splits["val"]
    assert images_dir == (tmp_path / "valid" / "images").resolve()
    assert labels_dir == (tmp_path / "valid" / "labels").resolve()
