"""Validate a YOLO instance-segmentation dataset before training.

The checks a dataset needs before it is worth spending GPU hours on:

* image ↔ label pairing (missing labels, orphan labels, ambiguous stems)
* label format — ``class x1 y1 x2 y2 ...`` rows
* normalised coordinates inside [0, 1], and polygons with three distinct points
* class ids within range, plus per-class instance counts
* byte-identical duplicate images

The core is standard-library only, so a broken dataset can be diagnosed before ``ultralytics`` or
``opencv`` are importable. ``--overlay`` additionally needs ``numpy`` and ``opencv-python``.

This lives in the package rather than beside an agent skill because running it is something a user
does, and everything a user runs should be reachable from the CLI.
"""

from __future__ import annotations

import hashlib
import logging
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from litterbug.common.constants import (
    CLASS_COLORS,
    CLASS_NAMES,
    DATA_YAML,
    IMAGE_EXTS,
    PROJECT_ROOT,
)
from litterbug.common.runtime import setup_logging

log = logging.getLogger("litterbug.validate")

# Keys Ultralytics allows for the validation split, and the one this project's data.yaml uses.
SPLIT_KEYS = ("train", "val", "valid", "test")
COORD_EPS = 1e-6
OVERLAY_DIR = PROJECT_ROOT / "validation_overlays"


@dataclass(frozen=True)
class ValidationConfig:
    data_yaml: Path = DATA_YAML
    data_root: Path | None = None
    require_labels: bool = False
    overlay: int = 0
    output: Path = OVERLAY_DIR
    seed: int = 42
    max_examples: int = 5


@dataclass
class SplitReport:
    """Everything found in one split, whether or not it was a problem."""

    name: str
    images_dir: Path
    labels_dir: Path
    images: int = 0
    labels: int = 0
    instances: int = 0
    per_class: Counter = field(default_factory=Counter)
    missing_labels: list[str] = field(default_factory=list)
    orphan_labels: list[str] = field(default_factory=list)
    ambiguous: list[str] = field(default_factory=list)
    bad_labels: list[tuple[str, str]] = field(default_factory=list)
    duplicate_images: list[tuple[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Images with no label file are reported but are not errors: they may be empties."""
        return not (
            self.orphan_labels or self.ambiguous or self.bad_labels or self.duplicate_images
        )

    @property
    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.name,
            "images": self.images,
            "labels": self.labels,
            "instances": self.instances,
            "instances_per_image": round(self.instances / self.images, 3) if self.images else None,
            "per_class": {name: self.per_class.get(i, 0) for i, name in enumerate(CLASS_NAMES)},
            "images_without_labels": len(self.missing_labels),
            "orphan_labels": len(self.orphan_labels),
            "ambiguous_stems": len(self.ambiguous),
            "invalid_labels": len(self.bad_labels),
            "duplicate_images": len(self.duplicate_images),
            "ok": self.ok,
        }


def label_dir_for(images_dir: Path) -> Path:
    """Derive the labels directory from an images directory, as the YOLO layout implies."""
    parts = list(images_dir.parts)
    if "images" in parts:
        index = len(parts) - 1 - parts[::-1].index("images")
        parts[index] = "labels"
        return Path(*parts)
    return images_dir.parent / "labels"


def _split_from_root(root: Path, split: str) -> tuple[Path, Path] | None:
    """Support both ``<root>/<split>/images`` and ``<root>/images/<split>``."""
    for images_dir in (root / split / "images", root / "images" / split):
        if images_dir.is_dir():
            return images_dir, label_dir_for(images_dir)
    return None


def collect_splits(
    data_yaml: Path | None = None, data_root: Path | None = None
) -> dict[str, tuple[Path, Path]]:
    """Resolve each split to its (images, labels) directories.

    Split entries are resolved relative to ``data.yaml``'s own directory when the file declares no
    ``path:`` key, which is how Ultralytics treats it and how this project's data.yaml is written.
    """
    splits: dict[str, tuple[Path, Path]] = {}

    if data_root is not None:
        for split in SPLIT_KEYS:
            found = _split_from_root(data_root, split)
            if found:
                splits[split] = found
        return splits

    if data_yaml is None or not data_yaml.is_file():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    import yaml

    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    base = data_yaml.parent.resolve()
    if payload.get("path"):
        declared = Path(str(payload["path"]))
        base = declared if declared.is_absolute() else (data_yaml.parent / declared).resolve()

    for split in SPLIT_KEYS:
        entry = payload.get(split)
        if not entry:
            continue
        images_dir = Path(str(entry))
        if not images_dir.is_absolute():
            images_dir = (base / images_dir).resolve()
        if images_dir.is_dir():
            splits[split] = (images_dir, label_dir_for(images_dir))
        else:
            log.warning("%s: images directory does not exist: %s", split, images_dir)
    return splits


def _hash_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def check_split(
    name: str,
    images_dir: Path,
    labels_dir: Path,
    require_labels: bool,
    max_examples: int,
) -> SplitReport:
    """Run every check over one split."""
    report = SplitReport(name=name, images_dir=images_dir, labels_dir=labels_dir)
    class_count = len(CLASS_NAMES)

    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    label_paths = sorted(labels_dir.glob("*.txt")) if labels_dir.is_dir() else []
    report.images, report.labels = len(image_paths), len(label_paths)

    by_stem: dict[str, list[Path]] = {}
    for path in image_paths:
        by_stem.setdefault(path.stem, []).append(path)
    report.ambiguous = [
        f"{stem} -> {', '.join(p.name for p in paths)}"
        for stem, paths in by_stem.items()
        if len(paths) > 1
    ]

    label_stems = {p.stem for p in label_paths}
    for stem, paths in by_stem.items():
        if stem in label_stems:
            continue
        if require_labels:
            report.bad_labels.append((paths[0].name, "missing label file"))
        else:
            report.missing_labels.append(paths[0].name)

    for label_path in label_paths:
        if label_path.stem not in by_stem:
            report.orphan_labels.append(label_path.name)
            continue

        issues: list[str] = []
        for lineno, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw.strip()
            if not line:
                continue
            tokens = line.split()
            try:
                class_id = int(float(tokens[0]))
            except (ValueError, IndexError):
                issues.append(f"line {lineno}: unparsable class id {raw!r}")
                continue

            coords = tokens[1:]
            if not 0 <= class_id < class_count:
                issues.append(
                    f"line {lineno}: class id {class_id} out of range [0,{class_count - 1}]"
                )
                continue
            if len(coords) % 2 != 0 or len(coords) < 6:
                issues.append(f"line {lineno}: need >=3 (x,y) points, got {len(coords)} values")
                continue

            try:
                values = [float(value) for value in coords]
            except ValueError:
                issues.append(f"line {lineno}: non-numeric coordinate in {raw!r}")
                continue
            if any(not (-COORD_EPS <= value <= 1.0 + COORD_EPS) for value in values):
                offender = next(
                    v for v in values if not (-COORD_EPS <= v <= 1.0 + COORD_EPS)
                )
                issues.append(f"line {lineno}: coordinate {offender} outside [0,1]")
                continue

            xs, ys = values[0::2], values[1::2]
            if len(set(zip(xs, ys, strict=True))) < 3:
                issues.append(
                    f"line {lineno}: degenerate polygon (fewer than 3 distinct points)"
                )
                continue
            report.instances += 1
            report.per_class[class_id] += 1

        if issues:
            report.bad_labels.append((label_path.name, "; ".join(issues[:max_examples])))

    seen: dict[str, str] = {}
    for path in image_paths:
        digest = _hash_file(path)
        if digest in seen:
            report.duplicate_images.append((seen[digest], path.name))
        else:
            seen[digest] = path.name

    return report


def render_overlays(
    reports: list[SplitReport], out_dir: Path, count: int, seed: int
) -> int:
    """Draw polygon outlines on a random sample of images, for a visual spot-check."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("\n! Overlay skipped: needs numpy and opencv-python (`uv add numpy opencv-python`).")
        return 0

    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    for report in reports:
        images = sorted(p for p in report.images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not images:
            continue
        for image_path in rng.sample(images, min(count, len(images))):
            label_path = report.labels_dir / f"{image_path.stem}.txt"
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            height, width = image.shape[:2]
            scale = np.array([width, height], dtype="float32")
            for raw in (
                label_path.read_text(encoding="utf-8").splitlines()
                if label_path.is_file()
                else []
            ):
                tokens = raw.split()
                if len(tokens) < 7:
                    continue
                class_id = int(float(tokens[0]))
                points = np.array([float(v) for v in tokens[1:]], dtype="float32")
                points = (points.reshape(-1, 2) * scale).astype("int32")
                colour = CLASS_COLORS[class_id % len(CLASS_COLORS)]
                cv2.polylines(image, [points], True, colour, 2)
                cv2.putText(
                    image,
                    CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else str(class_id),
                    tuple(points[0]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    colour,
                    2,
                    cv2.LINE_AA,
                )
            if cv2.imwrite(str(out_dir / f"{report.name}_{image_path.name}"), image):
                written += 1

    print(f"\nWrote {written} overlay image(s) to {out_dir}")
    return written


def print_report(reports: list[SplitReport], max_examples: int) -> bool:
    clean = True
    for report in reports:
        clean &= report.ok
        print(f"\n=== {report.name} ===")
        print(
            f"  images: {report.images}   labels: {report.labels}   "
            f"instances: {report.instances}"
        )
        if report.images:
            print(f"  instances/image: {report.instances / report.images:.2f}")
        counts = ", ".join(
            f"{name}={report.per_class.get(i, 0)}" for i, name in enumerate(CLASS_NAMES)
        )
        print(f"  per class: {counts}")

        def show(title: str, items: list[str]) -> None:
            if items:
                print(f"  ERROR {title}: {len(items)}")
                for item in items[:max_examples]:
                    print(f"    - {item}")

        show("orphan labels (no matching image)", report.orphan_labels)
        show("ambiguous image stems (same name, multiple files)", report.ambiguous)
        show("invalid label files", [f"{name}: {why}" for name, why in report.bad_labels])
        show("duplicate images", [f"{a} == {b}" for a, b in report.duplicate_images])
        if report.missing_labels:
            print(
                f"  NOTE images without a label file: {len(report.missing_labels)} "
                "(background images?)"
            )
            for item in report.missing_labels[:max_examples]:
                print(f"    - {item}")

        print(f"  -> {'OK' if report.ok else 'ISSUES FOUND'}")

    return clean


def validate_dataset(cfg: ValidationConfig | None = None) -> dict[str, Any]:
    """Validate the dataset and return a structured result. Also prints a human-readable report."""
    cfg = cfg or ValidationConfig()

    splits = collect_splits(cfg.data_yaml, cfg.data_root)
    if not splits:
        raise ValueError(
            "no splits found. Expected <root>/<split>/images, or a data.yaml whose split entries "
            "point at existing directories."
        )

    print(f"Validating {len(splits)} split(s): {', '.join(sorted(splits))}")
    print(f"Classes: {', '.join(f'{i}={n}' for i, n in enumerate(CLASS_NAMES))}")

    reports = [
        check_split(name, images, labels, cfg.require_labels, cfg.max_examples)
        for name, (images, labels) in sorted(splits.items())
    ]
    clean = print_report(reports, cfg.max_examples)

    if cfg.overlay:
        render_overlays(reports, cfg.output, cfg.overlay, cfg.seed)

    print("\nRESULT:", "PASS - no errors found" if clean else "FAIL - see errors above")

    return {
        "clean": clean,
        "splits": {report.name: report.to_dict for report in reports},
        "totals": {
            "images": sum(r.images for r in reports),
            "instances": sum(r.instances for r in reports),
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Allow ``python -m training.validate_data ...`` without a second argument parser."""
    from litterbug.cli.main import main as cli_main

    return cli_main(["validate", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    setup_logging()
    raise SystemExit(main())
