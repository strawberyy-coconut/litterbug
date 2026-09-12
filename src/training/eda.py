"""Exploratory data analysis over the YOLO segmentation splits.

Produces the dataset numbers the report needs: per-class instance counts and imbalance, image
geometry, and lighting. Everything is measured from the annotations and the pixels rather than
transcribed from an earlier run, so the figures can be regenerated after any change to the data.

`cv2` and `numpy` are imported inside the functions that need them, so importing this module — for
the notebook, or for a fast `--help` — stays cheap.
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.constants import CLASS_NAMES, IMAGE_EXTS, RUNS_DIR
from common.runtime import setup_logging
from training.error_analysis import split_paths

log = logging.getLogger("litterbug.eda")

# Keys as declared in data.yaml, which are *not* the directory names: `val` points at the
# `valid/` directory. Resolving through the YAML keeps that mismatch out of the rest of the code.
SPLITS = ("train", "val", "test")

DARK = 60.0
BRIGHT = 180.0


@dataclass(frozen=True)
class EdaConfig:
    splits: tuple[str, ...] = SPLITS
    project: Path = RUNS_DIR
    name: str = "dataset-eda"
    sample: int = 200
    seed: int = 42


def scan_split(split: str) -> dict[str, Any]:
    """Count images, instances and per-class instances for one split, from the labels alone."""
    images_dir, labels_dir = split_paths(split)
    images = sorted(path for path in images_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTS)

    counts: Counter[int] = Counter()
    without_instances = 0
    malformed = 0
    for image in images:
        label = labels_dir / f"{image.stem}.txt"
        rows = (
            [line for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
            if label.is_file()
            else []
        )
        if not rows:
            without_instances += 1
        for row in rows:
            try:
                class_id = int(row.split()[0])
            except (IndexError, ValueError):
                malformed += 1
                continue
            counts[class_id] += 1

    total = sum(counts.values())
    per_class = {name: counts.get(index, 0) for index, name in enumerate(CLASS_NAMES)}
    smallest = min(per_class.values()) if per_class else 0
    return {
        "split": split,
        "images": len(images),
        "instances": total,
        "instances_per_image": round(total / len(images), 2) if images else None,
        "images_without_instances": without_instances,
        "malformed_rows": malformed,
        "per_class": per_class,
        "per_class_share": {
            name: round(value / total, 4) if total else None for name, value in per_class.items()
        },
        # 1.0 is perfectly balanced. Reported because class imbalance is one of the things the
        # brief asks the EDA to establish rather than assume.
        "imbalance_ratio": round(max(per_class.values()) / smallest, 3)
        if per_class and smallest
        else None,
    }


def image_stats(split: str, sample: int, seed: int) -> dict[str, Any]:
    """Geometry and lighting over a random sample of a split.

    Brightness is the mean greyscale level; sharpness is the variance of the Laplacian, the same
    measure used to characterise the video footage in §8, so the two are directly comparable.
    Medians are reported because a few outlier images move a mean.
    """
    import cv2
    import numpy as np

    images_dir, _ = split_paths(split)
    paths = sorted(path for path in images_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTS)
    if not paths:
        return {"sampled": 0}

    rng = np.random.default_rng(seed)
    take = min(sample, len(paths))
    chosen = [paths[index] for index in rng.choice(len(paths), size=take, replace=False)]

    sizes: Counter[tuple[int, int]] = Counter()
    channels: Counter[int] = Counter()
    brightness: list[float] = []
    sharpness: list[float] = []
    for path in chosen:
        image = cv2.imread(str(path))
        if image is None:
            continue
        height, width = image.shape[:2]
        sizes[(width, height)] += 1
        channels[image.shape[2]] += 1
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness.append(float(grey.mean()))
        sharpness.append(float(cv2.Laplacian(grey, cv2.CV_64F).var()))

    levels = np.asarray(brightness, dtype=np.float64)
    edges = np.asarray(sharpness, dtype=np.float64)
    if not len(levels):
        return {"sampled": 0}

    return {
        "sampled": len(levels),
        "resolutions": [f"{width}x{height}" for (width, height), _ in sizes.most_common()],
        "single_resolution": len(sizes) == 1,
        "channels": {str(key): value for key, value in channels.items()},
        "brightness_median": round(float(np.median(levels)), 2),
        "brightness_p10": round(float(np.percentile(levels, 10)), 2),
        "brightness_p90": round(float(np.percentile(levels, 90)), 2),
        "exposure": {
            f"dark (mean < {DARK:.0f})": round(float((levels < DARK).mean()), 4),
            f"mid ({DARK:.0f}-{BRIGHT:.0f})": round(
                float(((levels >= DARK) & (levels <= BRIGHT)).mean()), 4
            ),
            f"bright (mean > {BRIGHT:.0f})": round(float((levels > BRIGHT).mean()), 4),
        },
        "sharpness_median": round(float(np.median(edges)), 1),
    }


def eda(cfg: EdaConfig) -> dict[str, Any]:
    """Run the dataset EDA and write it as JSON. Returns the same structure it writes."""
    report: dict[str, Any] = {
        "splits": {},
        "sampled_images_per_split": cfg.sample,
        "seed": cfg.seed,
    }
    for split in cfg.splits:
        report["splits"][split] = {
            **scan_split(split),
            "image_stats": image_stats(split, cfg.sample, cfg.seed),
        }

    out_dir = cfg.project / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = out_dir / "dataset_eda.json"
    payload.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("Wrote %s", payload)
    print_summary(report)
    return report


def print_summary(report: dict[str, Any]) -> None:
    splits = report["splits"]
    print(f"\ndataset EDA | seed {report['seed']} | up to "
          f"{report['sampled_images_per_split']} images sampled per split for pixel statistics")

    print(f"\n  {'split':<7}{'images':>8}{'instances':>11}{'inst/image':>12}{'no instances':>14}"
          f"{'imbalance':>11}")
    for name, row in splits.items():
        print(f"  {name:<7}{row['images']:>8}{row['instances']:>11}"
              f"{row['instances_per_image']:>12}{row['images_without_instances']:>14}"
              f"{row['imbalance_ratio']:>11}")

    header = f"\n  {'split':<7}" + "".join(f"{name:>10}" for name in CLASS_NAMES)
    print(f"{header}{'malformed':>11}")
    for name, row in splits.items():
        cells = "".join(f"{row['per_class'][cls]:>10}" for cls in CLASS_NAMES)
        print(f"  {name:<7}{cells}{row['malformed_rows']:>11}")

    print(f"\n  {'split':<7}" + "".join(f"{name + ' %':>10}" for name in CLASS_NAMES))
    for name, row in splits.items():
        cells = "".join(f"{row['per_class_share'][cls] * 100:>9.1f}%" for cls in CLASS_NAMES)
        print(f"  {name:<7}{cells}")

    print(f"\n  {'split':<7}{'resolution':>12}{'median bright':>15}{'dark':>8}{'bright':>9}"
          f"{'sharpness':>11}")
    for name, row in splits.items():
        stats = row["image_stats"]
        if not stats.get("sampled"):
            print(f"  {name:<7}{'n/a':>12}")
            continue
        exposure = stats["exposure"]
        dark = next(value for key, value in exposure.items() if key.startswith("dark"))
        bright = next(value for key, value in exposure.items() if key.startswith("bright"))
        print(f"  {name:<7}{stats['resolutions'][0]:>12}{stats['brightness_median']:>15}"
              f"{dark * 100:>7.0f}%{bright * 100:>8.0f}%{stats['sharpness_median']:>11}")


def main(argv: list[str] | None = None) -> int:
    """Allow ``python -m training.eda ...`` without a second argument parser."""
    from cli.main import main as cli_main

    return cli_main(["eda", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    setup_logging()
    raise SystemExit(main())
