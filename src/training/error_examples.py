"""Annotated examples of false positives and false negatives.

The brief asks for *commented* examples, not merely counts, so this module selects the images where
each error type is most visible and renders them as side-by-side panels: ground truth on the left,
predictions on the right, with the unmatched instances outlined so a reader can see at a glance what
was missed and what was invented.

The commentary it writes is **measured, not interpreted** — instance counts, class composition, the
size quartile of the misses, the confidences of the false positives. Those are facts the figures
show. Why the model behaves that way is a judgement call and is left to the report, because a
generated sentence asserting a cause would be indistinguishable from one the evidence supports.

Figures default to `report/figures/` rather than `runs/`, unlike every other command here: they are
report deliverables, not run outputs, and `runs/` is git-ignored, so anything written there is
unavailable to the write-up.
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common.config import IMGSZ
from common.constants import CLASS_COLORS, CLASS_NAMES, PROJECT_ROOT
from common.runtime import setup_logging
from training.error_analysis import (
    DEFAULT_CONF,
    DEFAULT_IOU,
    AnalysisConfig,
    TruthInstance,
    _rasterize,
    best_iou,
    load_labels,
    match_with_iou,
    predict_split,
    split_paths,
)

log = logging.getLogger("litterbug.examples")

FIGURES_DIR = PROJECT_ROOT / "report" / "figures"

# BGR. Red marks what the model missed; magenta marks what it invented. Neither appears in
# CLASS_COLORS, so a highlight can never be confused with a class.
MISSED_COLOUR = (0, 0, 255)
INVENTED_COLOUR = (255, 0, 255)
HEADER = 26

KINDS = ("misses", "false-positives", "confusion")


@dataclass(frozen=True)
class ExampleConfig:
    weights: Path
    split: str = "val"
    conf: float = DEFAULT_CONF
    iou_mode: str = "mask"
    iou_threshold: float = DEFAULT_IOU
    imgsz: int = IMGSZ
    batch: int | None = None
    workers: int | None = None
    device: str | None = None
    project: Path = FIGURES_DIR
    name: str | None = None
    per_kind: int = 3
    limit: int | None = None

    @property
    def run_name(self) -> str:
        suffix = "" if self.iou_mode == "mask" else f"-{self.iou_mode}"
        return self.name or f"examples-{self.split}{suffix}"


@dataclass(frozen=True)
class ImageErrors:
    """One image's ground truth, predictions and the three ways they can disagree."""

    image: str
    path: Path
    width: int
    height: int
    truths: list[TruthInstance]
    predictions: list[dict[str, Any]]
    matched: list[tuple[int, int, float]]
    missed: list[int]
    false_positives: list[int]
    confused: list[tuple[int, int]]
    truth_areas: list[float]
    duplicate_predictions: list[int] = field(default_factory=list)
    background_predictions: list[int] = field(default_factory=list)

    @property
    def true_positives(self) -> int:
        return len(self.matched) - len(self.confused)

    @property
    def recall(self) -> float | None:
        return len(self.matched) / len(self.truths) if self.truths else None

    @property
    def precision(self) -> float | None:
        return len(self.matched) / len(self.predictions) if self.predictions else None


def collect(cfg: ExampleConfig) -> tuple[list[ImageErrors], Any]:
    """Run the checkpoint over the split and record every image's matches and mismatches.

    Returns the per-image records and the ground-truth size-quartile edges for the split, which the
    commentary needs in order to say where a missed object sits in the size distribution.
    """
    import numpy as np

    files, grouped, sizes = predict_split(
        AnalysisConfig(
            weights=cfg.weights,
            split=cfg.split,
            conf=cfg.conf,
            iou_mode=cfg.iou_mode,
            iou_threshold=cfg.iou_threshold,
            imgsz=cfg.imgsz,
            batch=cfg.batch,
            workers=cfg.workers,
            device=cfg.device,
            project=cfg.project,
            limit=cfg.limit,
        )
    )
    _, labels_dir = split_paths(cfg.split)

    records: list[ImageErrors] = []
    areas: list[float] = []
    for path in files:
        width, height = sizes[path.stem]
        truths = load_labels(labels_dir / f"{path.stem}.txt", width, height)
        predictions = grouped.get(path.stem, [])
        matched = match_with_iou(
            truths, predictions, width, height, cfg.iou_threshold, cfg.iou_mode
        )
        matched_truth = {truth for truth, _, _ in matched}
        matched_prediction = {prediction for _, prediction, _ in matched}
        gt_masks = [_rasterize(item.polygon, width, height) for item in truths]
        truth_areas = [float(mask.sum()) for mask in gt_masks]
        areas.extend(truth_areas)

        false_positives = [i for i in range(len(predictions)) if i not in matched_prediction]
        # Matching is one-to-one, so a second prediction on an already-detected object is scored
        # as a false positive even though it covers a real object. Separate those from predictions
        # covering nothing at all: identical in the counts, different in cause and in remedy.
        duplicate_predictions: list[int] = []
        background_predictions: list[int] = []
        for index in false_positives:
            overlap = best_iou(
                truths, predictions[index], width, height, cfg.iou_mode, gt_masks
            )
            target = (
                duplicate_predictions
                if overlap >= cfg.iou_threshold
                else background_predictions
            )
            target.append(index)

        records.append(
            ImageErrors(
                image=path.stem,
                path=path,
                width=width,
                height=height,
                truths=truths,
                predictions=predictions,
                matched=matched,
                missed=[i for i in range(len(truths)) if i not in matched_truth],
                false_positives=false_positives,
                confused=[
                    (truth, prediction)
                    for truth, prediction, _ in matched
                    if truths[truth].class_id != predictions[prediction]["class_id"]
                ],
                truth_areas=truth_areas,
                duplicate_predictions=duplicate_predictions,
                background_predictions=background_predictions,
            )
        )
    edges = (
        np.quantile(np.asarray(areas, dtype=np.float64), [0.25, 0.5, 0.75])
        if areas
        else np.zeros(3)
    )
    return records, edges


def select_examples(records: list[ImageErrors], per_kind: int) -> dict[str, list[ImageErrors]]:
    """Pick the clearest examples of each error type.

    Images with nothing of the relevant kind are dropped rather than padding the gallery with
    empty panels, and ties break on image name so re-running on unchanged data selects the same
    images.
    """
    return {
        "misses": sorted(
            [r for r in records if r.missed], key=lambda r: (-len(r.missed), r.image)
        )[:per_kind],
        "false-positives": sorted(
            [r for r in records if r.false_positives],
            key=lambda r: (-len(r.false_positives), r.image),
        )[:per_kind],
        "confusion": sorted(
            [r for r in records if r.confused], key=lambda r: (-len(r.confused), r.image)
        )[:per_kind],
    }


def _draw_panel(
    image: Any,
    entries: list[tuple[Any, int, str]],
    highlight: set[int],
    highlight_colour: tuple[int, int, int],
    title: str,
) -> Any:
    """Tint each mask by class, outline the highlighted ones, and label above the panel."""
    import cv2
    import numpy as np

    canvas = image.copy()
    tint = image.copy()
    for polygon, class_id, _ in entries:
        if polygon is None or len(polygon) < 3:
            continue
        points = np.round(np.asarray(polygon, dtype=np.float64)).astype(np.int32)
        cv2.fillPoly(tint, [points], CLASS_COLORS.get(class_id, (255, 255, 255)))
    cv2.addWeighted(tint, 0.35, canvas, 0.65, 0, canvas)

    for index, (polygon, class_id, label) in enumerate(entries):
        if polygon is None or len(polygon) < 3:
            continue
        points = np.round(np.asarray(polygon, dtype=np.float64)).astype(np.int32)
        marked = index in highlight
        colour = highlight_colour if marked else CLASS_COLORS.get(class_id, (255, 255, 255))
        cv2.polylines(canvas, [points], True, colour, 3 if marked else 2)
        anchor = (int(points[:, 0].min()), max(12, int(points[:, 1].min()) - 3))
        # Outlined text: a single colour legible against both dark and pale regions of the belt.
        cv2.putText(
            canvas, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            canvas, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA
        )

    strip = np.zeros((HEADER, canvas.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        strip, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
    )
    return np.vstack([strip, canvas])


def _render(record: ImageErrors, conf: float) -> Any:
    """Ground truth beside predictions, with misses and unmatched predictions flagged."""
    import cv2
    import numpy as np

    image = cv2.imread(str(record.path))
    if image is None:
        raise FileNotFoundError(f"could not read {record.path}")

    truth_entries = [
        (item.polygon, item.class_id, CLASS_NAMES[item.class_id]) for item in record.truths
    ]
    prediction_entries = [
        (
            item["polygon"],
            item["class_id"],
            f"{CLASS_NAMES[item['class_id']]} {item['confidence']:.2f}",
        )
        for item in record.predictions
    ]

    left = _draw_panel(
        image,
        truth_entries,
        set(record.missed),
        MISSED_COLOUR,
        f"ground truth - {len(record.truths)} instances - {len(record.missed)} MISSED (red)",
    )
    right = _draw_panel(
        image,
        prediction_entries,
        set(record.false_positives),
        INVENTED_COLOUR,
        f"prediction conf>={conf} - {len(record.predictions)} masks - "
        f"{len(record.false_positives)} UNMATCHED (magenta)",
    )
    return np.hstack([left, right])


def _commentary(record: ImageErrors, edges: Any, figure: str) -> list[str]:
    """Measured description of one example. No causes are asserted here."""
    import numpy as np

    lines = [f"![{record.image}]({figure})", ""]
    lines.append(
        f"`{record.image}` - ground truth {len(record.truths)} instances, "
        f"{len(record.predictions)} predictions. True positives **{record.true_positives}**, "
        f"missed **{len(record.missed)}**, false positives **{len(record.false_positives)}**, "
        f"confused **{len(record.confused)}**."
    )
    if record.recall is not None and record.precision is not None:
        lines.append(
            f"On this image: recall **{record.recall:.2f}**, precision **{record.precision:.2f}**."
        )
    lines.append("")

    if record.missed:
        classes = Counter(CLASS_NAMES[record.truths[i].class_id] for i in record.missed)
        areas = np.asarray([record.truth_areas[i] for i in record.missed], dtype=np.float64)
        quartile = int(np.searchsorted(edges, float(np.median(areas)), side="right")) + 1
        lines.append(
            f"- **Missed ({len(record.missed)}):** "
            + ", ".join(f"{name} {count}" for name, count in sorted(classes.items()))
            + f". Median ground-truth area {np.median(areas):,.0f} px, which is size quartile "
            f"**q{quartile}** of 4 for this split."
        )

    if record.false_positives:
        described = ", ".join(
            f"{CLASS_NAMES[record.predictions[i]['class_id']]} "
            f"{record.predictions[i]['confidence']:.2f}"
            for i in record.false_positives
        )
        lines.append(f"- **Unmatched predictions ({len(record.false_positives)}):** {described}.")
        lines.append(
            f"  **{len(record.duplicate_predictions)} of these sit on an instance that was "
            f"already matched** — a duplicate detection — and "
            f"**{len(record.background_predictions)} overlap no instance at all**. Matching is "
            "one-to-one, so a second prediction on a correctly detected object cannot match and "
            "is counted as a false positive although it covers a real object."
        )

    if record.confused:
        pairs = Counter(
            f"{CLASS_NAMES[record.truths[t].class_id]} -> "
            f"{CLASS_NAMES[record.predictions[p]['class_id']]}"
            for t, p in record.confused
        )
        lines.append(
            "- **Confused:** "
            + ", ".join(f"{pair} x{count}" for pair, count in sorted(pairs.items()))
            + "."
        )
    lines.append("")
    return lines


def examples(cfg: ExampleConfig) -> dict[str, Any]:
    """Select, render and document the clearest false positives and false negatives."""
    if cfg.iou_mode not in {"mask", "box"}:
        raise ValueError(f"iou_mode must be 'mask' or 'box', got {cfg.iou_mode!r}")

    import cv2

    records, edges = collect(cfg)
    chosen = select_examples(records, cfg.per_kind)

    out_dir = cfg.project / cfg.run_name
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    body: list[str] = [
        "# Commented error examples",
        "",
        f"Split `{cfg.split}`, matching on {cfg.iou_mode} IoU >= {cfg.iou_threshold} at "
        f"confidence >= {cfg.conf}, class-agnostic, greedy one-to-one (the convention of §7.1).",
        "",
        "Left panel is ground truth, right panel the predictions. **Red** outlines mark "
        "ground-truth instances the model did not find; **magenta** outlines mark predictions "
        "that matched no ground-truth instance. Class fills use the dataset's own colour "
        "convention.",
        "",
        "**Licence.** Source images are from the BUU Waste Occlusion Dataset (BUU-WOD), "
        "VisionLab, Burapha University, used under CC BY 4.0 "
        "(<https://creativecommons.org/licenses/by/4.0/>). **Modified:** polygon fills, outlines "
        "and class labels were drawn over the originals by this project. The underlying images "
        "and annotations were not altered.",
        "",
    ]
    summary: dict[str, Any] = {
        "split": cfg.split,
        "per_kind": cfg.per_kind,
        "kinds": {},
        # Split-wide totals, so the gallery is anchored to the same numbers as §7 rather than
        # only describing its nine selected images.
        "totals": {
            "matched": sum(len(r.matched) for r in records),
            "missed": sum(len(r.missed) for r in records),
            "false_positives": sum(len(r.false_positives) for r in records),
            "duplicate_false_positives": sum(len(r.duplicate_predictions) for r in records),
            "background_false_positives": sum(len(r.background_predictions) for r in records),
            "confused": sum(len(r.confused) for r in records),
        },
    }

    for kind in KINDS:
        selected = chosen[kind]
        summary["kinds"][kind] = [record.image for record in selected]
        if not selected:
            body += [f"## {kind.replace('-', ' ').title()}", "", "_No example found._", ""]
            continue

        heading = {
            "misses": "False negatives - objects the model did not find",
            "false-positives": "False positives - predictions matching no object",
            "confusion": "Class confusion - matched but mislabelled",
        }[kind]
        body += [f"## {heading}", ""]

        for index, record in enumerate(selected, start=1):
            figure = figures_dir / f"{index:02d}_{kind}_{record.image}.png"
            cv2.imwrite(str(figure), _render(record, cfg.conf))
            body += [f"### {index}. `{record.image}`", ""]
            body += _commentary(record, edges, f"figures/{figure.name}")
            log.info("Wrote %s", figure)

    payload = out_dir / "error_examples.json"
    payload.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    commentary = out_dir / "examples.md"
    commentary.write_text("\n".join(body) + "\n", encoding="utf-8")
    log.info("Wrote %s and %s", commentary, payload)

    print(f"\n{'kind':<18}{'examples':>9}  images")
    for kind in KINDS:
        names = summary["kinds"][kind]
        print(f"  {kind:<16}{len(names):>9}  {', '.join(names) if names else '-'}")

    totals = summary["totals"]
    print(
        f"\nsplit totals: {totals['matched']} matched, {totals['missed']} missed, "
        f"{totals['confused']} confused\n"
        f"  false positives {totals['false_positives']} = "
        f"{totals['duplicate_false_positives']} duplicates + "
        f"{totals['background_false_positives']} on background"
    )
    print(f"\nfigures   {figures_dir}")
    print(f"commentary {commentary}")

    summary["figures_dir"] = str(figures_dir)
    summary["commentary"] = str(commentary)
    return summary


def main(argv: list[str] | None = None) -> int:
    """Allow ``python -m training.error_examples ...`` without a second argument parser."""
    from cli.main import main as cli_main

    return cli_main(["examples", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    setup_logging()
    raise SystemExit(main())
