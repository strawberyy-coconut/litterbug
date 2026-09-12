"""Instance-level error analysis for a trained checkpoint."""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from common.config import IMGSZ, resolve_inference_batch
from common.constants import CLASS_NAMES, DATA_YAML, IMAGE_EXTS, RUNS_DIR
from common.runtime import resolve_device, resolve_workers, setup_logging

log = logging.getLogger("litterbug.analysis")

DEFAULT_CONF = 0.34  
DEFAULT_IOU = 0.5
MISSED = "missed"

@dataclass(frozen=True)
class AnalysisConfig:
    weights: Path
    split: str = "val"
    conf: float = DEFAULT_CONF
    iou_mode: str = "mask"  # "mask" # "box"
    iou_threshold: float = DEFAULT_IOU
    imgsz: int = IMGSZ
    batch: int | None = None
    workers: int | None = None
    device: str | None = None
    project: Path = RUNS_DIR
    name: str | None = None
    limit: int | None = None

    @property
    def run_name(self) -> str:
        # The split and the matching mode are part of the name because they are not part of the
        # path. Without them, analysing `test` overwrites the `valid` analysis, and a box run
        # overwrites the mask run, both silently and both unrecoverably.
        stem = Path(self.weights).resolve().parent.parent.name
        suffix = "" if self.iou_mode == "mask" else f"-{self.iou_mode}"
        return self.name or f"{stem}-{self.split}{suffix}-analysis"


@dataclass(frozen=True)
class TruthInstance:
    class_id: int
    polygon: np.ndarray  

def split_paths(split: str) -> tuple[Path, Path]:
    import yaml

    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"data.yaml not found: {DATA_YAML}")

    payload = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8")) or {}
    if split not in payload:
        raise ValueError(f"split {split!r} not declared in {DATA_YAML.name}")

    images = (DATA_YAML.parent / str(payload[split])).resolve()
    labels = images.parent / "labels"
    if not images.is_dir():
        raise FileNotFoundError(f"images directory not found: {images}")
    if not labels.is_dir():
        raise FileNotFoundError(f"labels directory not found: {labels}")
    return images, labels


def load_labels(label_path: Path, width: int, height: int) -> list[TruthInstance]:
    """Parse a YOLO segmentation label file into pixel-space polygons."""
    if not label_path.is_file():
        return []

    instances: list[TruthInstance] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 7:  # class + at least three points
            continue
        points = np.asarray(parts[1:], dtype=np.float64)
        if points.size % 2:
            continue
        points = points.reshape(-1, 2)
        points[:, 0] *= width
        points[:, 1] *= height
        instances.append(TruthInstance(class_id=int(parts[0]), polygon=points))
    return instances



def _rasterize(polygon: np.ndarray | None, width: int, height: int) -> np.ndarray:
    import cv2

    mask = np.zeros((height, width), dtype=np.uint8)
    if polygon is not None:
        points = np.round(np.asarray(polygon, dtype=np.float64)).astype(np.int32)
        if points.shape[0] >= 3:
            cv2.fillPoly(mask, [points], 1)
    return mask.astype(bool)


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    union = np.logical_or(first, second).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(first, second).sum()) / float(union)


def _box_iou(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    if right <= left or bottom <= top:
        return 0.0
    inter = (right - left) * (bottom - top)
    area_a = (first[2] - first[0]) * (first[3] - first[1])
    area_b = (second[2] - second[0]) * (second[3] - second[1])
    union = area_a + area_b - inter
    return float(inter) / float(union) if union > 0 else 0.0


def _polygon_box(polygon: np.ndarray) -> tuple[float, float, float, float]:
    xs, ys = polygon[:, 0], polygon[:, 1]
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def solidity_scores(masks: list[np.ndarray]) -> list[float]:
    """Mask area divided by convex-hull area, per instance.

    A shape-irregularity proxy for occlusion: a partially hidden object tends to produce a
    ragged, non-convex visible region, so lower solidity *may* indicate occlusion.

    Two honest caveats. It is not a measurement of occlusion — see the module docstring.
    And it is confounded by naturally irregular materials (crumpled paper, plastic film),
    which show low solidity whether or not they are occluded. It is reported alongside size
    rather than instead of it, and its correlation with class is part of what it reveals.
    """
    import cv2

    scores: list[float] = []
    for mask in masks:
        area = float(mask.sum())
        if area <= 0:
            scores.append(1.0)
            continue
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        hull_area = float(cv2.contourArea(cv2.convexHull(np.vstack(contours)))) if contours else 0.0
        scores.append(min(1.0, area / hull_area) if hull_area > 0 else 1.0)
    return scores



def predict_split(
    cfg: AnalysisConfig,
) -> tuple[list[Path], dict[str, list[dict[str, Any]]], dict[str, tuple[int, int]]]:
    """Run the checkpoint over one split and return its predictions, grouped by image.

    Public because the commented-example figures need exactly this, and running the model twice
    under two slightly different conventions is how two reports of the same model start to
    disagree.
    """
    images_dir, _ = split_paths(cfg.split)
    files = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if cfg.limit:
        files = files[: cfg.limit]
    if not files:
        raise FileNotFoundError(f"no images found under {images_dir}")

    from ultralytics import YOLO

    model = YOLO(str(cfg.weights))
    device = resolve_device(cfg.device, purpose=f"{cfg.split} error analysis")

    grouped: dict[str, list[dict[str, Any]]] = {}
    sizes: dict[str, tuple[int, int]] = {}
    # Ultralytics ignores `batch` for a list source and feeds the whole list through at once,
    # which OOMs on any real-sized directory. Chunking the list ourselves is what actually
    # bounds memory; the chunk size is the effective batch.
    chunk = resolve_inference_batch(cfg.batch)
    for start in range(0, len(files), chunk):
        chunk_files = files[start : start + chunk]
        stream = model.predict(
            source=[str(path) for path in chunk_files],
            conf=cfg.conf,
            imgsz=cfg.imgsz,
            workers=resolve_workers(cfg.workers),
            device=device,
            stream=True,
            verbose=False,
        )
        for result in stream:
            stem = Path(getattr(result, "path", "unknown")).stem
            height, width = result.orig_shape[:2]
            masks = getattr(result, "masks", None)
            polygons = None if masks is None else getattr(masks, "xy", None)
            boxes = getattr(result, "boxes", None)

            found: list[dict[str, Any]] = []
            for index in range(0 if boxes is None else len(boxes)):
                polygon = None
                if polygons is not None and index < len(polygons):
                    points = np.asarray(polygons[index], dtype=np.float64)
                    if points.shape[0] >= 3:
                        polygon = points
                found.append(
                    {
                        "class_id": int(boxes.cls[index]),
                        "confidence": float(boxes.conf[index]),
                        "box": tuple(float(v) for v in boxes.xyxy[index]),
                        "polygon": polygon,
                    }
                )
            grouped[stem] = found
            sizes[stem] = (width, height)

    return files, grouped, sizes



def match_with_iou(
    truths: list[TruthInstance],
    predictions: list[dict[str, Any]],
    width: int,
    height: int,
    iou_threshold: float,
    iou_mode: str,
) -> list[tuple[int, int, float]]:
    """Greedy one-to-one matching, by descending confidence.

    Returns ``(truth_index, prediction_index, iou)`` triples. The IoU of each accepted pair is
    carried out because it is a reported metric in its own right, not merely a gate: it measures
    how well a matched mask fits the object, independently of whether the object was found at all.
    A model can have high recall and still score poorly here.
    """
    gt_masks = [_rasterize(item.polygon, width, height) for item in truths]
    pred_masks = [_rasterize(item["polygon"], width, height) for item in predictions]

    order = sorted(
        range(len(predictions)), key=lambda i: predictions[i]["confidence"], reverse=True
    )
    taken_truth: set[int] = set()
    matched: list[tuple[int, int, float]] = []

    for pred_index in order:
        best_iou, best_truth = 0.0, None
        for truth_index, truth in enumerate(truths):
            if truth_index in taken_truth:
                continue
            if iou_mode == "mask" and predictions[pred_index]["polygon"] is not None:
                iou = _iou(gt_masks[truth_index], pred_masks[pred_index])
            else:
                iou = _box_iou(_polygon_box(truth.polygon), predictions[pred_index]["box"])
            if iou > best_iou:
                best_iou, best_truth = iou, truth_index
        if best_truth is not None and best_iou >= iou_threshold:
            taken_truth.add(best_truth)
            matched.append((best_truth, pred_index, best_iou))
    return matched


def match_image(
    truths: list[TruthInstance],
    predictions: list[dict[str, Any]],
    width: int,
    height: int,
    iou_threshold: float,
    iou_mode: str,
) -> list[tuple[int, int]]:
    """Greedy one-to-one matching, by descending confidence. Returns (truth, prediction) pairs."""
    return [
        (truth_index, prediction_index)
        for truth_index, prediction_index, _ in match_with_iou(
            truths, predictions, width, height, iou_threshold, iou_mode
        )
    ]


def best_iou(
    truths: list[TruthInstance],
    prediction: dict[str, Any],
    width: int,
    height: int,
    iou_mode: str,
    gt_masks: np.ndarray | None = None,
) -> float:
    """Highest IoU between one prediction and any ground-truth instance, matched or not.

    This is what separates a duplicated detection from a genuine background error. Matching is
    one-to-one, so a second prediction sitting on an object that was already matched cannot match
    and is counted as a false positive — even though it covers a real object. Counting those
    alongside predictions that overlap nothing at all would conflate two different failures with
    two different fixes.
    """
    if gt_masks is None:
        gt_masks = [_rasterize(item.polygon, width, height) for item in truths]
    prediction_mask = (
        _rasterize(prediction["polygon"], width, height)
        if prediction["polygon"] is not None
        else None
    )

    best = 0.0
    for truth, mask in zip(truths, gt_masks, strict=False):
        if iou_mode == "mask" and prediction_mask is not None:
            value = _iou(mask, prediction_mask)
        else:
            value = _box_iou(_polygon_box(truth.polygon), prediction["box"])
        best = max(best, value)
    return best



def _quartile_edges(values: list[float]) -> np.ndarray:
    return np.quantile(np.asarray(values, dtype=np.float64), [0.25, 0.5, 0.75])


def _quartiles(records: list[tuple[float, bool]], label: str) -> list[dict[str, Any]]:
    """Recall within each quartile of a continuous attribute."""
    if not records:
        return []
    values = np.asarray([value for value, _ in records], dtype=np.float64)
    edges = _quartile_edges([value for value, _ in records])
    buckets: list[list[bool]] = [[], [], [], []]
    for value, hit in records:
        buckets[int(np.searchsorted(edges, value, side="right"))].append(hit)

    summary: list[dict[str, Any]] = []
    lower = float(values.min())
    for index, hits in enumerate(buckets):
        upper = float(edges[index]) if index < len(edges) else float(values.max())
        summary.append(
            {
                "quartile": index + 1,
                "attribute": label,
                "from": round(lower, 3),
                "to": round(upper, 3),
                "instances": len(hits),
                "recall": round(sum(hits) / len(hits), 4) if hits else None,
            }
        )
        lower = upper
    return summary


def recall_by_class_and_size(
    records: list[tuple[int, float, bool]], edges: np.ndarray
) -> dict[str, dict[str, Any]]:
    """Per-class recall within each overall size quartile, plus the class's median size.

    This is what separates "Paper is a hard class" from "Paper objects happen to be small".
    If every class behaves similarly inside a size bucket, size is the driver; if Paper
    underperforms its peers *within* the same bucket, something class-specific is going on.
    """
    class_count = len(CLASS_NAMES)
    buckets: dict[int, list[list[bool]]] = {index: [[], [], [], []] for index in range(class_count)}
    areas: dict[int, list[float]] = {index: [] for index in range(class_count)}
    for class_id, area, hit in records:
        buckets[class_id][int(np.searchsorted(edges, area, side="right"))].append(hit)
        areas[class_id].append(area)

    table: dict[str, dict[str, Any]] = {}
    for class_id, name in enumerate(CLASS_NAMES):
        per_bucket = [
            round(sum(hits) / len(hits), 4) if hits else None for hits in buckets[class_id]
        ]
        table[name] = {
            "instances": len(areas[class_id]),
            "median_area_px": (
                round(float(np.median(areas[class_id])), 1) if areas[class_id] else None
            ),
            "recall_by_quartile": per_bucket,
        }
    return table


def iou_summary(
    records: list[tuple[int, int, float, float]], edges: np.ndarray
) -> dict[str, Any]:
    """Mean IoU of matched pairs, overall, per class and per size quartile.

    ``records`` are ``(truth_class, predicted_class, iou, truth_mask_area_px)``.

    Per-class figures use only pairs where the predicted class equals the ground-truth class.
    A cross-class match measures how well a Plastic prediction happens to cover a Paper object,
    which says nothing about that class's mask quality; folding those in would quietly deflate
    every class's figure and make the per-class column mean something other than its label.
    """
    if not records:
        return {"matched_pairs": 0, "mean": None, "median": None, "by_class": {},
                "by_size_quartile": []}

    ious = np.asarray([record[2] for record in records], dtype=np.float64)

    by_class: dict[str, dict[str, Any]] = {}
    for class_id, name in enumerate(CLASS_NAMES):
        subset = [r[2] for r in records if r[0] == class_id and r[1] == class_id]
        by_class[name] = {
            "pairs": len(subset),
            "mean": round(float(np.mean(subset)), 4) if subset else None,
            "median": round(float(np.median(subset)), 4) if subset else None,
        }

    buckets: list[list[float]] = [[], [], [], []]
    for _, _, iou, area in records:
        buckets[int(np.searchsorted(edges, area, side="right"))].append(iou)

    by_quartile = [
        {
            "quartile": index + 1,
            "pairs": len(values),
            "mean": round(float(np.mean(values)), 4) if values else None,
        }
        for index, values in enumerate(buckets)
    ]

    return {
        "matched_pairs": len(records),
        "mean": round(float(ious.mean()), 4),
        "median": round(float(np.median(ious)), 4),
        "p10": round(float(np.percentile(ious, 10)), 4),
        "p90": round(float(np.percentile(ious, 90)), 4),
        "share_above_0_75": round(float((ious >= 0.75).mean()), 4),
        "by_class": by_class,
        "by_size_quartile": by_quartile,
    }


def analyse(cfg: AnalysisConfig) -> dict[str, Any]:
    """Run the full error analysis and write it as JSON."""
    if cfg.iou_mode not in {"mask", "box"}:
        raise ValueError(f"iou_mode must be 'mask' or 'box', got {cfg.iou_mode!r}")

    files, grouped, sizes = predict_split(cfg)
    _, labels_dir = split_paths(cfg.split)

    class_count = len(CLASS_NAMES)
    confusion = np.zeros((class_count, class_count + 1), dtype=int)  # last column = missed
    predicted_counts: Counter[int] = Counter()
    truth_counts: Counter[int] = Counter()
    matched_counts: Counter[int] = Counter()
    area_records: list[tuple[float, bool]] = []
    class_area_records: list[tuple[int, float, bool]] = []
    solidity_records: list[tuple[float, bool]] = []
    iou_records: list[tuple[int, int, float, float]] = []
    total_predictions = 0

    for path in files:
        width, height = sizes[path.stem]
        truths = load_labels(labels_dir / f"{path.stem}.txt", width, height)
        predictions = grouped.get(path.stem, [])

        total_predictions += len(predictions)
        for prediction in predictions:
            predicted_counts[prediction["class_id"]] += 1
        for truth in truths:
            truth_counts[truth.class_id] += 1

        gt_masks = [_rasterize(item.polygon, width, height) for item in truths]
        solidities = solidity_scores(gt_masks)

        matched = match_with_iou(
            truths, predictions, width, height, cfg.iou_threshold, cfg.iou_mode
        )
        matched_truth = {truth_index for truth_index, _, _ in matched}

        for truth_index, prediction_index, pair_iou in matched:
            truth_class = truths[truth_index].class_id
            predicted_class = predictions[prediction_index]["class_id"]
            confusion[truth_class][predicted_class] += 1
            if truth_class == predicted_class:
                matched_counts[truth_class] += 1
            iou_records.append(
                (truth_class, predicted_class, pair_iou, float(gt_masks[truth_index].sum()))
            )

        for truth_index, truth in enumerate(truths):
            hit = truth_index in matched_truth
            if not hit:
                confusion[truth.class_id][class_count] += 1
            area = float(gt_masks[truth_index].sum())
            area_records.append((area, hit))
            class_area_records.append((truth.class_id, area, hit))
            solidity_records.append((solidities[truth_index], hit))

    def per_class() -> dict[str, dict[str, Any]]:
        rows: dict[str, dict[str, Any]] = {}
        for class_id, name in enumerate(CLASS_NAMES):
            gt = truth_counts.get(class_id, 0)
            predicted = predicted_counts.get(class_id, 0)
            matched = matched_counts.get(class_id, 0)
            rows[name] = {
                "ground_truth": gt,
                "predicted": predicted,
                "true_positives": matched,
                "recall": round(matched / gt, 4) if gt else None,
                "precision": round(matched / predicted, 4) if predicted else None,
            }
        return rows

    total_truth = int(confusion[:, :class_count].sum() + confusion[:, class_count].sum())
    total_matched = int(confusion[:, :class_count].sum())
    area_edges = _quartile_edges([area for area, _ in area_records])

    report: dict[str, Any] = {
        "weights": str(Path(cfg.weights).resolve()),
        "split": cfg.split,
        "images": len(files),
        "matching": {
            "convention": "descending confidence, greedy one-to-one, class-agnostic",
            "iou_mode": cfg.iou_mode,
            "iou_threshold": cfg.iou_threshold,
            "confidence_threshold": cfg.conf,
        },
        "ground_truth_instances": total_truth,
        "predictions": total_predictions,
        "matched": total_matched,
        "overall": {
            "recall": round(total_matched / total_truth, 4) if total_truth else None,
            "precision": round(total_matched / total_predictions, 4) if total_predictions else None,
        },
        "per_class": per_class(),
        "confusion_matrix": {
            "rows": "ground_truth",
            "columns": [*CLASS_NAMES, MISSED],
            "matrix": confusion.tolist(),
        },
        "iou": {"iou_mode": cfg.iou_mode, **iou_summary(iou_records, area_edges)},
        "recall_by_area_quartile": _quartiles(area_records, "ground_truth_mask_pixels"),
        "recall_by_solidity_quartile": _quartiles(
            solidity_records, "mask_area_over_convex_hull_area"
        ),
        "recall_by_class_and_size_quartile": recall_by_class_and_size(
            class_area_records, area_edges
        ),
        "occlusion_note": (
            "True occlusion is unmeasurable here: annotations record only visible boundaries and "
            "exclude objects below ~30% visibility, so the hidden fraction cannot be recovered. "
            "solidity is a shape-irregularity proxy, confounded by naturally irregular materials."
        ),
    }

    out_dir = cfg.project / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = out_dir / "error_analysis.json"
    payload.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("Wrote %s", payload)
    print_summary(report)
    return report


def print_summary(report: dict[str, Any]) -> None:
    matching = report["matching"]
    print(
        f"\n{report['split']} split | {report['images']} images | "
        f"{report['ground_truth_instances']} ground-truth instances\n"
        f"matching: {matching['iou_mode']} IoU >= {matching['iou_threshold']}, "
        f"conf >= {matching['confidence_threshold']}, {matching['convention']}"
    )
    print(
        f"overall: recall {report['overall']['recall']} "
        f"precision {report['overall']['precision']} "
        f"({report['matched']} matched of {report['predictions']} predictions)"
    )

    print("\nper class")
    print(f"  {'class':<8} {'gt':>6} {'pred':>6} {'tp':>6} {'recall':>8} {'precision':>10}")
    for name, row in report["per_class"].items():
        print(
            f"  {name:<8} {row['ground_truth']:>6} {row['predicted']:>6} "
            f"{row['true_positives']:>6} {row['recall']:>8} {row['precision']:>10}"
        )

    cols = report["confusion_matrix"]["columns"]
    print("\nconfusion matrix (rows = truth, count of instances)")
    print(f"  {'':<8}" + "".join(f"{name:>8}" for name in cols))
    for name, row in zip(CLASS_NAMES, report["confusion_matrix"]["matrix"], strict=False):
        print(f"  {name:<8}" + "".join(f"{value:>8}" for value in row))

    iou = report.get("iou") or {}
    if iou.get("matched_pairs"):
        print(
            f"\nmean IoU of matched pairs ({iou.get('iou_mode')}): {iou['mean']} "
            f"(median {iou['median']}, p10 {iou['p10']}, p90 {iou['p90']}, "
            f"{iou['share_above_0_75'] * 100:.0f}% >= 0.75)"
        )
        print(f"  {'class':<8} {'pairs':>6} {'mean IoU':>9} {'median':>8}")
        for name, row in iou["by_class"].items():
            if row["pairs"]:
                print(f"  {name:<8} {row['pairs']:>6} {row['mean']:>9} {row['median']:>8}")
        print("  mean IoU by size quartile (q1 = smallest)")
        for bucket in iou["by_size_quartile"]:
            print(f"    q{bucket['quartile']}  n={bucket['pairs']:>5}  mean IoU={bucket['mean']}")

    for key in ("recall_by_area_quartile", "recall_by_solidity_quartile"):
        print(f"\nrecall by {report[key][0]['attribute'] if report[key] else key} (quartiles)")
        for bucket in report[key]:
            print(
                f"  q{bucket['quartile']}  [{bucket['from']:>9.2f}, {bucket['to']:>9.2f}]  "
                f"n={bucket['instances']:>5}  recall={bucket['recall']}"
            )

    print("\nrecall by class within overall size quartiles (q1 = smallest)")
    print(f"  {'class':<8} {'n':>6} {'median px':>10}    q1     q2     q3     q4")
    for name, row in report["recall_by_class_and_size_quartile"].items():
        cells = "  ".join(
            f"{value:.3f}" if value is not None else "  n/a" for value in row["recall_by_quartile"]
        )
        print(f"  {name:<8} {row['instances']:>6} {row['median_area_px']:>10}   {cells}")


def main(argv: list[str] | None = None) -> int:
    """Allow ``python -m training.error_analysis ...`` without a second argument parser."""
    from cli.main import main as cli_main

    return cli_main(["analyze", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    setup_logging()
    raise SystemExit(main())
