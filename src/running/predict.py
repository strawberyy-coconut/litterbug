"""Run a trained model over images or video and return structured detections.

The return value is deliberately structured rather than just a pile of files: the demo
API imports :func:`predict` and serialises the result, so it must never have to shell
out or re-parse annotated images.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from common.config import IMGSZ, resolve_inference_batch
from common.constants import (
    CLASS_COLORS,
    CLASS_NAMES,
    DEFAULT_COLOR,
    IMAGE_EXTS,
    RUNS_DIR,
    VIDEO_EXTS,
    class_name,
)
from common.runtime import resolve_device, resolve_workers, setup_logging

log = logging.getLogger("litterbug.predict")

MASK_ALPHA = 0.35
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.7


@dataclass(frozen=True)
class Detection:
    """A single detected object, in pixel coordinates of the source frame."""

    class_id: int
    class_name: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]
    mask_xy: list[list[float]] | None = None
    frame: int | None = None
    source: str | None = None
    track_id: int | None = None


@dataclass(frozen=True)
class PredictConfig:
    """Everything an inference run needs."""

    weights: Path
    source: Path
    output: Path = RUNS_DIR / "predict"
    conf: float = DEFAULT_CONF
    iou: float = DEFAULT_IOU
    imgsz: int = IMGSZ
    batch: int | None = None
    workers: int | None = None
    device: str | None = None
    vid_stride: int = 1
    limit: int | None = None
    save_txt: bool = False
    name: str | None = None
    save_artifacts: bool = True
    # A tracker name, e.g. "bytetrack.yaml". Set only for video: linking detections across frames
    # needs a temporal stream, and the image path deliberately chunks its source into batches.
    tracker: str | None = None


@dataclass
class PredictResult:
    """Structured inference output: detections, counts, per-frame totals, artifacts."""

    source: Path
    kind: str
    detections: list[Detection] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    per_frame: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    output_dir: str = ""
    tracks: dict[str, Any] = field(default_factory=dict)


def classify_source(source: Path) -> str:
    """Decide whether a path is a video, a single image, or a directory of images."""
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"source not found: {path}")
    if path.is_dir():
        return "directory"
    suffix = path.suffix.lower()
    if suffix in VIDEO_EXTS:
        return "video"
    if suffix in IMAGE_EXTS:
        return "image"
    raise ValueError(f"unsupported source type: {path.suffix!r}")


def collect_images(source: Path, limit: int | None = None) -> list[Path]:
    """List the images to process, sorted so runs are reproducible."""
    path = Path(source)
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    else:
        files = [path]
    if not files:
        raise FileNotFoundError(f"no images found under {path}")
    return files[:limit] if limit else files


def _track_ids(boxes: Any) -> list[int | None]:
    """Per-detection track ids, or ``None`` for each when no tracker was used."""
    ids = getattr(boxes, "id", None)
    if ids is None:
        return [None] * len(boxes)
    return [int(value) if value is not None else None for value in ids]


def _detections(
    result: Any, frame: int | None = None, source: str | None = None
) -> list[Detection]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    masks = getattr(result, "masks", None)
    polygons = None if masks is None else getattr(masks, "xy", None)
    track_ids = _track_ids(boxes)

    found: list[Detection] = []
    for index in range(len(boxes)):
        class_id = int(boxes.cls[index])
        mask = None
        if polygons is not None and index < len(polygons):
            mask = [[round(float(x), 2), round(float(y), 2)] for x, y in polygons[index]]
        found.append(
            Detection(
                class_id=class_id,
                class_name=class_name(class_id),
                confidence=round(float(boxes.conf[index]), 4),
                box_xyxy=tuple(round(float(value), 2) for value in boxes.xyxy[index]),
                mask_xy=mask,
                frame=frame,
                source=source,
                track_id=track_ids[index],
            )
        )
    return found


def _draw(result: Any) -> Any:
    """Overlay masks and boxes using the project's own class colours."""
    import cv2
    import numpy as np

    canvas = result.orig_img.copy()
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return canvas

    masks = getattr(result, "masks", None)
    polygons = None if masks is None else getattr(masks, "xy", None)
    if polygons is not None and len(polygons):
        overlay = canvas.copy()
        for index, polygon in enumerate(polygons):
            points = np.asarray(polygon, dtype=np.int32)
            if points.size < 6:  # fewer than three points cannot form a polygon
                continue
            cv2.fillPoly(overlay, [points], CLASS_COLORS.get(int(boxes.cls[index]), DEFAULT_COLOR))
        canvas = cv2.addWeighted(overlay, MASK_ALPHA, canvas, 1 - MASK_ALPHA, 0)

    track_ids = _track_ids(boxes)
    for index in range(len(boxes)):
        x1, y1, x2, y2 = (int(value) for value in boxes.xyxy[index])
        class_id = int(boxes.cls[index])
        colour = CLASS_COLORS.get(class_id, DEFAULT_COLOR)
        label = f"{class_name(class_id)} {float(boxes.conf[index]):.2f}"
        if track_ids[index] is not None:
            label = f"{label} #{track_ids[index]}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
        (width, height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (x1, max(0, y1 - height - 6)), (x1 + width + 4, y1), colour, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 2, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return canvas


def _counts(detections: list[Detection]) -> dict[str, int]:
    counter = Counter(detection.class_name for detection in detections)
    return {name: counter.get(name, 0) for name in CLASS_NAMES}


def _track_summary(
    detections: list[Detection], per_frame: list[dict[str, Any]]
) -> dict[str, Any]:
    """How long detections persist once a tracker links them across frames.

    Per-frame counts cannot tell an object detected in forty consecutive frames from forty
    one-frame flickers — both look like forty detections. Track length separates them, and unlike
    mAP it needs no ground truth, so it is the one measurement available for footage we cannot
    label. Short tracks are the signature of an unstable detector on video.
    """
    lengths = Counter(
        detection.track_id for detection in detections if detection.track_id is not None
    )
    if not lengths:
        return {}

    ordered = sorted(lengths.values())
    count = len(ordered)
    median = (
        float(ordered[count // 2])
        if count % 2
        else (ordered[count // 2 - 1] + ordered[count // 2]) / 2
    )
    total = sum(ordered)

    per_class: dict[str, dict[str, Any]] = {}
    for name in CLASS_NAMES:
        seen = Counter(
            detection.track_id
            for detection in detections
            if detection.class_name == name and detection.track_id is not None
        )
        if seen:
            values = sorted(seen.values())
            per_class[name] = {
                "tracks": len(values),
                "median_length_frames": values[len(values) // 2],
            }

    return {
        "unique_tracks": count,
        "frames": len(per_frame),
        "mean_track_length_frames": round(total / count, 2),
        "median_track_length_frames": median,
        "max_track_length_frames": max(ordered),
        "single_frame_tracks": sum(1 for value in ordered if value == 1),
        "share_single_frame": round(sum(1 for value in ordered if value == 1) / count, 4),
        "per_class": per_class,
    }


def _run_images(model: Any, cfg: PredictConfig, out_dir: Path, device: str) -> PredictResult:
    import cv2

    files = collect_images(cfg.source, cfg.limit)
    detections: list[Detection] = []
    artifacts: list[str] = []

    # Ultralytics ignores `batch` for a list source and feeds the whole list at once, which
    # OOMs on any directory larger than a handful of images. Chunking bounds the memory; the
    # chunk size is the effective batch.
    chunk = resolve_inference_batch(cfg.batch)
    for start in range(0, len(files), chunk):
        stream = model.predict(
            source=[str(path) for path in files[start : start + chunk]],
            conf=cfg.conf,
            iou=cfg.iou,
            imgsz=cfg.imgsz,
            workers=resolve_workers(cfg.workers),
            device=device,
            stream=True,
            verbose=False,
        )
        for result in stream:
            found = _detections(result, source=Path(getattr(result, "path", "frame")).stem)
            detections.extend(found)
            if not cfg.save_artifacts:
                continue
            # Trust each result's own source path instead of assuming stream order, so an
            # unreadable or skipped image can never misalign a filename with its detections.
            stem = Path(getattr(result, "path", "frame")).stem
            target = out_dir / f"{stem}.jpg"
            cv2.imwrite(str(target), _draw(result))
            artifacts.append(str(target))
            if cfg.save_txt:
                artifacts.append(str(_write_txt(out_dir / f"{stem}.txt", found)))

    return PredictResult(
        source=Path(cfg.source),
        kind=classify_source(cfg.source),
        detections=detections,
        counts=_counts(detections),
        artifacts=artifacts,
        output_dir=str(out_dir),
    )


def _video_fps(source: Path) -> float:
    import cv2

    capture = cv2.VideoCapture(str(source))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    finally:
        capture.release()
    return fps if fps > 1 else 25.0


def _run_video(model: Any, cfg: PredictConfig, out_dir: Path, device: str) -> PredictResult:
    import cv2

    writer = None
    detections: list[Detection] = []
    per_frame: list[dict[str, Any]] = []
    target = out_dir / f"{Path(cfg.source).stem}_annotated.mp4"

    # One unbroken stream. ByteTrack links detections frame to frame, so chunking the source the
    # way the image path does would reset its state between batches and fabricate spurious ids.
    arguments = {
        "source": str(cfg.source),
        "conf": cfg.conf,
        "iou": cfg.iou,
        "imgsz": cfg.imgsz,
        "workers": resolve_workers(cfg.workers),
        "device": device,
        "stream": True,
        "vid_stride": cfg.vid_stride,
        "verbose": False,
    }
    stream = (
        model.track(**arguments, tracker=cfg.tracker, persist=True)
        if cfg.tracker
        else model.predict(**arguments)
    )
    try:
        for index, result in enumerate(stream, start=1):
            found = _detections(result, frame=index)
            detections.extend(found)
            per_frame.append({"frame": index, "total": len(found), "counts": _counts(found)})

            frame = _draw(result)
            if writer is None:
                height, width = frame.shape[:2]
                # Frames consumed at vid_stride arrive at a correspondingly lower rate.
                fps = max(1.0, _video_fps(cfg.source) / max(1, cfg.vid_stride))
                writer = cv2.VideoWriter(
                    str(target), cv2.VideoWriter.fourcc(*"mp4v"), fps, (width, height)
                )
            writer.write(frame)
    finally:
        if writer is not None:
            writer.release()

    return PredictResult(
        source=Path(cfg.source),
        kind="video",
        detections=detections,
        counts=_counts(detections),
        per_frame=per_frame,
        artifacts=[str(target)] if writer is not None else [],
        output_dir=str(out_dir),
        tracks=_track_summary(detections, per_frame) if cfg.tracker else {},
    )


def _write_txt(path: Path, detections: list[Detection]) -> Path:
    lines = [
        f"{detection.class_name} {detection.confidence:.4f} "
        f"{detection.box_xyxy[0]:.2f} {detection.box_xyxy[1]:.2f} "
        f"{detection.box_xyxy[2]:.2f} {detection.box_xyxy[3]:.2f}"
        for detection in detections
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def _write_counts_csv(path: Path, per_frame: list[dict[str, Any]]) -> Path:
    fieldnames = ["frame", *CLASS_NAMES, "total"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_frame:
            writer.writerow(
                {
                    "frame": row["frame"],
                    "total": row["total"],
                    **{name: row["counts"].get(name, 0) for name in CLASS_NAMES},
                }
            )
    return path


def to_json(result: PredictResult) -> str:
    """Serialise a result for the CLI and, later, the demo API."""
    return json.dumps(asdict(result), default=str, indent=2)


def predict(cfg: PredictConfig) -> PredictResult:
    """Run inference and return structured detections plus written artifacts."""
    if not Path(cfg.weights).is_file():
        raise FileNotFoundError(f"weights not found: {cfg.weights}")

    kind = classify_source(cfg.source)
    device = resolve_device(cfg.device, purpose="inference")

    from ultralytics import YOLO  # imported late so --help stays cheap

    model = YOLO(str(cfg.weights))
    suffix = "-tracked" if cfg.tracker else ""
    run = cfg.name or f"{Path(cfg.weights).parent.parent.name}-{Path(cfg.source).stem}{suffix}"
    out_dir = Path(cfg.output) / run
    out_dir.mkdir(parents=True, exist_ok=True)

    result = (
        _run_video(model, cfg, out_dir, device)
        if kind == "video"
        else _run_images(model, cfg, out_dir, device)
    )

    if result.per_frame:
        counts_csv = _write_counts_csv(out_dir / "per_frame_counts.csv", result.per_frame)
        result.artifacts.append(str(counts_csv))

    report = out_dir / "detections.json"
    report.write_text(to_json(result), encoding="utf-8")
    result.artifacts.append(str(report))

    log.info(
        "%s: %d detections %s -> %s", result.kind, len(result.detections), result.counts, out_dir
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Allow ``python -m running.predict ...`` without a second argument parser."""
    from cli.main import main as cli_main

    return cli_main(["predict", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    setup_logging()
    raise SystemExit(main())
