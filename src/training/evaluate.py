"""Evaluate a trained checkpoint on the validation or test split.

``val`` is for iteration; ``test`` is touched once, at the end, so the reported numbers
are not the ones the model was selected on.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.config import IMGSZ, SEED
from common.constants import CLASS_NAMES, DATA_YAML, RUNS_DIR
from common.runtime import resolve_device, resolve_workers, seed_everything, setup_logging

log = logging.getLogger("litterbug.eval")

SPLITS: tuple[str, ...] = ("val", "test")


def run_name_for(weights: Path, split: str) -> str:
    """Name an evaluation run after the training run the weights came from.

    Ultralytics stores checkpoints as ``<project>/<run>/weights/<ckpt>``. The path is
    resolved first so this still works when the command runs from inside the weights
    directory with a bare filename — without resolution there are no parent components
    and the name collapses to something like ``-val``.
    """
    resolved = Path(weights).resolve()
    if resolved.parent.name == "weights":
        return f"{resolved.parent.parent.name}-{split}"
    return f"{resolved.stem}-{split}"


@dataclass(frozen=True)
class EvalConfig:
    """Everything an evaluation run needs."""

    weights: Path
    data: Path = DATA_YAML
    split: str = "val"
    imgsz: int = IMGSZ
    batch: int = 8
    workers: int | None = None
    device: str | None = None
    project: Path = RUNS_DIR
    name: str | None = None
    seed: int = SEED
    plots: bool = True

    @property
    def run_name(self) -> str:
        return self.name or run_name_for(self.weights, self.split)


def _round(value: Any) -> float | None:
    try:
        return round(float(value), 5)
    except (TypeError, ValueError):
        return None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    try:
        return list(value)
    except TypeError:
        return []


def per_class_map(metric: Any) -> dict[str, float | None]:
    """Per-class mAP50-95, labelled with our own class contract."""
    maps = _as_list(getattr(metric, "maps", None))
    ids = _as_list(getattr(metric, "ap_class_index", None))
    out: dict[str, float | None] = {}
    for class_id, value in zip(ids, maps, strict=False):
        try:
            index = int(class_id)
        except (TypeError, ValueError):
            continue
        label = CLASS_NAMES[index] if 0 <= index < len(CLASS_NAMES) else str(index)
        out[label] = _round(value)
    return out


def _metric_block(metric: Any) -> dict[str, Any] | None:
    if metric is None:
        return None
    return {
        "map50_95": _round(getattr(metric, "map", None)),
        "map50": _round(getattr(metric, "map50", None)),
        "precision": _round(getattr(metric, "mp", None)),
        "recall": _round(getattr(metric, "mr", None)),
        "per_class_map50_95": per_class_map(metric),
    }


def summarise(results: Any, cfg: EvalConfig, device: str) -> dict[str, Any]:
    """Turn Ultralytics' metrics object into a JSON-serialisable report."""
    box = _metric_block(getattr(results, "box", None))
    seg = _metric_block(getattr(results, "seg", None))
    return {
        "task": "segment" if seg is not None else "detect",
        "weights": str(Path(cfg.weights).resolve()),
        "data": str(cfg.data),
        "split": cfg.split,
        "imgsz": cfg.imgsz,
        "batch": cfg.batch,
        "device": device,
        "class_names": CLASS_NAMES,
        "box": box,
        "seg": seg,
        "speed_ms_per_image": _speed(results),
    }


def _speed(results: Any) -> dict[str, float | None]:
    """Per-image timings, guarded because the attribute is not always present."""
    speed = getattr(results, "speed", None)
    if not isinstance(speed, dict):
        return {}
    return {key: _round(value) for key, value in speed.items()}


def _print_summary(report: dict[str, Any]) -> None:
    print(f"\n{report['split']} split | {report['task']} | {report['weights']}")
    for key in ("box", "seg"):
        block = report.get(key)
        if not block:
            continue
        print(
            f"  {key:<4} mAP50-95 {block['map50_95']}  mAP50 {block['map50']}  "
            f"P {block['precision']}  R {block['recall']}"
        )
        for name, value in (block["per_class_map50_95"] or {}).items():
            print(f"       {name:<8} {value}")


def evaluate(cfg: EvalConfig) -> dict[str, Any]:
    """Run validation and write ``metrics.json`` next to the confusion matrix."""
    if cfg.split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {cfg.split!r}")
    if not Path(cfg.weights).is_file():
        raise FileNotFoundError(f"weights not found: {cfg.weights}")

    seed_everything(cfg.seed)
    device = resolve_device(cfg.device, purpose=f"{cfg.split} evaluation")

    from ultralytics import YOLO  # imported late so --help stays cheap

    model = YOLO(str(cfg.weights))
    results = model.val(
        data=str(cfg.data),
        split=cfg.split,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        workers=resolve_workers(cfg.workers),
        device=device,
        plots=cfg.plots,
        project=str(cfg.project),
        name=cfg.run_name,
        verbose=False,
    )

    report = summarise(results, cfg, device)
    save_dir = Path(getattr(results, "save_dir", cfg.project / cfg.run_name))
    payload = save_dir / "metrics.json"
    if payload.parent.exists():
        payload.write_text(json.dumps(report, indent=2), encoding="utf-8")
        log.info("Wrote %s", payload)

    _print_summary(report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Allow ``python -m training.evaluate ...`` without a second argument parser."""
    from cli.main import main as cli_main

    return cli_main(["val", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    setup_logging()
    raise SystemExit(main())
