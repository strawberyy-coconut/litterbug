from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from litterbug.common.config import EPOCHS, IMGSZ, SEED
from litterbug.common.constants import DATA_YAML, RUNS_DIR, TASKS
from litterbug.common.runtime import setup_logging

log = logging.getLogger("litterbug.cli")

EXIT_OK = 0
EXIT_ERROR = 1

RUNS_HELP = "Directory that run outputs are created under."


def _handle_train(args: argparse.Namespace) -> int:
    from litterbug.common.config import TrainConfig
    from litterbug.training.train import train

    train(
        TrainConfig(
            task=args.task,
            data=args.data,
            name=args.name,
            project=args.project,
            batch=args.batch,
            workers=args.workers,
            epochs=args.epochs,
            imgsz=args.imgsz,
            seed=args.seed,
            resume=args.resume,
            device=args.device,
            dry_run=args.dry_run,
        )
    )
    return EXIT_OK


def _handle_val(args: argparse.Namespace) -> int:
    from litterbug.training.evaluate import EvalConfig, evaluate

    evaluate(
        EvalConfig(
            weights=args.weights,
            data=args.data,
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            project=args.project,
            name=args.name,
            seed=args.seed,
            plots=args.plots,
        )
    )
    return EXIT_OK


def _handle_predict(args: argparse.Namespace) -> int:
    from litterbug.running.predict import PredictConfig, predict, to_json

    result = predict(
        PredictConfig(
            weights=args.weights,
            source=args.source,
            output=args.output,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            vid_stride=args.vid_stride,
            limit=args.limit,
            save_txt=args.save_txt,
            name=args.name,
            tracker="bytetrack.yaml" if args.track else None,
        )
    )
    if args.json:
        print(to_json(result))
    return EXIT_OK


def _handle_analyze(args: argparse.Namespace) -> int:
    from litterbug.training.error_analysis import DEFAULT_CONF, DEFAULT_IOU, AnalysisConfig, analyse

    analyse(
        AnalysisConfig(
            weights=args.weights,
            split=args.split,
            conf=args.conf if args.conf is not None else DEFAULT_CONF,
            iou_mode=args.iou_mode,
            iou_threshold=args.iou_threshold if args.iou_threshold is not None else DEFAULT_IOU,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            project=args.project,
            name=args.name,
            limit=args.limit,
        )
    )
    return EXIT_OK


def _handle_eda(args: argparse.Namespace) -> int:
    from litterbug.training.eda import SPLITS, EdaConfig, eda

    eda(
        EdaConfig(
            splits=tuple(args.split) if args.split else SPLITS,
            project=args.project,
            name=args.name,
            sample=args.sample,
            seed=args.seed,
        )
    )
    return EXIT_OK


def _handle_examples(args: argparse.Namespace) -> int:
    from litterbug.training.error_analysis import DEFAULT_CONF, DEFAULT_IOU
    from litterbug.training.error_examples import FIGURES_DIR, ExampleConfig, examples

    examples(
        ExampleConfig(
            weights=args.weights,
            split=args.split,
            conf=args.conf if args.conf is not None else DEFAULT_CONF,
            iou_mode=args.iou_mode,
            iou_threshold=args.iou_threshold if args.iou_threshold is not None else DEFAULT_IOU,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            project=args.project if args.project is not None else FIGURES_DIR,
            name=args.name,
            per_kind=args.per_kind,
            limit=args.limit,
        )
    )
    return EXIT_OK


def _handle_validate(args: argparse.Namespace) -> int:
    from litterbug.training.validate_data import OVERLAY_DIR, ValidationConfig, validate_dataset

    result = validate_dataset(
        ValidationConfig(
            data_yaml=args.data_yaml,
            data_root=args.data_root,
            require_labels=args.require_labels,
            overlay=args.overlay,
            output=args.output if args.output is not None else OVERLAY_DIR,
            seed=args.seed,
            max_examples=args.max_examples,
        )
    )
    # A dataset with errors is a failed check, so it must not exit 0 in a script.
    return EXIT_OK if result["clean"] else EXIT_ERROR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="litterbug",
        description="Train, evaluate and run waste detection and segmentation models.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--device", default=None, help="Torch device. Defaults to cuda:0.")
    common.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Data loader workers. Defaults to a /dev/shm-aware value.",
    )
    common.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    common.add_argument("--verbose", action="store_true", help="Enable debug logging.")

    train = subparsers.add_parser("train", parents=[common], help="Fine-tune a model.")
    train.add_argument("--task", choices=TASKS, default="segment", help="Which head to train.")
    train.add_argument("--data", type=Path, default=DATA_YAML, help="Path to data.yaml.")
    train.add_argument("--name", default=None, help="Run name. Defaults to <task>-<model>.")
    train.add_argument("--project", type=Path, default=RUNS_DIR, help=RUNS_HELP)
    train.add_argument("--batch", type=int, default=None, help="Defaults to a VRAM-derived value.")
    train.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Smoke-test override only. The default is the configured schedule.",
    )
    train.add_argument("--imgsz", type=int, default=IMGSZ)
    train.add_argument("--resume", action="store_true", help="Resume an interrupted run.")
    train.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print the configuration without training.",
    )
    train.set_defaults(handler=_handle_train)

    val = subparsers.add_parser("val", parents=[common], help="Evaluate a checkpoint.")
    val.add_argument("--weights", type=Path, required=True, help="Checkpoint to evaluate.")
    val.add_argument("--data", type=Path, default=DATA_YAML, help="Path to data.yaml.")
    val.add_argument("--split", choices=("val", "test"), default="val", help="Split to evaluate.")
    val.add_argument("--imgsz", type=int, default=IMGSZ)
    val.add_argument("--batch", type=int, default=8)
    val.add_argument("--project", type=Path, default=RUNS_DIR, help=RUNS_HELP)
    val.add_argument("--name", default=None, help="Run name. Defaults to <run>-<split>.")
    val.add_argument("--no-plots", action="store_false", dest="plots", help="Skip plots.")
    val.set_defaults(handler=_handle_val)

    predict = subparsers.add_parser(
        "predict", parents=[common], help="Run a checkpoint over images or video."
    )
    predict.add_argument("--weights", type=Path, required=True, help="Checkpoint to run.")
    predict.add_argument("--source", type=Path, required=True, help="Image, directory or video.")
    predict.add_argument("--output", type=Path, default=RUNS_DIR / "predict", help="Output root.")
    predict.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    predict.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    predict.add_argument("--imgsz", type=int, default=IMGSZ)
    predict.add_argument(
        "--batch", type=int, default=None, help="Defaults to a VRAM-derived value."
    )
    predict.add_argument("--vid-stride", type=int, default=1, help="Process every Nth video frame.")
    predict.add_argument("--limit", type=int, default=None, help="Process at most N images.")
    predict.add_argument("--save-txt", action="store_true", help="Also write per-image text.")
    predict.add_argument(
        "--track",
        action="store_true",
        help="Link detections across video frames with ByteTrack, adding track ids and "
        "persistence statistics.",
    )
    predict.add_argument("--json", action="store_true", help="Print structured results as JSON.")
    predict.add_argument("--name", default=None, help="Output run name.")
    predict.set_defaults(handler=_handle_predict)

    analyze = subparsers.add_parser(
        "analyze", parents=[common], help="Per-instance error analysis of a checkpoint."
    )
    analyze.add_argument("--weights", type=Path, required=True, help="Checkpoint to analyse.")
    analyze.add_argument(
        "--split", choices=("val", "test"), default="val", help="Split to analyse."
    )
    analyze.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Confidence threshold. Defaults to the F1-optimal value.",
    )
    analyze.add_argument(
        "--iou-mode",
        choices=("mask", "box"),
        default="mask",
        help="Criterion for matching predictions to ground truth.",
    )
    analyze.add_argument(
        "--iou-threshold",
        type=float,
        default=None,
        help="Minimum IoU for a match. Defaults to 0.5.",
    )
    analyze.add_argument("--imgsz", type=int, default=IMGSZ)
    analyze.add_argument(
        "--batch", type=int, default=None, help="Defaults to a VRAM-derived value."
    )
    analyze.add_argument("--project", type=Path, default=RUNS_DIR, help=RUNS_HELP)
    analyze.add_argument("--name", default=None, help="Output run name.")
    analyze.add_argument("--limit", type=int, default=None, help="Analyse at most N images.")
    analyze.set_defaults(handler=_handle_analyze)

    eda_parser = subparsers.add_parser(
        "eda", parents=[common], help="Exploratory analysis of the dataset splits."
    )
    eda_parser.add_argument(
        "--split",
        action="append",
        choices=("train", "val", "test"),
        default=None,
        help="Split to analyse. Repeatable; defaults to all three.",
    )
    eda_parser.add_argument("--project", type=Path, default=RUNS_DIR, help=RUNS_HELP)
    eda_parser.add_argument("--name", default="dataset-eda", help="Output run name.")
    eda_parser.add_argument(
        "--sample",
        type=int,
        default=200,
        help="Images sampled per split for pixel statistics.",
    )
    eda_parser.set_defaults(handler=_handle_eda)

    validate = subparsers.add_parser(
        "validate", parents=[common], help="Check the dataset before training."
    )
    source = validate.add_mutually_exclusive_group()
    source.add_argument(
        "--data-yaml", type=Path, default=DATA_YAML, help="data.yaml to read the splits from."
    )
    source.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Dataset root containing <split>/images, instead of a data.yaml.",
    )
    validate.add_argument(
        "--require-labels",
        action="store_true",
        help="Treat images with no label file as errors rather than as possible blanks.",
    )
    validate.add_argument(
        "--overlay",
        type=int,
        default=0,
        metavar="N",
        help="Render N random polygon overlays per split for a visual spot-check.",
    )
    validate.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Overlay output directory. Defaults to validation_overlays/ at the project root.",
    )
    validate.add_argument(
        "--max-examples", type=int, default=5, help="Max example problems printed per category."
    )
    validate.set_defaults(handler=_handle_validate)

    examples_parser = subparsers.add_parser(
        "examples",
        parents=[common],
        help="Render commented false-positive and false-negative examples.",
    )
    examples_parser.add_argument(
        "--weights", type=Path, required=True, help="Checkpoint to analyse."
    )
    examples_parser.add_argument(
        "--split", choices=("val", "test"), default="val", help="Split to analyse."
    )
    examples_parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Confidence threshold. Defaults to the F1-optimal value.",
    )
    examples_parser.add_argument(
        "--iou-mode",
        choices=("mask", "box"),
        default="mask",
        help="Criterion for matching predictions to ground truth.",
    )
    examples_parser.add_argument(
        "--iou-threshold",
        type=float,
        default=None,
        help="Minimum IoU for a match. Defaults to 0.5.",
    )
    examples_parser.add_argument("--imgsz", type=int, default=IMGSZ)
    examples_parser.add_argument(
        "--batch", type=int, default=None, help="Defaults to a VRAM-derived value."
    )
    examples_parser.add_argument(
        "--per-kind", type=int, default=3, help="Examples rendered per error type."
    )
    # Resolved in the handler: importing the module here would pull numpy into `--help`.
    examples_parser.add_argument(
        "--project",
        type=Path,
        default=None,
        help="Directory the figures are created under. Defaults to report/figures.",
    )
    examples_parser.add_argument("--name", default=None, help="Output run name.")
    examples_parser.add_argument(
        "--limit", type=int, default=None, help="Analyse at most N images."
    )
    examples_parser.set_defaults(handler=_handle_examples)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    try:
        return args.handler(args)
    except (FileNotFoundError, ValueError) as error:
        log.error("%s", error)
        return EXIT_ERROR
    except KeyboardInterrupt:
        log.warning("Interrupted.")
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
