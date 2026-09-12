from __future__ import annotations

from pathlib import Path

CLASS_NAMES: list[str] = ["Glass", "Metal", "Paper", "Plastic"]
CLASS_IDS: dict[str, int] = {name: index for index, name in enumerate(CLASS_NAMES)}

# BGR, indexed by class id. Matches the colour convention used by the dataset's own
# reference script, so our overlays agree with the ones shipped with the data.
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (0, 255, 255),  # Glass   — cyan
    1: (192, 192, 192),  # Metal   — silver
    2: (0, 165, 255),  # Paper   — orange
    3: (0, 255, 0),  # Plastic — green
}
DEFAULT_COLOR: tuple[int, int, int] = (255, 0, 255)


MODELS: dict[str, str] = {
    "detect": "yolo26s.pt",
    "segment": "yolo26s-seg.pt",
}

MODELS_FALLBACK: dict[str, str] = {
    "detect": "yolo11s.pt",
    "segment": "yolo11s-seg.pt",
}

TASKS: tuple[str, ...] = ("detect", "segment")

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_YAML: Path = PROJECT_ROOT / "dataset" / "1_Model_Training_Data" / "data.yaml"
RUNS_DIR: Path = PROJECT_ROOT / "runs"

IMAGE_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})
VIDEO_EXTS: frozenset[str] = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"})


def class_name(class_id: int) -> str:
    """Map a class id to its name, falling back to the raw id when out of range."""
    if 0 <= class_id < len(CLASS_NAMES):
        return CLASS_NAMES[class_id]
    return str(class_id)
