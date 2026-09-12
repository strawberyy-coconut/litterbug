"""Runtime guards: seeding, device selection, environment reporting, logging."""

from __future__ import annotations

import logging
import os
import random
import sys

DEFAULT_SEED = 42
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"

log = logging.getLogger("litterbug.runtime")

SHM_PATH = "/dev/shm"
MIN_SHM_PER_WORKER_GB = 0.5
MAX_WORKERS = 8


def shm_free_gb() -> float | None:
    try:
        stats = os.statvfs(SHM_PATH)
    except OSError:
        return None
    return (stats.f_bavail * stats.f_frsize) / 1024**3


def resolve_workers(requested: int | None = None) -> int:
    if requested is not None:
        return max(0, int(requested))

    available = shm_free_gb()
    if available is None:
        return min(MAX_WORKERS, os.cpu_count() or 1)

    if available < MIN_SHM_PER_WORKER_GB:
        log.warning(
            "/dev/shm has only %.0f MiB free, so data loading runs in the main process "
            "(workers=0). Training will be slower. Fix properly by giving the container "
            "more shared memory (shm_size / --shm-size) and re-running with --workers.",
            available * 1024,
        )
        return 0

    budget = int(available // MIN_SHM_PER_WORKER_GB)
    return max(1, min(MAX_WORKERS, budget, os.cpu_count() or 1))


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once, to stdout — this code always runs headless."""
    logging.basicConfig(
        level=level, format=_LOG_FORMAT, datefmt="%H:%M:%S", stream=sys.stdout, force=True
    )


def seed_everything(seed: int = DEFAULT_SEED) -> int:
    """Seed every RNG we can reach, so runs are reproducible without hand-tuning."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)

    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    return seed


def cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def vram_gb() -> float | None:
    """Total VRAM of device 0 in GiB, or ``None`` when CUDA is unavailable."""
    if not cuda_available():
        return None
    import torch

    return torch.cuda.get_device_properties(0).total_memory / 1024**3


def _no_gpu_message(purpose: str, requested: str | None) -> str:
    hint = f" (--device {requested} was requested)" if requested else ""
    return (
        f"No CUDA device is visible, refusing to run {purpose} on CPU{hint}.\n"
        "  - Run `nvidia-smi` and confirm the GPU is visible inside the container.\n"
        "  - Check that compose.dev.yaml still requests the nvidia device reservation.\n"
        "  - Pass --device cpu only if a CPU run is genuinely intended."
    )


def resolve_device(
    requested: str | None = None,
    *,
    require_cuda: bool = False,
    purpose: str = "training",
) -> str:
    """Pick a device, failing loudly when CUDA is required but missing.

    A silent CPU fallback turns a two hour job into several days, so training refuses
    to start rather than quietly degrading to the CPU.
    """
    if requested:
        if require_cuda and str(requested).lower().startswith("cpu"):
            raise SystemExit(_no_gpu_message(purpose, requested))
        return requested
    if cuda_available():
        return "cuda:0"
    if require_cuda:
        raise SystemExit(_no_gpu_message(purpose, None))
    return "cpu"


def describe_environment() -> str:
    """One-block environment summary — useful in logs, the report and bug reports."""
    lines = [f"python      {sys.version.split()[0]}"]
    try:
        import torch
    except ImportError:
        lines.append("torch       not installed")
        return "\n".join(lines)

    lines.append(f"torch       {torch.__version__}")
    lines.append(f"cuda        {torch.version.cuda or 'n/a'}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        lines.append(f"gpu         {props.name} ({props.total_memory / 1024**3:.1f} GiB)")
    else:
        lines.append("gpu         none visible")
    return "\n".join(lines)
