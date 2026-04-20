"""Utility functions and helpers."""

import logging
import sys
from pathlib import Path
from typing import Optional

from miniforge.utils.config import M7Config
from miniforge.utils.hardware import HardwareProfile, auto_config, detect_hardware
from miniforge.utils.monitoring import GenerationMetrics, MemoryMonitor, PerformanceMonitor

__all__ = [
    "M7Config",
    "PerformanceMonitor",
    "MemoryMonitor",
    "GenerationMetrics",
    "detect_hardware",
    "auto_config",
    "HardwareProfile",
]


def setup_logging(
    level: str = "INFO",
    format_string: str | None = None,
) -> None:
    """Setup logging configuration."""
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def get_cache_dir() -> Path:
    """Get default cache directory."""
    return Path.home() / ".cache" / "miniforge"


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path
