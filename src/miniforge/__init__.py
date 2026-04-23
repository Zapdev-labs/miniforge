"""
Miniforge: High-performance MiniMax M2.7 inference library for GMKtech M7.

Optimized for 28GB RAM constraint with GGUF quantization and TurboQuant KV cache compression.
"""

__version__ = "0.1.0"
__author__ = "Miniforge User"

from miniforge.core.engine import InferenceEngine
from miniforge.models.minimax import Miniforge
from miniforge.utils.config import M7Config
from miniforge.utils.hardware import HardwareProfile, auto_config, detect_hardware

__all__ = [
    "Miniforge",
    "InferenceEngine",
    "M7Config",
    "detect_hardware",
    "auto_config",
    "HardwareProfile",
]
