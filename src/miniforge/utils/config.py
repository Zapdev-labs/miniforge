"""Configuration management for Miniforge."""

from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "MiniMaxAI/MiniMax-M2.7"


@dataclass
class M7Config:
    """
    Configuration optimized for GMKtech M7 hardware.

    Defaults tuned for:
    - AMD Ryzen 7 PRO 6850H (8 cores)
    - 28GB total RAM (4GB to VRAM)
    - CPU-only inference with llama.cpp
    """

    # Hardware limits
    max_memory_gb: float = 24.0  # Leave 4GB for OS
    reserve_memory_gb: float = 4.0

    # CPU settings
    n_threads: int = 8  # Physical cores - Ryzen 7 PRO 6850H
    n_batch: int = 2048  # Increased for better throughput with 192K context
    n_ubatch: int = 512  # Micro-batch size for processing

    # Model settings
    model_id: str = DEFAULT_MODEL_ID
    model_weights_path: Optional[str] = None
    n_ctx: int = 194_560  # Full 192K context minus safety headroom (196608 - 2048)
    quantization: str = "Q4_K_M"

    # KV cache compression (TurboQuant)
    cache_type_k: str = "turbo3"  # 3-bit compression
    cache_type_v: str = "turbo3"

    # GPU offloading (for AMD Radeon 680M iGPU)
    n_gpu_layers: int = 0  # Set to 15-20 to offload some layers to 4GB VRAM
    main_gpu: int = 0  # Primary GPU device
    tensor_split: Optional[list[float]] = None  # For multi-GPU setups

    # Performance features
    flash_attn: bool = True
    use_mmap: bool = True
    use_mlock: bool = False  # WSL2 doesn't support mlock
    rope_scaling_type: Optional[str] = "linear"  # For long context scaling
    rope_freq_base: float = 10000.0  # Default RoPE base frequency
    rope_freq_scale: float = 1.0  # RoPE scaling factor (1.0 = no scaling)

    # Backend selection
    backend: str = "llama_cpp"  # llama_cpp or transformers

    # Generation defaults
    default_max_tokens: int = 512
    default_temperature: float = 1.0
    default_top_p: float = 0.95
    default_top_k: int = 40

    # Feature flags
    enable_tools: bool = True
    enable_vision: bool = True
    enable_streaming: bool = True

    # Paths
    cache_dir: Optional[str] = None
    llama_cpp_path: Optional[str] = None

    # Logging
    verbose: bool = False
    log_level: str = "INFO"

    def __post_init__(self):
        """Validate configuration."""
        if self.n_ctx < 512:
            logger.warning(f"Context window {self.n_ctx} very small, minimum 512 recommended")

        valid_quants = [
            "Q2_K",
            "Q3_K_M",
            "Q4_K_M",
            "Q5_K_M",
            "Q6_K",
            "Q8_0",
            "F16",
            "UD-IQ2_XXS",
            "UD-IQ2_M",
            "UD-Q2_K_XL",
        ]
        if self.quantization not in valid_quants:
            logger.warning(f"Unknown quantization {self.quantization}, using Q4_K_M")
            self.quantization = "Q4_K_M"

        valid_backends = ["llama_cpp", "transformers"]
        if self.backend not in valid_backends:
            logger.warning(f"Unknown backend {self.backend}, using llama_cpp")
            self.backend = "llama_cpp"

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "M7Config":
        """Load configuration from YAML file."""
        path = Path(path)

        if not path.exists():
            # Return default config
            logger.info(f"Config file not found at {path}, using defaults")
            return cls()

        with open(path) as f:
            data = yaml.safe_load(f)

        # Filter to only valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        return cls(**filtered_data)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

        logger.info(f"Configuration saved to {path}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def get_backend_config(self) -> Dict[str, Any]:
        """Get backend-specific configuration."""
        config = {
            "n_ctx": self.n_ctx,
            "n_threads": self.n_threads,
            "n_batch": self.n_batch,
            "n_ubatch": self.n_ubatch,
            "cache_type_k": self.cache_type_k,
            "cache_type_v": self.cache_type_v,
            "flash_attn": self.flash_attn,
            "use_mmap": self.use_mmap,
            "use_mlock": self.use_mlock,
            "verbose": self.verbose,
            "n_gpu_layers": self.n_gpu_layers,
            "main_gpu": self.main_gpu,
        }
        if self.tensor_split is not None:
            config["tensor_split"] = self.tensor_split
        if self.rope_scaling_type is not None:
            config["rope_scaling_type"] = self.rope_scaling_type
        config["rope_freq_base"] = self.rope_freq_base
        config["rope_freq_scale"] = self.rope_freq_scale
        return config

    def get_generation_defaults(self) -> Dict[str, Any]:
        """Get default generation parameters."""
        return {
            "max_tokens": self.default_max_tokens,
            "temperature": self.default_temperature,
            "top_p": self.default_top_p,
            "top_k": self.default_top_k,
        }

    def resolved_model_weights_dir(self) -> Optional[Path]:
        """If set, weights are stored and loaded only from this folder (no duplicate hub cache)."""
        if not self.model_weights_path:
            return None
        return Path(self.model_weights_path).expanduser()


def load_config(config_path: Optional[Union[str, Path]] = None) -> M7Config:
    """
    Load configuration from file or return defaults.

    Searches in order:
    1. Provided path
    2. ~/.config/miniforge/config.yaml
    3. ./miniforge.yaml
    4. Default configuration
    """
    if config_path:
        return M7Config.from_yaml(config_path)

    # Search paths
    search_paths = [
        Path.home() / ".config" / "miniforge" / "config.yaml",
        Path("miniforge.yaml"),
        Path("config.yaml"),
    ]

    for path in search_paths:
        if path.exists():
            return M7Config.from_yaml(path)

    return M7Config()


def create_default_config_file(path: Optional[Union[str, Path]] = None) -> Path:
    """Create a default configuration file."""
    if path is None:
        path = Path.home() / ".config" / "miniforge" / "config.yaml"

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    config = M7Config()
    config.to_yaml(path)

    return path
