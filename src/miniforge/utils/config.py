"""Configuration management for Miniforge.

Auto-detects hardware on first use. Users can override any field manually
or pick a performance preset. Environment overrides are supported so the CLI,
server, and WebUI share the same runtime configuration model.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "MiniMaxAI/MiniMax-M2.7"
VALID_PRESETS = ("speed", "balanced", "memory", "quality", "moe")


def _read_bool_env(name: str) -> bool | None:
    """Parse a boolean environment variable when present."""
    value = os.environ.get(name)
    if value is None:
        return None

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    logger.warning("Ignoring invalid boolean env var %s=%r", name, value)
    return None


def _read_int_env(name: str) -> int | None:
    """Parse an integer environment variable when present."""
    value = os.environ.get(name)
    if value is None:
        return None

    try:
        return int(value)
    except ValueError:
        logger.warning("Ignoring invalid integer env var %s=%r", name, value)
        return None


def _read_float_env(name: str) -> float | None:
    """Parse a float environment variable when present."""
    value = os.environ.get(name)
    if value is None:
        return None

    try:
        return float(value)
    except ValueError:
        logger.warning("Ignoring invalid float env var %s=%r", name, value)
        return None


def _read_list_env(name: str) -> list[str] | None:
    """Parse a path/list environment variable when present."""
    value = os.environ.get(name)
    if value is None:
        return None

    separators = [";", ","]
    if os.pathsep in value and not (len(value) > 2 and value[1] == ":"):
        separators.append(os.pathsep)

    for separator in separators:
        if separator in value:
            items = [item.strip() for item in value.split(separator) if item.strip()]
            return items or None

    stripped = value.strip()
    return [stripped] if stripped else None


@dataclass
class M7Config:
    """Inference configuration — auto-detected by default, overridable field-by-field."""

    # Hardware limits (auto-detected from actual RAM)
    max_memory_gb: float = 24.0
    reserve_memory_gb: float = 4.0

    # CPU settings
    n_threads: int = 8
    n_batch: int = 2048
    n_ubatch: int = 512
    cpu_mask: str = "0-7"
    cpu_strict: bool = False
    priority: str = "normal"

    # CPU ISA extensions (auto-detected if None)
    use_avx: bool | None = None
    use_avx2: bool | None = None
    use_avx512: bool | None = None
    use_fma: bool | None = None
    use_f16c: bool | None = None

    # Model settings
    model_id: str = DEFAULT_MODEL_ID
    model_weights_path: str | None = None
    n_ctx: int = 194_560
    quantization: str = "UD-IQ2_XXS"

    # KV cache compression
    cache_type_k: str = "q4_0"
    cache_type_v: str = "q4_0"

    # Advanced KV cache settings
    cache_quantization_group: int = 128
    cache_mixed_precision: bool = True

    # Speculative decoding
    speculative_n_max: int = 16
    speculative_n_min: int = 5
    speculative_p_min: float = 0.8

    # GPU offloading
    n_gpu_layers: int = 0
    main_gpu: int = 0
    tensor_split: list[float] | None = None

    # Performance features
    flash_attn: bool = True
    use_mmap: bool = True
    use_mlock: bool = False
    rope_scaling_type: str | None = "linear"
    rope_freq_base: float = 10000.0
    rope_freq_scale: float = 1.0

    # Backend selection
    backend: str = "llama_cpp"

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
    cache_dir: str | None = None
    download_dir: str | None = None
    llama_cpp_path: str | None = None
    model_dirs: list[str] | None = None
    offline: bool = False

    # Per-model overrides (set automatically from registry, or manually)
    max_model_ctx: int | None = None
    is_moe: bool = False

    # Logging
    verbose: bool = False
    log_level: str = "INFO"

    def __post_init__(self):
        """Validate configuration."""
        if self.n_ctx < 512:
            logger.warning("Context window %s very small, minimum 512 recommended", self.n_ctx)

        valid_quants = [
            "Q2_K",
            "Q3_K",
            "Q3_K_S",
            "Q3_K_M",
            "Q4_K_M",
            "Q5_K_M",
            "Q6_K",
            "Q8_0",
            "F16",
            "UD-TQ1_0",
            "UD-IQ1_S",
            "UD-IQ1_M",
            "UD-IQ2_XXS",
            "UD-IQ2_M",
            "UD-Q2_K_XL",
            "UD-IQ3_XXS",
            "UD-IQ3_S",
            "UD-IQ3_K_S",
            "UD-Q3_K_M",
            "UD-Q3_K_XL",
            "UD-IQ4_XS",
            "UD-Q4_K_S",
            "MXFP4_MOE",
            "UD-Q4_NL",
            "UD-Q4_K_M",
            "UD-Q4_K_XL",
            "UD-Q5_K_S",
            "UD-Q5_K_M",
            "UD-Q5_K_XL",
            "UD-Q6_K",
            "UD-Q6_K_XL",
            "UD-Q8_K_XL",
            "BF16",
        ]
        if self.quantization not in valid_quants:
            logger.warning("Unknown quantization %s, using Q4_K_M", self.quantization)
            self.quantization = "Q4_K_M"

        valid_backends = ["llama_cpp", "transformers"]
        if self.backend not in valid_backends:
            logger.warning("Unknown backend %s, using llama_cpp", self.backend)
            self.backend = "llama_cpp"

    @classmethod
    def auto(cls, model_params_b: float = 2.7, is_moe: bool = False) -> M7Config:
        """Auto-detect hardware and return a tuned config."""
        return cls.from_hardware(model_params_b=model_params_b, is_moe=is_moe)

    @classmethod
    def performance_preset(cls, preset: str = "balanced") -> M7Config:
        """Create a config optimized for a specific performance profile.

        Presets:
            - "speed": Maximum throughput, higher memory use
            - "balanced": Good speed with reasonable memory (default)
            - "memory": Minimize RAM usage, may be slower
            - "quality": Best output quality, no compression
            - "moe": Optimized for Mixture-of-Experts models
        """
        if preset not in VALID_PRESETS:
            logger.warning("Unknown preset %s, using balanced", preset)
            preset = "balanced"

        config = cls()

        if preset == "speed":
            config.n_threads = 16
            config.n_batch = 4096
            config.n_ubatch = 1024
            config.cache_type_k = "f16"
            config.cache_type_v = "f16"
            config.flash_attn = True
            config.priority = "high"
        elif preset == "balanced":
            config.n_threads = 12
            config.n_batch = 2048
            config.n_ubatch = 512
            config.cache_type_k = "q8_0"
            config.cache_type_v = "q8_0"
            config.flash_attn = True
        elif preset == "memory":
            config.n_threads = 8
            config.n_batch = 512
            config.n_ubatch = 256
            config.cache_type_k = "q4_0"
            config.cache_type_v = "q4_0"
            config.flash_attn = True
            config.quantization = "UD-IQ2_XXS"
        elif preset == "quality":
            config.n_threads = 8
            config.n_batch = 1024
            config.cache_type_k = "f16"
            config.cache_type_v = "f16"
            config.flash_attn = False
            config.quantization = "Q8_0"
        elif preset == "moe":
            config.n_threads = 16
            config.n_batch = 2048
            config.n_ubatch = 512
            config.cache_type_k = "q4_0"
            config.cache_type_v = "q4_0"
            config.flash_attn = True
            config.use_mmap = True
            config.use_mlock = False
            config.quantization = "UD-IQ2_XXS"

        return config

    @classmethod
    def from_env(cls, base: M7Config | None = None) -> M7Config:
        """Build a config from environment variables layered over auto config."""
        preset = os.environ.get("MINIFORGE_PRESET")
        if base is None:
            config = cls.performance_preset(preset) if preset else cls.auto()
        else:
            config = base

        config.apply_overrides(
            model_id=os.environ.get("MINIFORGE_MODEL"),
            backend=os.environ.get("MINIFORGE_BACKEND"),
            quantization=os.environ.get("MINIFORGE_QUANTIZATION"),
            download_dir=os.environ.get("MINIFORGE_DOWNLOAD_DIR"),
            cache_dir=os.environ.get("MINIFORGE_CACHE_DIR"),
            model_weights_path=os.environ.get("MINIFORGE_MODEL_WEIGHTS_PATH"),
            model_dirs=_read_list_env("MINIFORGE_MODEL_DIRS"),
            llama_cpp_path=os.environ.get("MINIFORGE_LLAMA_CPP"),
            offline=_read_bool_env("MINIFORGE_OFFLINE"),
            log_level=os.environ.get("MINIFORGE_LOG_LEVEL"),
            n_ctx=_read_int_env("MINIFORGE_N_CTX"),
            n_threads=_read_int_env("MINIFORGE_N_THREADS"),
            max_tokens=_read_int_env("MINIFORGE_MAX_TOKENS"),
            temperature=_read_float_env("MINIFORGE_TEMPERATURE"),
            top_p=_read_float_env("MINIFORGE_TOP_P"),
            verbose=_read_bool_env("MINIFORGE_VERBOSE"),
        )
        if os.environ.get("MINIFORGE_QUANTIZATION") is None:
            config.apply_model_metadata()
        return config

    @classmethod
    def from_hardware(
        cls,
        model_params_b: float = 2.7,
        is_moe: bool = False,
    ) -> M7Config:
        """Create a configuration auto-tuned to the current hardware."""
        from miniforge.utils.hardware import auto_config, detect_hardware

        profile = detect_hardware()
        settings = auto_config(model_params_b=model_params_b, is_moe=is_moe, profile=profile)
        return cls(**settings)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def apply_model_metadata(self) -> M7Config:
        """Apply known model registry metadata without changing explicit user paths."""
        try:
            from miniforge.models.registry import get_registry

            info = get_registry(Path(self.cache_dir) if self.cache_dir else None).get_model_info(
                self.model_id
            )
        except Exception:
            info = None

        if info is None:
            return self

        self.quantization = info.default_quantization
        self.max_model_ctx = info.max_context
        self.is_moe = info.is_moe
        return self

    def apply_overrides(
        self,
        *,
        model_id: str | None = None,
        backend: str | None = None,
        quantization: str | None = None,
        download_dir: str | None = None,
        cache_dir: str | None = None,
        model_weights_path: str | None = None,
        model_dirs: list[str] | None = None,
        llama_cpp_path: str | None = None,
        offline: bool | None = None,
        log_level: str | None = None,
        n_ctx: int | None = None,
        n_threads: int | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        verbose: bool | None = None,
    ) -> M7Config:
        """Apply non-None overrides in place and return the config."""
        if model_id is not None:
            self.model_id = model_id
        if backend is not None:
            self.backend = backend
        if quantization is not None:
            self.quantization = quantization
        if download_dir is not None:
            self.download_dir = download_dir
        if cache_dir is not None:
            self.cache_dir = cache_dir
        if model_weights_path is not None:
            self.model_weights_path = model_weights_path
        if model_dirs is not None:
            self.model_dirs = model_dirs
        if llama_cpp_path is not None:
            self.llama_cpp_path = llama_cpp_path
        if offline is not None:
            self.offline = offline
        if log_level is not None:
            self.log_level = log_level
        if n_ctx is not None:
            self.n_ctx = n_ctx
        if n_threads is not None:
            self.n_threads = n_threads
        if max_tokens is not None:
            self.default_max_tokens = max_tokens
        if temperature is not None:
            self.default_temperature = temperature
        if top_p is not None:
            self.default_top_p = top_p
        if verbose is not None:
            self.verbose = verbose

        self.__post_init__()
        return self

    def summary(self) -> dict[str, Any]:
        """Return a compact runtime summary for UX surfaces."""
        return {
            "model_id": self.model_id,
            "backend": self.backend,
            "quantization": self.quantization,
            "n_ctx": self.n_ctx,
            "n_threads": self.n_threads,
            "cache_type_k": self.cache_type_k,
            "cache_type_v": self.cache_type_v,
            "flash_attn": self.flash_attn,
            "download_dir": self.download_dir,
            "cache_dir": self.cache_dir,
            "model_weights_path": self.model_weights_path,
            "model_dirs": self.model_dirs,
            "offline": self.offline,
            "generation": self.get_generation_defaults(),
        }

    def get_backend_config(self) -> dict[str, Any]:
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
            "cpu_mask": self.cpu_mask,
            "cpu_strict": self.cpu_strict,
            "priority": self.priority,
        }
        if self.use_avx is not None:
            config["use_avx"] = self.use_avx
        if self.use_avx2 is not None:
            config["use_avx2"] = self.use_avx2
        if self.use_avx512 is not None:
            config["use_avx512"] = self.use_avx512
        if self.use_fma is not None:
            config["use_fma"] = self.use_fma
        if self.use_f16c is not None:
            config["use_f16c"] = self.use_f16c
        if self.tensor_split is not None:
            config["tensor_split"] = self.tensor_split
        if self.rope_scaling_type is not None:
            config["rope_scaling_type"] = self.rope_scaling_type
        config["rope_freq_base"] = self.rope_freq_base
        config["rope_freq_scale"] = self.rope_freq_scale
        if self.max_model_ctx is not None:
            config["max_model_ctx"] = self.max_model_ctx
        if self.is_moe:
            config["is_moe"] = True
        return config

    def get_generation_defaults(self) -> dict[str, Any]:
        """Get default generation parameters."""
        return {
            "max_tokens": self.default_max_tokens,
            "temperature": self.default_temperature,
            "top_p": self.default_top_p,
            "top_k": self.default_top_k,
        }

    def resolved_model_weights_dir(self) -> Path | None:
        """If set, weights are stored and loaded only from this folder."""
        if not self.model_weights_path:
            return None
        return Path(self.model_weights_path).expanduser()
