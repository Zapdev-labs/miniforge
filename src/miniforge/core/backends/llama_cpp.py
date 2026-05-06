"""Llama.cpp backend for high-performance CPU inference."""

import asyncio
import contextlib
import inspect
import logging
import re
import time
from collections.abc import AsyncIterator
from functools import partial
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MAX_MINIMAX_TRAINED_CTX = 196_608  # MiniMax trained context window
CTX_SAFETY_HEADROOM = 2_048  # Reserve for generation
DEFAULT_OPTIMAL_CTX = 194_560  # 192K usable context (196608 - 2048)
GGML_TYPE_MAP: dict[str, int] = {
    "F32": 0,
    "F16": 1,
    "Q4_0": 2,
    "Q4_1": 3,
    "Q5_0": 6,
    "Q5_1": 7,
    "Q8_0": 8,
    "Q8_1": 9,
    "Q2_K": 10,
    "Q3_K": 11,
    "Q4_K": 12,
    "Q5_K": 13,
    "Q6_K": 14,
    "IQ4_NL": 20,
}
KV_ALIAS_MAP: dict[str, str] = {
    "f16": "F16",
    "fp16": "F16",
    "f32": "F32",
    "fp32": "F32",
    "q8_0": "Q8_0",
    "q5_0": "Q5_0",
    "q5_1": "Q5_1",
    "q4_0": "Q4_0",
    "q4_1": "Q4_1",
}
KV_FALLBACK_CHAINS: dict[str, list[str]] = {
    "turbo3": ["q4_0", "q8_0"],
    "turbo4": ["q4_0", "q8_0"],
}


def _resolve_kv_cache_types(cache_type_k: Any, cache_type_v: Any) -> tuple[int | None, int | None]:
    """Convert user-facing KV cache names to llama-cpp enum values."""
    try:
        import llama_cpp

        ggml_enum = getattr(llama_cpp, "GGMLType", None)
    except Exception:
        ggml_enum = None

    def _convert(value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, int):
            return value
        if not isinstance(value, str):
            return None

        normalized = value.strip().lower()

        candidates = [normalized] + KV_FALLBACK_CHAINS.get(normalized, [])

        for candidate in candidates:
            enum_name = KV_ALIAS_MAP.get(candidate, candidate.upper())

            if ggml_enum is not None:
                enum_value = getattr(ggml_enum, enum_name, None)
                if enum_value is not None:
                    if candidate != normalized:
                        logger.info("KV cache: '%s' -> '%s' (via runtime enum)", value, candidate)
                    return int(enum_value)

            if enum_name in GGML_TYPE_MAP:
                if candidate != normalized:
                    logger.info("KV cache: '%s' -> '%s' (via hardcoded enum)", value, candidate)
                return GGML_TYPE_MAP[enum_name]

        return None

    return _convert(cache_type_k), _convert(cache_type_v)


def _context_fallbacks(requested_n_ctx: int, max_safe_ctx: int) -> list[int]:
    first = min(requested_n_ctx, max_safe_ctx)
    candidates = [
        first,
        131_072,
        98_304,
        65_536,
        49_152,
        32_768,
        24_576,
        16_384,
        8_192,
        4_096,
    ]
    unique: list[int] = []
    for value in candidates:
        if 1024 <= value <= max_safe_ctx and value not in unique:
            unique.append(value)
    return unique


def _infer_model_params_from_path(model_path: Path) -> float | None:
    """Infer parameter count from common local GGUF filenames like Qwen3-8B."""
    haystack = " ".join([model_path.name, model_path.parent.name])
    matches = re.findall(r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)\s*[bB](?![A-Za-z])", haystack)
    for raw in matches:
        try:
            value = float(raw)
        except ValueError:
            continue
        if 0.1 <= value <= 2000.0:
            return value
    return None


def _auto_dense_context_limit(model_params_b: float) -> int:
    """Protect dense models from paying huge KV-cache costs by default."""
    if model_params_b <= 12.0:
        return 32_768
    if model_params_b <= 34.0:
        return 24_576
    if model_params_b <= 80.0:
        return 16_384
    return 8_192


def _apply_hardware_auto_tuning(
    config: dict[str, Any], model_params_b: float | None, is_moe: bool
) -> None:
    """Fill default backend settings from hardware detection for direct GGUF usage."""
    if not bool(config.get("auto_context", True)) or model_params_b is None:
        return

    try:
        from miniforge.utils.config import M7Config
        from miniforge.utils.hardware import auto_config
    except Exception as exc:
        logger.debug("Hardware auto-tuning unavailable: %s", exc)
        return

    defaults = M7Config().get_backend_config()
    try:
        tuned = auto_config(model_params_b=model_params_b, is_moe=is_moe)
    except Exception as exc:
        logger.debug("Hardware auto-tuning failed: %s", exc)
        return

    for key, tuned_value in tuned.items():
        if key in {"n_ctx", "model_params_b", "is_moe"}:
            continue
        current_value = config.get(key)
        if key not in config or current_value == defaults.get(key):
            config[key] = tuned_value


def _usage_int(usage: dict[str, Any], key: str, default: int) -> int:
    value = usage.get(key, default)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default


def _supports_kwargs(callable_obj: Any) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return True
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _filter_supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Filter keyword arguments to those accepted by callable_obj."""
    if _supports_kwargs(callable_obj):
        return kwargs
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs
    accepted = set(signature.parameters.keys())
    filtered: dict[str, Any] = {}
    dropped: list[str] = []
    for key, value in kwargs.items():
        if key in accepted:
            filtered[key] = value
        else:
            dropped.append(key)
    if dropped:
        logger.debug("Dropping unsupported llama-cpp kwargs: %s", ", ".join(sorted(dropped)))
    return filtered


class LlamaCppBackend:
    """
    Backend using llama-cpp-python for optimized CPU inference.

    Features:
    - GGUF quantization support
    - TurboQuant KV cache compression
    - AMD ZenDNN acceleration (when available)
    - Flash Attention
    """

    def __init__(self, model_path: Path, config: dict[str, Any]):
        self.model_path = model_path
        self.config = config
        self._llm: Any | None = None
        self._lock = asyncio.Lock()
        self._last_tps: float = 0.0
        self._last_generation: dict[str, Any] = {}
        self._runtime_config: dict[str, Any] = {}

    def _build_completion_kwargs(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        stop: list[str] | None,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "stop": stop or [],
            "stream": stream,
        }
        optional_generation: dict[str, Any] = {
            "min_p": self.config.get("min_p"),
            "typical_p": self.config.get("typical_p"),
            "repeat_penalty": self.config.get("repeat_penalty"),
            "presence_penalty": self.config.get("presence_penalty"),
            "frequency_penalty": self.config.get("frequency_penalty"),
            "mirostat_mode": self.config.get("mirostat_mode"),
            "mirostat_tau": self.config.get("mirostat_tau"),
            "mirostat_eta": self.config.get("mirostat_eta"),
            "seed": self.config.get("seed"),
            "tfs_z": self.config.get("tfs_z"),
            "speculative_n_max": self.config.get("speculative_n_max"),
            "speculative_n_min": self.config.get("speculative_n_min"),
            "speculative_p_min": self.config.get("speculative_p_min"),
        }
        for key, value in optional_generation.items():
            if value is not None:
                kwargs[key] = value
        return kwargs

    async def initialize(self) -> None:
        """Initialize the Llama model with performance optimizations."""
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ImportError(
                "llama-cpp-python not installed. Install with: uv pip install llama-cpp-python"
            ) from exc

        requested_n_ctx = int(self.config.get("n_ctx", DEFAULT_OPTIMAL_CTX))
        model_params_raw = self.config.get("model_params_b")
        model_params_b = (
            float(model_params_raw)
            if isinstance(model_params_raw, (int, float))
            else _infer_model_params_from_path(self.model_path)
        )
        is_moe = bool(self.config.get("is_moe", False))
        _apply_hardware_auto_tuning(self.config, model_params_b, is_moe)
        if (
            bool(self.config.get("auto_context", True))
            and model_params_b is not None
            and not is_moe
        ):
            auto_limit = _auto_dense_context_limit(model_params_b)
            if requested_n_ctx > auto_limit:
                logger.info(
                    "Auto context: reducing dense %.1fB model n_ctx %s -> %s. "
                    "Set auto_context=False or MINIFORGE_N_CTX to override.",
                    model_params_b,
                    requested_n_ctx,
                    auto_limit,
                )
                requested_n_ctx = auto_limit
        model_max_ctx = int(self.config.get("max_model_ctx", MAX_MINIMAX_TRAINED_CTX))
        max_safe_ctx = model_max_ctx - CTX_SAFETY_HEADROOM
        n_ctx = min(requested_n_ctx, max_safe_ctx)
        if n_ctx != requested_n_ctx:
            logger.warning(
                "Requested n_ctx=%s exceeds safe model context window (%s); clamped to n_ctx=%s",
                requested_n_ctx,
                model_max_ctx,
                n_ctx,
            )

        n_threads = self.config.get("n_threads", 8)
        n_batch = self.config.get("n_batch", 2048)
        n_ubatch = self.config.get("n_ubatch", 512)
        n_gpu_layers = self.config.get("n_gpu_layers", 0)
        main_gpu = self.config.get("main_gpu", 0)

        cache_type_k = self.config.get("cache_type_k", "turbo3")
        cache_type_v = self.config.get("cache_type_v", "turbo3")

        flash_attn = self.config.get("flash_attn", True)

        use_mmap = self.config.get("use_mmap", True)
        use_mlock = self.config.get("use_mlock", False)

        memory_mode = str(self.config.get("memory_mode", "auto"))
        if memory_mode == "resident" and not is_moe:
            use_mmap = False
            use_mlock = False
        elif memory_mode in {"mmap", "paged_moe"}:
            use_mmap = True
            use_mlock = False

        if is_moe:
            use_mmap = True
            use_mlock = False
            n_batch = min(n_batch, 512)
            n_ubatch = min(n_ubatch, 256)
            if n_ctx > 32_768:
                logger.warning(
                    "MoE model with n_ctx=%s: KV cache will consume significant RAM. "
                    "AirLLM context sizing should have clamped this already.",
                    n_ctx,
                )

        rope_freq_base = self.config.get("rope_freq_base", 10000.0)
        rope_freq_scale = self.config.get("rope_freq_scale", 1.0)

        logger.info("Loading model from %s", self.model_path)
        logger.info("Context: %s tokens (model max: %s)", n_ctx, model_max_ctx)
        logger.info("Threads: %s, Batch: %s/%s", n_threads, n_batch, n_ubatch)
        logger.info("Memory mode: %s, mmap=%s, mlock=%s", memory_mode, use_mmap, use_mlock)
        logger.info("KV cache: k=%s, v=%s, FlashAttn=%s", cache_type_k, cache_type_v, flash_attn)
        if is_moe:
            logger.info(
                "MoE model: mmap=True, mlock=False, batch=%d/%d (active experts paged from disk)",
                n_batch,
                n_ubatch,
            )
        if n_gpu_layers > 0:
            logger.info("GPU layers: %s on device %s", n_gpu_layers, main_gpu)

        kwargs: dict[str, Any] = {
            "model_path": str(self.model_path),
            "n_ctx": n_ctx,
            "n_threads": n_threads,
            "n_batch": n_batch,
            "verbose": self.config.get("verbose", False),
            "use_mmap": use_mmap,
            "use_mlock": use_mlock,
            "rope_freq_base": rope_freq_base,
            "rope_freq_scale": rope_freq_scale,
            "chat_format": self.config.get("chat_format", None),
        }
        optional_init_kwargs: dict[str, Any] = {
            "n_threads_batch": self.config.get("n_threads_batch"),
            "numa": self.config.get("numa"),
            "seed": self.config.get("seed"),
            "offload_kqv": self.config.get("offload_kqv"),
            "mul_mat_q": self.config.get("mul_mat_q"),
            "f16_kv": self.config.get("f16_kv"),
        }
        for key, value in optional_init_kwargs.items():
            if value is not None:
                kwargs[key] = value

        if n_ubatch != n_batch:
            kwargs["n_ubatch"] = n_ubatch

        if flash_attn:
            kwargs["flash_attn"] = flash_attn

        type_k, type_v = _resolve_kv_cache_types(cache_type_k, cache_type_v)
        if type_k is not None:
            kwargs["type_k"] = type_k
        if type_v is not None:
            kwargs["type_v"] = type_v
        if (type_k is None or type_v is None) and (
            cache_type_k is not None or cache_type_v is not None
        ):
            logger.warning(
                "KV cache type mapping failed for k=%s, v=%s — using f16 defaults. "
                "This DRAMATICALLY increases memory usage. Install a newer llama-cpp-python "
                "build with q4_0/q8_0 KV cache support.",
                cache_type_k,
                cache_type_v,
            )

        if n_gpu_layers > 0:
            kwargs["n_gpu_layers"] = n_gpu_layers
            kwargs["main_gpu"] = main_gpu
            if "tensor_split" in self.config:
                kwargs["tensor_split"] = self.config["tensor_split"]

        _orig_env_init: Any | None = None
        try:
            import jinja2

            _orig_env_init = jinja2.Environment.__init__

            def _env_init_with_loopcontrols(self: Any, *args: Any, **kwargs: Any) -> None:
                ext = kwargs.get("extensions") or []
                lc = "jinja2.ext.loopcontrols"
                if lc not in ext:
                    ext = list(ext) + [lc]
                kwargs["extensions"] = ext
                _orig_env_init(self, *args, **kwargs)

            jinja2.Environment.__init__ = _env_init_with_loopcontrols  # type: ignore[method-assign]
        except Exception:
            pass

        loop = asyncio.get_running_loop()
        init_errors: list[str] = []
        initialized_ctx: int | None = None
        initialized_batch: int | None = None
        try:
            for candidate_ctx in _context_fallbacks(requested_n_ctx, max_safe_ctx):
                candidate_kwargs = dict(kwargs)
                candidate_kwargs["n_ctx"] = candidate_ctx
                candidate_kwargs["n_batch"] = min(n_batch, candidate_ctx)
                candidate_kwargs = _filter_supported_kwargs(Llama, candidate_kwargs)
                try:
                    self._llm = await loop.run_in_executor(None, partial(Llama, **candidate_kwargs))
                    initialized_ctx = candidate_ctx
                    initialized_batch = int(candidate_kwargs["n_batch"])
                    if candidate_ctx != n_ctx:
                        logger.warning(
                            "Initialized with fallback n_ctx=%s after allocation failure at higher context",
                            candidate_ctx,
                        )
                    break
                except ValueError as exc:
                    message = str(exc)
                    init_errors.append(f"n_ctx={candidate_ctx}: {message}")
                    logger.warning("Llama init failed with n_ctx=%s: %s", candidate_ctx, message)
                    continue
                except TypeError as exc:
                    message = str(exc)
                    init_errors.append(f"n_ctx={candidate_ctx}: {message}")
                    logger.warning(
                        "Llama init rejected kwarg set at n_ctx=%s: %s",
                        candidate_ctx,
                        message,
                    )
                    continue
        finally:
            if _orig_env_init is not None:
                with contextlib.suppress(Exception):
                    jinja2.Environment.__init__ = _orig_env_init  # type: ignore[method-assign]

        if self._llm is None:
            errors = "; ".join(init_errors) if init_errors else "unknown initialization error"
            raise RuntimeError(f"Failed to initialize llama.cpp context after fallbacks: {errors}")

        self._runtime_config = {
            "n_ctx": initialized_ctx or n_ctx,
            "n_threads": n_threads,
            "n_batch": initialized_batch or min(n_batch, n_ctx),
            "n_ubatch": n_ubatch,
            "n_gpu_layers": n_gpu_layers,
            "main_gpu": main_gpu,
            "cache_type_k": cache_type_k,
            "cache_type_v": cache_type_v,
            "flash_attn": flash_attn,
            "use_mmap": use_mmap,
            "use_mlock": use_mlock,
            "memory_mode": memory_mode,
            "model_params_b": model_params_b,
            "is_moe": is_moe,
        }

        logger.info("Llama model loaded successfully")

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 40,
        stop: list[str] | None = None,
    ) -> str:
        """Generate text synchronously."""
        if not self._llm:
            raise RuntimeError("Backend not initialized")
        llm = self._llm

        async with self._lock:
            loop = asyncio.get_running_loop()
            t0 = time.perf_counter()
            result = await loop.run_in_executor(
                None,
                lambda: llm.create_completion(
                    **_filter_supported_kwargs(
                        llm.create_completion,
                        self._build_completion_kwargs(
                            prompt=prompt,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            stop=stop,
                            stream=False,
                        ),
                    )
                ),
            )
            elapsed = time.perf_counter() - t0

        text = str(result["choices"][0]["text"])
        usage_raw = result.get("usage", {})
        usage = usage_raw if isinstance(usage_raw, dict) else {}
        n_tokens = _usage_int(usage, "completion_tokens", max(1, len(text.split())))
        prompt_tokens = _usage_int(usage, "prompt_tokens", 0)
        total_tokens = _usage_int(usage, "total_tokens", prompt_tokens + n_tokens)
        tps = n_tokens / elapsed if elapsed > 0 else 0.0
        total_tps = total_tokens / elapsed if elapsed > 0 else 0.0
        logger.info(
            "Generation: prompt=%d completion=%d total=%d in %.1fs "
            "(decode %.2f tok/s, total %.2f tok/s)",
            prompt_tokens,
            n_tokens,
            total_tokens,
            elapsed,
            tps,
            total_tps,
        )
        self._last_tps = tps
        self._last_generation = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": n_tokens,
            "total_tokens": total_tokens,
            "elapsed_seconds": elapsed,
            "decode_tps": tps,
            "total_tps": total_tps,
        }

        return text

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 40,
        stop: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """Generate text with streaming."""
        if not self._llm:
            raise RuntimeError("Backend not initialized")
        llm = self._llm

        async with self._lock:
            loop = asyncio.get_running_loop()
            stream = await loop.run_in_executor(
                None,
                lambda: llm.create_completion(
                    **_filter_supported_kwargs(
                        llm.create_completion,
                        self._build_completion_kwargs(
                            prompt=prompt,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            stop=stop,
                            stream=True,
                        ),
                    )
                ),
            )

            t0 = time.perf_counter()
            first_token_at: float | None = None
            n_tokens = 0
            for chunk in stream:
                text = chunk["choices"][0].get("text", "")
                if text:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    n_tokens += 1
                    yield text
                await asyncio.sleep(0)

            elapsed = time.perf_counter() - t0
            tps = n_tokens / elapsed if elapsed > 0 else 0.0
            first_token_seconds = first_token_at - t0 if first_token_at is not None else None
            logger.info(
                "Stream: %d chunks in %.1fs (%.2f chunk/s, first token %s)",
                n_tokens,
                elapsed,
                tps,
                f"{first_token_seconds:.2f}s" if first_token_seconds is not None else "n/a",
            )
            self._last_tps = tps
            self._last_generation = {
                "completion_tokens": n_tokens,
                "elapsed_seconds": elapsed,
                "decode_tps": tps,
                "first_token_seconds": first_token_seconds,
                "streaming": True,
            }

    async def get_info(self) -> dict[str, Any]:
        """Get model information."""
        if not self._llm:
            return {"status": "not_initialized"}

        return {
            "backend": "llama_cpp",
            "n_vocab": self._llm.n_vocab(),
            "n_ctx": self._llm.n_ctx(),
            "n_embd": self._llm.n_embd(),
            "n_params": getattr(self._llm, "n_params", "unknown"),
            "model_path": str(self.model_path),
            "last_tps": self._last_tps,
            "last_generation": self._last_generation,
            "runtime_config": self._runtime_config,
        }

    async def cleanup(self) -> None:
        """Cleanup model resources."""
        if self._llm:
            del self._llm
            self._llm = None
            logger.info("Llama model cleaned up")
