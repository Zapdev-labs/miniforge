"""Llama.cpp backend for high-performance CPU inference."""

from typing import Optional, Dict, Any, List, AsyncIterator, Tuple
from pathlib import Path
import asyncio
import logging

logger = logging.getLogger(__name__)

MAX_MINIMAX_TRAINED_CTX = 196_608  # MiniMax trained context window
CTX_SAFETY_HEADROOM = 2_048  # Reserve for generation
DEFAULT_OPTIMAL_CTX = 194_560  # 192K usable context (196608 - 2048)


def _resolve_kv_cache_types(
    cache_type_k: Any, cache_type_v: Any
) -> Tuple[Optional[int], Optional[int]]:
    """
    Convert user-friendly KV cache type values to llama-cpp enum ints.

    Newer llama-cpp builds require integer ggml type enums for type_k/type_v.
    If values cannot be mapped, return None to let llama-cpp use defaults.
    """
    try:
        import llama_cpp
    except Exception:
        return None, None

    ggml_enum = getattr(llama_cpp, "GGMLType", None)
    if ggml_enum is None:
        return None, None

    alias_map = {
        "f16": "F16",
        "fp16": "F16",
        "f32": "F32",
        "fp32": "F32",
        "q8_0": "Q8_0",
        "q5_0": "Q5_0",
        "q5_1": "Q5_1",
        "q4_0": "Q4_0",
        "q4_1": "Q4_1",
        # TurboQuant aliases are not standard ggml cache enums.
        "turbo3": None,
    }

    def _convert(value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, int):
            return value
        if not isinstance(value, str):
            return None

        normalized = value.strip().lower()
        enum_name = alias_map.get(normalized, normalized.upper())
        if enum_name is None:
            return None

        enum_value = getattr(ggml_enum, enum_name, None)
        return int(enum_value) if enum_value is not None else None

    return _convert(cache_type_k), _convert(cache_type_v)


def _context_fallbacks(requested_n_ctx: int, max_safe_ctx: int) -> List[int]:
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
    unique: List[int] = []
    for value in candidates:
        if 1024 <= value <= max_safe_ctx and value not in unique:
            unique.append(value)
    return unique


class LlamaCppBackend:
    """
    Backend using llama-cpp-python for optimized CPU inference.

    Features:
    - GGUF quantization support
    - TurboQuant KV cache compression
    - AMD ZenDNN acceleration (when available)
    - Flash Attention
    """

    def __init__(self, model_path: Path, config: Dict[str, Any]):
        self.model_path = model_path
        self.config = config
        self._llm = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the Llama model with performance optimizations."""
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed. Install with: uv pip install llama-cpp-python"
            )

        # Extract configuration with M7-optimized defaults for full 192K context
        requested_n_ctx = int(self.config.get("n_ctx", DEFAULT_OPTIMAL_CTX))
        max_safe_ctx = MAX_MINIMAX_TRAINED_CTX - CTX_SAFETY_HEADROOM
        n_ctx = min(requested_n_ctx, max_safe_ctx)
        if n_ctx != requested_n_ctx:
            logger.warning(
                "Requested n_ctx=%s exceeds safe model context window; clamped to n_ctx=%s",
                requested_n_ctx,
                n_ctx,
            )

        # CPU/GPU settings
        n_threads = self.config.get("n_threads", 8)
        n_batch = self.config.get("n_batch", 2048)  # Larger batch for 192K context
        n_ubatch = self.config.get("n_ubatch", 512)  # Micro-batch for chunked processing
        n_gpu_layers = self.config.get("n_gpu_layers", 0)
        main_gpu = self.config.get("main_gpu", 0)

        # KV cache quantization (TurboQuant - aggressive compression for 192K)
        cache_type_k = self.config.get("cache_type_k", "turbo3")
        cache_type_v = self.config.get("cache_type_v", "turbo3")

        # Performance features
        flash_attn = self.config.get("flash_attn", True)

        # Memory mapping (optimize for large context)
        use_mmap = self.config.get("use_mmap", True)
        use_mlock = self.config.get("use_mlock", False)

        # RoPE settings for long context
        rope_freq_base = self.config.get("rope_freq_base", 10000.0)
        rope_freq_scale = self.config.get("rope_freq_scale", 1.0)

        logger.info(f"Loading model from {self.model_path}")
        logger.info(f"Context: {n_ctx} tokens (optimal for 192K)")
        logger.info(f"Threads: {n_threads}, Batch: {n_batch}/{n_ubatch}")
        logger.info(f"KV cache: k={cache_type_k}, v={cache_type_v}, FlashAttn={flash_attn}")
        if n_gpu_layers > 0:
            logger.info(f"GPU layers: {n_gpu_layers} on device {main_gpu}")

        # Build kwargs for Llama with all optimizations
        kwargs = {
            "model_path": str(self.model_path),
            "n_ctx": n_ctx,
            "n_threads": n_threads,
            "n_batch": n_batch,
            "verbose": self.config.get("verbose", False),
            "use_mmap": use_mmap,
            "use_mlock": use_mlock,
            "rope_freq_base": rope_freq_base,
            "rope_freq_scale": rope_freq_scale,
        }

        # Add micro-batch size if supported (helps with 192K context)
        if n_ubatch != n_batch:
            kwargs["n_ubatch"] = n_ubatch

        # Flash Attention - critical for 192K context performance
        if flash_attn:
            kwargs["flash_attn"] = flash_attn

        # KV cache types (TurboQuant for memory efficiency)
        type_k, type_v = _resolve_kv_cache_types(cache_type_k, cache_type_v)
        if type_k is not None:
            kwargs["type_k"] = type_k
        if type_v is not None:
            kwargs["type_v"] = type_v
        if (type_k is None or type_v is None) and (
            cache_type_k is not None or cache_type_v is not None
        ):
            logger.info(
                "KV cache type mapping unavailable for this llama-cpp build; using defaults"
            )

        # GPU layers (for AMD iGPU or discrete GPU)
        if n_gpu_layers > 0:
            kwargs["n_gpu_layers"] = n_gpu_layers
            kwargs["main_gpu"] = main_gpu
            if "tensor_split" in self.config:
                kwargs["tensor_split"] = self.config["tensor_split"]

        # Initialize in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        init_errors: List[str] = []
        for candidate_ctx in _context_fallbacks(requested_n_ctx, max_safe_ctx):
            kwargs["n_ctx"] = candidate_ctx
            kwargs["n_batch"] = min(n_batch, candidate_ctx)
            try:
                self._llm = await loop.run_in_executor(None, lambda: Llama(**kwargs))
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

        if self._llm is None:
            errors = "; ".join(init_errors) if init_errors else "unknown initialization error"
            raise RuntimeError(f"Failed to initialize llama.cpp context after fallbacks: {errors}")

        logger.info("Llama model loaded successfully")

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 40,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Generate text synchronously."""
        if not self._llm:
            raise RuntimeError("Backend not initialized")

        async with self._lock:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._llm.create_completion(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    stop=stop or [],
                ),
            )

        return result["choices"][0]["text"]

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 40,
        stop: Optional[List[str]] = None,
    ) -> AsyncIterator[str]:
        """Generate text with streaming."""
        if not self._llm:
            raise RuntimeError("Backend not initialized")

        async with self._lock:
            loop = asyncio.get_event_loop()

            # Create completion in thread pool
            stream = await loop.run_in_executor(
                None,
                lambda: self._llm.create_completion(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    stop=stop or [],
                    stream=True,
                ),
            )

            # Stream tokens
            for chunk in stream:
                text = chunk["choices"][0].get("text", "")
                if text:
                    yield text
                await asyncio.sleep(0)  # Allow other tasks

    async def get_info(self) -> Dict[str, Any]:
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
        }

    async def cleanup(self) -> None:
        """Cleanup model resources."""
        if self._llm:
            del self._llm
            self._llm = None
            logger.info("Llama model cleaned up")
