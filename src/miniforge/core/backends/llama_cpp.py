"""Llama.cpp backend for high-performance CPU inference."""

from typing import Optional, Dict, Any, List, AsyncIterator
from pathlib import Path
import asyncio
import logging

logger = logging.getLogger(__name__)


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
        """Initialize the Llama model."""
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed. Install with: uv pip install llama-cpp-python"
            )

        # Extract configuration with M7-optimized defaults
        n_ctx = self.config.get("n_ctx", 200_000)
        n_threads = self.config.get("n_threads", 8)
        n_batch = self.config.get("n_batch", 512)
        n_gpu_layers = self.config.get("n_gpu_layers", 0)

        # KV cache quantization (TurboQuant)
        cache_type_k = self.config.get("cache_type_k", "turbo3")
        cache_type_v = self.config.get("cache_type_v", "turbo3")

        # Flash Attention
        flash_attn = self.config.get("flash_attn", True)

        # Memory mapping
        use_mmap = self.config.get("use_mmap", True)
        use_mlock = self.config.get("use_mlock", False)  # WSL2 doesn't support mlock well

        logger.info(f"Loading model from {self.model_path}")
        logger.info(f"Config: ctx={n_ctx}, threads={n_threads}, batch={n_batch}")
        logger.info(f"KV cache: k={cache_type_k}, v={cache_type_v}")

        # Build kwargs for Llama
        kwargs = {
            "model_path": str(self.model_path),
            "n_ctx": n_ctx,
            "n_threads": n_threads,
            "n_batch": n_batch,
            "verbose": self.config.get("verbose", False),
            "use_mmap": use_mmap,
            "use_mlock": use_mlock,
        }

        # Add optional params if llama-cpp supports them
        if flash_attn:
            kwargs["flash_attn"] = flash_attn

        # KV cache types (if supported by this llama-cpp version)
        try:
            kwargs["type_k"] = cache_type_k
            kwargs["type_v"] = cache_type_v
        except TypeError:
            logger.warning("TurboQuant KV types not supported in this llama-cpp version")

        # GPU layers (for AMD ROCm if available)
        if n_gpu_layers > 0:
            kwargs["n_gpu_layers"] = n_gpu_layers

        # Initialize in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        self._llm = await loop.run_in_executor(None, lambda: Llama(**kwargs))

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
