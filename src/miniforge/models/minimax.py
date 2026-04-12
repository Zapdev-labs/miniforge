"""Main MiniMax model interface."""

from typing import Optional, Union, List, Dict, Any, AsyncIterator
from pathlib import Path
import asyncio
import logging

from miniforge.core.engine import InferenceEngine
from miniforge.core.memory import MemoryManager
from miniforge.models.gguf_convert import auto_convert_safetensors_to_gguf
from miniforge.models.registry import get_registry
from miniforge.utils.config import M7Config, load_config
from miniforge.generation.tools import Tool, ToolExecutor
from miniforge.multimodal.vision import VisionProcessor

logger = logging.getLogger(__name__)


class Miniforge:
    """
    High-level interface for MiniMax M2.7 inference.

    Optimized for GMKtech M7 hardware:
    - 28GB RAM (4GB to VRAM)
    - AMD Ryzen 7 PRO 6850H
    - WSL2 environment

    Features:
    - GGUF quantization (Q4_K_M recommended)
    - TurboQuant KV cache (turbo3 = 3-bit)
    - Tool calling
    - Vision/multimodal
    - Streaming generation
    - Async support
    """

    DEFAULT_MODEL = "MiniMaxAI/MiniMax-M2.7"

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        config: Optional[M7Config] = None,
        backend: Optional[str] = None,
    ):
        """
        Initialize MiniMax model.

        Args:
            model_path: Path to model file or HuggingFace ID
            config: Configuration object
            backend: Override backend (llama_cpp or transformers)
        """
        self.config = config or load_config()
        self.model_path = model_path or self.DEFAULT_MODEL
        self.backend_name = backend or self.config.backend

        self._engine: Optional[InferenceEngine] = None
        self._memory_manager = MemoryManager(
            target_utilization=self.config.max_memory_gb / MemoryManager.TOTAL_RAM_GB
        )
        self._registry = get_registry(self.config.cache_dir)
        self._tool_executor: Optional[ToolExecutor] = None
        self._vision_processor: Optional[VisionProcessor] = None

        self._initialized = False

    @classmethod
    async def from_pretrained(
        cls,
        model_id: str = "MiniMaxAI/MiniMax-M2.7",
        quantization: Optional[str] = None,
        config: Optional[M7Config] = None,
        backend: str = "llama_cpp",
    ) -> "Miniforge":
        """
        Load model from HuggingFace or cache.

        Args:
            model_id: HuggingFace model ID
            quantization: Quantization type (auto-selected if None)
            config: Configuration object
            backend: Backend to use

        Returns:
            Initialized Miniforge instance
        """
        instance = cls(model_id, config, backend)
        await instance._load_model(quantization)
        return instance

    @classmethod
    def from_gguf(
        cls,
        gguf_path: Union[str, Path],
        config: Optional[M7Config] = None,
    ) -> "Miniforge":
        """
        Load from local GGUF file.

        Args:
            gguf_path: Path to GGUF file
            config: Configuration object

        Returns:
            Miniforge instance (call initialize() to load)
        """
        return cls(gguf_path, config, "llama_cpp")

    async def _load_model(self, quantization: Optional[str] = None) -> None:
        """Load and prepare model."""
        model_id = str(self.model_path)

        # Get model info
        model_info = self._registry.get_model_info(model_id)

        if model_info:
            params = model_info.params_billions
        else:
            # Assume 2.7B for MiniMax
            params = 2.7

        # Determine quantization
        if quantization is None:
            if self.config.quantization:
                quantization = self.config.quantization
            else:
                quantization = self._memory_manager.select_quantization(params)

        # Check cache for GGUF
        cached_path = self._registry.get_cached_gguf_path(model_id, quantization)

        if cached_path:
            logger.info(f"Using cached GGUF: {cached_path}")
            model_path = cached_path
        elif self.backend_name == "llama_cpp":
            # Try to download GGUF from HuggingFace
            try:
                from huggingface_hub import hf_hub_download

                # Look for unsloth or other GGUF repos
                repo_variants = [
                    model_id.replace("MiniMaxAI/", "unsloth/"),
                    model_id + "-GGUF",
                ]

                gguf_file = None
                for repo in repo_variants:
                    try:
                        # Try common GGUF filenames
                        for q in [quantization, "Q4_K_M", "Q5_K_M"]:
                            try:
                                filename = f"{model_id.split('/')[-1]}-{q}.gguf"

                                def _hub_dl() -> str:
                                    return hf_hub_download(
                                        repo_id=repo,
                                        filename=filename,
                                        local_dir=str(self._registry.gguf_dir),
                                    )

                                gguf_file = await asyncio.get_event_loop().run_in_executor(
                                    None,
                                    _hub_dl,
                                )
                                break
                            except Exception:
                                continue
                        if gguf_file:
                            break
                    except Exception:
                        continue

                if gguf_file:
                    model_path = Path(gguf_file)
                else:
                    loop = asyncio.get_event_loop()

                    def _to_gguf() -> Path:
                        return auto_convert_safetensors_to_gguf(
                            self._registry,
                            model_id,
                            quantization,
                            llama_cpp_root=self.config.llama_cpp_path,
                        )

                    model_path = await loop.run_in_executor(None, _to_gguf)

            except Exception as e:
                logger.warning(f"Could not get GGUF (download or convert): {e}")
                logger.info("Falling back to transformers backend")
                self.backend_name = "transformers"
                model_path = model_id
        else:
            trial = Path(model_id)
            model_path = trial if trial.exists() else model_id

        # Initialize engine
        backend_config = self.config.get_backend_config()

        self._engine = InferenceEngine(
            model_path=model_path,
            backend=self.backend_name,
            config=backend_config,
        )

        # Initialize
        await self._engine.initialize()
        self._initialized = True

        # Register memory usage
        # Estimate based on quantization
        quant_ratios = {
            "Q8_0": 1.0,
            "Q6_K": 0.75,
            "Q5_K_M": 0.625,
            "Q4_K_M": 0.5,
            "Q3_K_M": 0.375,
            "Q2_K": 0.25,
        }
        ratio = quant_ratios.get(quantization, 0.5)
        model_gb = params * 2 * ratio

        self._memory_manager.register_model_memory(model_gb)
        self._memory_manager.register_kv_memory(
            self.config.n_ctx,
            self.config.cache_type_k,
        )

        logger.info(f"Model loaded: {model_id} ({quantization}, ~{model_gb:.1f}GB)")

    async def initialize(self) -> None:
        """Explicit initialization if not done in constructor."""
        if not self._initialized:
            await self._load_model()

    async def chat(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        tools: Optional[List[Tool]] = None,
        stream: bool = False,
    ) -> Union[str, AsyncIterator[str]]:
        """
        Send a chat message.

        Args:
            message: User message
            history: Previous conversation history
            system_prompt: System prompt/instructions
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            tools: Available tools for the model
            stream: Whether to stream response

        Returns:
            Response string or async iterator for streaming
        """
        if not self._engine:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        # Build messages
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": message})

        # Add tool instructions if provided
        if tools and self.config.enable_tools:
            if not self._tool_executor:
                self._tool_executor = ToolExecutor(tools)
            else:
                for tool in tools:
                    self._tool_executor.register(tool)

            tool_instruction = self._tool_executor.format_tools_for_prompt(tools)
            messages.insert(0, {"role": "system", "content": tool_instruction})

        # Get generation parameters
        gen_params = self.config.get_generation_defaults()
        if max_tokens is not None:
            gen_params["max_tokens"] = max_tokens
        if temperature is not None:
            gen_params["temperature"] = temperature

        # Generate
        response = await self._engine.chat(
            messages=messages,
            stream=stream,
            **gen_params,
        )

        if stream:
            return response

        # Check for tool calls in non-streaming response
        if tools and self._tool_executor:
            tool_calls = self._tool_executor.parse_tool_calls(response)
            if tool_calls:
                # Execute tools
                results = await self._tool_executor.execute(tool_calls)

                # Build follow-up prompt with tool results
                tool_messages = []
                for result in results:
                    tool_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": result.tool_call_id,
                            "content": result.content,
                        }
                    )

                # Get final response
                messages.append({"role": "assistant", "content": response})
                messages.extend(tool_messages)
                messages.append(
                    {
                        "role": "user",
                        "content": "Based on the tool results, please provide your final response.",
                    }
                )

                final_response = await self._engine.chat(
                    messages=messages,
                    stream=False,
                    **gen_params,
                )
                return final_response

        return response

    async def chat_vision(
        self,
        message: str,
        image: Union[str, Path],
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> str:
        """
        Chat with image input.

        Args:
            message: Text prompt about the image
            image: Path to image file
            history: Previous conversation
            **kwargs: Additional generation parameters

        Returns:
            Model response
        """
        if not self.config.enable_vision:
            raise RuntimeError("Vision not enabled in config")

        if not self._vision_processor:
            self._vision_processor = VisionProcessor()

        # Process image
        multimodal_message = self._vision_processor.create_vision_message(
            text=message,
            image=image,
        )

        return await self.chat(
            message=multimodal_message,
            history=history,
            **kwargs,
        )

    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
    ) -> Union[str, AsyncIterator[str]]:
        """
        Raw text generation.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream

        Returns:
            Generated text or async iterator
        """
        if not self._engine:
            raise RuntimeError("Model not initialized")

        gen_params = self.config.get_generation_defaults()
        if max_tokens is not None:
            gen_params["max_tokens"] = max_tokens
        if temperature is not None:
            gen_params["temperature"] = temperature

        return await self._engine.generate(
            prompt=prompt,
            stream=stream,
            **gen_params,
        )

    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        return self._memory_manager.get_stats().__dict__

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self._engine:
            await self._engine.cleanup()
            self._engine = None
        self._initialized = False
        logger.info("Model resources cleaned up")

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
