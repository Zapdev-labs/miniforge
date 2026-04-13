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
        cache_dir: Optional[str] = None,
    ) -> "Miniforge":
        """
        Load model from HuggingFace or cache.

        Args:
            model_id: HuggingFace model ID
            quantization: Quantization type (auto-selected if None)
            config: Configuration object
            backend: Backend to use
            cache_dir: Directory to cache downloaded models

        Returns:
            Initialized Miniforge instance
        """
        # Create or update config with cache_dir
        if config is None:
            config = load_config()
        if cache_dir is not None:
            config.cache_dir = cache_dir

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
                # If user already provided a unsloth GGUF repo, use it directly
                if "unsloth" in model_id or "-GGUF" in model_id:
                    repo_variants = [model_id]
                else:
                    repo_variants = [
                        model_id.replace("MiniMaxAI/", "unsloth/"),
                        model_id + "-GGUF",
                    ]

                gguf_files = []
                for repo in repo_variants:
                    logger.info(f"Trying repo: {repo}")
                    try:
                        # Try common GGUF filenames with different patterns
                        # unsloth uses subdirectories like "2-bit/UD-IQ2_XXS/"
                        base_name = model_id.split("/")[-1].replace("-GGUF", "")

                        # Map quantization to bit-depth folder
                        bit_depth_map = {
                            "UD-IQ1_M": "1-bit",
                            "UD-IQ2_XXS": "2-bit",
                            "UD-IQ2_M": "2-bit",
                            "UD-Q2_K_XL": "2-bit",
                            "UD-IQ3_XXS": "3-bit",
                            "UD-IQ3_S": "3-bit",
                            "UD-IQ3_K_S": "3-bit",
                            "UD-Q3_K_M": "3-bit",
                            "UD-Q3_K_XL": "3-bit",
                            "UD-IQ4_XS": "4-bit",
                            "UD-Q4_K_S": "4-bit",
                            "MXFP4_MOE": "4-bit",
                            "UD-Q4_NL": "4-bit",
                            "UD-Q4_K_M": "4-bit",
                            "UD-Q4_K_XL": "4-bit",
                            "UD-Q5_K_S": "5-bit",
                            "UD-Q5_K_M": "5-bit",
                            "UD-Q5_K_XL": "5-bit",
                            "UD-Q6_K": "6-bit",
                            "UD-Q6_K_XL": "6-bit",
                            "Q8_0": "8-bit",
                            "UD-Q8_K_XL": "8-bit",
                            "BF16": "BF16",
                        }

                        for q in [quantization, "Q4_K_M"]:
                            bit_depth = bit_depth_map.get(q, "")
                            if not bit_depth:
                                continue

                            # Try patterns for unsloth repo structure
                            # Pattern: {bit-depth}/{quant}/MiniMax-M2.7-{quant}-00001-of-0000N.gguf
                            search_patterns = [
                                f"{bit_depth}/{q}/MiniMax-M2.7-{q}-00001-of-*.gguf",
                                f"{bit_depth}/{q}/*.gguf",
                                f"{q}/MiniMax-M2.7-{q}-00001-of-*.gguf",
                                f"MiniMax-M2.7-{q}.gguf",
                            ]

                            for pattern in search_patterns:
                                try:
                                    from huggingface_hub import list_repo_files

                                    files = list(list_repo_files(repo))
                                    matching = [f for f in files if f.endswith(".gguf") and q in f]

                                    if matching:
                                        # Found files with this quantization
                                        # Sort to get the first shard
                                        matching.sort()
                                        gguf_files = matching
                                        logger.info(
                                            f"Found {len(gguf_files)} GGUF files in {repo}: {gguf_files[:3]}..."
                                        )
                                        break
                                except Exception as e:
                                    logger.debug(f"    Pattern {pattern} failed: {e}")
                                    continue

                            if gguf_files:
                                break
                        if gguf_files:
                            break
                    except Exception as e:
                        logger.warning(f"Repo {repo} failed: {e}")
                        continue

                if gguf_files:
                    # Download the first GGUF file (or all if needed)
                    # For now, download just the first one to check
                    first_file = gguf_files[0]
                    logger.info(f"Downloading {first_file}...")

                    def _hub_dl(repo_id=repo, fname=first_file) -> str:
                        return hf_hub_download(
                            repo_id=repo_id,
                            filename=fname,
                            local_dir=str(self._registry.gguf_dir),
                        )

                    gguf_path = await asyncio.get_event_loop().run_in_executor(
                        None,
                        _hub_dl,
                    )

                    # If there are multiple shards, we need all of them
                    if len(gguf_files) > 1:
                        logger.info(f"Model has {len(gguf_files)} shards, downloading all...")
                        for fname in gguf_files[1:]:

                            def _hub_dl_shard(repo_id=repo, filename=fname) -> str:
                                return hf_hub_download(
                                    repo_id=repo_id,
                                    filename=filename,
                                    local_dir=str(self._registry.gguf_dir),
                                )

                            await asyncio.get_event_loop().run_in_executor(None, _hub_dl_shard)

                    # llama.cpp can load split GGUF files with the first file
                    model_path = Path(gguf_path)
                else:
                    # If user explicitly specified a GGUF repo but we couldn't find the file,
                    # list available GGUF files to help the user
                    if "unsloth" in model_id or "-GGUF" in model_id:
                        available_files = []
                        try:
                            from huggingface_hub import list_repo_files

                            for repo in repo_variants:
                                try:
                                    files = list(list_repo_files(repo))
                                    available_files = [f for f in files if f.endswith(".gguf")]
                                    if available_files:
                                        break
                                except Exception:
                                    continue
                        except Exception:
                            pass

                        msg = f"Could not find GGUF file for quantization '{quantization}' in {model_id}."
                        if available_files:
                            msg += f"\nAvailable GGUF files: {', '.join(available_files[:10])}"
                        msg += f"\nCheck all files at: https://huggingface.co/{repo_variants[0]}/tree/main"
                        raise ValueError(msg)

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
                logger.error(f"Could not get GGUF: {e}")
                # If it's a GGUF-specific repo, don't fall back to transformers
                if "unsloth" in model_id or "-GGUF" in model_id:
                    raise RuntimeError(
                        f"Failed to download GGUF from {model_id}. Error: {e}"
                    ) from e
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
