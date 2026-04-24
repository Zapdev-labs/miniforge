"""Main MiniMax model interface."""

import asyncio
import logging
from collections.abc import AsyncIterator
from pathlib import Path

import psutil

from miniforge.core.engine import InferenceEngine
from miniforge.core.memory import MemoryManager
from miniforge.generation.tools import Tool, ToolExecutor
from miniforge.models.gguf_convert import auto_convert_safetensors_to_gguf
from miniforge.models.registry import get_registry
from miniforge.multimodal.vision import VisionProcessor
from miniforge.utils.config import M7Config

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
        model_path: str | Path | None = None,
        config: M7Config | None = None,
        backend: str | None = None,
    ):
        """
        Initialize MiniMax model.

        Args:
            model_path: Path to model file or HuggingFace ID
            config: Configuration object
            backend: Override backend (llama_cpp or transformers)
        """
        self.config = config or M7Config.auto()
        self.model_path = model_path or self.DEFAULT_MODEL
        self.backend_name = backend or self.config.backend

        self._engine: InferenceEngine | None = None
        total_ram = psutil.virtual_memory().total / (1024**3)
        target_util = self.config.max_memory_gb / total_ram if total_ram > 0 else 0.9
        self._memory_manager = MemoryManager(target_utilization=target_util)
        self._registry = get_registry(self.config.cache_dir)
        self._tool_executor: ToolExecutor | None = None
        self._vision_processor: VisionProcessor | None = None

        self._initialized = False

    @classmethod
    async def from_pretrained(
        cls,
        model_id: str = "MiniMaxAI/MiniMax-M2.7",
        quantization: str | None = None,
        config: M7Config | None = None,
        backend: str | None = None,
        cache_dir: str | None = None,
        download_dir: str | None = None,
    ) -> "Miniforge":
        """
        Load model from HuggingFace or cache.

        Args:
            model_id: HuggingFace model ID
            quantization: Quantization type (auto-selected if None)
            config: Configuration object
            backend: Backend to use
            cache_dir: Directory to cache downloaded models
            download_dir: Directory where GGUF shards are stored/downloaded (e.g. "D:/AI").
                          Takes precedence over cache_dir for GGUF storage.

        Returns:
            Initialized Miniforge instance
        """
        # Create or update config with cache_dir / download_dir
        if config is None:
            registry = get_registry(Path(cache_dir) if cache_dir else None)
            model_info = registry.get_model_info(model_id)
            if model_info is not None:
                config = M7Config.auto(
                    model_params_b=model_info.params_billions,
                    is_moe=model_info.is_moe,
                )
                config.model_id = model_id
                config.model_params_b = model_info.params_billions
                config.quantization = model_info.default_quantization
                config.max_model_ctx = model_info.max_context
                config.is_moe = model_info.is_moe
            else:
                config = M7Config.auto()
        if cache_dir is not None:
            config.cache_dir = cache_dir
        # download_dir overrides where GGUF files land (e.g. "D:/AI" for large models)
        if download_dir is not None:
            config.download_dir = download_dir
        elif config.download_dir:
            download_dir = config.download_dir

        selected_backend = backend or config.backend
        instance = cls(model_id, config, selected_backend)
        await instance._load_model(quantization, download_dir=download_dir)
        return instance

    @classmethod
    def from_gguf(
        cls,
        gguf_path: str | Path,
        config: M7Config | None = None,
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

    async def _load_model(
        self, quantization: str | None = None, download_dir: str | None = None
    ) -> None:
        """Load and prepare model."""
        model_id = str(self.model_path)

        # Resolve download directory: explicit arg > config field > registry gguf_dir default
        gguf_download_dir = (
            Path(download_dir).expanduser() if download_dir else self._registry.gguf_dir
        )
        gguf_download_dir.mkdir(parents=True, exist_ok=True)

        # Get model info
        model_info = self._registry.get_model_info(model_id)

        # Model architecture mapping: model_id -> n_layers
        model_architecture = {
            # MiniMax M2 series: 62 layers
            "MiniMax-M2": 62,
            "MiniMax-M2.1": 62,
            "MiniMax-M2.5": 62,
            "MiniMax-M2.7": 62,
            # MiniMax 01 series (Text-01, VL-01, M1): 80 layers
            "MiniMax-Text-01": 80,
            "MiniMax-VL-01": 80,
            "MiniMax-M1": 80,
            # Llama 4 series: typically 48-80 layers depending on variant
            "Llama-4-Scout": 48,
            "Llama-4-Maverick": 80,
            "Llama-3": 80,  # 3.1, 3.2, 3.3 all use similar architecture
            # Mistral: varies by model
            "Mistral-Small-3.1": 40,
            "Mistral-Large-2": 80,
            "Mistral-Nemo": 40,
            # Qwen: varies by size
            "Qwen3-235B": 80,
            "Qwen3-32B": 64,
            "Qwen3-14B": 48,
            "Qwen3-8B": 36,
            "Qwen3-4B": 32,
            "Qwen3-1.7B": 24,
            "Qwen3-0.6B": 24,
            "QwQ-32B": 64,
            "Qwen2.5-VL": 64,
            "Qwen2.5-Coder": 64,
            # Kimi K2 series (K2.5 / K2.6)
            "Kimi-K2.5": 64,
            "Kimi-K2.6": 64,
        }

        def get_n_layers(model_id: str) -> int:
            """Get number of layers for a model, with fallback to 62."""
            model_id_lower = model_id.lower()
            for key, layers in model_architecture.items():
                if key.lower() in model_id_lower:
                    return layers
            return 62  # Default fallback

        if model_info:
            params = model_info.params_billions
            # Propagate per-model metadata to config
            self.config.model_params_b = params
            if model_info.max_context:
                self.config.max_model_ctx = model_info.max_context
            self.config.is_moe = model_info.is_moe
            if model_info.is_moe:
                # AirLLM-inspired: dynamically compute safe context from available RAM
                # instead of a hardcoded 8192 that may waste memory or still be too large.
                kv_type = self.config.cache_type_k
                # If turbo3/turbo4 will fall back to q8_0, plan for that
                effective_kv_type = "q4_0" if kv_type in ("turbo3", "turbo4") else kv_type

                # Estimate GGUF disk size from param count + quant ratio
                quant_bpw = {
                    "UD-TQ1_0": 1.0,
                    "UD-IQ1_S": 1.1,
                    "UD-IQ1_M": 1.2,
                    "UD-IQ2_XXS": 2.06,
                    "UD-IQ2_M": 2.3,
                    "Q2_K": 2.5,
                    "UD-IQ3_XXS": 3.0,
                    "Q3_K_M": 3.4,
                    "Q4_K_M": 4.5,
                    "Q5_K_M": 5.5,
                    "Q6_K": 6.5,
                    "Q8_0": 8.0,
                }
                bpw = quant_bpw.get(quantization or self.config.quantization, 2.5)
                disk_gb = (params * 1e9 * bpw) / (8 * 1024**3)

                # Get appropriate n_layers for this model architecture
                n_layers = get_n_layers(model_id)

                safe_ctx = self._memory_manager.compute_moe_context(
                    model_disk_gb=disk_gb,
                    n_layers=n_layers,
                    n_kv_heads=8,
                    head_dim=128,
                    kv_cache_type=effective_kv_type,
                    is_moe=True,
                )

                if self.config.n_ctx > safe_ctx:
                    logger.warning(
                        "MoE model %s: reducing n_ctx %s -> %s to fit KV cache in RAM. "
                        "Override with config=M7Config(n_ctx=N).",
                        model_id,
                        self.config.n_ctx,
                        safe_ctx,
                    )
                    self.config.n_ctx = safe_ctx
            # Warn when the model is far larger than available RAM
            if params > 100:
                min_gb = params * 2 * 0.083  # UD-TQ1_0 ≈ 8.3% of fp16 size
                logger.warning(
                    "%.0fB-parameter MoE model. Smallest GGUF ≈ %.0f GB (UD-TQ1_0). "
                    "Requires %.0f GB free NVMe disk. "
                    "llama.cpp will mmap experts from disk — inference speed depends on SSD bandwidth.",
                    params,
                    min_gb,
                    min_gb,
                )
        else:
            # Assume 2.7B for MiniMax
            params = 2.7

        # Determine quantization
        if quantization is None:
            if self.config.quantization:
                quantization = self.config.quantization
            else:
                quantization = self._memory_manager.select_quantization(params)

        configured_search_dirs: list[Path] = []
        if self.config.download_dir:
            configured_search_dirs.append(Path(self.config.download_dir).expanduser())
        if download_dir:
            configured_search_dirs.append(Path(download_dir).expanduser())
        if self.config.model_dirs:
            configured_search_dirs.extend(
                Path(path).expanduser() for path in self.config.model_dirs
            )

        for hosted in self._registry.list_hosted_models():
            if hosted.id == model_id:
                self.backend_name = hosted.backend
                break

        configured_weights = self.config.resolved_model_weights_dir()
        local_path = Path(model_id).expanduser()
        if configured_weights is not None:
            resolved_weights = self._registry.resolve_local_model(
                str(configured_weights),
                quantization,
                backend=self.backend_name,
                search_dirs=configured_search_dirs,
            )
            if resolved_weights is None:
                raise FileNotFoundError(
                    f"Configured model_weights_path was not found: {configured_weights}"
                )
            logger.info("Using configured local model weights: %s", resolved_weights)
            model_path = resolved_weights
        # Check if model_id is a local path first
        elif local_path.exists():
            logger.info(f"Using local model path: {local_path}")
            resolved_local = self._registry.resolve_local_model(
                model_id,
                quantization,
                backend=self.backend_name,
                search_dirs=configured_search_dirs,
            )
            model_path = resolved_local or local_path
        elif hosted_path := self._registry.resolve_local_model(
            model_id,
            quantization,
            backend=self.backend_name,
            search_dirs=configured_search_dirs,
        ):
            logger.info("Using hosted/local model: %s", hosted_path)
            model_path = hosted_path
        # Check cache for GGUF
        elif cached_path := self._registry.get_cached_gguf_path(model_id, quantization):
            logger.info(f"Using cached GGUF: {cached_path}")
            model_path = cached_path
        elif self.config.offline:
            searched = [str(path) for path in configured_search_dirs]
            searched.append(str(self._registry.gguf_dir))
            raise FileNotFoundError(
                "Offline mode is enabled and no local model was found for "
                f"{model_id!r} ({quantization}). Searched: {', '.join(searched)}. "
                "Register a local model with `miniforge register <id> <path>` or set "
                "MINIFORGE_MODEL_DIRS."
            )
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
                found_in_repo = repo_variants[0]  # track which repo had the files
                for repo in repo_variants:
                    logger.info(f"Trying repo: {repo}")
                    try:
                        from huggingface_hub import list_repo_files

                        # Map quantization to bit-depth folder (unsloth subdirectory layout)
                        bit_depth_map = {
                            "UD-TQ1_0": "1-bit",
                            "UD-IQ1_S": "1-bit",
                            "UD-IQ1_M": "1-bit",
                            "UD-IQ2_XXS": "2-bit",
                            "UD-IQ2_M": "2-bit",
                            "UD-Q2_K_XL": "2-bit",
                            "Q2_K": "2-bit",
                            "UD-IQ3_XXS": "3-bit",
                            "UD-IQ3_S": "3-bit",
                            "UD-IQ3_K_S": "3-bit",
                            "UD-Q3_K_M": "3-bit",
                            "UD-Q3_K_XL": "3-bit",
                            "Q3_K": "3-bit",
                            "Q3_K_S": "3-bit",
                            "Q3_K_M": "3-bit",
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

                        # Phase 1: try unsloth subdirectory layout ({bit-depth}/{quant}/*.gguf)
                        for q in [quantization, "UD-Q4_K_M", "Q4_K_M"]:
                            bit_depth = bit_depth_map.get(q, "")
                            if not bit_depth:
                                continue
                            try:
                                files = list(list_repo_files(repo))
                                matching = sorted(
                                    f for f in files if f.endswith(".gguf") and q in f
                                )
                                if matching:
                                    gguf_files = matching
                                    found_in_repo = repo
                                    logger.info(
                                        "Found %d GGUF files in %s (quant=%s, subdirectory layout): %s...",
                                        len(gguf_files),
                                        repo,
                                        q,
                                        gguf_files[:3],
                                    )
                                    break
                            except Exception as e:
                                logger.debug("Subdirectory scan failed for %s/%s: %s", repo, q, e)
                            if gguf_files:
                                break

                        # Phase 2: flat-structure fallback (e.g. bakosh/*, bartowski/*)
                        # These repos place .gguf files directly in the root with no subdirectories.
                        if not gguf_files:
                            try:
                                all_files = list(list_repo_files(repo))
                                # Try requested quant first, then common fallbacks
                                for q_try in [
                                    quantization,
                                    "UD-TQ1_0",
                                    "Q2_K",
                                    "Q3_K_M",
                                    "Q4_K_M",
                                ]:
                                    if not q_try:
                                        continue
                                    matching = sorted(
                                        f for f in all_files if f.endswith(".gguf") and q_try in f
                                    )
                                    if matching:
                                        gguf_files = matching
                                        found_in_repo = repo
                                        logger.info(
                                            "Found %d GGUF files in %s via flat scan (quant=%s): %s...",
                                            len(gguf_files),
                                            repo,
                                            q_try,
                                            gguf_files[:3],
                                        )
                                        break
                            except Exception as e:
                                logger.debug("Flat repo scan failed for %s: %s", repo, e)

                        if gguf_files:
                            break
                    except Exception as e:
                        logger.warning(f"Repo {repo} failed: {e}")
                        continue

                if gguf_files:
                    first_file = gguf_files[0]
                    logger.info(
                        "Downloading %s from %s to %s...",
                        first_file,
                        found_in_repo,
                        gguf_download_dir,
                    )

                    def _hub_dl(repo_id=found_in_repo, fname=first_file) -> str:
                        return hf_hub_download(
                            repo_id=repo_id,
                            filename=fname,
                            local_dir=str(gguf_download_dir),
                        )

                    gguf_path = await asyncio.get_event_loop().run_in_executor(
                        None,
                        _hub_dl,
                    )

                    # Multi-shard models (e.g. Kimi K2.5 Q2_K = 41 shards): download all.
                    # llama.cpp loads split GGUFs by pointing to the first shard.
                    if len(gguf_files) > 1:
                        logger.info(
                            "Multi-shard model: downloading remaining %d shards to %s...",
                            len(gguf_files) - 1,
                            gguf_download_dir,
                        )
                        for fname in gguf_files[1:]:

                            def _hub_dl_shard(repo_id=found_in_repo, filename=fname) -> str:
                                return hf_hub_download(
                                    repo_id=repo_id,
                                    filename=filename,
                                    local_dir=str(gguf_download_dir),
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

        if (
            self.backend_name == "llama_cpp"
            and isinstance(model_path, Path)
            and model_path.is_dir()
        ):
            gguf = self._registry.find_gguf_in_repo(model_path)
            if gguf is None:
                raise ValueError(
                    f"llama_cpp backend requires a GGUF file, but no .gguf was found in {model_path}"
                )
            model_path = gguf

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
        history: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        tools: list[Tool] | None = None,
        stream: bool = False,
    ) -> str | AsyncIterator[str]:
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

    async def chat_messages(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stream: bool = False,
    ) -> str | AsyncIterator[str]:
        """Run chat completion from an OpenAI-style message list."""
        if not self._engine:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        gen_params = self.config.get_generation_defaults()
        if max_tokens is not None:
            gen_params["max_tokens"] = max_tokens
        if temperature is not None:
            gen_params["temperature"] = temperature
        if top_p is not None:
            gen_params["top_p"] = top_p
        if top_k is not None:
            gen_params["top_k"] = top_k

        return await self._engine.chat(
            messages=messages,
            stream=stream,
            **gen_params,
        )

    async def chat_vision(
        self,
        message: str,
        image: str | Path,
        history: list[dict[str, str]] | None = None,
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
        max_tokens: int | None = None,
        temperature: float | None = None,
        stream: bool = False,
    ) -> str | AsyncIterator[str]:
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

    def get_memory_stats(self) -> dict[str, float]:
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
