"""Model registry for downloading and caching models."""

from typing import Optional, List
from pathlib import Path
from dataclasses import dataclass
import hashlib
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a model."""

    id: str
    params_billions: float
    default_quantization: str
    description: str
    tags: List[str]
    vision: bool = False
    tools: bool = True


# Known MiniMax models
MINIMAX_MODELS = {
    "MiniMaxAI/MiniMax-M2.7": ModelInfo(
        id="MiniMaxAI/MiniMax-M2.7",
        params_billions=2.7,
        default_quantization="Q4_K_M",
        description="MiniMax M2.7 dense multimodal model",
        tags=["multimodal", "vision", "tools"],
        vision=True,
        tools=True,
    ),
}


class ModelRegistry:
    """
    Manage model downloads and conversions.

    Handles:
    - HuggingFace model downloads
    - GGUF conversion caching
    - Model metadata
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        if cache_dir is None:
            self.cache_dir = Path.home() / ".cache" / "miniforge"
        else:
            self.cache_dir = Path(cache_dir)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.gguf_dir = self.cache_dir / "gguf"
        self.gguf_dir.mkdir(exist_ok=True)
        self.hf_dir = self.cache_dir / "huggingface"
        self.hf_dir.mkdir(exist_ok=True)

    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get information about a known model."""
        return MINIMAX_MODELS.get(model_id)

    def get_cached_gguf_path(
        self,
        model_id: str,
        quantization: str,
    ) -> Optional[Path]:
        """
        Get path to cached GGUF model if it exists.

        Args:
            model_id: HuggingFace model ID
            quantization: Quantization type (Q4_K_M, etc.)

        Returns:
            Path if cached, None otherwise
        """
        # Create deterministic filename
        safe_name = model_id.replace("/", "--")
        filename = f"{safe_name}_{quantization}.gguf"
        cache_path = self.gguf_dir / filename

        if cache_path.exists():
            return cache_path

        return None

    def register_gguf(
        self,
        model_id: str,
        quantization: str,
        gguf_path: Path,
    ) -> Path:
        """
        Register an external GGUF file in the cache.

        Args:
            model_id: Original model ID
            quantization: Quantization type
            gguf_path: Path to GGUF file

        Returns:
            Path in cache
        """
        safe_name = model_id.replace("/", "--")
        filename = f"{safe_name}_{quantization}.gguf"
        cache_path = self.gguf_dir / filename

        # Copy or symlink
        if gguf_path != cache_path:
            import shutil

            shutil.copy2(gguf_path, cache_path)
            logger.info(f"Cached GGUF: {cache_path}")

        return cache_path

    def download_hf_model(
        self,
        model_id: str,
        local_files_only: bool = False,
    ) -> Path:
        """
        Download model from HuggingFace.

        Args:
            model_id: HuggingFace model ID
            local_files_only: Only use local cache

        Returns:
            Path to downloaded model
        """
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ImportError(
                "huggingface-hub not installed. Install with: uv pip install huggingface-hub"
            )

        cache_path = self.hf_dir / model_id.replace("/", "--")

        if local_files_only and not cache_path.exists():
            raise FileNotFoundError(f"Model not found locally: {model_id}")

        if not local_files_only:
            logger.info(f"Downloading {model_id}...")
            snapshot_download(
                repo_id=model_id,
                local_dir=cache_path,
                local_dir_use_symlinks=False,
            )

        return cache_path

    def find_gguf_in_repo(self, repo_path: Path) -> Optional[Path]:
        """
        Find GGUF files in a repository.

        Args:
            repo_path: Path to model repository

        Returns:
            Path to GGUF file if found, None otherwise
        """
        # Search for GGUF files
        gguf_files = list(repo_path.rglob("*.gguf"))

        if not gguf_files:
            return None

        # Prefer specific quantization
        preferred_order = ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "Q3_K_M", "Q4_0"]

        for quant in preferred_order:
            for f in gguf_files:
                if quant in f.name:
                    return f

        # Return first GGUF
        return gguf_files[0]

    def list_cached_models(self) -> List[str]:
        """List all cached models."""
        models = []

        # List GGUF models
        for gguf_file in self.gguf_dir.glob("*.gguf"):
            name = gguf_file.stem.replace("--", "/")
            models.append(f"[GGUF] {name}")

        # List HF models
        for hf_dir in self.hf_dir.iterdir():
            if hf_dir.is_dir():
                name = hf_dir.name.replace("--", "/")
                models.append(f"[HF] {name}")

        return models

    def clear_cache(self, confirm: bool = False) -> None:
        """Clear model cache."""
        if not confirm:
            print("Set confirm=True to clear cache")
            return

        import shutil

        if self.gguf_dir.exists():
            shutil.rmtree(self.gguf_dir)
            self.gguf_dir.mkdir(exist_ok=True)

        if self.hf_dir.exists():
            shutil.rmtree(self.hf_dir)
            self.hf_dir.mkdir(exist_ok=True)

        logger.info("Model cache cleared")


# Global registry instance
_global_registry: Optional[ModelRegistry] = None


def get_registry(cache_dir: Optional[Path] = None) -> ModelRegistry:
    """Get global model registry instance."""
    global _global_registry

    if _global_registry is None:
        _global_registry = ModelRegistry(cache_dir)

    return _global_registry
