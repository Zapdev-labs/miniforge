"""Model registry for tracking and streaming model weights."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, AsyncIterator, Callable
import time

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a model available on the mesh."""
    model_id: str
    model_path: str
    file_size: int
    checksum: str
    quantization: str
    location_node: str  # node_id where model is stored
    layer_count: int
    layers_cached: List[int] = field(default_factory=list)  # For workers


class ModelRegistry:
    """
    Tracks model locations across the mesh and streams weights to workers.
    
    Host nodes:
    - Store full models locally
    - Serve weight chunks to workers on demand
    
    Worker nodes:
    - Cache frequently used layers
    - Request missing layers from host
    """
    
    CHUNK_SIZE = 16 * 1024 * 1024  # 16MB chunks for streaming
    
    def __init__(self, node_id: str, is_host: bool = False, cache_dir: Optional[Path] = None):
        self.node_id = node_id
        self.is_host = is_host
        self.cache_dir = cache_dir or (Path.home() / ".cache" / "miniforge" / "mesh_models")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Models this node has locally
        self._local_models: Dict[str, ModelInfo] = {}
        
        # Models available on other nodes
        self._remote_models: Dict[str, List[ModelInfo]] = {}
        
        # For workers: cached layers
        self._cached_layers: Dict[str, Dict[int, Path]] = {}
        
        # Progress callbacks
        self._download_callbacks: List[Callable[[str, int, int], None]] = []
        
    def register_local_model(
        self,
        model_id: str,
        model_path: str,
        quantization: str = "Q4_K_M",
        layer_count: int = 0,
    ) -> ModelInfo:
        """Register a model available locally (host mode)."""
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
            
        # Calculate checksum
        checksum = self._calculate_checksum(path)
        
        info = ModelInfo(
            model_id=model_id,
            model_path=str(path.absolute()),
            file_size=path.stat().st_size,
            checksum=checksum,
            quantization=quantization,
            location_node=self.node_id,
            layer_count=layer_count,
        )
        
        self._local_models[model_id] = info
        logger.info(f"Registered model {model_id}: {info.file_size / 1e9:.1f}GB at {path}")
        return info
        
    def get_local_model(self, model_id: str) -> Optional[ModelInfo]:
        """Get info about a local model."""
        return self._local_models.get(model_id)
        
    def list_local_models(self) -> List[ModelInfo]:
        """List all models available locally."""
        return list(self._local_models.values())
        
    def register_remote_model(self, model_info: ModelInfo) -> None:
        """Register a model available on another node."""
        if model_info.model_id not in self._remote_models:
            self._remote_models[model_info.model_id] = []
        
        # Remove existing entry for this node
        self._remote_models[model_info.model_id] = [
            m for m in self._remote_models[model_info.model_id]
            if m.location_node != model_info.location_node
        ]
        
        # Add new entry
        self._remote_models[model_info.model_id].append(model_info)
        logger.debug(f"Registered remote model {model_info.model_id} on {model_info.location_node}")
        
    def find_model_location(self, model_id: str) -> Optional[str]:
        """Find which node has a model. Returns node_id or None."""
        # Check local first
        if model_id in self._local_models:
            return self.node_id
            
        # Check remote
        if model_id in self._remote_models:
            # Return first available location
            if self._remote_models[model_id]:
                return self._remote_models[model_id][0].location_node
        return None
        
    def is_model_available(self, model_id: str) -> bool:
        """Check if a model is available locally or on the mesh."""
        return self.find_model_location(model_id) is not None
        
    async def stream_weights(
        self,
        model_id: str,
        layer_indices: List[int],
        chunk_callback: Callable[[bytes], None],
    ) -> None:
        """
        Stream weight chunks for specified layers.
        
        Args:
            model_id: Model to stream
            layer_indices: Specific layers to stream (empty = all)
            chunk_callback: Called with each chunk of bytes
        """
        model = self._local_models.get(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found locally")
            
        path = Path(model.model_path)
        
        # In a real implementation, we'd use GGUF metadata to extract
        # specific layers. For now, stream the whole file in chunks.
        
        logger.info(f"Streaming {model_id} ({model.file_size / 1e9:.1f}GB)")
        
        bytes_sent = 0
        with open(path, "rb") as f:
            while True:
                chunk = f.read(self.CHUNK_SIZE)
                if not chunk:
                    break
                    
                chunk_callback(chunk)
                bytes_sent += len(chunk)
                
                # Progress
                if bytes_sent % (100 * 1024 * 1024) == 0:  # Every 100MB
                    logger.info(f"Streamed {bytes_sent / 1e9:.1f}GB / {model.file_size / 1e9:.1f}GB")
                    
        logger.info(f"Streaming complete: {bytes_sent / 1e9:.1f}GB sent")
        
    async def download_model(
        self,
        model_id: str,
        source_node: str,
        request_layer_stream: Callable[[str, List[int]], AsyncIterator[bytes]],
        layer_indices: Optional[List[int]] = None,
    ) -> Path:
        """
        Download a model from another node (worker mode).
        
        Args:
            model_id: Model to download
            source_node: Node ID that has the model
            request_layer_stream: Async function to request weight stream
            layer_indices: Specific layers to download (None = all)
            
        Returns:
            Path to downloaded model
        """
        output_path = self.cache_dir / f"{model_id.replace('/', '_')}.gguf"
        
        logger.info(f"Downloading {model_id} from {source_node}...")
        
        bytes_received = 0
        with open(output_path, "wb") as f:
            async for chunk in request_layer_stream(model_id, layer_indices or []):
                f.write(chunk)
                bytes_received += len(chunk)
                
        logger.info(f"Downloaded {model_id}: {bytes_received / 1e9:.1f}GB to {output_path}")
        
        # Register as local model
        self.register_local_model(model_id, str(output_path))
        
        return output_path
        
    def cache_layer(self, model_id: str, layer_idx: int, weights: bytes) -> Path:
        """Cache a specific layer for fast access (worker mode)."""
        layer_dir = self.cache_dir / model_id.replace("/", "_") / "layers"
        layer_dir.mkdir(parents=True, exist_ok=True)
        
        layer_path = layer_dir / f"layer_{layer_idx}.bin"
        with open(layer_path, "wb") as f:
            f.write(weights)
            
        if model_id not in self._cached_layers:
            self._cached_layers[model_id] = {}
        self._cached_layers[model_id][layer_idx] = layer_path
        
        return layer_path
        
    def get_cached_layer(self, model_id: str, layer_idx: int) -> Optional[Path]:
        """Get path to cached layer if available."""
        if model_id in self._cached_layers:
            return self._cached_layers[model_id].get(layer_idx)
        return None
        
    def get_cache_size(self, model_id: str) -> int:
        """Get number of cached layers for a model."""
        if model_id in self._cached_layers:
            return len(self._cached_layers[model_id])
        return 0
        
    def _calculate_checksum(self, path: Path) -> str:
        """Calculate SHA256 checksum of file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()[:16]
        
    def get_mesh_model_info(self) -> Dict[str, Dict]:
        """Get info about all models available on mesh (for API)."""
        info = {}
        
        # Local models
        for model_id, model in self._local_models.items():
            info[model_id] = {
                "local": True,
                "location": self.node_id,
                "size_gb": model.file_size / 1e9,
                "quantization": model.quantization,
            }
            
        # Remote models
        for model_id, models in self._remote_models.items():
            if model_id not in info:
                info[model_id] = {
                    "local": False,
                    "locations": [m.location_node for m in models],
                    "size_gb": models[0].file_size / 1e9,
                    "quantization": models[0].quantization,
                }
                
        return info
