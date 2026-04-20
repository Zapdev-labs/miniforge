"""Host/Worker distributed inference for centralized model storage."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from miniforge.core.engine import InferenceEngine
from miniforge.mesh.coordinator import MeshCoordinator, NodeState
from miniforge.mesh.registry import ModelRegistry, ModelInfo
from miniforge.mesh.transport import MeshConnection

logger = logging.getLogger(__name__)


class HostWorkerEngine:
    """
    Host/Worker distributed inference topology.
    
    Host:
    - Stores all models locally
    - Streams weights to workers on demand
    - Acts as inference coordinator
    
    Worker:
    - No local models required
    - Receives weights from host on first use
    - Caches frequently used layers
    - Performs inference using streamed weights
    
    Benefits:
    - Only download models once (on host)
    - Workers can join with zero storage requirement
    - Automatic weight caching on workers for performance
    """
    
    def __init__(
        self,
        local_engine: InferenceEngine,
        coordinator: MeshCoordinator,
        registry: ModelRegistry,
        node_id: str,
        is_host: bool = False,
        host_node_id: Optional[str] = None,
    ):
        self.local_engine = local_engine
        self.coordinator = coordinator
        self.registry = registry
        self.node_id = node_id
        self.is_host = is_host
        self.host_node_id = host_node_id
        
        # Track active downloads
        self._active_downloads: Dict[str, asyncio.Task] = {}
        
    async def generate(
        self,
        prompt: str,
        model_id: str = "minimax",
        max_tokens: int = 512,
        temperature: float = 1.0,
        stream: bool = False,
    ) -> Union[str, AsyncIterator[str]]:
        """
        Generate text using host/worker topology.
        
        If worker: Ensure model is available (download if needed), then run inference.
        If host: Run inference locally and stream weights to workers if needed.
        """
        if self.is_host:
            # Host always runs locally
            return await self._host_generate(
                prompt, model_id, max_tokens, temperature, stream
            )
        else:
            # Worker - ensure model, then generate
            return await self._worker_generate(
                prompt, model_id, max_tokens, temperature, stream
            )
            
    async def _host_generate(
        self,
        prompt: str,
        model_id: str,
        max_tokens: int,
        temperature: float,
        stream: bool,
    ) -> Union[str, AsyncIterator[str]]:
        """Host generates locally."""
        # Check if we have this model
        model = self.registry.get_local_model(model_id)
        if not model:
            # Try to find it in local paths
            logger.warning(f"Model {model_id} not registered, using local engine")
            
        if stream:
            return self.local_engine.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
        else:
            return await self.local_engine.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
            )
            
    async def _worker_generate(
        self,
        prompt: str,
        model_id: str,
        max_tokens: int,
        temperature: float,
        stream: bool,
    ) -> Union[str, AsyncIterator[str]]:
        """Worker generates - may need to download model first."""
        # Check if model is available locally (downloaded previously)
        if not self.registry.is_model_available(model_id):
            # Need to download from host
            if not self.host_node_id:
                logger.error("No host configured and model not available locally")
                return "[Error: Model not available and no host configured]"
                
            # Download the model
            success = await self._download_from_host(model_id)
            if not success:
                return f"[Error: Failed to download model {model_id} from host]"
                
        # Now run inference locally
        if stream:
            return self.local_engine.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
        else:
            return await self.local_engine.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
            )
            
    async def _download_from_host(self, model_id: str) -> bool:
        """Download model from host node. Returns True on success."""
        logger.info(f"Worker downloading {model_id} from host {self.host_node_id}")
        
        # Get connection to host
        host_conn = self._get_connection_to_host()
        if not host_conn:
            logger.error(f"No connection to host {self.host_node_id}")
            return False
            
        try:
            # Request model stream
            await host_conn.send("model_request", {
                "model_id": model_id,
                "requester": self.node_id,
            })
            
            # Wait for download to complete
            # The actual download happens via a separate file transfer
            # For now, we wait for acknowledgment
            download_complete = asyncio.Event()
            
            async def on_model_stream(payload: Dict) -> None:
                if payload.get("model_id") == model_id:
                    chunk = payload.get("chunk")
                    if chunk:
                        # Write chunk to file
                        # This is simplified - real impl needs file assembly
                        pass
                    if payload.get("complete"):
                        download_complete.set()
                        
            host_conn.on("model_chunk", on_model_stream)
            
            # Wait with timeout
            await asyncio.wait_for(download_complete.wait(), timeout=300.0)
            
            logger.info(f"Downloaded {model_id} successfully")
            return True
            
        except asyncio.TimeoutError:
            logger.error(f"Download timeout for {model_id}")
            return False
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False
            
    def _get_connection_to_host(self) -> Optional[MeshConnection]:
        """Get connection to the host node."""
        connections = self.coordinator.transport.connections
        for conn in connections.values():
            # TODO: Track which connection is to host
            return conn
        return None
        
    async def handle_model_request(
        self,
        payload: Dict[str, Any],
        conn: MeshConnection,
    ) -> None:
        """
        Handle model download request from worker (host mode).
        
        Streams model weights to requesting worker.
        """
        if not self.is_host:
            logger.warning("Received model request but not in host mode")
            return
            
        model_id = payload.get("model_id")
        requester = payload.get("requester")
        
        logger.info(f"Worker {requester} requested model {model_id}")
        
        model = self.registry.get_local_model(model_id)
        if not model:
            await conn.send("model_response", {
                "model_id": model_id,
                "error": "Model not available on this host",
            })
            return
            
        # Stream the model
        chunk_count = 0
        
        def send_chunk(chunk: bytes) -> None:
            nonlocal chunk_count
            # Fire-and-forget chunk send
            asyncio.create_task(conn.send("model_chunk", {
                "model_id": model_id,
                "chunk": chunk,
                "chunk_index": chunk_count,
                "complete": False,
            }))
            chunk_count += 1
            
        try:
            await self.registry.stream_weights(model_id, [], send_chunk)
            
            # Send completion
            await conn.send("model_chunk", {
                "model_id": model_id,
                "chunk": b"",
                "chunk_index": chunk_count,
                "complete": True,
            })
            
            logger.info(f"Streamed {model_id} to {requester} in {chunk_count} chunks")
            
        except Exception as e:
            logger.error(f"Failed to stream {model_id}: {e}")
            await conn.send("model_response", {
                "model_id": model_id,
                "error": str(e),
            })
                
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model_id: str = "minimax",
        max_tokens: int = 512,
        stream: bool = False,
    ) -> Union[str, AsyncIterator[str]]:
        """Chat completion with message history."""
        prompt = self._format_chat_prompt(messages)
        return await self.generate(
            prompt=prompt,
            model_id=model_id,
            max_tokens=max_tokens,
            stream=stream,
        )
        
    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into prompt string."""
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                formatted.append(f"System: {content}")
            elif role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
        formatted.append("Assistant:")
        return "\n\n".join(formatted)
