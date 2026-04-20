"""Distributed inference engine: routes or splits inference across mesh."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from miniforge.core.engine import InferenceEngine
from miniforge.mesh.coordinator import MeshCoordinator, InferenceJob
from miniforge.mesh.transport import MeshConnection

logger = logging.getLogger(__name__)


class DistributedInferenceEngine:
    """
    Transparently distributes inference across mesh nodes.
    
    Features:
    - Automatic routing to least-loaded node
    - Layer splitting for large models
    - Streaming support
    - Fallback to local execution
    """
    
    def __init__(
        self,
        local_engine: InferenceEngine,
        coordinator: MeshCoordinator,
        node_id: str,
    ):
        self.local_engine = local_engine
        self.coordinator = coordinator
        self.node_id = node_id
        
        # Track assigned jobs
        self._active_jobs: Dict[str, InferenceJob] = {}
        
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 40,
        stream: bool = False,
        strategy: str = "auto",
    ) -> Union[str, AsyncIterator[str]]:
        """
        Generate text, distributed across mesh if beneficial.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            stream: Whether to stream output
            strategy: "auto", "local", "remote", or "split"
            
        Returns:
            Generated text or async iterator for streaming
        """
        # Get mesh state
        resources = self.coordinator.total_resources
        nodes = self.coordinator.all_nodes
        
        # Decide execution strategy
        if strategy == "auto":
            if len(nodes) == 1:
                strategy = "local"
            else:
                # Use mesh for multi-node
                strategy = "remote"
                
        if strategy == "local":
            # Local execution
            if stream:
                return self.local_engine.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    stream=True,
                )
            else:
                return await self.local_engine.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    stream=False,
                )
                
        elif strategy == "remote":
            # Route to mesh
            job = await self.coordinator.schedule_inference(
                prompt=prompt,
                max_tokens=max_tokens,
                strategy="route",
            )
            self._active_jobs[job.job_id] = job
            
            if stream:
                return self._stream_remote(job)
            else:
                return await self._execute_remote(job)
                
        elif strategy == "split":
            # Split model across nodes
            job = await self.coordinator.schedule_inference(
                prompt=prompt,
                max_tokens=max_tokens,
                strategy="split",
            )
            self._active_jobs[job.job_id] = job
            
            if stream:
                return self._stream_split(job)
            else:
                return await self._execute_split(job)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
            
    async def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 1.0,
        stream: bool = False,
        strategy: str = "auto",
    ) -> Union[str, AsyncIterator[str]]:
        """Chat completion with message history."""
        # Format messages into prompt
        prompt = self._format_chat_prompt(messages)
        return await self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
            strategy=strategy,
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
        
    async def _execute_remote(self, job: InferenceJob) -> str:
        """Execute job remotely and wait for result."""
        target_node = job.assigned_node
        
        if target_node == self.node_id:
            # Local execution
            return await self.local_engine.generate(
                prompt=job.prompt,
                max_tokens=job.max_tokens,
            )
            
        # Send to remote node
        conn = self._get_connection_to_node(target_node)
        if not conn:
            logger.error(f"No connection to {target_node}, falling back to local")
            return await self.local_engine.generate(
                prompt=job.prompt,
                max_tokens=job.max_tokens,
            )
            
        await conn.send("inference_request", {
            "job_id": job.job_id,
            "prompt": job.prompt,
            "model_id": job.model_id,
            "max_tokens": job.max_tokens,
        })
        
        # Wait for response (with timeout)
        try:
            await asyncio.wait_for(
                self._wait_for_job_completion(job),
                timeout=120.0,
            )
        except asyncio.TimeoutError:
            logger.error(f"Job {job.job_id} timed out")
            job.status = "failed"
            return "[Error: Inference timeout]"
            
        return job.result or "[Error: No result]"
        
    async def _stream_remote(self, job: InferenceJob) -> AsyncIterator[str]:
        """Stream results from remote execution."""
        # TODO: Implement streaming protocol
        # For now, just yield final result
        result = await self._execute_remote(job)
        yield result
        
    async def _execute_split(self, job: InferenceJob) -> str:
        """Execute job split across multiple nodes."""
        nodes = self.coordinator.all_nodes
        
        if len(nodes) == 1:
            # Fallback to local
            return await self.local_engine.generate(
                prompt=job.prompt,
                max_tokens=job.max_tokens,
            )
            
        # Split layers across nodes
        # For now, simple approach: prefill on one node, decode on another
        logger.info(f"Splitting job {job.job_id} across {len(nodes)} nodes")
        
        # TODO: Implement proper layer splitting
        # This requires model-specific knowledge of layer structure
        
        # For now, route to least-loaded
        return await self._execute_remote(job)
        
    async def _stream_split(self, job: InferenceJob) -> AsyncIterator[str]:
        """Stream results from split execution."""
        # TODO: Implement proper streaming for split execution
        result = await self._execute_split(job)
        yield result
        
    async def _wait_for_job_completion(self, job: InferenceJob) -> None:
        """Wait for job to complete."""
        while job.status not in ("completed", "failed"):
            await asyncio.sleep(0.1)
            
    def _get_connection_to_node(self, node_id: str) -> Optional[MeshConnection]:
        """Get connection to a specific node."""
        connections = self.coordinator.transport.connections
        for conn in connections.values():
            # TODO: Track node_id -> connection mapping
            return conn
        return None
        
    async def handle_remote_request(
        self,
        job_id: str,
        prompt: str,
        model_id: str,
        max_tokens: int,
        conn: MeshConnection,
    ) -> None:
        """Handle incoming inference request from remote node."""
        logger.info(f"Handling remote request {job_id}")
        
        try:
            # Execute locally
            result = await self.local_engine.generate(
                prompt=prompt,
                max_tokens=max_tokens,
            )
            
            # Send response
            await conn.send("inference_response", {
                "job_id": job_id,
                "result": result,
                "status": "completed",
            })
            
        except Exception as e:
            logger.error(f"Remote inference error: {e}")
            await conn.send("inference_response", {
                "job_id": job_id,
                "result": f"[Error: {e}]",
                "status": "failed",
            })
