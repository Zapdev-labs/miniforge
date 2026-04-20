"""Mesh coordinator: manages distributed state and job scheduling."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
import uuid

from miniforge.mesh.discovery import MeshDiscovery, PeerInfo
from miniforge.mesh.transport import MeshConnection, MeshTransport

logger = logging.getLogger(__name__)


@dataclass
class NodeState:
    """State of a mesh node."""
    node_id: str
    node_name: str
    ip: str
    port: int
    ram_gb: float
    cpu_cores: int
    ram_available: float = 0.0
    cpu_percent: float = 0.0
    is_leader: bool = False
    last_heartbeat: float = field(default_factory=time.time)
    status: str = "active"  # active, busy, offline


@dataclass
class InferenceJob:
    """A distributed inference job."""
    job_id: str
    prompt: str
    model_id: str
    max_tokens: int
    assigned_node: Optional[str] = None
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


class MeshCoordinator:
    """
    Coordinates the mesh: leader election, state sync, job scheduling.
    
    Features:
    - Automatic leader election (oldest node wins)
    - State synchronization across mesh
    - Job scheduling: route or split based on model size
    - Failure detection and reconnection
    """
    
    LEADER_TIMEOUT = 30.0  # seconds before leader considered dead
    STATE_SYNC_INTERVAL = 5.0
    
    def __init__(
        self,
        node_id: str,
        node_name: str,
        discovery: MeshDiscovery,
        transport: MeshTransport,
        local_ram_gb: float = 28.0,
        local_cpu_cores: int = 8,
    ):
        self.node_id = node_id
        self.node_name = node_name
        self.discovery = discovery
        self.transport = transport
        self.local_ram_gb = local_ram_gb
        self.local_cpu_cores = local_cpu_cores
        
        # Mesh state
        self._nodes: Dict[str, NodeState] = {}
        self._jobs: Dict[str, InferenceJob] = {}
        self._is_leader = False
        self._leader_id: Optional[str] = None
        self._started_at = time.time()
        
        # Callbacks
        self._state_callbacks: List[Callable[[], None]] = []
        
        # Tasks
        self._running = False
        self._election_task: Optional[asyncio.Task] = None
        self._sync_task: Optional[asyncio.Task] = None
        self._discovery_task: Optional[asyncio.Task] = None
        
    @property
    def is_leader(self) -> bool:
        """Check if this node is the leader."""
        return self._is_leader
        
    @property
    def leader_id(self) -> Optional[str]:
        """Get current leader ID."""
        return self._leader_id
        
    @property
    def all_nodes(self) -> List[NodeState]:
        """Get all known nodes including self."""
        nodes = list(self._nodes.values())
        # Add self
        nodes.append(NodeState(
            node_id=self.node_id,
            node_name=self.node_name,
            ip="127.0.0.1",
            port=0,
            ram_gb=self.local_ram_gb,
            cpu_cores=self.local_cpu_cores,
            is_leader=self._is_leader,
        ))
        return nodes
        
    @property
    def total_resources(self) -> Dict[str, Any]:
        """Get aggregated resource info."""
        nodes = self.all_nodes
        return {
            "nodes": len(nodes),
            "total_ram_gb": sum(n.ram_gb for n in nodes),
            "total_cpu_cores": sum(n.cpu_cores for n in nodes),
            "available_ram_gb": sum(n.ram_available for n in nodes),
        }
        
    def on_state_change(self, callback: Callable[[], None]) -> None:
        """Register callback for state changes."""
        self._state_callbacks.append(callback)
        
    def _notify_state_change(self) -> None:
        """Notify all state change callbacks."""
        for cb in self._state_callbacks:
            try:
                cb()
            except Exception as e:
                logger.warning(f"State callback error: {e}")
                
    async def start(self) -> None:
        """Start coordinator."""
        if self._running:
            return
            
        self._running = True
        logger.info("Starting mesh coordinator")
        
        # Start discovery
        await self.discovery.start()
        self.discovery.on_peer_discovered(self._on_peer_event)
        
        # Start transport
        await self.transport.start()
        self.transport.on_connection_event(self._on_connection_event)
        
        # Start background tasks
        self._election_task = asyncio.create_task(self._leader_election_loop())
        self._sync_task = asyncio.create_task(self._state_sync_loop())
        self._discovery_task = asyncio.create_task(self._discovery_connect_loop())
        
        logger.info(f"Coordinator started, node_id={self.node_id}")
        
    async def stop(self) -> None:
        """Stop coordinator."""
        self._running = False
        
        # Cancel tasks
        for task in [self._election_task, self._sync_task, self._discovery_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                    
        # Stop transport and discovery
        await self.transport.stop()
        await self.discovery.stop()
        
        logger.info("Coordinator stopped")
        
    async def schedule_inference(
        self,
        prompt: str,
        model_id: str = "minimax",
        max_tokens: int = 512,
        strategy: str = "auto",
    ) -> InferenceJob:
        """
        Schedule an inference job.
        
        Strategies:
        - "auto": Choose based on model size and available resources
        - "local": Force local execution
        - "remote": Force remote execution
        - "split": Split model across nodes (for large models)
        """
        job_id = str(uuid.uuid4())[:8]
        job = InferenceJob(
            job_id=job_id,
            prompt=prompt,
            model_id=model_id,
            max_tokens=max_tokens,
        )
        self._jobs[job_id] = job
        
        # Determine strategy
        if strategy == "auto":
            # Simple heuristic: if we have multiple nodes and model is large
            nodes = self.all_nodes
            if len(nodes) > 1 and self._estimate_model_size(model_id) > 20:  # >20GB
                strategy = "split"
            else:
                strategy = "route"
                
        if strategy == "route":
            # Route to least-loaded node
            target = self._select_least_loaded_node()
            job.assigned_node = target
            job.status = "running"
            job.started_at = time.time()
            
            logger.info(f"Job {job_id} routed to {target}")
            
        elif strategy == "split":
            # Split across all nodes
            job.assigned_node = "split"
            job.status = "running"
            job.started_at = time.time()
            
            logger.info(f"Job {job_id} split across mesh")
            
        self._notify_state_change()
        return job
        
    def _estimate_model_size(self, model_id: str) -> float:
        """Estimate model size in GB based on model ID."""
        # Simple heuristic based on common model sizes
        if "70b" in model_id.lower():
            return 40.0  # Q4 quantized
        elif "30b" in model_id.lower() or "34b" in model_id.lower():
            return 20.0
        elif "13b" in model_id.lower():
            return 8.0
        elif "7b" in model_id.lower():
            return 4.0
        elif "minimax" in model_id.lower():
            return 15.0  # MoE model
        return 10.0  # Default
        
    def _select_least_loaded_node(self) -> str:
        """Select least-loaded node for job routing."""
        candidates = self.all_nodes
        
        # Filter to active nodes
        active = [n for n in candidates if n.status == "active"]
        if not active:
            return self.node_id  # Fallback to self
            
        # Sort by CPU usage then available RAM
        sorted_nodes = sorted(
            active,
            key=lambda n: (n.cpu_percent, -n.ram_available)
        )
        return sorted_nodes[0].node_id
        
    async def _on_peer_event(self, peer: PeerInfo, event: str) -> None:
        """Handle peer discovery events."""
        if event == "added":
            logger.info(f"New peer discovered: {peer.node_name} @ {peer.endpoint}")
            # Try to connect
            asyncio.create_task(self._connect_to_peer(peer))
        elif event == "removed":
            logger.info(f"Peer removed: {peer.node_name}")
            # Remove from state
            if peer.node_id in self._nodes:
                del self._nodes[peer.node_id]
                self._notify_state_change()
                
    async def _connect_to_peer(self, peer: PeerInfo) -> None:
        """Connect to a discovered peer."""
        conn = await self.transport.connect_to_peer(peer.ip, peer.port)
        if conn:
            # Register node state
            self._nodes[peer.node_id] = NodeState(
                node_id=peer.node_id,
                node_name=peer.node_name,
                ip=peer.ip,
                port=peer.port,
                ram_gb=peer.ram_gb,
                cpu_cores=peer.cpu_cores,
            )
            self._notify_state_change()
            
    def _on_connection_event(self, peer_id: str, conn: MeshConnection, event: str) -> None:
        """Handle transport connection events."""
        if event == "connected":
            logger.info(f"Connected to {peer_id}")
            # Register handlers for inference requests
            conn.on("inference_request", self._handle_inference_request)
            conn.on("inference_response", self._handle_inference_response)
            conn.on("heartbeat", self._handle_heartbeat)
        elif event == "disconnected":
            logger.info(f"Disconnected from {peer_id}")
            if peer_id in self._nodes:
                self._nodes[peer_id].status = "offline"
                self._notify_state_change()
                
    async def _handle_inference_request(self, payload: Dict[str, Any]) -> None:
        """Handle incoming inference request from peer."""
        job_id = payload.get("job_id")
        prompt = payload.get("prompt")
        model_id = payload.get("model_id", "minimax")
        max_tokens = payload.get("max_tokens", 512)
        
        logger.info(f"Received inference request {job_id} from peer")
        
        # TODO: Execute inference using local engine
        # For now, just acknowledge
        
    async def _handle_inference_response(self, payload: Dict[str, Any]) -> None:
        """Handle inference response from peer."""
        job_id = payload.get("job_id")
        result = payload.get("result")
        
        if job_id in self._jobs:
            job = self._jobs[job_id]
            job.result = result
            job.status = "completed"
            job.completed_at = time.time()
            self._notify_state_change()
            
    async def _handle_heartbeat(self, payload: Dict[str, Any]) -> None:
        """Handle heartbeat from peer."""
        node_id = payload.get("node_id")
        if node_id in self._nodes:
            node = self._nodes[node_id]
            node.last_heartbeat = time.time()
            node.ram_available = payload.get("ram_available", 0.0)
            node.cpu_percent = payload.get("cpu_percent", 0.0)
            
    async def _leader_election_loop(self) -> None:
        """Leader election: oldest node wins."""
        while self._running:
            try:
                # Check current leader
                if self._leader_id and self._leader_id != self.node_id:
                    # Verify leader is alive
                    leader = self._nodes.get(self._leader_id)
                    if leader:
                        time_since_heartbeat = time.time() - leader.last_heartbeat
                        if time_since_heartbeat > self.LEADER_TIMEOUT:
                            logger.warning(f"Leader {self._leader_id} timed out")
                            self._leader_id = None
                            
                # If no leader, trigger election
                if not self._leader_id:
                    await self._run_election()
                    
            except Exception as e:
                logger.error(f"Leader election error: {e}")
                
            await asyncio.sleep(self.LEADER_TIMEOUT / 2)
            
    async def _run_election(self) -> None:
        """Run leader election."""
        logger.info("Running leader election")
        
        # Collect all candidates (self + known nodes)
        candidates = [
            (self.node_id, self._started_at)
        ]
        for node in self._nodes.values():
            # Use node_id as tiebreaker (lexicographic)
            candidates.append((node.node_id, node.last_heartbeat))
            
        # Oldest node wins
        winner = min(candidates, key=lambda x: (x[1], x[0]))
        winner_id = winner[0]
        
        self._leader_id = winner_id
        self._is_leader = (winner_id == self.node_id)
        
        if self._is_leader:
            logger.info("Elected as leader")
        else:
            logger.info(f"Leader elected: {winner_id}")
            
        self._notify_state_change()
        
    async def _state_sync_loop(self) -> None:
        """Periodic state synchronization."""
        while self._running:
            try:
                # Remove stale nodes
                now = time.time()
                stale = [
                    nid for nid, node in self._nodes.items()
                    if now - node.last_heartbeat > self.LEADER_TIMEOUT
                ]
                for nid in stale:
                    logger.info(f"Removing stale node {nid}")
                    del self._nodes[nid]
                    self._notify_state_change()
                    
            except Exception as e:
                logger.error(f"State sync error: {e}")
                
            await asyncio.sleep(self.STATE_SYNC_INTERVAL)
            
    async def _discovery_connect_loop(self) -> None:
        """Continuously try to connect to newly discovered peers."""
        while self._running:
            try:
                # Get peers from discovery
                peers = self.discovery.peers
                for peer in peers:
                    if peer.node_id not in self._nodes:
                        await self._connect_to_peer(peer)
                        
            except Exception as e:
                logger.debug(f"Discovery connect error: {e}")
                
            await asyncio.sleep(5)
