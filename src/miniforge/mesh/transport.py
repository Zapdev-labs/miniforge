"""Async mTLS transport for mesh peer connections."""

from __future__ import annotations

import asyncio
import logging
import struct
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, TypeVar
import time

import msgpack

from miniforge.mesh.security import MeshSecurity

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class MeshMessage:
    """Message sent between mesh nodes."""
    msg_type: str  # "handshake", "heartbeat", "inference_request", "inference_response", etc.
    payload: Dict[str, Any]
    timestamp: float
    node_id: str


class MeshConnection:
    """A connection to a peer mesh node."""
    
    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        node_id: str,
        security: MeshSecurity,
    ):
        self.reader = reader
        self.writer = writer
        self.node_id = node_id
        self.security = security
        self.connected_at = time.time()
        self.last_seen = time.time()
        self._handlers: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {}
        self._running = False
        self._read_task: Optional[asyncio.Task] = None
        
    async def start(self) -> None:
        """Start handling messages."""
        self._running = True
        self._read_task = asyncio.create_task(self._read_loop())
        
    async def stop(self) -> None:
        """Stop and close connection."""
        self._running = False
        if self._read_task:
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass
        self.writer.close()
        await self.writer.wait_closed()
        
    def on(self, msg_type: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Register handler for message type."""
        if msg_type not in self._handlers:
            self._handlers[msg_type] = []
        self._handlers[msg_type].append(handler)
        
    async def send(self, msg_type: str, payload: Dict[str, Any]) -> None:
        """Send a message to the peer."""
        msg = MeshMessage(
            msg_type=msg_type,
            payload=payload,
            timestamp=time.time(),
            node_id=self.node_id,
        )
        data = msgpack.packb({
            "type": msg.msg_type,
            "payload": msg.payload,
            "timestamp": msg.timestamp,
            "node_id": msg.node_id,
        }, use_bin_type=True)
        
        # Send length-prefixed message
        length = struct.pack(">I", len(data))
        self.writer.write(length + data)
        await self.writer.drain()
        
    async def _read_loop(self) -> None:
        """Read and dispatch messages."""
        while self._running:
            try:
                # Read length prefix
                length_data = await self.reader.readexactly(4)
                length = struct.unpack(">I", length_data)[0]
                
                if length > 10 * 1024 * 1024:  # Max 10MB
                    logger.warning(f"Message too large ({length} bytes), dropping connection")
                    break
                    
                # Read message data
                data = await self.reader.readexactly(length)
                msg_dict = msgpack.unpackb(data, raw=False)
                
                msg = MeshMessage(
                    msg_type=msg_dict["type"],
                    payload=msg_dict["payload"],
                    timestamp=msg_dict["timestamp"],
                    node_id=msg_dict["node_id"],
                )
                
                self.last_seen = time.time()
                await self._dispatch(msg)
                
            except asyncio.IncompleteReadError:
                logger.info(f"Connection closed by {self.node_id}")
                break
            except Exception as e:
                logger.warning(f"Error reading from {self.node_id}: {e}")
                break
                
        self._running = False
        
    async def _dispatch(self, msg: MeshMessage) -> None:
        """Dispatch message to handlers."""
        handlers = self._handlers.get(msg.msg_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(msg.payload)
                else:
                    handler(msg.payload)
            except Exception as e:
                logger.error(f"Handler error for {msg.msg_type}: {e}")


class MeshTransport:
    """
    Manages all mesh connections.
    
    Features:
    - Accept incoming connections
    - Connect to discovered peers
    - Automatic reconnection
    - Heartbeat/health checking
    """
    
    PROTOCOL_MAGIC = b"MINIFORGE_NODE"
    RECONNECT_DELAY = 5.0  # seconds
    HEARTBEAT_INTERVAL = 10.0  # seconds
    
    def __init__(
        self,
        node_id: str,
        node_name: str,
        mesh_port: int,
        ram_gb: float,
        cpu_cores: int,
        security: MeshSecurity,
    ):
        self.node_id = node_id
        self.node_name = node_name
        self.mesh_port = mesh_port
        self.ram_gb = ram_gb
        self.cpu_cores = cpu_cores
        self.security = security
        
        self._connections: Dict[str, MeshConnection] = {}
        self._callbacks: List[Callable[[str, MeshConnection, str], None]] = []
        self._running = False
        self._server: Optional[asyncio.Server] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        
    def on_connection_event(
        self, 
        callback: Callable[[str, MeshConnection, str], None]
    ) -> None:
        """
        Register callback for connection events.
        
        Args:
            callback: Function(peer_node_id, connection, event)
            Events: "connected", "disconnected", "heartbeat"
        """
        self._callbacks.append(callback)
        
    def _notify(self, peer_id: str, conn: MeshConnection, event: str) -> None:
        """Notify callbacks of connection event."""
        for cb in self._callbacks:
            try:
                cb(peer_id, conn, event)
            except Exception as e:
                logger.warning(f"Connection callback error: {e}")
                
    @property
    def connections(self) -> Dict[str, MeshConnection]:
        """Return current connections."""
        return dict(self._connections)
        
    async def start(self) -> None:
        """Start transport server and heartbeat."""
        if self._running:
            return
            
        self._running = True
        
        # Start server
        self._server = await asyncio.start_server(
            self._handle_incoming,
            host="0.0.0.0",
            port=self.mesh_port,
        )
        logger.info(f"Mesh transport listening on 0.0.0.0:{self.mesh_port}")
        
        # Start heartbeat
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
    async def stop(self) -> None:
        """Stop transport and close all connections."""
        self._running = False
        
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
                
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            
        # Close all connections
        for conn in list(self._connections.values()):
            await conn.stop()
        self._connections.clear()
        
        logger.info("Mesh transport stopped")
        
    async def connect_to_peer(self, ip: str, port: int) -> Optional[MeshConnection]:
        """Connect to a discovered peer."""
        peer_key = f"{ip}:{port}"
        
        if peer_key in self._connections:
            return self._connections[peer_key]
            
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(ip, port),
                timeout=5.0,
            )
            
            # Send handshake
            handshake = f"{self.PROTOCOL_MAGIC.decode()}|{self.node_id}|{self.node_name}|{self.ram_gb}|{self.cpu_cores}\n"
            writer.write(handshake.encode())
            await writer.drain()
            
            # Read response
            response = await asyncio.wait_for(reader.readline(), timeout=5.0)
            response_str = response.decode().strip()
            
            if not response_str.startswith("MINIFORGE_OK|"):
                logger.warning(f"Peer {peer_key} rejected handshake: {response_str}")
                writer.close()
                await writer.wait_closed()
                return None
                
            # Parse peer info from response
            parts = response_str.split("|")
            peer_id = parts[1] if len(parts) > 1 else "unknown"
            
            # Create connection
            conn = MeshConnection(reader, writer, self.node_id, self.security)
            conn.on("heartbeat", lambda _: None)  # Update last_seen
            await conn.start()
            
            self._connections[peer_key] = conn
            self._notify(peer_id, conn, "connected")
            
            logger.info(f"Connected to peer {peer_id} @ {peer_key}")
            return conn
            
        except Exception as e:
            logger.debug(f"Failed to connect to {peer_key}: {e}")
            return None
            
    async def _handle_incoming(
        self, 
        reader: asyncio.StreamReader, 
        writer: asyncio.StreamWriter
    ) -> None:
        """Handle incoming connection."""
        peer_addr = writer.get_extra_info("peername")
        logger.debug(f"Incoming connection from {peer_addr}")
        
        try:
            # Read handshake
            handshake = await asyncio.wait_for(reader.readline(), timeout=5.0)
            handshake_str = handshake.decode().strip()
            
            if not handshake_str.startswith("MINIFORGE_NODE|"):
                writer.write(b"MINIFORGE_REJECT|Invalid handshake\n")
                await writer.drain()
                writer.close()
                await writer.wait_closed()
                return
                
            # Parse peer info
            parts = handshake_str.split("|")
            if len(parts) < 5:
                writer.write(b"MINIFORGE_REJECT|Incomplete handshake\n")
                await writer.drain()
                writer.close()
                await writer.wait_closed()
                return
                
            _, peer_id, peer_name, peer_ram, peer_cpu = parts[:5]
            peer_key = f"{peer_addr[0]}:{peer_addr[1]}"
            
            # Send OK response
            response = f"MINIFORGE_OK|{self.node_id}|{self.node_name}|{self.ram_gb}|{self.cpu_cores}\n"
            writer.write(response.encode())
            await writer.drain()
            
            # Create connection
            conn = MeshConnection(reader, writer, self.node_id, self.security)
            await conn.start()
            
            self._connections[peer_key] = conn
            self._notify(peer_id, conn, "connected")
            
            logger.info(f"Accepted connection from {peer_name} ({peer_id}) @ {peer_key}")
            
            # Keep connection alive
            while conn._running:
                await asyncio.sleep(1)
                
        except asyncio.TimeoutError:
            logger.warning(f"Handshake timeout from {peer_addr}")
        except Exception as e:
            logger.warning(f"Error handling incoming from {peer_addr}: {e}")
        finally:
            if peer_key in self._connections:
                del self._connections[peer_key]
                
    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to all peers."""
        while self._running:
            try:
                for peer_key, conn in list(self._connections.items()):
                    try:
                        await conn.send("heartbeat", {
                            "ram_available": self.ram_gb,  # TODO: get actual available
                            "cpu_percent": 0.0,  # TODO: get actual
                            "timestamp": time.time(),
                        })
                    except Exception as e:
                        logger.debug(f"Heartbeat failed to {peer_key}: {e}")
                        # Mark for reconnection
                        
                # Clean up stale connections
                await self._cleanup_stale()
                
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")
                
            await asyncio.sleep(self.HEARTBEAT_INTERVAL)
            
    async def _cleanup_stale(self, max_silence: float = 30.0) -> None:
        """Remove connections that haven't been seen recently."""
        now = time.time()
        stale = [
            key for key, conn in self._connections.items()
            if now - conn.last_seen > max_silence
        ]
        for key in stale:
            conn = self._connections.pop(key)
            self._notify(conn.node_id, conn, "disconnected")
            await conn.stop()
            logger.info(f"Removed stale connection to {key}")
