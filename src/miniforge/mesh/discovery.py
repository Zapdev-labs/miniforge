"""Mesh auto-discovery using mDNS and IP scanning."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
from dataclasses import dataclass
from typing import List, Optional, Set, Callable
import time

logger = logging.getLogger(__name__)

# Try to import zeroconf for mDNS, fallback to scanning if not available
try:
    from zeroconf import ServiceBrowser, ServiceListener, Zeroconf, ServiceInfo
    from zeroconf.asyncio import AsyncZeroconf
    ZEROCONF_AVAILABLE = True
except ImportError:
    ZEROCONF_AVAILABLE = False
    logger.warning("zeroconf not installed, mDNS discovery disabled. Using IP scanning only.")


@dataclass
class PeerInfo:
    """Information about a discovered peer."""
    ip: str
    port: int
    node_id: str
    node_name: str
    ram_gb: float
    cpu_cores: int
    last_seen: float

    @property
    def endpoint(self) -> str:
        return f"{self.ip}:{self.port}"


class MeshDiscovery:
    """
    Auto-discovery for mesh nodes using mDNS + IP scanning.
    
    Uses both methods in parallel for maximum reliability:
    - mDNS: Fast, efficient, but blocked on some networks
    - IP scanning: Slower but works on any LAN
    """
    
    SERVICE_TYPE = "_miniforge._tcp.local."
    DEFAULT_MESH_PORT = 9999
    SCAN_TIMEOUT = 5  # seconds
    
    def __init__(
        self,
        node_id: str,
        node_name: str,
        mesh_port: int = DEFAULT_MESH_PORT,
        ram_gb: float = 28.0,
        cpu_cores: int = 8,
    ):
        self.node_id = node_id
        self.node_name = node_name
        self.mesh_port = mesh_port
        self.ram_gb = ram_gb
        self.cpu_cores = cpu_cores
        
        self._peers: dict[str, PeerInfo] = {}
        self._callbacks: List[Callable[[PeerInfo, str], None]] = []
        self._running = False
        self._zeroconf: Optional[AsyncZeroconf] = None
        self._browser: Optional[ServiceBrowser] = None
        self._scan_task: Optional[asyncio.Task] = None
        self._broadcast_task: Optional[asyncio.Task] = None
        
    def on_peer_discovered(self, callback: Callable[[PeerInfo, str], None]) -> None:
        """Register callback for peer events. Events: 'added', 'updated', 'removed'."""
        self._callbacks.append(callback)
        
    def _notify(self, peer: PeerInfo, event: str) -> None:
        """Notify all callbacks of peer event."""
        for cb in self._callbacks:
            try:
                cb(peer, event)
            except Exception as e:
                logger.warning(f"Discovery callback error: {e}")
                
    @property
    def peers(self) -> List[PeerInfo]:
        """Return list of currently known peers."""
        return list(self._peers.values())
    
    async def start(self) -> None:
        """Start discovery services."""
        if self._running:
            return
            
        self._running = True
        logger.info(f"Starting mesh discovery on port {self.mesh_port}")
        
        # Start mDNS if available
        if ZEROCONF_AVAILABLE:
            await self._start_mdns()
            
        # Start IP scanning (always works)
        self._scan_task = asyncio.create_task(self._ip_scan_loop())
        
        logger.info("Discovery started")
        
    async def stop(self) -> None:
        """Stop discovery services."""
        self._running = False
        
        if self._broadcast_task:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass
                
        if self._scan_task:
            self._scan_task.cancel()
            try:
                await self._scan_task
            except asyncio.CancelledError:
                pass
                
        if self._zeroconf:
            await self._zeroconf.async_close()
            
        logger.info("Discovery stopped")
        
    async def _start_mdns(self) -> None:
        """Start mDNS service broadcast and browser."""
        try:
            self._zeroconf = AsyncZeroconf()
            
            # Register our service
            info = ServiceInfo(
                type_=self.SERVICE_TYPE,
                name=f"{self.node_name}.{self.SERVICE_TYPE}",
                addresses=[self._get_local_ip().encode()],
                port=self.mesh_port,
                properties={
                    b"node_id": self.node_id.encode(),
                    b"node_name": self.node_name.encode(),
                    b"ram_gb": str(self.ram_gb).encode(),
                    b"cpu_cores": str(self.cpu_cores).encode(),
                },
            )
            await self._zeroconf.async_register_service(info)
            logger.info(f"mDNS service registered: {self.node_name}.{self.SERVICE_TYPE}")
            
            # Start browser for other services
            listener = MeshServiceListener(self)
            self._browser = ServiceBrowser(
                self._zeroconf.zeroconf,
                self.SERVICE_TYPE,
                listener,
            )
        except Exception as e:
            logger.warning(f"Failed to start mDNS: {e}")
            self._zeroconf = None
            
    async def _ip_scan_loop(self) -> None:
        """Continuously scan local network for peers."""
        while self._running:
            try:
                await self._scan_network()
            except Exception as e:
                logger.warning(f"IP scan error: {e}")
            await asyncio.sleep(10)  # Scan every 10 seconds
            
    async def _scan_network(self) -> None:
        """Scan local subnet for mesh peers."""
        local_ip = self._get_local_ip()
        if not local_ip or local_ip == "127.0.0.1":
            return
            
        # Get subnet from local IP
        network = ipaddress.ip_network(f"{local_ip}/24", strict=False)
        
        # Scan common ports on each host (limited to avoid flooding)
        ports_to_check = [self.mesh_port]
        
        # Create scan tasks for all hosts
        tasks = []
        hosts = list(network.hosts())[:50]  # Limit to first 50 hosts
        
        for host in hosts:
            host_str = str(host)
            if host_str == local_ip:
                continue
            for port in ports_to_check:
                tasks.append(self._check_host(host_str, port))
                
        if not tasks:
            return
            
        # Run checks with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(20)
        
        async def check_with_limit(host: str, port: int) -> None:
            async with semaphore:
                await self._check_host(host, port)
                
        await asyncio.gather(*[
            check_with_limit(str(h), p) 
            for h in hosts[:50] 
            for p in ports_to_check
            if str(h) != local_ip
        ], return_exceptions=True)
        
    async def _check_host(self, host: str, port: int) -> None:
        """Check if a host:port is a mesh node."""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=1.0
            )
            
            # Send handshake probe
            probe = b"MINIFORGE_PROBE\n"
            writer.write(probe)
            await writer.drain()
            
            # Read response
            response = await asyncio.wait_for(reader.read(1024), timeout=1.0)
            writer.close()
            await writer.wait_closed()
            
            if b"MINIFORGE_NODE" in response:
                # Parse node info from response
                # Format: MINIFORGE_NODE|node_id|node_name|ram|cpu
                parts = response.decode().strip().split("|")
                if len(parts) >= 5:
                    _, node_id, node_name, ram_gb, cpu_cores = parts[:5]
                    peer = PeerInfo(
                        ip=host,
                        port=port,
                        node_id=node_id,
                        node_name=node_name,
                        ram_gb=float(ram_gb),
                        cpu_cores=int(cpu_cores),
                        last_seen=time.time(),
                    )
                    await self._add_or_update_peer(peer)
                    
        except (asyncio.TimeoutError, ConnectionRefusedError, OSError):
            pass  # Host not available or not a mesh node
        except Exception as e:
            logger.debug(f"Host check error for {host}:{port}: {e}")
            
    async def _add_or_update_peer(self, peer: PeerInfo) -> None:
        """Add new peer or update existing."""
        key = f"{peer.ip}:{peer.port}"
        existing = self._peers.get(key)
        
        if existing is None:
            self._peers[key] = peer
            logger.info(f"Discovered peer: {peer.node_name} @ {peer.endpoint}")
            self._notify(peer, "added")
        else:
            # Update last_seen
            existing.last_seen = peer.last_seen
            self._notify(peer, "updated")
            
    async def remove_stale_peers(self, max_age: float = 60.0) -> None:
        """Remove peers not seen recently."""
        now = time.time()
        stale = [
            key for key, peer in self._peers.items()
            if now - peer.last_seen > max_age
        ]
        for key in stale:
            peer = self._peers.pop(key)
            logger.info(f"Removed stale peer: {peer.node_name}")
            self._notify(peer, "removed")
            
    def _get_local_ip(self) -> str:
        """Get local IP address."""
        try:
            # Try to connect to a public DNS to determine local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0)
            try:
                s.connect(("8.8.8.8", 1))
                ip = s.getsockname()[0]
            except Exception:
                ip = "127.0.0.1"
            finally:
                s.close()
            return ip
        except Exception:
            return "127.0.0.1"


class MeshServiceListener(ServiceListener if ZEROCONF_AVAILABLE else object):  # type: ignore
    """Listener for mDNS service announcements."""
    
    def __init__(self, discovery: MeshDiscovery):
        self.discovery = discovery
        
    def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """Called when a new service is discovered."""
        info = zc.get_service_info(type_, name)
        if info and info.addresses:
            self._handle_service(info, "added")
            
    def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """Called when a service is updated."""
        info = zc.get_service_info(type_, name)
        if info and info.addresses:
            self._handle_service(info, "updated")
            
    def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """Called when a service is removed."""
        # Parse name to get node info for removal
        pass
        
    def _handle_service(self, info: ServiceInfo, event: str) -> None:
        """Process discovered service."""
        try:
            ip = str(ipaddress.ip_address(info.addresses[0]))
            props = info.properties
            
            peer = PeerInfo(
                ip=ip,
                port=info.port,
                node_id=props.get(b"node_id", b"unknown").decode(),
                node_name=props.get(b"node_name", b"unknown").decode(),
                ram_gb=float(props.get(b"ram_gb", b"0").decode()),
                cpu_cores=int(props.get(b"cpu_cores", b"0").decode()),
                last_seen=time.time(),
            )
            
            asyncio.create_task(self.discovery._add_or_update_peer(peer))
        except Exception as e:
            logger.warning(f"Error handling mDNS service: {e}")
