"""FastAPI dashboard with OpenWebUI-style interface."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from aiohttp import web
import aiohttp_jinja2
import jinja2
import socketio

from miniforge.mesh.coordinator import MeshCoordinator
from miniforge.mesh.engine import DistributedInferenceEngine

logger = logging.getLogger(__name__)


class MeshDashboard:
    """
    Web dashboard for mesh monitoring and chat.
    
    Features:
    - Real-time mesh status visualization
    - Chat interface with distributed inference
    - Model management
    - Logs viewer
    """
    
    def __init__(
        self,
        coordinator: MeshCoordinator,
        engine: DistributedInferenceEngine,
        host: str = "0.0.0.0",
        port: int = 8000,
    ):
        self.coordinator = coordinator
        self.engine = engine
        self.host = host
        self.port = port
        
        # Create app
        self.app = web.Application()
        self.sio = socketio.AsyncServer(async_mode="aiohttp", cors_allowed_origins="*")
        self.sio.attach(self.app)
        
        # Setup templates - use PackageLoader for better compatibility
        template_dir = Path(__file__).parent / "static"
        try:
            # Try FileSystemLoader first
            env = aiohttp_jinja2.setup(
                self.app,
                loader=jinja2.FileSystemLoader(str(template_dir)),
                context_processors=[aiohttp_jinja2.request_processor],
            )
            # Add global url function
            env.globals['url'] = lambda name, **kwargs: f"/static/{kwargs.get('filename', '')}"
        except Exception as e:
            logger.warning(f"Jinja2 setup warning: {e}")
        
        # Store template dir for fallback
        self._template_dir = template_dir
        
        # State
        self._site: Optional[web.TCPSite] = None
        self._runner: Optional[web.AppRunner] = None
        
        # Setup routes
        self._setup_routes()
        self._setup_socketio()
        
        # Register for state updates
        coordinator.on_state_change(self._on_state_change)
        
    def _setup_routes(self) -> None:
        """Setup HTTP routes."""
        # Static files must be added BEFORE template routes
        static_dir = Path(__file__).parent / "static"
        self.app.router.add_static("/static", static_dir, name="static")
        
        # API routes
        self.app.router.add_get("/api/status", self.api_status)
        self.app.router.add_post("/api/chat", self.api_chat)
        self.app.router.add_get("/api/nodes", self.api_nodes)
        self.app.router.add_post("/api/connect", self.api_connect)
        
        # Main page - must be last to not conflict with static
        self.app.router.add_get("/", self.index)
        
    def _setup_socketio(self) -> None:
        """Setup Socket.IO event handlers."""
        @self.sio.event
        async def connect(sid: str, environ: Dict) -> None:
            logger.debug(f"Client connected: {sid}")
            await self.sio.emit("status", self._get_status(), room=sid)
            
        @self.sio.event
        async def disconnect(sid: str) -> None:
            logger.debug(f"Client disconnected: {sid}")
            
        @self.sio.event
        async def chat_message(sid: str, data: Dict[str, Any]) -> None:
            """Handle chat message from client."""
            prompt = data.get("message", "")
            stream = data.get("stream", True)
            
            try:
                if stream:
                    # Stream response
                    response_stream = await self.engine.generate(
                        prompt=prompt,
                        stream=True,
                    )
                    async for chunk in response_stream:
                        await self.sio.emit("chat_chunk", {
                            "chunk": chunk,
                        }, room=sid)
                    await self.sio.emit("chat_complete", {}, room=sid)
                else:
                    # Non-streaming
                    response = await self.engine.generate(
                        prompt=prompt,
                        stream=False,
                    )
                    await self.sio.emit("chat_response", {
                        "response": response,
                    }, room=sid)
                    
            except Exception as e:
                logger.error(f"Chat error: {e}")
                await self.sio.emit("chat_error", {
                    "error": str(e),
                }, room=sid)
                
    async def start(self) -> None:
        """Start the dashboard server."""
        self._runner = web.AppRunner(self.app)
        await self._runner.setup()
        
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()
        
        logger.info(f"Dashboard running at http://{self.host}:{self.port}")
        
    async def stop(self) -> None:
        """Stop the dashboard server."""
        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()
        logger.info("Dashboard stopped")
        
    async def index(self, request: web.Request) -> web.Response:
        """Serve main dashboard page."""
        try:
            context = {
                "node_id": self.coordinator.node_id,
                "node_name": self.coordinator.node_name,
                "is_leader": self.coordinator.is_leader,
            }
            return aiohttp_jinja2.render_template("index.html", request, context)
        except Exception as e:
            logger.error(f"Template render error: {e}")
            # Return simple HTML as fallback
            return web.Response(
                text=f"""
<!DOCTYPE html>
<html>
<head><title>Miniforge Mesh</title></head>
<body>
    <h1>Miniforge Mesh - {self.coordinator.node_name}</h1>
    <p>Node ID: {self.coordinator.node_id}</p>
    <p>Leader: {self.coordinator.is_leader}</p>
    <p><a href="/static/index.html">Try static fallback</a></p>
</body>
</html>
""",
                content_type="text/html"
            )
        
    async def api_status(self, request: web.Request) -> web.Response:
        """Get current mesh status."""
        return web.json_response(self._get_status())
        
    async def api_nodes(self, request: web.Request) -> web.Response:
        """Get list of mesh nodes."""
        nodes = [
            {
                "node_id": n.node_id,
                "node_name": n.node_name,
                "ip": n.ip,
                "port": n.port,
                "ram_gb": n.ram_gb,
                "ram_available": n.ram_available,
                "cpu_cores": n.cpu_cores,
                "cpu_percent": n.cpu_percent,
                "is_leader": n.is_leader,
                "status": n.status,
            }
            for n in self.coordinator.all_nodes
        ]
        return web.json_response({"nodes": nodes})
        
    async def api_chat(self, request: web.Request) -> web.Response:
        """API endpoint for chat completion."""
        try:
            data = await request.json()
            prompt = data.get("prompt", "")
            stream = data.get("stream", False)
            
            if stream:
                # Return stream response
                response = web.StreamResponse(
                    status=200,
                    reason="OK",
                    headers={
                        "Content-Type": "text/plain",
                    },
                )
                await response.prepare(request)
                
                result_stream = await self.engine.generate(
                    prompt=prompt,
                    stream=True,
                )
                async for chunk in result_stream:
                    await response.write(chunk.encode())
                await response.write_eof()
                return response
            else:
                # Non-streaming
                result = await self.engine.generate(
                    prompt=prompt,
                    stream=False,
                )
                return web.json_response({
                    "response": result,
                    "node_id": self.coordinator.node_id,
                })
                
        except Exception as e:
            logger.error(f"API chat error: {e}")
            return web.json_response(
                {"error": str(e)},
                status=500,
            )
            
    async def api_connect(self, request: web.Request) -> web.Response:
        """Manually connect to a peer."""
        try:
            data = await request.json()
            ip = data.get("ip")
            port = data.get("port", 9999)
            
            if not ip:
                return web.json_response(
                    {"error": "IP address required"},
                    status=400,
                )
                
            # Try to connect
            from miniforge.mesh.discovery import PeerInfo
            peer = PeerInfo(
                ip=ip,
                port=port,
                node_id="unknown",
                node_name="manual",
                ram_gb=0,
                cpu_cores=0,
                last_seen=0,
            )
            
            # This will trigger connection through discovery callback
            await self.coordinator._connect_to_peer(peer)
            
            return web.json_response({
                "status": "connecting",
                "endpoint": f"{ip}:{port}",
            })
            
        except Exception as e:
            logger.error(f"API connect error: {e}")
            return web.json_response(
                {"error": str(e)},
                status=500,
            )
            
    def _get_status(self) -> Dict[str, Any]:
        """Get current status for API/Socket.IO."""
        resources = self.coordinator.total_resources
        return {
            "node_id": self.coordinator.node_id,
            "node_name": self.coordinator.node_name,
            "is_leader": self.coordinator.is_leader,
            "leader_id": self.coordinator.leader_id,
            "resources": resources,
            "nodes_count": len(self.coordinator.all_nodes),
        }
        
    def _on_state_change(self) -> None:
        """Broadcast state change to all clients."""
        # Schedule in event loop
        asyncio.create_task(self._broadcast_status())
        
    async def _broadcast_status(self) -> None:
        """Broadcast status to all connected clients."""
        try:
            status = self._get_status()
            await self.sio.emit("status", status)
        except Exception as e:
            logger.debug(f"Broadcast error: {e}")
