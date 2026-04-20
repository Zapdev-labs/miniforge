"""CLI commands for mesh operations."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from miniforge.mesh import (
    MeshDiscovery,
    MeshTransport,
    MeshCoordinator,
    MeshSecurity,
    DistributedInferenceEngine,
    MeshDashboard,
)
from miniforge.core.engine import InferenceEngine
from miniforge.utils.config import _default_config_dir

logger = logging.getLogger(__name__)


def generate_node_id() -> str:
    """Generate unique node ID."""
    import uuid
    return str(uuid.uuid4())[:8]


async def mesh_up_command(args: argparse.Namespace) -> None:
    """
    Start miniforge mesh node.
    
    This command starts a mesh node that can:
    - Auto-discover other nodes on the network
    - Accept connections from other nodes
    - Distribute inference jobs across the mesh
    - Serve a web dashboard for monitoring and chat
    """
    # Generate or load node identity
    config_dir = _default_config_dir()
    node_id_file = config_dir / "mesh_node_id"
    
    if node_id_file.exists() and not args.new_identity:
        node_id = node_id_file.read_text().strip()
    else:
        node_id = generate_node_id()
        node_id_file.parent.mkdir(parents=True, exist_ok=True)
        node_id_file.write_text(node_id)
    
    node_name = args.name or f"node-{node_id[:4]}"
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if not args.verbose else logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    print(f"🔥 Miniforge Mesh Starting...")
    print(f"   Node ID: {node_id}")
    print(f"   Name: {node_name}")
    print(f"   Port: {args.port}")
    print()
    
    # Initialize security
    security = MeshSecurity(config_dir)
    fingerprint = security.get_cert_fingerprint()
    print(f"🔐 TLS Certificate: {fingerprint}")
    print()
    
    # Initialize discovery
    discovery = MeshDiscovery(
        node_id=node_id,
        node_name=node_name,
        mesh_port=args.port,
        ram_gb=args.ram or 28.0,
        cpu_cores=args.cores or 8,
    )
    
    # Initialize transport
    transport = MeshTransport(
        node_id=node_id,
        node_name=node_name,
        mesh_port=args.port,
        ram_gb=args.ram or 28.0,
        cpu_cores=args.cores or 8,
        security=security,
    )
    
    # Initialize coordinator
    coordinator = MeshCoordinator(
        node_id=node_id,
        node_name=node_name,
        discovery=discovery,
        transport=transport,
        local_ram_gb=args.ram or 28.0,
        local_cpu_cores=args.cores or 8,
    )
    
    # If specific IP provided, connect to it
    if args.ip:
        print(f"🔗 Connecting to {args.ip}:{args.port}...")
        from miniforge.mesh.discovery import PeerInfo
        peer = PeerInfo(
            ip=args.ip,
            port=args.port,
            node_id="manual",
            node_name="manual",
            ram_gb=0,
            cpu_cores=0,
            last_seen=0,
        )
        # Will be connected by coordinator
    
    # Initialize local inference engine (placeholder)
    # In real implementation, load actual model
    local_engine = InferenceEngine(
        model_path=args.model or "minimax",
        backend="llama_cpp",
    )
    
    # Initialize distributed engine
    engine = DistributedInferenceEngine(
        local_engine=local_engine,
        coordinator=coordinator,
        node_id=node_id,
    )
    
    # Initialize dashboard
    dashboard = MeshDashboard(
        coordinator=coordinator,
        engine=engine,
        host=args.host,
        port=args.dashboard_port,
    )
    
    # Start everything
    try:
        print("📡 Starting discovery...")
        await coordinator.start()
        
        print("🌐 Starting dashboard...")
        await dashboard.start()
        
        print()
        print("✅ Mesh is running!")
        print()
        
        # Show status
        await asyncio.sleep(2)  # Give time for discovery
        nodes = coordinator.all_nodes
        resources = coordinator.total_resources
        
        print(f"📊 Mesh Status ({len(nodes)} nodes):")
        for node in nodes:
            leader_marker = " 👑" if node.is_leader else ""
            you_marker = " ⭐ You" if node.node_id == node_id else ""
            print(f"   • {node.node_name}{leader_marker}{you_marker}")
            print(f"     {node.ip}:{node.port} | {node.ram_gb}GB RAM | {node.cpu_cores} cores")
        print()
        print(f"💾 Combined Resources: {resources['total_ram_gb']:.0f}GB RAM | {resources['total_cpu_cores']} CPU cores")
        print()
        
        if coordinator.is_leader:
            print("👑 This node is the leader")
        else:
            print(f"👑 Leader: {coordinator.leader_id}")
        print()
        
        print(f"🌐 Dashboard: http://{args.host}:{args.dashboard_port}")
        print(f"💬 Chat Interface: http://{args.host}:{args.dashboard_port}/chat")
        print()
        print("Press Ctrl+C to stop")
        print()
        
        # Keep running
        while True:
            await asyncio.sleep(1)
            
    except asyncio.CancelledError:
        print("\n\n🛑 Shutting down...")
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
    finally:
        await dashboard.stop()
        await coordinator.stop()
        print("✅ Mesh stopped")


def add_mesh_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Add the 'up' subparser for mesh commands."""
    up_parser = subparsers.add_parser(
        "up",
        help="Start mesh node and join distributed network",
        description="Start a miniforge mesh node that auto-discovers peers and shares compute resources.",
    )
    
    up_parser.add_argument(
        "--ip",
        help="IP address of a known peer to connect to (optional, auto-discovery used if not provided)",
        type=str,
    )
    
    up_parser.add_argument(
        "--port",
        "-p",
        help="Mesh protocol port (default: 9999)",
        type=int,
        default=9999,
    )
    
    up_parser.add_argument(
        "--host",
        help="Dashboard bind host (default: 0.0.0.0)",
        type=str,
        default="0.0.0.0",
    )
    
    up_parser.add_argument(
        "--dashboard-port",
        "-d",
        help="Dashboard web port (default: 8000)",
        type=int,
        default=8000,
    )
    
    up_parser.add_argument(
        "--name",
        "-n",
        help="Node name for display in mesh",
        type=str,
    )
    
    up_parser.add_argument(
        "--model",
        "-m",
        help="Model to load for inference",
        type=str,
        default="minimax",
    )
    
    up_parser.add_argument(
        "--ram",
        "-r",
        help="RAM available on this node in GB (default: 28)",
        type=float,
    )
    
    up_parser.add_argument(
        "--cores",
        "-c",
        help="CPU cores available (default: 8)",
        type=int,
    )
    
    up_parser.add_argument(
        "--new-identity",
        action="store_true",
        help="Generate new node identity (ignore saved node ID)",
    )
    
    up_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    up_parser.set_defaults(func=lambda args: asyncio.run(mesh_up_command(args)))
