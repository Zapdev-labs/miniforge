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
    
    Host/Worker Mode:
    - Host: Stores all models, streams to workers on-demand
    - Worker: No local models, downloads from host when needed
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
    
    # Determine mode
    is_host = args.mode == "host"
    host_node_id: Optional[str] = None
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if not args.verbose else logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    print(f"🔥 Miniforge Mesh Starting...")
    print(f"   Node ID: {node_id}")
    print(f"   Name: {node_name}")
    print(f"   Mode: {args.mode}")
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
    
    # Initialize model registry
    from miniforge.mesh.registry import ModelRegistry
    registry = ModelRegistry(
        node_id=node_id,
        is_host=is_host,
        cache_dir=config_dir / "mesh_models",
    )
    
    # If specific IP provided for host, connect to it
    if args.host_ip:
        print(f"🔗 Will connect to host at {args.host_ip}:{args.port}...")
    
    # If specific peer IP provided, connect to it
    if args.ip:
        print(f"🔗 Connecting to peer {args.ip}:{args.port}...")
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
    
    # Initialize local inference engine
    local_engine = InferenceEngine(
        model_path=args.model or "minimax",
        backend="llama_cpp",
    )
    
    # Register model if host mode and model path provided
    if is_host and args.model_path:
        try:
            registry.register_local_model(
                model_id=args.model or "minimax",
                model_path=args.model_path,
                quantization=args.quantization or "Q4_K_M",
            )
            print(f"📦 Registered model: {args.model_path}")
        except FileNotFoundError as e:
            print(f"⚠️  Warning: {e}")
    
    # Initialize engine based on mode
    if is_host:
        # Host uses distributed engine with registry
        from miniforge.mesh.host_worker import HostWorkerEngine
        engine = HostWorkerEngine(
            local_engine=local_engine,
            coordinator=coordinator,
            registry=registry,
            node_id=node_id,
            is_host=True,
        )
    else:
        # Worker uses host/worker engine
        from miniforge.mesh.host_worker import HostWorkerEngine
        engine = HostWorkerEngine(
            local_engine=local_engine,
            coordinator=coordinator,
            registry=registry,
            node_id=node_id,
            is_host=False,
            host_node_id=host_node_id,
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
        
        # Get model info
        models = registry.list_local_models()
        
        print(f"📊 Mesh Status ({len(nodes)} nodes):")
        for node in nodes:
            leader_marker = " 👑" if node.is_leader else ""
            you_marker = " ⭐ You" if node.node_id == node_id else ""
            host_marker = " 💾 HOST" if is_host and node.node_id == node_id else ""
            worker_marker = " 🖥️  WORKER" if not is_host and node.node_id == node_id else ""
            print(f"   • {node.node_name}{leader_marker}{you_marker}{host_marker}{worker_marker}")
            print(f"     {node.ip}:{node.port} | {node.ram_gb}GB RAM | {node.cpu_cores} cores")
        print()
        print(f"💾 Combined Resources: {resources['total_ram_gb']:.0f}GB RAM | {resources['total_cpu_cores']} CPU cores")
        
        if models:
            print()
            print("📦 Local Models:")
            for model in models:
                print(f"   • {model.model_id} ({model.file_size / 1e9:.1f}GB)")
        elif not is_host:
            print()
            print("📦 Models: Will download from host on first use")
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
        "--mode",
        choices=["host", "worker", "auto"],
        default="auto",
        help="Node mode: host (stores models), worker (no models, downloads from host), auto (peer-to-peer)"
    )
    
    up_parser.add_argument(
        "--host-ip",
        help="IP address of the host node (required for worker mode if not using auto-discovery)",
        type=str,
    )
    
    up_parser.add_argument(
        "--ip",
        help="IP address of any peer to connect to (for auto-discovery bypass)",
        type=str,
    )
    
    up_parser.add_argument(
        "--model-path",
        help="Path to model file (required for host mode to serve models)",
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
