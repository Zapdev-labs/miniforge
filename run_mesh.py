#!/usr/bin/env python3
"""
Miniforge Mesh Distributed Inference Runner

Usage:
    # On host/leader (this machine):
    python run_mesh.py --model /path/to/model.gguf --name m7-local

    # On worker (remote machine):
    python run_mesh.py --model /path/to/model.gguf --name loveruffles \
                       --peer 100.111.229.67 --port 9999

Both machines load the model locally; inference jobs are routed to the
least-loaded node for performance scaling.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure src is on path when running from repo root
sys.path.insert(0, str(Path(__file__).parent / "src"))

from miniforge.core.engine import InferenceEngine
from miniforge.mesh import (
    MeshCoordinator,
    MeshDiscovery,
    MeshSecurity,
    MeshTransport,
    DistributedInferenceEngine,
)
from miniforge.utils.config import M7Config

logger = logging.getLogger("mesh_runner")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def get_local_ip() -> str:
    import socket
    try:
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


async def load_model(model_path: str, config: M7Config) -> InferenceEngine:
    """Load a GGUF model into an inference engine."""
    path = Path(model_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    backend_config = config.get_backend_config()
    engine = InferenceEngine(
        model_path=path,
        backend="llama_cpp",
        config=backend_config,
    )
    await engine.initialize()
    info = await engine.get_info()
    logger.info(f"Model loaded: {path.name}")
    logger.info(f"  Backend: {info.get('backend')}")
    logger.info(f"  Context: {info.get('n_ctx')} tokens")
    logger.info(f"  Params: {info.get('n_params')}")
    return engine


async def run_mesh_node(
    model_path: str,
    node_name: str,
    mesh_port: int,
    peer_ip: Optional[str],
    config: M7Config,
) -> None:
    """Start a mesh node with local model loaded."""
    import uuid
    node_id = str(uuid.uuid4())[:8]
    local_ip = get_local_ip()

    print(f"🔥 Miniforge Mesh Node Starting...")
    print(f"   Node ID: {node_id}")
    print(f"   Name: {node_name}")
    print(f"   Local IP: {local_ip}")
    print(f"   Mesh Port: {mesh_port}")
    print()

    # Load model locally first
    print(f"📦 Loading model: {model_path}")
    engine = await load_model(model_path, config)
    print("✅ Model loaded locally\n")

    # Initialize mesh components
    config_dir = Path.home() / ".config" / "miniforge"
    config_dir.mkdir(parents=True, exist_ok=True)

    security = MeshSecurity(config_dir)
    discovery = MeshDiscovery(
        node_id=node_id,
        node_name=node_name,
        mesh_port=mesh_port,
        ram_gb=config.max_memory_gb,
        cpu_cores=config.n_threads or 8,
    )
    transport = MeshTransport(
        node_id=node_id,
        node_name=node_name,
        mesh_port=mesh_port,
        ram_gb=config.max_memory_gb,
        cpu_cores=config.n_threads or 8,
        security=security,
    )
    coordinator = MeshCoordinator(
        node_id=node_id,
        node_name=node_name,
        discovery=discovery,
        transport=transport,
        local_ram_gb=config.max_memory_gb,
        local_cpu_cores=config.n_threads or 8,
    )
    dist_engine = DistributedInferenceEngine(
        local_engine=engine,
        coordinator=coordinator,
        node_id=node_id,
    )

    # Start coordinator (which starts discovery and transport)
    await coordinator.start()
    print("📡 Mesh coordinator started\n")

    # If peer IP specified, manually trigger connection
    if peer_ip:
        print(f"🔗 Connecting to peer {peer_ip}:{mesh_port}...")
        from miniforge.mesh.discovery import PeerInfo
        peer = PeerInfo(
            ip=peer_ip,
            port=mesh_port,
            node_id="manual",
            node_name="manual",
            ram_gb=0,
            cpu_cores=0,
            last_seen=0,
        )
        # Give discovery a moment, then force connection attempt
        await asyncio.sleep(1)
        conn = await transport.connect_to_peer(peer_ip, mesh_port)
        if conn:
            print(f"✅ Connected to peer at {peer_ip}:{mesh_port}\n")
        else:
            print(f"⚠️  Could not connect to {peer_ip}:{mesh_port} yet; discovery will retry\n")

    # Wait for mesh to stabilize
    print("⏳ Waiting for mesh discovery...")
    for _ in range(10):
        await asyncio.sleep(1)
        nodes = coordinator.all_nodes
        if len(nodes) > 1:
            break

    nodes = coordinator.all_nodes
    resources = coordinator.total_resources

    print(f"\n📊 Mesh Status ({len(nodes)} nodes):")
    for node in nodes:
        leader_marker = " 👑" if node.is_leader else ""
        you_marker = " ⭐ You" if node.node_id == node_id else ""
        print(f"   • {node.node_name}{leader_marker}{you_marker}")
        print(f"     {node.ip}:{node.port} | {node.ram_gb:.0f}GB RAM | {node.cpu_cores} cores")

    print(f"\n💾 Combined Resources: {resources['total_ram_gb']:.0f}GB RAM | {resources['total_cpu_cores']} CPU cores")

    if coordinator.is_leader:
        print("\n👑 This node is the leader")
    else:
        print(f"\n👑 Leader: {coordinator.leader_id}")

    # Run test inferences
    print("\n" + "=" * 50)
    print("🧪 RUNNING DISTRIBUTED INFERENCE TESTS")
    print("=" * 50 + "\n")

    prompts = [
        "Explain quantum computing in one paragraph.",
        "What are the main differences between Python and Rust?",
        "Describe the architecture of a transformer neural network briefly.",
    ]

    # Test 1: Local inference
    print("Test 1: Local inference (strategy='local')")
    t0 = time.perf_counter()
    result = await dist_engine.generate(
        prompt=prompts[0],
        max_tokens=128,
        temperature=0.7,
        strategy="local",
    )
    elapsed = time.perf_counter() - t0
    print(f"Result: {result[:200]}...")
    print(f"Time: {elapsed:.2f}s\n")

    # Test 2: Remote inference if we have peers
    if len(nodes) > 1:
        print("Test 2: Remote inference (strategy='remote')")
        t0 = time.perf_counter()
        result = await dist_engine.generate(
            prompt=prompts[1],
            max_tokens=128,
            temperature=0.7,
            strategy="remote",
        )
        elapsed = time.perf_counter() - t0
        print(f"Result: {result[:200]}...")
        print(f"Time: {elapsed:.2f}s\n")

        print("Test 3: Auto strategy (routes to least-loaded)")
        t0 = time.perf_counter()
        result = await dist_engine.generate(
            prompt=prompts[2],
            max_tokens=128,
            temperature=0.7,
            strategy="auto",
        )
        elapsed = time.perf_counter() - t0
        print(f"Result: {result[:200]}...")
        print(f"Time: {elapsed:.2f}s\n")

        # Benchmark: run multiple requests in parallel
        print("Test 4: Parallel distributed requests")
        async def run_one(i: int, prompt: str) -> tuple[int, float, str]:
            t0 = time.perf_counter()
            res = await dist_engine.generate(
                prompt=prompt,
                max_tokens=64,
                temperature=0.7,
                strategy="auto",
            )
            elapsed = time.perf_counter() - t0
            return i, elapsed, res

        parallel_prompts = [
            "List three benefits of async programming.",
            "What is RAID and what are its levels?",
            "Explain the CAP theorem.",
            "What is the difference between TCP and UDP?",
        ]

        t0 = time.perf_counter()
        results = await asyncio.gather(*[
            run_one(i, p) for i, p in enumerate(parallel_prompts)
        ])
        total_elapsed = time.perf_counter() - t0

        for i, elapsed, res in results:
            print(f"  Request {i+1}: {elapsed:.2f}s | {res[:100]}...")
        print(f"\nTotal time for {len(parallel_prompts)} parallel requests: {total_elapsed:.2f}s")
        print(f"Effective throughput: {len(parallel_prompts) / total_elapsed:.2f} req/s")
    else:
        print("⚠️  No peers discovered yet. Remote tests skipped.")
        print("   Make sure the peer node is running and reachable.")

    print("\n" + "=" * 50)
    print("✅ Tests complete. Mesh node is running.")
    print("   Press Ctrl+C to stop")
    print("=" * 50 + "\n")

    try:
        while True:
            await asyncio.sleep(1)
    except (asyncio.CancelledError, KeyboardInterrupt):
        print("\n🛑 Shutting down...")
    finally:
        await coordinator.stop()
        await engine.cleanup()
        print("✅ Mesh stopped")


def main() -> None:
    parser = argparse.ArgumentParser(description="Miniforge Mesh Distributed Inference")
    parser.add_argument("--model", "-m", required=True, help="Path to GGUF model file")
    parser.add_argument("--name", "-n", default="mesh-node", help="Node name")
    parser.add_argument("--port", "-p", type=int, default=9999, help="Mesh port")
    parser.add_argument("--peer", help="IP address of peer to connect to")
    parser.add_argument("--preset", choices=["speed", "balanced", "memory", "quality", "moe"],
                        default="balanced", help="Performance preset")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()
    setup_logging(args.verbose)

    config = M7Config.performance_preset(args.preset)

    try:
        asyncio.run(run_mesh_node(
            model_path=args.model,
            node_name=args.name,
            mesh_port=args.port,
            peer_ip=args.peer,
            config=config,
        ))
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.exception("Mesh runner failed")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
