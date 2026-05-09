#!/usr/bin/env python3
"""Simple mesh runner — loads model, connects peer, runs distributed inference."""
import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from miniforge.core.engine import InferenceEngine
from miniforge.mesh import (
    MeshCoordinator, MeshDiscovery, MeshSecurity, MeshTransport, DistributedInferenceEngine,
)
from miniforge.utils.config import M7Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("mesh")


def get_local_ip():
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


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", required=True)
    parser.add_argument("--name", "-n", default="node")
    parser.add_argument("--port", "-p", type=int, default=9999)
    parser.add_argument("--peer")
    parser.add_argument("--preset", default="balanced")
    parser.add_argument("--prompt", default="Explain the advantages of distributed inference in one sentence.")
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    import uuid
    node_id = str(uuid.uuid4())[:8]
    local_ip = get_local_ip()
    print(f"🔥 {args.name} | {node_id} | {local_ip}:{args.port}")

    config = M7Config.performance_preset(args.preset)
    print(f"📦 Loading {Path(args.model).name} ...")
    engine = InferenceEngine(
        model_path=Path(args.model),
        backend="llama_cpp",
        config=config.get_backend_config(),
    )
    await engine.initialize()
    info = await engine.get_info()
    print(f"✅ Loaded | ctx={info.get('n_ctx')} | backend={info.get('backend')}")

    config_dir = Path.home() / ".config" / "miniforge"
    config_dir.mkdir(parents=True, exist_ok=True)

    security = MeshSecurity(config_dir)
    discovery = MeshDiscovery(node_id=node_id, node_name=args.name, mesh_port=args.port,
                              ram_gb=config.max_memory_gb, cpu_cores=config.n_threads or 8)
    transport = MeshTransport(node_id=node_id, node_name=args.name, mesh_port=args.port,
                              ram_gb=config.max_memory_gb, cpu_cores=config.n_threads or 8,
                              security=security)
    coordinator = MeshCoordinator(node_id=node_id, node_name=args.name,
                                  discovery=discovery, transport=transport,
                                  local_ram_gb=config.max_memory_gb, local_cpu_cores=config.n_threads or 8)
    dist = DistributedInferenceEngine(local_engine=engine, coordinator=coordinator, node_id=node_id)

    await coordinator.start()
    print("📡 Coordinator started")

    if args.peer:
        print(f"🔗 Connecting to {args.peer} ...")
        await asyncio.sleep(2)
        conn = await transport.connect_to_peer(args.peer, args.port)
        if conn:
            print("✅ Connected to peer")
        else:
            print("⚠️  Direct connect failed; discovery will retry")

    # Wait for mesh
    print("⏳ Waiting for mesh ...")
    for _ in range(30):
        await asyncio.sleep(2)
        nodes = coordinator.all_nodes
        if len(nodes) > 1:
            break

    nodes = coordinator.all_nodes
    res = coordinator.total_resources
    print(f"\n📊 {len(nodes)} nodes | {res['total_ram_gb']:.0f}GB RAM | {res['total_cpu_cores']} cores")
    for n in nodes:
        mark = "⭐ You" if n.node_id == node_id else ""
        print(f"   • {n.node_name} {n.ip}:{n.port} {mark}")

    # Run inference
    print(f"\n🧪 Inference: strategy=auto | max_tokens={args.max_tokens}")
    t0 = time.perf_counter()
    try:
        result = await dist.generate(prompt=args.prompt, max_tokens=args.max_tokens, temperature=0.7, strategy="auto")
        elapsed = time.perf_counter() - t0
        print(f"Result: {result[:300]}...")
        print(f"Time: {elapsed:.1f}s")
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()

    # If multi-node, run one more explicitly remote
    if len(nodes) > 1:
        print(f"\n🧪 Inference: strategy=remote")
        t0 = time.perf_counter()
        try:
            result = await dist.generate(prompt=args.prompt, max_tokens=args.max_tokens, temperature=0.7, strategy="remote")
            elapsed = time.perf_counter() - t0
            print(f"Result: {result[:300]}...")
            print(f"Time: {elapsed:.1f}s")
        except Exception as e:
            print(f"❌ Remote inference failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n💤 Keeping alive. Press Ctrl+C to stop.")
    try:
        while True:
            await asyncio.sleep(10)
            nodes = coordinator.all_nodes
            peers = [n.node_name for n in nodes if n.node_id != node_id]
            if peers:
                print(f"   [alive] peers: {', '.join(peers)}")
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        await coordinator.stop()
        await engine.cleanup()
        print("🛑 Stopped")


if __name__ == "__main__":
    asyncio.run(main())
