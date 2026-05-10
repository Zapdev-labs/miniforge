#!/usr/bin/env python3
"""Mesh benchmark: both machines, small context, concurrent requests."""
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
logger = logging.getLogger("meshbench")


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


async def load_model(model_path: str, config: M7Config) -> InferenceEngine:
    path = Path(model_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    engine = InferenceEngine(
        model_path=path, backend="llama_cpp", config=config.get_backend_config()
    )
    await engine.initialize()
    info = await engine.get_info()
    logger.info(f"Loaded {path.name} | ctx={info.get('n_ctx')} | backend={info.get('backend')}")
    return engine


async def benchmark_run(dist_engine, coordinator, node_id, prompt, max_tokens, strategy, label):
    """Run one inference and report which node handled it."""
    t0 = time.perf_counter()
    try:
        result = await dist_engine.generate(
            prompt=prompt, max_tokens=max_tokens, temperature=0.7, strategy=strategy
        )
        elapsed = time.perf_counter() - t0
        # Figure out which node ran it
        jobs = coordinator._jobs
        # Get most recent job
        if jobs:
            latest_job = max(jobs.values(), key=lambda j: j.created_at)
            handler = latest_job.assigned_node or node_id
            if handler == node_id:
                handler_name = "LOCAL"
            else:
                # Look up node name
                handler_name = None
                for n in coordinator.all_nodes:
                    if n.node_id == handler:
                        handler_name = n.node_name
                        break
                if not handler_name:
                    handler_name = handler[:8]
        else:
            handler_name = "?"
        return {
            "label": label,
            "strategy": strategy,
            "handler": handler_name,
            "elapsed": elapsed,
            "result": result[:200] if isinstance(result, str) else str(result)[:200],
            "tokens": max_tokens,
            "tps": max_tokens / elapsed if elapsed > 0 else 0,
        }
    except Exception as e:
        return {"label": label, "strategy": strategy, "handler": "ERROR", "elapsed": time.perf_counter() - t0, "error": str(e)}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", required=True)
    parser.add_argument("--name", "-n", default="node")
    parser.add_argument("--port", "-p", type=int, default=9999)
    parser.add_argument("--peer")
    parser.add_argument("--preset", default="moe")
    parser.add_argument("--ctx", type=int, default=4096, help="Context window (default 4096 for weight caching)")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--requests", type=int, default=4, help="Concurrent requests for benchmark")
    args = parser.parse_args()

    import uuid
    node_id = str(uuid.uuid4())[:8]
    local_ip = get_local_ip()
    print(f"\n🔥 {args.name} | {node_id} | {local_ip}:{args.port} | ctx={args.ctx}\n")

    # Build config with SMALL context so weights cache in RAM
    config = M7Config.performance_preset(args.preset)
    config.n_ctx = args.ctx
    config.n_batch = min(config.n_batch, args.ctx)
    config.n_ubatch = min(config.n_ubatch, args.ctx // 2)
    # Turn off prompt lookup for cleaner benchmark
    config.prompt_lookup = False

    print(f"📦 Loading model with n_ctx={args.ctx} ...")
    engine = await load_model(args.model, config)

    config_dir = Path.home() / ".config" / "miniforge"
    config_dir.mkdir(parents=True, exist_ok=True)

    security = MeshSecurity(config_dir)
    discovery = MeshDiscovery(
        node_id=node_id, node_name=args.name, mesh_port=args.port,
        ram_gb=config.max_memory_gb, cpu_cores=config.n_threads or 8,
    )
    transport = MeshTransport(
        node_id=node_id, node_name=args.name, mesh_port=args.port,
        ram_gb=config.max_memory_gb, cpu_cores=config.n_threads or 8,
        security=security,
    )
    coordinator = MeshCoordinator(
        node_id=node_id, node_name=args.name,
        discovery=discovery, transport=transport,
        local_ram_gb=config.max_memory_gb, local_cpu_cores=config.n_threads or 8,
    )
    dist = DistributedInferenceEngine(local_engine=engine, coordinator=coordinator, node_id=node_id)

    await coordinator.start()

    if args.peer:
        print(f"🔗 Connecting to {args.peer} ...")
        await asyncio.sleep(2)
        conn = await transport.connect_to_peer(args.peer, args.port)
        if conn:
            print("✅ Connected")
        else:
            print("⚠️  Direct connect failed; discovery retrying")

    # Wait for mesh
    print("⏳ Waiting for mesh ...")
    for _ in range(20):
        await asyncio.sleep(2)
        if len(coordinator.all_nodes) > 1:
            break

    nodes = coordinator.all_nodes
    print(f"\n📊 {len(nodes)} nodes online")
    for n in nodes:
        mark = "⭐ YOU" if n.node_id == node_id else ""
        print(f"   • {n.node_name} ({n.node_id[:8]}) @ {n.ip}:{n.port} {mark}")
    print()

    if len(nodes) <= 1:
        print("⚠️  No peers found. Exiting.")
        await coordinator.stop()
        await engine.cleanup()
        return

    prompts = [
        "What is quantum computing?",
        "Explain neural networks briefly.",
        "What are the benefits of async programming?",
        "Describe the Python GIL.",
        "What is transformer architecture?",
        "Explain RAID levels.",
        "What is the CAP theorem?",
        "Describe TCP vs UDP.",
    ][:args.requests]

    # --- Single request local ---
    print("=" * 60)
    print("TEST 1: Single request (strategy='local')")
    print("=" * 60)
    r = await benchmark_run(dist, coordinator, node_id, prompts[0], args.max_tokens, "local", "local-1")
    print(f"  [{r['handler']}] {r['elapsed']:.1f}s | {r.get('tps', 0):.2f} tok/s")
    print(f"  {r.get('result', r.get('error', '?'))[:120]}...")
    print()

    # --- Single request remote ---
    print("=" * 60)
    print("TEST 2: Single request (strategy='remote')")
    print("=" * 60)
    r = await benchmark_run(dist, coordinator, node_id, prompts[1], args.max_tokens, "remote", "remote-1")
    print(f"  [{r['handler']}] {r['elapsed']:.1f}s | {r.get('tps', 0):.2f} tok/s")
    print(f"  {r.get('result', r.get('error', '?'))[:120]}...")
    print()

    # --- Concurrent requests (auto strategy = round-robin) ---
    print("=" * 60)
    print(f"TEST 3: {len(prompts)} concurrent requests (strategy='auto', round-robin)")
    print("=" * 60)
    t0 = time.perf_counter()
    results = await asyncio.gather(*[
        benchmark_run(dist, coordinator, node_id, p, args.max_tokens, "auto", f"auto-{i+1}")
        for i, p in enumerate(prompts)
    ])
    total_elapsed = time.perf_counter() - t0

    total_tokens = 0
    for r in results:
        total_tokens += r.get("tokens", 0)
        status = "✅" if "error" not in r else "❌"
        print(f"  {status} [{r['handler']:10s}] {r['elapsed']:6.1f}s | {r.get('tps', 0):5.2f} tok/s | {r['label']}")
    print()
    print(f"  Total time: {total_elapsed:.1f}s for {len(prompts)} requests")
    print(f"  Aggregate throughput: {len(prompts)/total_elapsed:.2f} req/s")
    print(f"  Token throughput: {total_tokens/total_elapsed:.2f} tok/s")
    print()

    # --- Sequential requests (auto strategy) ---
    print("=" * 60)
    print(f"TEST 4: {len(prompts)} sequential requests (strategy='auto')")
    print("=" * 60)
    t0 = time.perf_counter()
    seq_results = []
    for i, p in enumerate(prompts):
        r = await benchmark_run(dist, coordinator, node_id, p, args.max_tokens, "auto", f"seq-{i+1}")
        seq_results.append(r)
        print(f"  [{r['handler']:10s}] {r['elapsed']:6.1f}s | {r.get('tps', 0):5.2f} tok/s | {r['label']}")
    seq_elapsed = time.perf_counter() - t0
    print()
    print(f"  Total time: {seq_elapsed:.1f}s for {len(prompts)} sequential requests")
    print(f"  Throughput: {len(prompts)/seq_elapsed:.2f} req/s")
    print()

    print("💤 Mesh running. Press Ctrl+C to stop.")
    try:
        while True:
            await asyncio.sleep(30)
            peers = [n.node_name for n in coordinator.all_nodes if n.node_id != node_id]
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
