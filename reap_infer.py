#!/usr/bin/env python3
"""Local MiniMax-M2.7 REAP GGUF runner and tuner.

This avoids the Hugging Face download path and talks directly to the local
GGUF through Miniforge's llama.cpp backend.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from miniforge.core.engine import InferenceEngine
from miniforge.utils.config import M7Config


DEFAULT_MODEL = Path(
    "/run/media/dih/8CEDA5F938E73A48/AI/"
    "MiniMax-M2.7-161B-REAP-GGUF/MiniMax-M2.7-161B-REAP-Q3_K_M.gguf"
)

PROMPT = """]~!b[]~b]system
You are MiniMax-M2.7. Be concise and direct.[e~[
]~b]user
Say hello, count to five, and stop.[e~[
]~b]ai
<think>
"""


@dataclass
class RunResult:
    threads: int
    batch: int
    ubatch: int
    ctx: int
    cache_type_k: str
    cache_type_v: str
    flash_attn: bool
    load_seconds: float
    elapsed_seconds: float
    prompt_tokens: int
    completion_tokens: int
    decode_tps: float
    total_tps: float
    output: str


def apply_process_tuning(args: argparse.Namespace) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads_batch or args.threads))
    os.environ.setdefault("OMP_PROC_BIND", "close")
    os.environ.setdefault("OMP_PLACES", "cores")
    os.environ.setdefault("GGML_OPENMP_THREADS", str(args.threads_batch or args.threads))

    if args.cpu_mask:
        cpus = parse_cpu_mask(args.cpu_mask)
        if cpus:
            with contextlib.suppress(OSError, AttributeError):
                os.sched_setaffinity(0, cpus)

    with contextlib.suppress(OSError):
        os.nice(args.nice)

    if args.ionice:
        with contextlib.suppress(Exception):
            subprocess.run(
                ["ionice", "-c", "2", "-n", "0", "-p", str(os.getpid())],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )


def parse_cpu_mask(raw: str) -> set[int]:
    cpus: set[int] = set()
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start_raw, end_raw = item.split("-", 1)
            cpus.update(range(int(start_raw), int(end_raw) + 1))
        else:
            cpus.add(int(item))
    return cpus


def prefetch_model(path: Path) -> None:
    if not hasattr(os, "posix_fadvise"):
        return
    with path.open("rb") as model_file:
        os.posix_fadvise(model_file.fileno(), 0, 0, os.POSIX_FADV_WILLNEED)


def build_config(args: argparse.Namespace) -> M7Config:
    return M7Config(
        quantization="Q3_K_M",
        cache_type_k=args.cache_k,
        cache_type_v=args.cache_v,
        n_ctx=args.ctx,
        n_threads=args.threads,
        n_threads_batch=args.threads_batch or args.threads,
        n_batch=args.batch,
        n_ubatch=args.ubatch,
        n_gpu_layers=args.gpu_layers,
        flash_attn=not args.no_flash_attn,
        use_mmap=True,
        use_mlock=False,
        memory_mode="mmap",
        prompt_lookup=args.prompt_lookup,
        prompt_lookup_ngram=args.prompt_lookup_ngram,
        prompt_lookup_tokens=args.prompt_lookup_tokens,
        numa=args.numa,
        offload_kqv=not args.no_offload_kqv,
        verbose=args.verbose,
        is_moe=True,
        model_params_b=161.97,
        max_model_ctx=196_608,
    )


async def run_once(args: argparse.Namespace) -> RunResult:
    model_path = args.model.expanduser()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if args.prefetch:
        prefetch_model(model_path)

    config = build_config(args)
    engine = InferenceEngine(model_path, backend="llama_cpp", config=config.get_backend_config())

    started_load = time.perf_counter()
    await engine.initialize()
    load_seconds = time.perf_counter() - started_load

    started = time.perf_counter()
    output = await engine.generate(
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        stop=["[e~["],
    )
    elapsed_seconds = time.perf_counter() - started

    info = await engine.get_info()
    generation = info.get("last_generation", {})
    await engine.cleanup()

    return RunResult(
        threads=args.threads,
        batch=args.batch,
        ubatch=args.ubatch,
        ctx=args.ctx,
        cache_type_k=args.cache_k,
        cache_type_v=args.cache_v,
        flash_attn=not args.no_flash_attn,
        load_seconds=load_seconds,
        elapsed_seconds=elapsed_seconds,
        prompt_tokens=int(generation.get("prompt_tokens", 0)),
        completion_tokens=int(generation.get("completion_tokens", 0)),
        decode_tps=float(generation.get("decode_tps", 0.0)),
        total_tps=float(generation.get("total_tps", 0.0)),
        output=output,
    )


async def run_with_engine(args: argparse.Namespace, engine: InferenceEngine) -> RunResult:
    started = time.perf_counter()
    output = await engine.generate(
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        stop=["[e~["],
    )
    elapsed_seconds = time.perf_counter() - started

    info = await engine.get_info()
    generation = info.get("last_generation", {})

    return RunResult(
        threads=args.threads,
        batch=args.batch,
        ubatch=args.ubatch,
        ctx=args.ctx,
        cache_type_k=args.cache_k,
        cache_type_v=args.cache_v,
        flash_attn=not args.no_flash_attn,
        load_seconds=0.0,
        elapsed_seconds=elapsed_seconds,
        prompt_tokens=int(generation.get("prompt_tokens", 0)),
        completion_tokens=int(generation.get("completion_tokens", 0)),
        decode_tps=float(generation.get("decode_tps", 0.0)),
        total_tps=float(generation.get("total_tps", 0.0)),
        output=output,
    )


async def run_repeat(args: argparse.Namespace) -> None:
    model_path = args.model.expanduser()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if args.prefetch:
        prefetch_model(model_path)

    config = build_config(args)
    engine = InferenceEngine(model_path, backend="llama_cpp", config=config.get_backend_config())

    started_load = time.perf_counter()
    await engine.initialize()
    load_seconds = time.perf_counter() - started_load
    results: list[RunResult] = []

    try:
        for run_index in range(args.repeat):
            print(f"\n=== run {run_index + 1}/{args.repeat} ===", flush=True)
            result = await run_with_engine(args, engine)
            if run_index == 0:
                result.load_seconds = load_seconds
            results.append(result)
            print(
                f"decode={result.decode_tps:.3f} tok/s total={result.total_tps:.3f} tok/s "
                f"gen={result.elapsed_seconds:.1f}s",
                flush=True,
            )
    finally:
        await engine.cleanup()

    best = max(results, key=lambda item: item.decode_tps)
    print("\nBest loaded-engine run:")
    print(json.dumps(asdict(best), indent=2))


def _build_worker_cmd(args: argparse.Namespace, worker_idx: int) -> list[str]:
    cmd: list[str] = [
        "python",
        "reap_infer.py",
        "--model",
        str(args.model),
        "--ctx",
        str(args.ctx),
        "--threads",
        str(args.threads),
        "--threads-batch",
        str(args.threads_batch),
        "--batch",
        str(args.batch),
        "--ubatch",
        str(args.ubatch),
        "--cache-k",
        args.cache_k,
        "--cache-v",
        args.cache_v,
        "--gpu-layers",
        str(args.gpu_layers),
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--top-k",
        str(args.top_k),
        "--prompt",
        args.prompt,
        "--cpu-mask",
        args.cpu_mask,
        "--nice",
        str(args.nice),
        "--repeat",
        str(args.repeat),
    ]
    if not args.no_flash_attn:
        pass
    else:
        cmd.append("--no-flash-attn")
    if args.prompt_lookup:
        cmd.append("--prompt-lookup")
        cmd.extend(
            [
                "--prompt-lookup-ngram",
                str(args.prompt_lookup_ngram),
                "--prompt-lookup-tokens",
                str(args.prompt_lookup_tokens),
            ]
        )
    if args.numa:
        cmd.append("--numa")
    if args.no_offload_kqv:
        cmd.append("--no-offload-kqv")
    if args.verbose:
        cmd.append("--verbose")
    if args.ionice:
        cmd.append("--ionice")
    else:
        cmd.append("--no-ionice")
    if args.prefetch:
        cmd.append("--prefetch")
    else:
        cmd.append("--no-prefetch")
    if args.full_throttle:
        cmd.append("--full-throttle")
    cmd.extend(["--worker-label", f"worker-{worker_idx}"])
    return cmd


def run_workers(args: argparse.Namespace) -> None:
    env = os.environ.copy()
    procs: list[subprocess.Popen[str]] = []
    try:
        for idx in range(args.workers):
            cmd = _build_worker_cmd(args, idx + 1)
            procs.append(subprocess.Popen(cmd, text=True, env=env))
        for proc in procs:
            proc.wait()
    except KeyboardInterrupt:
        for proc in procs:
            with contextlib.suppress(Exception):
                proc.send_signal(signal.SIGINT)
    finally:
        for proc in procs:
            if proc.poll() is None:
                with contextlib.suppress(Exception):
                    proc.terminate()


def parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


async def run_sweep(args: argparse.Namespace) -> None:
    results: list[RunResult] = []
    base_threads = args.threads
    base_batch = args.batch
    base_ubatch = args.ubatch

    for threads in parse_int_list(args.sweep_threads):
        for batch in parse_int_list(args.sweep_batches):
            for ubatch in parse_int_list(args.sweep_ubatches):
                args.threads = threads
                args.batch = batch
                args.ubatch = ubatch
                print(f"\n=== threads={threads} batch={batch} ubatch={ubatch} ===", flush=True)
                result = await run_once(args)
                results.append(result)
                print(
                    f"decode={result.decode_tps:.3f} tok/s total={result.total_tps:.3f} tok/s "
                    f"load={result.load_seconds:.1f}s gen={result.elapsed_seconds:.1f}s",
                    flush=True,
                )

    args.threads = base_threads
    args.batch = base_batch
    args.ubatch = base_ubatch

    best = max(results, key=lambda item: item.decode_tps)
    print("\nBest:")
    print(json.dumps(asdict(best), indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run/tune local MiniMax-M2.7 REAP GGUF")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--ctx", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--threads-batch", type=int, default=16)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--ubatch", type=int, default=128)
    parser.add_argument("--cache-k", default="q4_0")
    parser.add_argument("--cache-v", default="q4_0")
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-flash-attn", action="store_true")
    parser.add_argument("--prompt-lookup", action="store_true")
    parser.add_argument("--prompt-lookup-ngram", type=int, default=4)
    parser.add_argument("--prompt-lookup-tokens", type=int, default=12)
    parser.add_argument("--numa", action="store_true")
    parser.add_argument("--no-offload-kqv", action="store_true")
    parser.add_argument("--cpu-mask", default="0-15")
    parser.add_argument("--nice", type=int, default=-10)
    parser.add_argument("--ionice", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--full-throttle", action="store_true")
    parser.add_argument("--worker-label", default="")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--sweep-threads", default="6,8,10,12")
    parser.add_argument("--sweep-batches", default="64,128,256")
    parser.add_argument("--sweep-ubatches", default="32,64,128")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args()
    if args.full_throttle:
        args.threads = 16
        args.threads_batch = 16
        args.batch = max(args.batch, 512)
        args.ubatch = max(args.ubatch, 256)
        args.cpu_mask = "0-15"
        args.nice = -15
    if args.workers > 1:
        run_workers(args)
        return
    apply_process_tuning(args)

    if args.repeat > 1:
        asyncio.run(run_repeat(args))
        return

    if args.sweep:
        asyncio.run(run_sweep(args))
        return

    result = asyncio.run(run_once(args))
    print(result.output)
    print(
        f"\nload={result.load_seconds:.1f}s elapsed={result.elapsed_seconds:.1f}s "
        f"decode={result.decode_tps:.3f} tok/s total={result.total_tps:.3f} tok/s "
        f"tokens={result.completion_tokens}"
    )


if __name__ == "__main__":
    main()
