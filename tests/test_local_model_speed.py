"""Opt-in local GGUF speed test.

Run with:
  MINIFORGE_RUN_LOCAL_MODEL_SPEED_TEST=1 uv run pytest tests/test_local_model_speed.py -s

Optional overrides:
  MINIFORGE_SPEED_TEST_MODEL=/path/to/model-or-dir
  MINIFORGE_SPEED_TEST_GPU_LAYERS=999
  MINIFORGE_SPEED_TEST_AB=1
"""

from __future__ import annotations

import importlib.util
import os
import time
from pathlib import Path
from typing import Any

import pytest

DEFAULT_FAST_MODEL = Path(
    "/run/media/dih/8CEDA5F938E73A48/AI/unsloth-nvidia-nemotron-3-nano-4b/"
    "NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf"
)


def _find_gguf(path: Path) -> Path:
    if path.is_file() and path.suffix.lower() == ".gguf":
        return path

    ggufs = sorted(path.glob("*.gguf"))
    if not ggufs:
        raise FileNotFoundError(f"No .gguf files found under {path}")
    return ggufs[0]


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


def _bool_env(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _run_variant(
    model_path: Path,
    *,
    label: str,
    threads: int,
    ctx: int,
    batch: int,
    ubatch: int,
    max_tokens: int,
    gpu_layers: int,
    use_mmap: bool,
    cache_type_k: str | None,
    cache_type_v: str | None,
) -> dict[str, float]:
    from llama_cpp import Llama

    load_start = time.perf_counter()
    kwargs: dict[str, Any] = {
        "model_path": str(model_path),
        "n_ctx": ctx,
        "n_threads": threads,
        "n_batch": batch,
        "n_ubatch": ubatch,
        "n_gpu_layers": gpu_layers,
        "flash_attn": True,
        "use_mmap": use_mmap,
        "use_mlock": False,
        "verbose": False,
    }
    if cache_type_k is not None:
        kwargs["type_k"] = cache_type_k
    if cache_type_v is not None:
        kwargs["type_v"] = cache_type_v
    llm = Llama(**kwargs)
    load_seconds = time.perf_counter() - load_start

    prompt = "Q: Name three practical ways to speed up local GGUF inference.\nA:"
    run_start = time.perf_counter()
    result = llm.create_completion(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        stop=["\nQ:"],
    )
    elapsed = time.perf_counter() - run_start

    usage = result.get("usage", {})
    assert isinstance(usage, dict)
    completion_tokens = int(usage.get("completion_tokens", 0))
    total_tokens = int(usage.get("total_tokens", completion_tokens))
    decode_tps = completion_tokens / elapsed if elapsed > 0 else 0.0
    total_tps = total_tokens / elapsed if elapsed > 0 else 0.0

    print(
        f"\n[{label}]"
        f"\nload_seconds={load_seconds:.2f}"
        f"\nctx={ctx} threads={threads} batch={batch} ubatch={ubatch} gpu_layers={gpu_layers}"
        f"\nuse_mmap={use_mmap} type_k={cache_type_k} type_v={cache_type_v}"
        f"\ncompletion_tokens={completion_tokens} total_tokens={total_tokens}"
        f"\ndecode_tps={decode_tps:.2f} total_tps={total_tps:.2f}"
    )

    return {
        "completion_tokens": float(completion_tokens),
        "decode_tps": decode_tps,
        "total_tps": total_tps,
        "load_seconds": load_seconds,
    }


@pytest.mark.skipif(
    os.environ.get("MINIFORGE_RUN_LOCAL_MODEL_SPEED_TEST") != "1",
    reason="opt-in local benchmark; set MINIFORGE_RUN_LOCAL_MODEL_SPEED_TEST=1",
)
def test_local_gguf_runs_as_fast_as_possible() -> None:
    """Run a short real-model decode and print throughput diagnostics."""
    if importlib.util.find_spec("llama_cpp") is None:
        pytest.skip("llama-cpp-python is not installed")

    model_path = _find_gguf(Path(os.environ.get("MINIFORGE_SPEED_TEST_MODEL", DEFAULT_FAST_MODEL)))
    if not model_path.exists():
        pytest.skip(f"Local model does not exist: {model_path}")

    threads = _int_env("MINIFORGE_SPEED_TEST_THREADS", os.cpu_count() or 8)
    ctx = _int_env("MINIFORGE_SPEED_TEST_CTX", 2048)
    batch = _int_env("MINIFORGE_SPEED_TEST_BATCH", 1024)
    ubatch = _int_env("MINIFORGE_SPEED_TEST_UBATCH", 512)
    max_tokens = _int_env("MINIFORGE_SPEED_TEST_MAX_TOKENS", 96)
    gpu_layers = _int_env("MINIFORGE_SPEED_TEST_GPU_LAYERS", 0)

    print(f"\nlocal GGUF speed test\nmodel={model_path}")
    baseline = _run_variant(
        model_path,
        label="baseline",
        threads=threads,
        ctx=ctx,
        batch=batch,
        ubatch=ubatch,
        max_tokens=max_tokens,
        gpu_layers=gpu_layers,
        use_mmap=False,
        cache_type_k=None,
        cache_type_v=None,
    )
    if _bool_env("MINIFORGE_SPEED_TEST_AB"):
        tuned = _run_variant(
            model_path,
            label="tuned",
            threads=threads,
            ctx=ctx,
            batch=batch,
            ubatch=ubatch,
            max_tokens=max_tokens,
            gpu_layers=gpu_layers,
            use_mmap=True,
            cache_type_k="q8_0",
            cache_type_v="q8_0",
        )
        if baseline["decode_tps"] > 0:
            uplift = ((tuned["decode_tps"] - baseline["decode_tps"]) / baseline["decode_tps"]) * 100
            print(f"\n[A/B] decode_tps_uplift_percent={uplift:.2f}")

    assert baseline["completion_tokens"] > 0
    assert baseline["decode_tps"] > 0.0
