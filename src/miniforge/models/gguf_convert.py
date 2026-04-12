"""Convert HuggingFace SafeTensors weights to GGUF via llama.cpp's convert_hf_to_gguf.py."""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from miniforge.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


def is_hf_weights_directory(path: Path) -> bool:
    if not (path / "config.json").is_file():
        return False
    if list(path.glob("*.safetensors")):
        return True
    if list(path.glob("model*.safetensors")):
        return True
    return (path / "pytorch_model.bin").is_file()


def resolve_llama_cpp_root(explicit: str | Path | None = None) -> Path | None:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if (p / "convert_hf_to_gguf.py").is_file():
            return p
        return None
    for key in ("MINIFORGE_LLAMA_CPP", "LLAMA_CPP_PATH", "LLAMA_CPP_ROOT"):
        raw = os.environ.get(key)
        if not raw:
            continue
        p = Path(raw).expanduser().resolve()
        if (p / "convert_hf_to_gguf.py").is_file():
            return p
    return None


def find_llama_quantize_binary(llama_root: Path) -> str | None:
    found = shutil.which("llama-quantize") or shutil.which("llama-quantize.exe")
    if found:
        return found
    for rel in (
        ("build", "bin", "llama-quantize"),
        ("build", "bin", "llama-quantize.exe"),
        ("build", "bin", "Release", "llama-quantize.exe"),
        ("build", "bin", "Debug", "llama-quantize.exe"),
    ):
        cand = llama_root.joinpath(*rel)
        if cand.is_file():
            return str(cand)
    return None


def _run_hf_to_gguf(convert_script: Path, model_dir: Path, outfile: Path, outtype: str) -> None:
    outfile.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(convert_script),
        str(model_dir),
        "--outfile",
        str(outfile),
        "--outtype",
        outtype,
    ]
    logger.info("Running llama.cpp convert_hf_to_gguf (can take several minutes)...")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "convert_hf_to_gguf failed:\n"
            f"{proc.stdout}\n{proc.stderr}"
        )


def auto_convert_safetensors_to_gguf(
    registry: ModelRegistry,
    model_id: str,
    quantization: str,
    *,
    llama_cpp_root: str | Path | None = None,
) -> Path:
    """
    Produce a GGUF file from a HF repo or local weights dir, cached under registry.gguf_dir.

    Requires a llama.cpp checkout: set MINIFORGE_LLAMA_CPP or pass llama_cpp_root.
    For non-F16 targets, llama-quantize must be on PATH or under llama_root/build/.
    """
    hit = registry.get_cached_gguf_path(model_id, quantization)
    if hit:
        return hit

    local = Path(model_id).expanduser()
    if local.is_dir() and is_hf_weights_directory(local):
        repo_path = local.resolve()
    else:
        repo_path = registry.download_hf_model(model_id)

    found_gguf = registry.find_gguf_in_repo(repo_path)
    if found_gguf:
        return registry.register_gguf(model_id, quantization, found_gguf)

    if not is_hf_weights_directory(repo_path):
        raise ValueError(f"No HuggingFace safetensors layout at {repo_path}")

    root = resolve_llama_cpp_root(llama_cpp_root)
    if root is None:
        raise RuntimeError(
            "SafeTensors→GGUF needs a llama.cpp tree with convert_hf_to_gguf.py. "
            "Clone https://github.com/ggml-org/llama.cpp and set MINIFORGE_LLAMA_CPP "
            "to that directory (or pass llama_cpp_root in config)."
        )

    convert_script = root / "convert_hf_to_gguf.py"
    stem = registry.model_id_to_cache_stem(model_id)
    gguf_dir = registry.gguf_dir
    final_path = gguf_dir / f"{stem}_{quantization}.gguf"
    quant_upper = quantization.upper()

    if quant_upper == "F16":
        _run_hf_to_gguf(convert_script, repo_path, final_path, "f16")
        logger.info("Wrote GGUF: %s", final_path)
        return final_path

    f16_path = gguf_dir / f"{stem}_F16.gguf"
    if not f16_path.is_file():
        _run_hf_to_gguf(convert_script, repo_path, f16_path, "f16")
    else:
        logger.info("Reusing existing F16 GGUF: %s", f16_path)

    quant_bin = find_llama_quantize_binary(root)
    if quant_bin is None:
        logger.warning(
            "llama-quantize not found; loading F16 GGUF instead of %s (larger on disk).",
            quantization,
        )
        return f16_path

    logger.info("Quantizing to %s via %s", quantization, quant_bin)
    proc = subprocess.run(
        [quant_bin, str(f16_path), str(final_path), quantization],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"llama-quantize failed:\n{proc.stdout}\n{proc.stderr}"
        )

    with contextlib.suppress(OSError):
        f16_path.unlink()

    logger.info("Wrote GGUF: %s", final_path)
    return final_path
