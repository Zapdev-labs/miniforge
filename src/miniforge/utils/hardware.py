"""Hardware auto-detection for Miniforge.

Probes CPU capabilities, RAM, GPU, and OS environment to produce an
M7Config tuned to the actual machine rather than hard-coded M7 defaults.
"""

from __future__ import annotations

import contextlib
import ctypes
import logging
import platform
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil

logger = logging.getLogger(__name__)


@dataclass
class CpuInfo:
    """Detected CPU characteristics."""

    vendor: str = "unknown"
    brand: str = "unknown"
    physical_cores: int = 4
    logical_cores: int = 8
    max_freq_mhz: float = 0.0
    flags: list[str] = field(default_factory=list)

    @property
    def has_avx(self) -> bool:
        return "avx" in self.flags

    @property
    def has_avx2(self) -> bool:
        return "avx2" in self.flags

    @property
    def has_avx512(self) -> bool:
        return any(f.startswith("avx512") for f in self.flags)

    @property
    def has_fma(self) -> bool:
        return "fma" in self.flags

    @property
    def has_f16c(self) -> bool:
        return "f16c" in self.flags


@dataclass
class GpuInfo:
    """Detected GPU characteristics."""

    name: str = "unknown"
    vendor: str = "unknown"
    vram_gb: float = 0.0
    device_id: int = 0


@dataclass
class HardwareProfile:
    """Complete hardware profile."""

    cpu: CpuInfo = field(default_factory=CpuInfo)
    total_ram_gb: float = 16.0
    available_ram_gb: float = 8.0
    gpus: list[GpuInfo] = field(default_factory=list)
    os_name: str = "unknown"
    is_wsl: bool = False
    is_linux: bool = False
    is_windows: bool = False
    is_darwin: bool = False

    @property
    def has_igpu(self) -> bool:
        """Check if an integrated GPU is present."""
        return any(gpu.vendor in ("AMD", "Intel") and gpu.vram_gb <= 8.0 for gpu in self.gpus)

    @property
    def has_dgpu(self) -> bool:
        """Check if a discrete GPU is present."""
        return any(gpu.vram_gb > 8.0 for gpu in self.gpus)


def _read_cpu_flags_linux() -> list[str]:
    """Read CPU flags from /proc/cpuinfo on Linux."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    return line.split(":", 1)[1].strip().split()
    except Exception:
        pass
    return []


def _read_cpu_flags_windows() -> list[str]:
    """Detect CPU feature flags on Windows via CPUID-like heuristics."""
    flags: list[str] = []
    try:
        # IsProcessorFeaturePresent constants
        PF_AVX_INSTRUCTIONS_AVAILABLE = 39  # noqa: N806
        PF_AVX2_INSTRUCTIONS_AVAILABLE = 40  # noqa: N806
        PF_AVX512F_INSTRUCTIONS_AVAILABLE = 41  # noqa: N806

        kernel32 = ctypes.windll.kernel32
        if kernel32.IsProcessorFeaturePresent(PF_AVX_INSTRUCTIONS_AVAILABLE):
            flags.append("avx")
        if kernel32.IsProcessorFeaturePresent(PF_AVX2_INSTRUCTIONS_AVAILABLE):
            flags.append("avx2")
        if kernel32.IsProcessorFeaturePresent(PF_AVX512F_INSTRUCTIONS_AVAILABLE):
            flags.append("avx512f")
    except Exception:
        pass
    return flags


def _read_cpu_flags_darwin() -> list[str]:
    """Detect CPU feature flags on macOS."""
    flags: list[str] = []
    try:
        result = subprocess.run(
            ["sysctl", "-a"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            output = result.stdout.lower()
            if "avx1.0" in output or "avx" in output:
                flags.append("avx")
            if "avx2" in output:
                flags.append("avx2")
            if "avx512f" in output:
                flags.append("avx512f")
            if "fma" in output:
                flags.append("fma")
            if "f16c" in output:
                flags.append("f16c")
    except Exception:
        pass
    return flags


def detect_cpu() -> CpuInfo:
    """Detect CPU information."""
    info = CpuInfo()

    # Core counts
    info.physical_cores = psutil.cpu_count(logical=False) or 4
    info.logical_cores = psutil.cpu_count(logical=True) or 8

    # Frequency
    try:
        freq = psutil.cpu_freq()
        if freq:
            info.max_freq_mhz = freq.max or freq.current or 0.0
    except Exception:
        pass

    # OS-specific flag detection
    if sys.platform.startswith("linux"):
        info.flags = _read_cpu_flags_linux()
        try:
            with open("/proc/cpuinfo") as f:
                content = f.read()
            vendor_match = re.search(r"vendor_id\s*:\s*(.+)", content)
            if vendor_match:
                info.vendor = vendor_match.group(1).strip()
            name_match = re.search(r"model name\s*:\s*(.+)", content)
            if name_match:
                info.brand = name_match.group(1).strip()
        except Exception:
            pass
    elif sys.platform == "win32":
        info.flags = _read_cpu_flags_windows()
        with contextlib.suppress(Exception):
            info.brand = platform.processor()
    elif sys.platform == "darwin":
        info.flags = _read_cpu_flags_darwin()
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                info.brand = result.stdout.strip()
        except Exception:
            pass

    # Normalize flags
    info.flags = [f.lower() for f in info.flags]
    logger.info(
        "CPU detected: %s, %d cores (%d threads), flags=%s",
        info.brand,
        info.physical_cores,
        info.logical_cores,
        info.flags,
    )
    return info


def _detect_nvidia_gpus() -> list[GpuInfo]:
    """Detect NVIDIA GPUs via nvidia-smi."""
    gpus: list[GpuInfo] = []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                parts = line.split(",")
                if len(parts) >= 2:
                    name = parts[0].strip()
                    mem_str = parts[1].strip()
                    vram_mb = 0
                    if "MiB" in mem_str:
                        m = re.search(r"\d+", mem_str)
                        if m:
                            vram_mb = int(m.group())
                    elif "GB" in mem_str:
                        m = re.search(r"[\d.]+", mem_str)
                        if m:
                            vram_mb = int(float(m.group()) * 1024)
                    gpus.append(
                        GpuInfo(
                            name=name,
                            vendor="NVIDIA",
                            vram_gb=vram_mb / 1024,
                        )
                    )
    except Exception:
        pass
    return gpus


def _detect_amd_gpus_linux() -> list[GpuInfo]:
    """Detect AMD GPUs on Linux via rocminfo or lspci."""
    gpus: list[GpuInfo] = []
    # Try rocminfo first
    try:
        result = subprocess.run(
            ["rocminfo"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            # Very basic parsing
            names = re.findall(r"Name:\s+(.*AMD.*|.*Radeon.*)", result.stdout)
            for name in names:
                gpus.append(GpuInfo(name=name.strip(), vendor="AMD", vram_gb=0.0))
    except Exception:
        pass

    # Fallback to lspci
    if not gpus:
        try:
            result = subprocess.run(
                ["lspci"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if ("VGA" in line or "3D controller" in line) and (
                        "AMD" in line or "ATI" in line
                    ):
                        gpus.append(
                            GpuInfo(
                                name=line.split(":")[-1].strip(),
                                vendor="AMD",
                                vram_gb=0.0,
                            )
                        )
        except Exception:
            pass
    return gpus


def _detect_amd_gpus_windows() -> list[GpuInfo]:
    """Detect AMD GPUs on Windows via WMI or registry."""
    gpus: list[GpuInfo] = []
    try:
        import wmi  # type: ignore[import-not-found]

        c = wmi.WMI()
        for gpu in c.Win32_VideoController():
            if gpu.AdapterCompatibility and "AMD" in gpu.AdapterCompatibility:
                vram_mb = getattr(gpu, "AdapterRAM", 0) // (1024 * 1024)
                gpus.append(
                    GpuInfo(
                        name=gpu.Name or "AMD GPU",
                        vendor="AMD",
                        vram_gb=vram_mb / 1024,
                    )
                )
    except Exception:
        pass
    return gpus


def detect_gpus() -> list[GpuInfo]:
    """Detect available GPUs."""
    gpus: list[GpuInfo] = []

    if sys.platform == "win32":
        gpus.extend(_detect_nvidia_gpus())
        gpus.extend(_detect_amd_gpus_windows())
    else:
        gpus.extend(_detect_nvidia_gpus())
        gpus.extend(_detect_amd_gpus_linux())

    # Deduplicate by name
    seen: set = set()
    unique: list[GpuInfo] = []
    for gpu in gpus:
        key = f"{gpu.vendor}:{gpu.name}"
        if key not in seen:
            seen.add(key)
            unique.append(gpu)

    logger.info("GPUs detected: %s", [f"{g.vendor} {g.name} ({g.vram_gb:.1f}GB)" for g in unique])
    return unique


def detect_os() -> tuple[str, bool, bool, bool, bool]:
    """Detect OS and WSL status.

    Returns (os_name, is_wsl, is_linux, is_windows, is_darwin)
    """
    is_linux = sys.platform.startswith("linux")
    is_windows = sys.platform == "win32"
    is_darwin = sys.platform == "darwin"
    os_name = platform.system()

    is_wsl = False
    if is_linux:
        try:
            with open("/proc/version") as f:
                version = f.read().lower()
                is_wsl = "microsoft" in version or "wsl" in version
        except Exception:
            pass
        try:
            if not is_wsl:
                is_wsl = Path("/proc/sys/fs/binfmt_misc/WSLInterop").exists()
        except Exception:
            pass

    logger.info(
        "OS: %s (wsl=%s, linux=%s, windows=%s, darwin=%s)",
        os_name,
        is_wsl,
        is_linux,
        is_windows,
        is_darwin,
    )
    return os_name, is_wsl, is_linux, is_windows, is_darwin


def detect_hardware() -> HardwareProfile:
    """Detect full hardware profile."""
    profile = HardwareProfile()

    os_name, is_wsl, is_linux, is_windows, is_darwin = detect_os()
    profile.os_name = os_name
    profile.is_wsl = is_wsl
    profile.is_linux = is_linux
    profile.is_windows = is_windows
    profile.is_darwin = is_darwin

    profile.cpu = detect_cpu()

    mem = psutil.virtual_memory()
    profile.total_ram_gb = mem.total / (1024**3)
    profile.available_ram_gb = mem.available / (1024**3)

    profile.gpus = detect_gpus()

    logger.info(
        "Hardware profile: RAM=%.1fGB, CPU=%s, GPUs=%d",
        profile.total_ram_gb,
        profile.cpu.brand,
        len(profile.gpus),
    )
    return profile


def _infer_quantization(ram_gb: float, model_params_b: float, is_moe: bool = False) -> str:
    """Infer best quantization given RAM and model size."""
    # Rough FP16 size in GB
    fp16_gb = model_params_b * 2.0
    if is_moe:
        # MoE models use mmap heavily; we can tolerate larger disk size
        # but active params matter. Rough heuristic: treat as ~30% of fp16_gb
        fp16_gb = model_params_b * 0.6

    # Reserve OS + working memory
    usable = max(ram_gb * 0.75, 4.0)

    # Quant ratios vs FP16
    candidates = [
        ("Q8_0", 1.0),
        ("Q6_K", 0.75),
        ("Q5_K_M", 0.625),
        ("Q4_K_M", 0.5),
        ("Q3_K_M", 0.375),
        ("UD-IQ2_XXS", 0.25),
    ]

    for quant, ratio in candidates:
        model_gb = fp16_gb * ratio
        # Rough KV overhead for 32k context
        kv_gb = 1.5
        if model_gb + kv_gb <= usable:
            return quant

    return "UD-IQ2_XXS"


def auto_config(
    model_params_b: float = 2.7,
    is_moe: bool = False,
    profile: HardwareProfile | None = None,
) -> dict[str, Any]:
    """Generate configuration dict tuned to detected hardware.

    Returns a dict that can be splatted into M7Config(**auto_config(...)).
    """
    if profile is None:
        profile = detect_hardware()

    config: dict[str, Any] = {}

    # Threads: physical cores for compute-bound inference
    config["n_threads"] = max(1, profile.cpu.physical_cores)
    # On systems with SMT, we can use logical cores for batch work
    if profile.cpu.logical_cores > profile.cpu.physical_cores:
        config["n_threads"] = max(1, profile.cpu.physical_cores + profile.cpu.physical_cores // 2)

    # RAM-based safety
    total_ram = profile.total_ram_gb
    config["max_memory_gb"] = max(4.0, total_ram - 4.0)
    config["reserve_memory_gb"] = min(4.0, total_ram * 0.15)

    # Batch sizes scale with available RAM
    if total_ram >= 64:
        config["n_batch"] = 4096
        config["n_ubatch"] = 1024
    elif total_ram >= 32:
        config["n_batch"] = 2048
        config["n_ubatch"] = 512
    elif total_ram >= 16:
        config["n_batch"] = 1024
        config["n_ubatch"] = 512
    else:
        config["n_batch"] = 512
        config["n_ubatch"] = 256

    # Context window: aim high but safe
    if total_ram >= 64:
        config["n_ctx"] = 194_560
    elif total_ram >= 32:
        config["n_ctx"] = 98_304
    else:
        config["n_ctx"] = 32_768

    # Quantization
    config["quantization"] = _infer_quantization(total_ram, model_params_b, is_moe)

    # KV cache: q4_0 is the safest aggressive default
    config["cache_type_k"] = "q4_0"
    config["cache_type_v"] = "q4_0"

    # CPU ISA
    config["use_avx"] = profile.cpu.has_avx
    config["use_avx2"] = profile.cpu.has_avx2
    config["use_avx512"] = profile.cpu.has_avx512
    config["use_fma"] = profile.cpu.has_fma
    config["use_f16c"] = profile.cpu.has_f16c

    # OS-specific tweaks
    if profile.is_wsl:
        config["use_mlock"] = False
        config["use_mmap"] = True
        # WSL2 memory reclaim can be aggressive; be conservative
        config["max_memory_gb"] = max(4.0, total_ram * 0.65)
    elif profile.is_linux:
        config["use_mlock"] = False  # Most users won't have unlimited memlock
        config["use_mmap"] = True
    elif profile.is_windows or profile.is_darwin:
        config["use_mlock"] = False
        config["use_mmap"] = True

    # GPU offloading
    config["n_gpu_layers"] = 0
    if profile.gpus:
        for gpu in profile.gpus:
            if gpu.vendor == "NVIDIA" and gpu.vram_gb >= 4.0:
                config["n_gpu_layers"] = min(20, int(gpu.vram_gb * 3))
                break
            elif gpu.vendor == "AMD" and gpu.vram_gb >= 2.0:
                config["n_gpu_layers"] = min(15, int(gpu.vram_gb * 3))
                break

    # Priority
    config["priority"] = "normal"
    if not profile.is_windows:
        config["priority"] = "high"

    # MoE
    if is_moe:
        config["is_moe"] = True
        config["use_mmap"] = True
        config["use_mlock"] = False
        config["n_batch"] = min(config.get("n_batch", 2048), 512)
        config["n_ubatch"] = min(config.get("n_ubatch", 512), 256)

    logger.info("Auto-config: %s", config)
    return config
