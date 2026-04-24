"""Tests for hardware auto-detection."""

from miniforge.utils.hardware import (
    CpuInfo,
    GpuInfo,
    HardwareProfile,
    auto_config,
    detect_cpu,
    detect_gpus,
    detect_hardware,
    detect_os,
    recommend_optimizations,
)


def test_cpu_info_defaults() -> None:
    """Test CpuInfo dataclass defaults."""
    cpu = CpuInfo()
    assert cpu.physical_cores == 4
    assert cpu.logical_cores == 8
    assert cpu.vendor == "unknown"
    assert cpu.has_avx is False


def test_cpu_info_flags() -> None:
    """Test CPU flag detection."""
    cpu = CpuInfo(flags=["avx", "avx2", "fma", "f16c", "avx512f"])
    assert cpu.has_avx is True
    assert cpu.has_avx2 is True
    assert cpu.has_avx512 is True
    assert cpu.has_fma is True
    assert cpu.has_f16c is True


def test_gpu_info() -> None:
    """Test GpuInfo dataclass."""
    gpu = GpuInfo(name="RTX 4090", vendor="NVIDIA", vram_gb=24.0)
    assert gpu.vendor == "NVIDIA"
    assert gpu.vram_gb == 24.0


def test_hardware_profile() -> None:
    """Test HardwareProfile properties."""
    profile = HardwareProfile(
        cpu=CpuInfo(physical_cores=8, logical_cores=16),
        total_ram_gb=32.0,
        gpus=[GpuInfo(name="Radeon 680M", vendor="AMD", vram_gb=4.0)],
    )
    assert profile.has_igpu is True
    assert profile.has_dgpu is False


def test_detect_os() -> None:
    """Test OS detection returns expected tuple."""
    os_name, is_wsl, is_linux, is_windows, is_darwin = detect_os()
    assert isinstance(os_name, str)
    assert isinstance(is_wsl, bool)
    assert isinstance(is_linux, bool)
    assert isinstance(is_windows, bool)
    assert isinstance(is_darwin, bool)
    # Only one should be true (or linux + wsl)
    assert sum([is_linux, is_windows, is_darwin]) >= 1 or is_wsl


def test_detect_cpu() -> None:
    """Test CPU detection returns valid CpuInfo."""
    cpu = detect_cpu()
    assert cpu.physical_cores >= 1
    assert cpu.logical_cores >= cpu.physical_cores
    assert isinstance(cpu.flags, list)


def test_detect_gpus() -> None:
    """Test GPU detection returns a list."""
    gpus = detect_gpus()
    assert isinstance(gpus, list)
    for gpu in gpus:
        assert isinstance(gpu, GpuInfo)


def test_detect_hardware() -> None:
    """Test full hardware detection."""
    profile = detect_hardware()
    assert isinstance(profile, HardwareProfile)
    assert profile.total_ram_gb > 0
    assert profile.cpu.physical_cores >= 1
    assert isinstance(profile.gpus, list)


def test_auto_config_returns_dict() -> None:
    """Test auto_config produces a valid config dict."""
    profile = HardwareProfile(
        cpu=CpuInfo(physical_cores=8, logical_cores=16),
        total_ram_gb=32.0,
        gpus=[],
    )
    config = auto_config(model_params_b=2.7, is_moe=False, profile=profile)
    assert isinstance(config, dict)
    assert "n_threads" in config
    assert "n_batch" in config
    assert "quantization" in config
    assert config["n_threads"] > 0


def test_auto_config_wsl() -> None:
    """Test WSL-specific adjustments."""
    profile = HardwareProfile(
        cpu=CpuInfo(physical_cores=8, logical_cores=16),
        total_ram_gb=28.0,
        is_wsl=True,
        is_linux=True,
    )
    config = auto_config(model_params_b=2.7, is_moe=True, profile=profile)
    assert config["use_mlock"] is False
    assert config["use_mmap"] is True
    assert config["is_moe"] is True


def test_auto_config_memory_presets() -> None:
    """Test batch sizes scale with RAM."""
    small = HardwareProfile(total_ram_gb=8.0, cpu=CpuInfo())
    large = HardwareProfile(total_ram_gb=64.0, cpu=CpuInfo())

    cfg_small = auto_config(profile=small)
    cfg_large = auto_config(profile=large)

    assert cfg_small["n_batch"] < cfg_large["n_batch"]


def test_recommend_optimizations_explains_settings() -> None:
    """Optimization reports should include config, tier, and explanations."""
    profile = HardwareProfile(
        cpu=CpuInfo(physical_cores=8, logical_cores=16, flags=["avx", "avx2"]),
        total_ram_gb=32.0,
        available_ram_gb=20.0,
        gpus=[GpuInfo(name="RTX 4060", vendor="NVIDIA", vram_gb=8.0)],
    )

    report = recommend_optimizations(model_params_b=7.0, profile=profile)

    assert report.tier == "balanced"
    assert report.settings["n_threads"] == 12
    assert report.settings["cpu_mask"] == "0-15"
    assert report.settings["n_gpu_layers"] > 0
    assert report.reasons
    assert report.to_dict()["settings"] == report.settings
