"""Tests for local model hosting and offline resolution."""

from pathlib import Path

from miniforge.models.registry import ModelRegistry


def test_register_local_gguf_model(tmp_path: Path) -> None:
    """Registered GGUF models should resolve without external services."""
    gguf = tmp_path / "MiniMax-M2.7-Q4_K_M.gguf"
    gguf.write_bytes(b"gguf")
    registry = ModelRegistry(tmp_path / "cache")

    hosted = registry.register_local_model("local/minimax", gguf)

    assert hosted.id == "local/minimax"
    assert hosted.backend == "llama_cpp"
    assert hosted.quantization == "Q4_K_M"
    assert registry.resolve_local_model("local/minimax", "Q4_K_M") == gguf.resolve()
    assert any("[Hosted] local/minimax" in model for model in registry.list_cached_models())


def test_resolve_model_from_custom_directory(tmp_path: Path) -> None:
    """Configured model roots should be searched for matching GGUF files."""
    model_root = tmp_path / "models"
    model_root.mkdir()
    gguf = model_root / "acme-test-Q5_K_M-00001-of-00002.gguf"
    gguf.write_bytes(b"gguf")
    registry = ModelRegistry(tmp_path / "cache")

    resolved = registry.resolve_local_model(
        "acme/test",
        "Q5_K_M",
        backend="llama_cpp",
        search_dirs=[model_root],
    )

    assert resolved == gguf


def test_register_transformers_directory(tmp_path: Path) -> None:
    """Local HF-style directories should be hostable with the transformers backend."""
    model_dir = tmp_path / "hf-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    registry = ModelRegistry(tmp_path / "cache")

    hosted = registry.register_local_model("local/hf", model_dir)

    assert hosted.backend == "transformers"
    assert hosted.format == "transformers"
    assert registry.resolve_local_model("local/hf", backend="transformers") == model_dir.resolve()
